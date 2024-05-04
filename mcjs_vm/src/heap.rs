use std::collections::HashMap;
use std::ops::Deref;
use std::{borrow::Cow, cell::RefCell};

use crate::interpreter::{Closure, Value};

/// Common interface implemented by types that are sufficently "object-like" to be exposed
/// as objects in JavaScript code.   This includes all objects: ordinary and exotic,
/// immediate and heap-stored.
///
/// This interface only allows access to the object's "own" properties, without using the
/// prototype.  For that, you need use a `Heap` and its methods.
// TODO Will this design work even though this trait is not object-safe?
pub trait Object {
    /// Get an owned property (or array element) of this object.
    ///
    /// If you want to include inherited properties in the lookup, use
    /// `Heap::get_property_chained` instead.
    fn get_own(&self, index_or_key: IndexOrKey) -> Option<Property>;

    /// Set the value of an owned property (or an array element).
    fn set_own(&mut self, index_or_key: IndexOrKey, value: Property);

    /// Remove an owned property (or array element) of this object.
    ///
    /// This does NOT affect inherited properties and does NOT access
    /// the prototype chain in any way.
    fn delete_own(&mut self, index_or_key: IndexOrKey);

    // I know, inefficient, but so, *so* simple
    fn own_properties(&self, only_enumerable: bool, out: &mut Vec<String>);

    fn type_of(&self) -> Typeof;
    fn proto(&self, heap: &Heap) -> Option<ObjectId>;
    fn set_proto(&mut self, proto_id: Option<ObjectId>);
}

#[derive(Clone, Copy)]
pub enum Typeof {
    Object,
    Function,
    String,
    Number,
    Boolean,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IndexOrKey<'a> {
    Index(usize),
    Key(&'a str),
}
impl<'a> IndexOrKey<'a> {
    fn ensure_number_is_index(self) -> Self {
        match self {
            IndexOrKey::Index(_) => self,
            IndexOrKey::Key(key) => {
                if let Ok(num_key) = key.parse() {
                    IndexOrKey::Index(num_key)
                } else {
                    self
                }
            }
        }
    }

    fn ensure_number_is_key(&self) -> Cow<'a, str> {
        match self {
            IndexOrKey::Index(num) => Cow::Owned(num.to_string()),
            IndexOrKey::Key(key) => Cow::Borrowed(key),
        }
    }
}
impl<'a> From<&'a str> for IndexOrKey<'a> {
    fn from(value: &'a str) -> Self {
        IndexOrKey::Key(value)
    }
}
impl<'a> From<usize> for IndexOrKey<'a> {
    fn from(value: usize) -> Self {
        IndexOrKey::Index(value)
    }
}

#[derive(PartialEq, Eq)]
pub enum IndexOrKeyOwned {
    Index(usize),
    Key(String),
}
impl IndexOrKeyOwned {
    pub fn to_ref(&self) -> IndexOrKey {
        match self {
            IndexOrKeyOwned::Index(ndx) => IndexOrKey::Index(*ndx),
            IndexOrKeyOwned::Key(key) => IndexOrKey::Key(key),
        }
    }
}

//
// Ordinary objects
//

slotmap::new_key_type! { pub struct ObjectId; }

pub struct Heap {
    objects: slotmap::SlotMap<ObjectId, RefCell<HeapObject>>,

    object_proto: ObjectId,
    number_proto: ObjectId,
    string_proto: ObjectId,
    func_proto: ObjectId,
    array_proto: ObjectId,
    bool_proto: ObjectId,
}

impl Heap {
    pub(crate) fn new() -> Heap {
        let new_ordinary = || HeapObject {
            proto_id: None,
            properties: HashMap::new(),
            order: Vec::new(),
            exotic_part: Exotic::None,
        };

        let mut objects = slotmap::SlotMap::with_key();
        let object_proto = objects.insert(RefCell::new(new_ordinary()));
        let number_proto = objects.insert(RefCell::new(new_ordinary()));
        let string_proto = objects.insert(RefCell::new(new_ordinary()));
        let func_proto = objects.insert(RefCell::new(new_ordinary()));
        let bool_proto = objects.insert(RefCell::new(new_ordinary()));
        let array_proto = objects.insert(RefCell::new(new_ordinary()));

        Heap {
            objects,
            object_proto,
            number_proto,
            string_proto,
            array_proto,
            func_proto,
            bool_proto,
        }
    }

    pub fn number_proto(&self) -> ObjectId {
        self.number_proto
    }
    pub fn func_proto(&self) -> ObjectId {
        self.func_proto
    }
    pub fn array_proto(&self) -> ObjectId {
        self.array_proto
    }
    // These are going to be used at some point. No use in delaying their writing
    #[allow(dead_code)]
    pub fn string_proto(&self) -> ObjectId {
        self.string_proto
    }
    #[allow(dead_code)]
    pub fn bool_proto(&self) -> ObjectId {
        self.bool_proto
    }

    pub(crate) fn new_ordinary_object(&mut self) -> ObjectId {
        self.objects.insert(RefCell::new(HeapObject {
            proto_id: Some(self.object_proto),
            properties: HashMap::new(),
            order: Vec::new(),
            exotic_part: Exotic::None,
        }))
    }

    // Weird property of these functions: their purpose is to *create* an exotic
    // object, but they actually *modify* an existing object. This is to align
    // them to the property of JavaScript constructors, which act on a
    // pre-created (ordinary) object passed as `this`.

    fn init_exotic(&mut self, oid: ObjectId, proto_oid: ObjectId, exotic_part: Exotic) {
        let mut obj = self.objects.get(oid).unwrap().borrow_mut();
        assert!(matches!(obj.exotic_part, Exotic::None));
        obj.proto_id = Some(proto_oid);
        obj.exotic_part = exotic_part;
    }

    pub(crate) fn init_array(&mut self, oid: ObjectId, elements: Vec<Value>) {
        self.init_exotic(oid, self.array_proto, Exotic::Array { elements });
    }
    pub(crate) fn init_function(&mut self, oid: ObjectId, closure: Closure) {
        self.init_exotic(oid, self.func_proto, Exotic::Function { closure });
    }
    pub(crate) fn init_string(&mut self, oid: ObjectId, string: String) {
        self.init_exotic(oid, self.string_proto, Exotic::String { string });
    }

    pub(crate) fn new_array(&mut self, elements: Vec<Value>) -> ObjectId {
        let oid = self.new_ordinary_object();
        self.init_array(oid, elements);
        oid
    }
    pub(crate) fn new_function(&mut self, closure: Closure) -> ObjectId {
        let oid = self.new_ordinary_object();
        self.init_function(oid, closure);
        oid
    }
    pub(crate) fn new_string(&mut self, string: String) -> ObjectId {
        let oid = self.new_ordinary_object();
        self.init_string(oid, string);
        oid
    }

    pub fn is_instance_of<O: ?Sized + Object>(&self, obj: &O, sup_oid: ObjectId) -> bool {
        let mut cur_proto = match obj.proto(self) {
            Some(proto_id) if proto_id == sup_oid => return true,
            Some(proto_id) => self.get(proto_id),
            None => return false,
        };

        #[cfg(test)]
        let mut trace = vec![];

        while let Some(proto) = cur_proto {
            let proto = proto.borrow();
            match proto.proto(self) {
                Some(proto_id) if proto_id == sup_oid => return true,
                Some(proto_id) => {
                    #[cfg(test)]
                    {
                        if trace.contains(&proto_id) {
                            panic!("Circular prototype chain!")
                        }
                        trace.push(proto_id);
                    }

                    cur_proto = self.get(proto_id);
                }
                None => return false,
            };
        }

        false
    }

    pub fn get(&self, oid: ObjectId) -> Option<&RefCell<HeapObject>> {
        self.objects.get(oid)
    }

    pub fn get_property_chained<O: ?Sized + Object>(&self, obj: &O, key: &str) -> Option<Property> {
        let value = obj.get_own(IndexOrKey::Key(key));

        if value.is_some() {
            value
        } else if let Some(proto_id) = obj.proto(self) {
            let proto = self.get(proto_id)?.borrow();
            self.get_property_chained(proto.deref(), key)
        } else {
            None
        }
    }

    pub fn list_properties_prototypes<O: ?Sized + Object>(
        &self,
        obj: &O,
        only_enumerable: bool,
        out: &mut Vec<String>,
    ) {
        if let Some(proto_oid) = obj.proto(self) {
            let proto = self.get(proto_oid).unwrap().borrow();
            proto.own_properties(only_enumerable, out);
            self.list_properties_prototypes(proto.deref(), only_enumerable, out);
        }
    }
}

/// An ordinary object, i.e. one that you can create in JavaScript with the `{a: 1, b: 2}`
/// syntax.
///
/// It stores key-value pairs where keys ("properties") are strings, and any JS value can
/// be stored as value. It also stores the order in which keys were inserted, so that they
/// are visited in the same order on iteration.
#[derive(Debug, Clone)]
pub struct HeapObject {
    proto_id: Option<ObjectId>,
    properties: HashMap<String, Property>,
    order: Vec<String>,
    exotic_part: Exotic,
}
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Property {
    pub value: Value,
    pub is_enumerable: bool,
}
impl Property {
    pub fn non_enumerable(value: Value) -> Property {
        Property {
            value,
            is_enumerable: false,
        }
    }
    pub fn enumerable(value: Value) -> Property {
        Property {
            value,
            is_enumerable: true,
        }
    }
}
impl HeapObject {
    pub fn as_str(&self) -> Option<&str> {
        match &self.exotic_part {
            Exotic::String { string } => Some(string),
            _ => None,
        }
    }

    pub fn as_closure(&self) -> Option<&Closure> {
        match &self.exotic_part {
            Exotic::Function { closure } => Some(closure),
            _ => None,
        }
    }

    pub fn array_elements(&self) -> Option<&[Value]> {
        match &self.exotic_part {
            Exotic::Array { elements } => Some(elements),
            _ => None,
        }
    }

    pub fn array_push(&mut self, value: Value) -> bool {
        match &mut self.exotic_part {
            Exotic::Array { elements } => {
                elements.push(value);
                true
            }
            _ => false,
        }
    }

    pub fn to_boolean(&self) -> bool {
        match &self.exotic_part {
            Exotic::None | Exotic::Array { .. } | Exotic::Function { .. } => true,
            Exotic::String { string } => !string.is_empty(),
        }
    }

    pub fn js_to_string(&self) -> String {
        match &self.exotic_part {
            Exotic::Array { .. } | Exotic::None => "<object>".to_owned(),
            Exotic::String { string } => string.clone(),
            Exotic::Function { .. } => "<closure>".to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
enum Exotic {
    /// An ordinary object
    None,
    Array {
        elements: Vec<Value>,
    },
    String {
        string: String,
    },
    Function {
        closure: Closure,
    },
}

impl Object for HeapObject {
    fn type_of(&self) -> Typeof {
        match self.exotic_part {
            Exotic::None => Typeof::Object,
            Exotic::Array { .. } => Typeof::Object,
            Exotic::String { .. } => Typeof::String,
            Exotic::Function { .. } => Typeof::Function,
        }
    }

    fn get_own(&self, index_or_key: IndexOrKey) -> Option<Property> {
        if index_or_key == IndexOrKey::Key("__proto__") {
            let proto_oid = self.proto_id?;
            return Some(Property::non_enumerable(Value::Object(proto_oid)));
        }

        match &self.exotic_part {
            Exotic::Array { elements } => match index_or_key.ensure_number_is_index() {
                IndexOrKey::Key("length") => Some(Property::non_enumerable(Value::Number(
                    elements.len() as f64,
                ))),
                IndexOrKey::Index(index) => elements.get(index).copied().map(Property::enumerable),
                _ => None,
            },
            Exotic::String { string } => match index_or_key {
                IndexOrKey::Key("length") => {
                    Some(Property::non_enumerable(Value::Number(string.len() as f64)))
                }
                _ => None,
            },
            Exotic::None | Exotic::Function { .. } => {
                let key = index_or_key.ensure_number_is_key();
                self.properties.get(key.as_ref()).copied()
            }
        }
    }
    fn set_own(&mut self, index_or_key: IndexOrKey, property: Property) {
        if index_or_key == IndexOrKey::Key("__proto__") {
            if let Value::Object(new_proto_id) = property.value {
                // property.is_enumerable is discarded. It's implicitly non-enumerable (see
                // `get_own`)
                self.proto_id = Some(new_proto_id);
            } else {
                // Apparently this is simply a nop in V8?
            }
            return;
        }

        match (&mut self.exotic_part, index_or_key) {
            (Exotic::None, IndexOrKey::Index(ndx)) => {
                let new_key = ndx.to_string();
                let new_key = IndexOrKey::Key(&new_key);
                self.set_own(new_key, property)
            }
            (Exotic::Array { elements }, IndexOrKey::Index(ndx)) => {
                if elements.len() < ndx + 1 {
                    elements.resize(ndx + 1, Value::Undefined);
                }
                // implicitly enumerable
                elements[ndx] = property.value;
            }
            (Exotic::String { .. } | Exotic::Function { .. }, IndexOrKey::Index(_)) => {
                // do nothing
            }
            (_, IndexOrKey::Key(key)) => {
                let prev = self.properties.insert(key.to_owned(), property);
                if prev.is_none() {
                    self.order.push(key.to_string());
                }
                debug_assert_eq!(1, self.order.iter().filter(|x| *x == key).count());
            }
        }
    }
    fn delete_own(&mut self, index_or_key: IndexOrKey) {
        if index_or_key == IndexOrKey::Key("__proto__") {
            return;
        }

        match (&mut self.exotic_part, index_or_key) {
            (Exotic::Array { .. } | Exotic::String { .. }, IndexOrKey::Key("length")) => {
                // do nothing
            }
            (Exotic::Array { elements }, IndexOrKey::Index(index)) => {
                if index < elements.len() {
                    elements[index] = Value::Undefined;

                    let mut new_len: usize = elements.len();
                    while new_len > 0 && elements[new_len - 1] == Value::Undefined {
                        new_len -= 1;
                    }
                    elements.truncate(new_len);
                }
            }
            (_, IndexOrKey::Index(_)) => {
                // do nothing
            }
            (_, IndexOrKey::Key(key)) => {
                self.properties.remove(key);
            }
        }
    }

    fn own_properties(&self, only_enumerable: bool, out: &mut Vec<String>) {
        match &self.exotic_part {
            Exotic::None | Exotic::Function { .. } => {}
            Exotic::Array { elements } => {
                for i in 0..elements.len() {
                    out.push(i.to_string());
                }
                if !only_enumerable {
                    out.push("length".to_owned());
                }
            }
            Exotic::String { .. } => {
                if !only_enumerable {
                    out.push("length".to_owned());
                }
            }
        }

        for key in &self.order {
            let prop = self.properties.get(key).unwrap();
            if !only_enumerable || prop.is_enumerable {
                out.push(key.clone());
            }
        }
    }

    fn proto(&self, _heap: &Heap) -> Option<ObjectId> {
        self.proto_id
    }
    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        self.proto_id = proto_id;
    }
}

// ValueObjectRef and ValueObjectMut  must exist, in order to carry either
// an immediate object (number, boolean) inline, or a Ref<HeapObject> (the
// HeapObject contains an OrdObject, or a StringObject, etc.).  This way you
// get a uniform "Object" interface, but you can also keep the runtime-checked
// borrowed ref of a heap-allocated object, and release it correctly when it's
// time.
pub enum ValueObjectRef<'h> {
    Number(f64, &'h Heap),
    Bool(bool, &'h Heap),
    Heap(&'h RefCell<HeapObject>),
}

impl<'h> ValueObjectRef<'h> {
    pub fn into_heap_cell(self) -> Option<&'h RefCell<HeapObject>> {
        match self {
            Self::Heap(ho) => Some(ho),
            _ => None,
        }
    }
}

impl<'h> Object for ValueObjectRef<'h> {
    fn get_own(&self, index_or_key: IndexOrKey) -> Option<Property> {
        match (self, index_or_key) {
            (
                ValueObjectRef::Number(_, heap) | ValueObjectRef::Bool(_, heap),
                IndexOrKey::Key("__proto__"),
            ) => self
                .proto(heap)
                .map(Value::Object)
                .map(Property::non_enumerable),
            (ValueObjectRef::Heap(ho), ik) => {
                let horef = ho.borrow();
                horef.get_own(ik)
            }
            (_, _) => None,
        }
    }

    fn set_own(&mut self, index_or_key: IndexOrKey, prop: Property) {
        if let ValueObjectRef::Heap(ho) = self {
            ho.borrow_mut().set_own(index_or_key, prop)
        }
    }

    fn delete_own(&mut self, index_or_key: IndexOrKey) {
        if let ValueObjectRef::Heap(ho) = self {
            ho.borrow_mut().delete_own(index_or_key)
        }
    }

    fn own_properties(&self, only_enumerable: bool, out: &mut Vec<String>) {
        match self {
            ValueObjectRef::Number(_, _) | ValueObjectRef::Bool(_, _) => {}
            ValueObjectRef::Heap(ho) => ho.borrow().own_properties(only_enumerable, out),
        }
    }

    fn type_of(&self) -> Typeof {
        match self {
            ValueObjectRef::Number(_, _) => Typeof::Number,
            ValueObjectRef::Bool(_, _) => Typeof::Boolean,
            ValueObjectRef::Heap(ho) => ho.borrow().type_of(),
        }
    }

    fn proto(&self, heap: &Heap) -> Option<ObjectId> {
        match self {
            ValueObjectRef::Number(_, heap) => Some(heap.number_proto),
            ValueObjectRef::Bool(_, heap) => Some(heap.bool_proto),
            ValueObjectRef::Heap(ho) => ho.borrow().proto(heap),
        }
    }

    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        if let ValueObjectRef::Heap(ho) = self {
            ho.borrow_mut().set_proto(proto_id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ops::DerefMut;

    #[test]
    fn test_number_properties() {
        let heap = Heap::new();
        {
            let mut number_proto = heap.get(heap.number_proto()).unwrap().borrow_mut();
            let number_proto = number_proto.deref_mut();
            number_proto.set_own("isNumber".into(), Property::enumerable(Value::Bool(true)));
            number_proto.set_own("isCool".into(), Property::enumerable(Value::Bool(true)));
        }

        let num = ValueObjectRef::Number(123.0, &heap);
        assert!(num.get_own("isNumber".into()).is_none());
        assert!(num.get_own("isCool".into()).is_none());
        assert_eq!(
            heap.get_property_chained(&num, "isNumber"),
            Some(Property::enumerable(Value::Bool(true)))
        );
        assert_eq!(
            heap.get_property_chained(&num, "isCool"),
            Some(Property::enumerable(Value::Bool(true)))
        );
    }

    #[test]
    fn test_bool_properties() {
        let heap = Heap::new();
        {
            let mut bool_proto = heap.get(heap.bool_proto()).unwrap().borrow_mut();
            let bool_proto = bool_proto.deref_mut();
            bool_proto.set_own("isNumber".into(), Property::enumerable(Value::Bool(false)));
            bool_proto.set_own("isCool".into(), Property::enumerable(Value::Bool(true)));
        }

        let num = ValueObjectRef::Bool(true, &heap);
        assert!(num.get_own("isNumber".into()).is_none());
        assert!(num.get_own("isCool".into()).is_none());
        assert_eq!(
            heap.get_property_chained(&num, "isNumber"),
            Some(Property::enumerable(Value::Bool(false)))
        );
        assert_eq!(
            heap.get_property_chained(&num, "isCool"),
            Some(Property::enumerable(Value::Bool(true)))
        );
    }

    #[test]
    fn test_array() {
        let mut heap = Heap::new();
        {
            let mut array_proto = heap.get(heap.array_proto()).unwrap().borrow_mut();
            let array_proto = array_proto.deref_mut();
            array_proto.set_own(
                "specialArrayProperty".into(),
                Property::enumerable(Value::Number(999.0)),
            );
        }

        let arr = heap.new_array(vec![
            Value::Number(9.0),
            Value::Number(6.0),
            Value::Number(3.0),
        ]);
        let arr = heap.get(arr).unwrap().borrow();
        assert!(arr.get_own("specialArrayProperty".into()).is_none());
        assert_eq!(
            heap.get_property_chained(arr.deref(), "specialArrayProperty"),
            Some(Property::enumerable(Value::Number(999.0)))
        );
        assert_eq!(heap.get_property_chained(arr.deref(), "isCool"), None);
    }
}
