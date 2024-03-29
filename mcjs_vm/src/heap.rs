use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;

use crate::interpreter::{Closure, Value};

/// Common interface implemented by types that are sufficently "object-like" to be exposed
/// as objects in JavaScript code.   This includes all objects: ordinary and exotic,
/// immediate and heap-stored.
///
/// This interface only allows access to the object's "own" properties, without using the
/// prototype.  For that, you need use a `Heap` and its methods.
// TODO Will this design work even though this trait is not object-safe?
pub trait Object {
    fn get_own_element_or_property(&self, index_or_key: IndexOrKey) -> Option<Value>;
    fn set_own_element_or_property(&mut self, index_or_key: IndexOrKey, value: Value);
    fn delete_own_element_or_property(&mut self, index_or_key: IndexOrKey);

    // I know, inefficient, but so, *so* simple, and enum_dispatch-able.
    // TODO: make it slightly better, return a Cow<'static, str>
    fn own_properties(&self) -> Vec<String>;

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

#[derive(PartialEq, Eq)]
pub enum IndexOrKey<'a> {
    Index(usize),
    Key(&'a str),
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

    pub(crate) fn new_ordinary_object(&mut self, properties: HashMap<String, Value>) -> ObjectId {
        self.objects.insert(RefCell::new(HeapObject {
            proto_id: Some(self.object_proto),
            properties,
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
        let oid = self.new_ordinary_object(HashMap::new());
        self.init_array(oid, elements);
        oid
    }
    pub(crate) fn new_function(&mut self, closure: Closure) -> ObjectId {
        let oid = self.new_ordinary_object(HashMap::new());
        self.init_function(oid, closure);
        oid
    }
    pub(crate) fn new_string(&mut self, string: String) -> ObjectId {
        let oid = self.new_ordinary_object(HashMap::new());
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

    pub fn get_property_chained<O: ?Sized + Object>(&self, obj: &O, key: &str) -> Option<Value> {
        let value = obj.get_own_element_or_property(IndexOrKey::Key(key));

        if value.is_some() {
            value
        } else if let Some(proto_id) = obj.proto(self) {
            let proto = self.get(proto_id)?.borrow();
            self.get_property_chained(proto.deref(), key)
        } else {
            None
        }
    }
}

/// An ordinary object, i.e. one that you can create in JavaScript with the `{a: 1, b: 2}`
/// syntax.
///
/// It stores key-value pairs where keys ("properties") are strings, and any JS value can
/// be stored as value.
#[derive(Debug, Clone)]
pub struct HeapObject {
    proto_id: Option<ObjectId>,
    properties: HashMap<String, Value>,
    exotic_part: Exotic,
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

    fn get_own_element_or_property(&self, index_or_key: IndexOrKey) -> Option<Value> {
        if index_or_key == IndexOrKey::Key("__proto__") {
            return self.proto_id.map(Value::Object);
        }

        match (&self.exotic_part, index_or_key) {
            (Exotic::Array { elements }, IndexOrKey::Key("length")) => {
                Some(Value::Number(elements.len() as f64))
            }
            (Exotic::Array { elements }, IndexOrKey::Index(index)) => elements.get(index).copied(),
            (Exotic::String { string }, IndexOrKey::Key("length")) => {
                Some(Value::Number(string.len() as f64))
            }
            (_, IndexOrKey::Key(key)) => self.properties.get(key).copied(),
            (_, IndexOrKey::Index(_)) => None,
        }
    }
    fn set_own_element_or_property(&mut self, index_or_key: IndexOrKey, value: Value) {
        if index_or_key == IndexOrKey::Key("__proto__") {
            if let Value::Object(new_proto_id) = value {
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
                self.set_own_element_or_property(new_key, value)
            }
            (Exotic::Array { elements }, IndexOrKey::Index(ndx)) => {
                if elements.len() < ndx + 1 {
                    elements.resize(ndx + 1, Value::Undefined);
                }
                elements[ndx] = value;
            }
            (Exotic::String { .. } | Exotic::Function { .. }, IndexOrKey::Index(_)) => {
                // do nothing
            }
            (_, IndexOrKey::Key(key)) => {
                self.properties.insert(key.to_owned(), value);
            }
        }
    }
    fn delete_own_element_or_property(&mut self, index_or_key: IndexOrKey) {
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

    fn own_properties(&self) -> Vec<String> {
        let mut props: Vec<_> = self.properties.keys().cloned().collect();

        match self.exotic_part {
            Exotic::None | Exotic::Function { .. } => {}
            Exotic::Array { .. } | Exotic::String { .. } => {
                props.push("length".to_owned());
            }
        }

        props
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
    fn get_own_element_or_property(&self, index_or_key: IndexOrKey) -> Option<Value> {
        match (self, index_or_key) {
            (
                ValueObjectRef::Number(_, heap) | ValueObjectRef::Bool(_, heap),
                IndexOrKey::Key("__proto__"),
            ) => self.proto(heap).map(Value::Object),
            (ValueObjectRef::Heap(ho), ik) => {
                let horef = ho.borrow();
                horef.get_own_element_or_property(ik)
            }
            (_, _) => None,
        }
    }

    fn set_own_element_or_property(&mut self, index_or_key: IndexOrKey, value: Value) {
        if let ValueObjectRef::Heap(ho) = self {
            ho.borrow_mut()
                .set_own_element_or_property(index_or_key, value)
        }
    }

    fn delete_own_element_or_property(&mut self, index_or_key: IndexOrKey) {
        if let ValueObjectRef::Heap(ho) = self {
            ho.borrow_mut().delete_own_element_or_property(index_or_key)
        }
    }

    fn own_properties(&self) -> Vec<String> {
        match self {
            ValueObjectRef::Number(_, _) => Vec::new(),
            ValueObjectRef::Bool(_, _) => Vec::new(),
            ValueObjectRef::Heap(ho) => ho.borrow().own_properties(),
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
            number_proto.set_own_element_or_property("isNumber".into(), Value::Bool(true));
            number_proto.set_own_element_or_property("isCool".into(), Value::Bool(true));
        }

        let num = ValueObjectRef::Number(123.0, &heap);
        assert!(num.get_own_element_or_property("isNumber".into()).is_none());
        assert!(num.get_own_element_or_property("isCool".into()).is_none());
        assert_eq!(
            heap.get_property_chained(&num, "isNumber"),
            Some(Value::Bool(true))
        );
        assert_eq!(
            heap.get_property_chained(&num, "isCool"),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_bool_properties() {
        let heap = Heap::new();
        {
            let mut bool_proto = heap.get(heap.bool_proto()).unwrap().borrow_mut();
            let bool_proto = bool_proto.deref_mut();
            bool_proto.set_own_element_or_property("isNumber".into(), Value::Bool(false));
            bool_proto.set_own_element_or_property("isCool".into(), Value::Bool(true));
        }

        let num = ValueObjectRef::Bool(true, &heap);
        assert!(num.get_own_element_or_property("isNumber".into()).is_none());
        assert!(num.get_own_element_or_property("isCool".into()).is_none());
        assert_eq!(
            heap.get_property_chained(&num, "isNumber"),
            Some(Value::Bool(false))
        );
        assert_eq!(
            heap.get_property_chained(&num, "isCool"),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_array() {
        let mut heap = Heap::new();
        {
            let mut array_proto = heap.get(heap.array_proto()).unwrap().borrow_mut();
            let array_proto = array_proto.deref_mut();
            array_proto
                .set_own_element_or_property("specialArrayProperty".into(), Value::Number(999.0));
        }

        let arr = heap.new_array(vec![
            Value::Number(9.0),
            Value::Number(6.0),
            Value::Number(3.0),
        ]);
        let arr = heap.get(arr).unwrap().borrow();
        assert!(arr
            .get_own_element_or_property("specialArrayProperty".into())
            .is_none());
        assert_eq!(
            heap.get_property_chained(arr.deref(), "specialArrayProperty"),
            Some(Value::Number(999.0))
        );
        assert_eq!(heap.get_property_chained(arr.deref(), "isCool"), None);
    }
}
