use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::interpreter::{Closure, Value};

/// Reference to a JavaScript object.
///
/// This covers all objects: ordinary and exotic, immediate (wrappers of
/// primitive values) and stored on the heap.
pub struct ObjectRefW<'a, R> {
    heap: &'a Heap,
    value: ObjectValue<R>,
}
pub type ObjectRef<'a> = ObjectRefW<'a, Ref<'a, HeapObject>>;
pub type ObjectRefMut<'a> = ObjectRefW<'a, RefMut<'a, HeapObject>>;

enum ObjectValue<O> {
    Number(f64),
    Bool(bool),
    Object { obj: O, oid: ObjectId },
    Symbol(&'static str),
}

impl<'a, R> ObjectRefW<'a, R>
where
    R: Deref<Target = HeapObject>,
{
    /// Get an owned property (or array element) of this object.
    ///
    /// If you want to include inherited properties in the lookup, use
    /// `get_chained` instead.
    pub fn get_own(&self, iok: IndexOrKey) -> Option<Property> {
        if iok == IndexOrKey::Key("__proto__") {
            let proto_oid = self.proto()?;
            return Some(Property::non_enumerable(Value::Object(proto_oid)));
        }

        let iok_index = iok.ensure_number_is_index();

        match &self.value {
            ObjectValue::Object { obj, .. } => match (&obj.exotic_part, iok, iok_index) {
                (Exotic::Array { elements }, _, IndexOrKey::Key("length")) => Some(
                    Property::non_enumerable(Value::Number(elements.len() as f64)),
                ),
                (Exotic::Array { elements }, _, IndexOrKey::Index(index)) => {
                    elements.get(index).copied().map(Property::enumerable)
                }

                (Exotic::String { string }, IndexOrKey::Key("length"), _) => {
                    Some(Property::non_enumerable(Value::Number(string.len() as f64)))
                }

                (_, iok, _) => match iok {
                    IndexOrKey::Index(num) => obj.properties.get(&num.to_string()).copied(),
                    IndexOrKey::Key(key) => obj.properties.get(key).copied(),
                    IndexOrKey::Symbol(sym) => obj.sym_properties.get(sym).copied(),
                },
            },
            ObjectValue::Number(_) | ObjectValue::Bool(_) | ObjectValue::Symbol(_) => None,
        }
    }

    pub fn get_chained(&self, index_or_key: IndexOrKey) -> Option<Property> {
        let value = self.get_own(index_or_key);

        // No chained lookup for numeric keys
        if let IndexOrKey::Index(_) = index_or_key {
            return value;
        }

        if value.is_some() {
            value
        } else if let Some(proto_id) = self.proto() {
            let proto = self.heap.get(Value::Object(proto_id))?;
            proto.get_chained(index_or_key)
        } else {
            None
        }
    }

    pub fn list_properties_prototypes(&self, only_enumerable: bool, out: &mut Vec<String>) {
        if let Some(proto_oid) = self.proto() {
            let proto = self.heap.get(Value::Object(proto_oid)).unwrap();
            proto.own_properties(only_enumerable, out);
            proto.list_properties_prototypes(only_enumerable, out);
        }
    }

    pub fn is_instance_of(&self, sup_oid: ObjectId) -> bool {
        if let ObjectValue::Object { oid, .. } = &self.value {
            if sup_oid == *oid {
                return true;
            }
        }

        let mut cur_proto_oid = self.proto();

        #[cfg(test)]
        let mut trace = vec![];

        loop {
            match cur_proto_oid {
                None => return false,
                Some(oid) if oid == sup_oid => return true,
                Some(oid) => {
                    #[cfg(test)]
                    {
                        if trace.contains(&oid) {
                            panic!("Circular prototype chain!")
                        }
                        trace.push(oid);
                    }

                    cur_proto_oid = self.heap.get(Value::Object(oid)).unwrap().proto();
                }
            }
        }
    }

    // I know, inefficient, but so, *so* simple
    pub fn own_properties(&self, only_enumerable: bool, out: &mut Vec<String>) {
        let obj = match &self.value {
            ObjectValue::Object { obj, .. } => obj,
            _ => return,
        };

        match &obj.exotic_part {
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
        };

        for key in &obj.order {
            if let Some(prop) = obj.properties.get(key) {
                if !only_enumerable || prop.is_enumerable {
                    out.push(key.clone());
                }
            }
        }
    }

    pub fn type_of(&self) -> Typeof {
        match &self.value {
            ObjectValue::Number(_) => Typeof::Number,
            ObjectValue::Bool(_) => Typeof::Boolean,
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::None | Exotic::Array { .. } | Exotic::String { .. } => Typeof::Object,
                Exotic::Function { .. } => Typeof::Function,
            },
            ObjectValue::Symbol(_) => Typeof::Symbol,
        }
    }
    pub fn proto(&self) -> Option<ObjectId> {
        let proto_oid = match &self.value {
            ObjectValue::Number(_) => self.heap.number_proto,
            ObjectValue::Bool(_) => self.heap.bool_proto,
            ObjectValue::Object { obj, .. } => obj.proto_id?,
            ObjectValue::Symbol(_) => todo!("symbol prototype"),
        };
        Some(proto_oid)
    }

    pub fn as_str(&self) -> Option<&str> {
        match &self.value {
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::String { string } => Some(string.as_str()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn as_closure(&self) -> Option<&Closure> {
        match &self.value {
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::Function { closure } => Some(closure),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn array_elements(&self) -> Option<&[Value]> {
        match &self.value {
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::Array { elements } => Some(elements),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn to_boolean(&self) -> bool {
        match &self.value {
            ObjectValue::Number(n) => *n != 0.0,
            ObjectValue::Bool(b) => *b,
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::None | Exotic::Array { .. } | Exotic::Function { .. } => true,
                Exotic::String { string } => !string.is_empty(),
            },
            ObjectValue::Symbol(_) => true,
        }
    }

    pub fn js_to_string(&self) -> String {
        match &self.value {
            ObjectValue::Number(n) => n.to_string(),
            ObjectValue::Bool(b) => b.to_string(),
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::Array { .. } | Exotic::None => "<object>".to_owned(),
                Exotic::String { string } => string.clone(),
                Exotic::Function { .. } => "<closure>".to_owned(),
            },
            ObjectValue::Symbol(_) => todo!(),
        }
    }

    pub fn show_debug(&self) -> String {
        match &self.value {
            ObjectValue::Number(n) => format!("object number {}", n),
            ObjectValue::Bool(b) => format!("object boolean {}", b),
            ObjectValue::Object { obj, .. } => match &obj.exotic_part {
                Exotic::None => format!("object ({} properties)", obj.properties.len()),
                Exotic::Array { elements } => {
                    format!("array ({} elements)", elements.len())
                }
                Exotic::String { string } => format!("{:?}", string),
                Exotic::Function { .. } => "<closure>".to_owned(),
            },
            ObjectValue::Symbol(_) => todo!(),
        }
    }
}

impl<'a, R> ObjectRefW<'a, R>
where
    R: DerefMut<Target = HeapObject>,
{
    /// Set the value of an owned property (or an array element).
    pub fn set_own(&mut self, index_or_key: IndexOrKey, property: Property) {
        if index_or_key == IndexOrKey::Key("__proto__") {
            if let Value::Object(new_proto_id) = property.value {
                if let ObjectValue::Object { obj, .. } = &mut self.value {
                    // property.is_enumerable is discarded. It's implicitly non-enumerable (see
                    // `get_own`)
                    obj.proto_id = Some(new_proto_id);
                }
            } else {
                // Apparently this is simply a nop in V8?
            }
            return;
        }

        let obj = match &mut self.value {
            ObjectValue::Object { obj, .. } => obj,
            ObjectValue::Number(_) | ObjectValue::Bool(_) | ObjectValue::Symbol(_) => return,
        };

        match (&mut obj.exotic_part, index_or_key) {
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
            (_, IndexOrKey::Index(_)) => {
                // do nothing
            }
            (_, IndexOrKey::Key(key)) => {
                let prev = obj.properties.insert(key.to_owned(), property);
                if prev.is_none() {
                    obj.order.push(key.to_string());
                }
                debug_assert_eq!(1, obj.order.iter().filter(|x| *x == key).count());
            }
            (_, IndexOrKey::Symbol(sym)) => {
                // Properties associated to Symbol keys are not enumerable
                obj.sym_properties.insert(sym, property);
            }
        }
    }

    /// Remove an owned property (or array element) of this object.
    ///
    /// This does NOT affect inherited properties and does NOT access
    /// the prototype chain in any way.
    pub fn delete_own(&mut self, index_or_key: IndexOrKey) {
        if index_or_key == IndexOrKey::Key("__proto__") {
            return;
        }

        let obj = match &mut self.value {
            ObjectValue::Object { obj, .. } => obj,
            ObjectValue::Number(_) | ObjectValue::Bool(_) | ObjectValue::Symbol(_) => return,
        };

        match (&mut obj.exotic_part, index_or_key) {
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
                // Deletion of properties are rare and only involve a small
                // minority of properties, so no space to be saved by removing
                // the key from `self.order`. Instead, only delete from the
                // hashmap, then the iteration procedure will skip it.
                obj.properties.remove(key);
            }
            (_, IndexOrKey::Symbol(sym)) => {
                obj.sym_properties.remove(sym);
            }
        }
    }

    pub fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        if let ObjectValue::Object { obj, .. } = &mut self.value {
            obj.proto_id = proto_id;
        }
    }

    pub fn array_push(&mut self, value: Value) -> bool {
        match &mut self.value {
            ObjectValue::Object { obj, .. } => match &mut obj.exotic_part {
                Exotic::Array { elements } => {
                    elements.push(value);
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Typeof {
    Object,
    Function,
    String,
    Number,
    Boolean,
    Symbol,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IndexOrKey<'a> {
    Index(usize),
    Key(&'a str),
    Symbol(&'static str),
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
            IndexOrKey::Symbol(_) => self,
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
    Symbol(&'static str),
}
impl IndexOrKeyOwned {
    pub fn to_ref(&self) -> IndexOrKey {
        match self {
            IndexOrKeyOwned::Index(ndx) => IndexOrKey::Index(*ndx),
            IndexOrKeyOwned::Key(key) => IndexOrKey::Key(key),
            IndexOrKeyOwned::Symbol(sym) => IndexOrKey::Symbol(sym),
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
        let mut objects = slotmap::SlotMap::with_key();
        let object_proto = objects.insert(RefCell::new(HeapObject::default()));
        let number_proto = objects.insert(RefCell::new(HeapObject::default()));
        let string_proto = objects.insert(RefCell::new(HeapObject::default()));
        let func_proto = objects.insert(RefCell::new(HeapObject::default()));
        let bool_proto = objects.insert(RefCell::new(HeapObject::default()));
        let array_proto = objects.insert(RefCell::new(HeapObject::default()));

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
            sym_properties: HashMap::new(),
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

    /// Access the JavaScript object represented by the given value.
    ///
    /// If the value is an object reference, the returned interface allows
    /// access to the referenced object. Otherwise, the value is a primitive;
    /// then an object-like interface to  the primitive is returned that allows
    /// access to its prototype and any intrinsic properties..
    ///
    /// The conversion fails (None is returned) for `null` and `undefined`.
    pub fn get(&self, value: Value) -> Option<ObjectRef> {
        let value = match value {
            Value::Number(n) => ObjectValue::Number(n),
            Value::Bool(b) => ObjectValue::Bool(b),
            Value::Object(oid) => {
                let obj = self.objects.get(oid)?.borrow();
                ObjectValue::Object { obj, oid }
            }
            Value::Symbol(s) => ObjectValue::Symbol(s),
            Value::Null | Value::Undefined => return None,
        };
        Some(ObjectRef { heap: self, value })
    }

    pub fn get_mut(&self, value: Value) -> Option<ObjectRefMut> {
        let value = match value {
            Value::Number(n) => ObjectValue::Number(n),
            Value::Bool(b) => ObjectValue::Bool(b),
            Value::Object(oid) => {
                let obj = self.objects.get(oid)?.borrow_mut();
                ObjectValue::Object { obj, oid }
            }
            Value::Symbol(s) => ObjectValue::Symbol(s),
            Value::Null | Value::Undefined => return None,
        };
        Some(ObjectRefMut { heap: self, value })
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
    sym_properties: HashMap<&'static str, Property>,
    order: Vec<String>,
    exotic_part: Exotic,
}
impl Default for HeapObject {
    fn default() -> HeapObject {
        HeapObject {
            proto_id: None,
            properties: HashMap::new(),
            sym_properties: HashMap::new(),
            order: Vec::new(),
            exotic_part: Exotic::None,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_properties() {
        let heap = Heap::new();
        {
            let mut number_proto = heap.get_mut(Value::Object(heap.number_proto())).unwrap();
            number_proto.set_own("isNumber".into(), Property::enumerable(Value::Bool(true)));
            number_proto.set_own("isCool".into(), Property::enumerable(Value::Bool(true)));
        }

        let num = heap.get(Value::Number(123.0)).unwrap();
        assert!(num.get_own("isNumber".into()).is_none());
        assert!(num.get_own("isCool".into()).is_none());
        assert_eq!(
            num.get_chained(IndexOrKey::Key("isNumber")),
            Some(Property::enumerable(Value::Bool(true)))
        );
        assert_eq!(
            num.get_chained(IndexOrKey::Key("isCool")),
            Some(Property::enumerable(Value::Bool(true)))
        );
    }

    #[test]
    fn test_bool_properties() {
        let heap = Heap::new();
        {
            let mut bool_proto = heap.get_mut(Value::Object(heap.bool_proto())).unwrap();
            bool_proto.set_own("isNumber".into(), Property::enumerable(Value::Bool(false)));
            bool_proto.set_own("isCool".into(), Property::enumerable(Value::Bool(true)));
        }

        let bool_ = heap.get(Value::Bool(true)).unwrap();
        assert!(bool_.get_own("isNumber".into()).is_none());
        assert!(bool_.get_own("isCool".into()).is_none());
        assert_eq!(
            bool_.get_chained(IndexOrKey::Key("isNumber")),
            Some(Property::enumerable(Value::Bool(false)))
        );
        assert_eq!(
            bool_.get_chained(IndexOrKey::Key("isCool")),
            Some(Property::enumerable(Value::Bool(true)))
        );
    }

    #[test]
    fn test_array() {
        let mut heap = Heap::new();
        {
            let mut array_proto = heap.get_mut(Value::Object(heap.array_proto())).unwrap();
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
        let arr = heap.get(Value::Object(arr)).unwrap();
        assert!(arr.get_own("specialArrayProperty".into()).is_none());
        assert_eq!(
            arr.get_chained(IndexOrKey::Key("specialArrayProperty")),
            Some(Property::enumerable(Value::Number(999.0)))
        );
        assert_eq!(arr.get_chained(IndexOrKey::Key("isCool")), None);
    }
}
