use std::collections::HashMap;
use std::rc::Rc;

use crate::interpreter::{Closure, Value};

//
// Object access API
//
impl Heap {
    /// Get an owned property (or array element) of this object.
    ///
    /// If you want to include inherited properties in the lookup, use
    /// `get_chained` instead.
    pub fn get_own(&self, obj: Value, iok: IndexOrKey) -> Option<Property> {
        if iok == IndexOrKey::Key("__proto__") {
            let proto_oid = self.proto(obj)?;
            return Some(Property::NonEnumerable(Value::Object(proto_oid)));
        }

        let iok_index = iok.ensure_number_is_index();

        let obj = match obj {
            Value::Object(obj) => obj,
            Value::Number(_)
            | Value::Bool(_)
            | Value::Symbol(_)
            | Value::Null
            | Value::Undefined => return None,
        };

        // TODO Separate the failure case where the object doesn't exist
        // from the one where the property does not exist
        let obj = self.get(obj).unwrap();
        match (&obj.exotic_part, iok, iok_index) {
            (Exotic::Array { elements }, _, IndexOrKey::Key("length")) => {
                let value = Value::Number(elements.len() as f64);
                Some(Property::NonEnumerable(value))
            }
            (Exotic::Array { elements }, _, IndexOrKey::Index(index)) => elements
                .get(index as usize)
                .copied()
                .map(Property::Enumerable),

            (Exotic::String { string }, IndexOrKey::Key("length"), _) => {
                let value = Value::Number(string.view().len() as f64);
                Some(Property::NonEnumerable(value))
            }
            (Exotic::String { string }, _, IndexOrKey::Index(index)) => {
                let substr = string.substring(index, index + 1);
                if substr.view().len() == 0 {
                    Some(Property::NonEnumerable(Value::Undefined))
                } else {
                    Some(Property::Substring(substr))
                }
            }

            (_, iok, _) => match iok {
                IndexOrKey::Index(num) => obj.properties.get(&num.to_string()).cloned(),
                IndexOrKey::Key(key) => obj.properties.get(key).cloned(),
                IndexOrKey::Symbol(sym) => obj.sym_properties.get(sym).cloned(),
            },
        }
    }

    pub fn get_chained(&self, obj: Value, index_or_key: IndexOrKey) -> Option<Property> {
        let value = self.get_own(obj, index_or_key);

        // No chained lookup for numeric keys
        if let IndexOrKey::Index(_) = index_or_key {
            return value;
        }

        if value.is_some() {
            value
        } else if let Some(proto_id) = self.proto(obj) {
            self.get_chained(Value::Object(proto_id), index_or_key)
        } else {
            None
        }
    }

    pub fn list_properties_prototypes(
        &self,
        obj: Value,
        only_enumerable: bool,
        out: &mut Vec<String>,
    ) {
        if let Some(proto_oid) = self.proto(obj) {
            self.own_properties(Value::Object(proto_oid), only_enumerable, out);
            self.list_properties_prototypes(Value::Object(proto_oid), only_enumerable, out);
        }
    }

    pub fn is_instance_of(&self, obj: Value, sup_oid: ObjectId) -> bool {
        if let Value::Object(oid) = obj {
            if sup_oid == oid {
                return true;
            }
        }

        let mut cur_proto_oid = self.proto(obj);

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

                    cur_proto_oid = self.proto(Value::Object(oid));
                }
            }
        }
    }

    // I know, inefficient, but so, *so* simple
    pub fn own_properties(&self, obj: Value, only_enumerable: bool, out: &mut Vec<String>) {
        let obj = match obj {
            Value::Object(obj) => obj,
            _ => return,
        };
        // TODO Handle specific failure?
        let obj = self.get(obj).unwrap();

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
                if !only_enumerable || prop.is_enumerable() {
                    out.push(key.clone());
                }
            }
        }
    }

    pub fn type_of(&self, obj: Value) -> Typeof {
        match obj {
            Value::Number(_) => Typeof::Number,
            Value::Bool(_) => Typeof::Boolean,
            Value::Object(obj) => {
                // TODO Handle specific failure
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::None | Exotic::Array { .. } => Typeof::Object,
                    Exotic::String { .. } => Typeof::String,
                    Exotic::Function { .. } => Typeof::Function,
                }
            }
            Value::Symbol(_) => Typeof::Symbol,
            Value::Null => Typeof::Symbol,
            Value::Undefined => Typeof::Undefined,
        }
    }
    pub fn proto(&self, obj: Value) -> Option<ObjectId> {
        match obj {
            Value::Number(_) => Some(self.number_proto),
            Value::Bool(_) => Some(self.bool_proto),
            Value::Object(obj) => self.get(obj).unwrap().proto_id,
            Value::Symbol(_) => todo!("symbol prototype"),
            Value::Null | Value::Undefined => None,
        }
    }

    pub fn as_str(&self, obj: Value) -> Option<&JSString> {
        match obj {
            Value::Object(obj) => {
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::String { string } => Some(string),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn as_closure(&self, obj: Value) -> Option<&Closure> {
        match obj {
            Value::Object(obj) => {
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::Function { ref closure } => Some(closure),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub fn array_elements(&self, obj: Value) -> Option<&[Value]> {
        match obj {
            Value::Object(obj) => {
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::Array { ref elements } => Some(elements),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Converts the given value to a boolean (e.g. for use by `if`,
    /// or operators `&&` and `||`)
    ///
    /// See: https://262.ecma-international.org/14.0/#sec-toboolean
    pub fn to_boolean(&self, obj: Value) -> bool {
        match obj {
            Value::Number(n) => n != 0.0,
            Value::Bool(b) => b,
            Value::Object(obj) => {
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::None | Exotic::Array { .. } | Exotic::Function { .. } => true,
                    Exotic::String { string } => !string.view().is_empty(),
                }
            }
            Value::Symbol(_) => true,
            Value::Null | Value::Undefined => false,
        }
    }

    pub fn js_to_string(&self, obj: Value) -> JSString {
        match obj {
            Value::Number(n) => JSString::new_from_str(&n.to_string()),
            Value::Bool(b) => JSString::new_from_str(&b.to_string()),
            Value::Object(obj) => {
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::Array { .. } | Exotic::None => JSString::new_from_str("<object>"),
                    Exotic::String { string } => string.clone(),
                    Exotic::Function { .. } => JSString::new_from_str("<closure>"),
                }
            }
            Value::Symbol(_) => todo!(),
            Value::Null => JSString::new_from_str("null"),
            Value::Undefined => JSString::new_from_str("undefined"),
        }
    }

    pub fn show_debug(&self, obj: Value) -> String {
        match obj {
            Value::Number(n) => format!("object number {}", n),
            Value::Bool(b) => format!("object boolean {}", b),
            Value::Object(obj) => {
                let obj = self.get(obj).unwrap();
                match &obj.exotic_part {
                    Exotic::None => format!("object ({} properties)", obj.properties.len()),
                    Exotic::Array { elements } => {
                        format!("array ({} elements)", elements.len())
                    }
                    Exotic::String { string } => format!("{:?}", string),
                    Exotic::Function { .. } => "<closure>".to_owned(),
                }
            }
            Value::Symbol(_) => todo!(),
            Value::Null => "null".to_owned(),
            Value::Undefined => "undefined".to_owned(),
        }
    }
}

//
// Object mutation API
//
impl Heap {
    /// Set the value of an owned property (or an array element).
    pub fn set_own(&mut self, obj: Value, index_or_key: IndexOrKey, property: Property) {
        if index_or_key == IndexOrKey::Key("__proto__") {
            if let (Value::Object(obj), Some(Value::Object(new_proto_id))) = (obj, property.value())
            {
                let obj = self.get_mut(obj).unwrap();
                // property.is_enumerable is discarded. It's implicitly non-enumerable (see
                // `get_own`)
                obj.proto_id = Some(new_proto_id);
            } else {
                // Apparently this is simply a nop in V8?
            }
            return;
        }

        let hobj = match obj {
            Value::Object(oid) => self.get_mut(oid).unwrap(),
            Value::Null
            | Value::Undefined
            | Value::Number(_)
            | Value::Bool(_)
            | Value::Symbol(_) => return,
        };

        match (&mut hobj.exotic_part, index_or_key) {
            (Exotic::None, IndexOrKey::Index(ndx)) => {
                let new_key = ndx.to_string();
                let new_key = IndexOrKey::Key(&new_key);
                self.set_own(obj, new_key, property)
            }
            (Exotic::Array { elements }, IndexOrKey::Index(ndx)) => {
                let ndx = ndx as usize;
                if elements.len() < ndx + 1 {
                    elements.resize(ndx + 1, Value::Undefined);
                }
                // implicitly enumerable
                elements[ndx] = property.value().unwrap();
            }
            (_, IndexOrKey::Index(_)) => {
                // do nothing
            }
            (_, IndexOrKey::Key(key)) => {
                let prev = hobj.properties.insert(key.to_owned(), property);
                if prev.is_none() {
                    hobj.order.push(key.to_string());
                }
                debug_assert_eq!(1, hobj.order.iter().filter(|x| *x == key).count());
            }
            (_, IndexOrKey::Symbol(sym)) => {
                // Properties associated to Symbol keys are not enumerable
                hobj.sym_properties.insert(sym, property);
            }
        }
    }

    /// Remove an owned property (or array element) of this object.
    ///
    /// This does NOT affect inherited properties and does NOT access the
    /// prototype chain in any way.
    ///
    /// Returns `true` iff `obj` was a non-exotic object. This does not depend
    /// on whether the property actually existed or not, or anything else.
    // TODO change this stinky return value type
    pub fn delete_own(&mut self, obj: Value, index_or_key: IndexOrKey) -> bool {
        if index_or_key == IndexOrKey::Key("__proto__") {
            return true;
        }

        let hobj = match obj {
            // TODO Handle specific failure case
            Value::Object(oid) => self.get_mut(oid).unwrap(),
            Value::Number(_)
            | Value::Bool(_)
            | Value::Symbol(_)
            | Value::Null
            | Value::Undefined => return false,
        };

        match (&mut hobj.exotic_part, index_or_key) {
            (Exotic::Array { .. } | Exotic::String { .. }, IndexOrKey::Key("length")) => {
                // do nothing
            }
            (Exotic::Array { elements }, IndexOrKey::Index(index)) => {
                let index = index as usize;
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
                hobj.properties.remove(key);
            }
            (_, IndexOrKey::Symbol(sym)) => {
                hobj.sym_properties.remove(sym);
            }
        }

        true
    }

    pub fn set_proto(&mut self, obj: Value, proto_id: Option<ObjectId>) {
        if let Value::Object(oid) = obj {
            self.get_mut(oid).unwrap().proto_id = proto_id;
        }
    }

    pub fn array_push(&mut self, obj: Value, value: Value) -> bool {
        if let Some(elements) = self.as_array_mut(obj) {
            elements.push(value);
            true
        } else {
            false
        }
    }

    pub(crate) fn as_array_mut(&mut self, obj: Value) -> Option<&mut Vec<Value>> {
        let obj = obj.expect_obj().ok()?;
        let obj = self.get_mut(obj)?;
        match &mut obj.exotic_part {
            Exotic::Array { elements } => Some(elements),
            _ => None,
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
    Undefined,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IndexOrKey<'a> {
    Index(u32),
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
        IndexOrKey::Index(value.try_into().unwrap())
    }
}

#[derive(PartialEq, Eq)]
pub enum IndexOrKeyOwned {
    Index(u32),
    // TODO Change this! Better if keys are JSString
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
    objects: slotmap::SlotMap<ObjectId, HeapObject>,

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
        let object_proto = objects.insert(HeapObject::default());
        let number_proto = objects.insert(HeapObject::default());
        let string_proto = objects.insert(HeapObject::default());
        let func_proto = objects.insert(HeapObject::default());
        let bool_proto = objects.insert(HeapObject::default());
        let array_proto = objects.insert(HeapObject::default());

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

    pub fn object_proto(&self) -> ObjectId {
        self.object_proto
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
        self.objects.insert(HeapObject {
            proto_id: Some(self.object_proto),
            properties: HashMap::new(),
            sym_properties: HashMap::new(),
            order: Vec::new(),
            exotic_part: Exotic::None,
        })
    }

    // Weird property of these functions: their purpose is to *create* an exotic
    // object, but they actually *modify* an existing object. This is to align
    // them to the property of JavaScript constructors, which act on a
    // pre-created (ordinary) object passed as `this`.

    fn init_exotic(&mut self, oid: ObjectId, proto_oid: ObjectId, exotic_part: Exotic) {
        let obj = self.objects.get_mut(oid).unwrap();
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
    pub(crate) fn new_string(&mut self, string: JSString) -> ObjectId {
        let oid = self.new_ordinary_object();
        self.init_exotic(oid, self.string_proto, Exotic::String { string: string });
        oid
    }

    fn get(&self, oid: ObjectId) -> Option<&HeapObject> {
        self.objects.get(oid)
    }

    fn get_mut(&mut self, oid: ObjectId) -> Option<&mut HeapObject> {
        self.objects.get_mut(oid)
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
#[derive(Debug, PartialEq, Clone)]
pub enum Property {
    Enumerable(Value),
    NonEnumerable(Value),
    // implicitly non-enumerable
    Substring(JSString),
}
impl Property {
    #[inline]
    pub fn is_enumerable(&self) -> bool {
        match self {
            Property::Enumerable(_) => true,
            Property::NonEnumerable(_) => false,
            Property::Substring(_) => false,
        }
    }

    #[inline]
    pub fn value(&self) -> Option<Value> {
        match self {
            Property::Enumerable(value) | Property::NonEnumerable(value) => Some(*value),
            Property::Substring(_) => None,
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
        string: JSString,
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
        let mut heap = Heap::new();
        {
            let number_proto = Value::Object(heap.number_proto());
            heap.set_own(number_proto, "isNumber".into(), {
                let value = Value::Bool(true);
                Property::Enumerable(value)
            });
            heap.set_own(number_proto, "isCool".into(), {
                let value = Value::Bool(true);
                Property::Enumerable(value)
            });
        }

        let num = Value::Number(123.0);
        assert!(heap.get_own(num, "isNumber".into()).is_none());
        assert!(heap.get_own(num, "isCool".into()).is_none());
        assert_eq!(
            heap.get_chained(num, IndexOrKey::Key("isNumber")),
            Some(Property::Enumerable(Value::Bool(true)))
        );
        assert_eq!(
            heap.get_chained(num, IndexOrKey::Key("isCool")),
            Some(Property::Enumerable(Value::Bool(true)))
        );
    }

    #[test]
    fn test_bool_properties() {
        let mut heap = Heap::new();
        {
            let bool_proto = Value::Object(heap.bool_proto());
            heap.set_own(bool_proto, "isNumber".into(), {
                let value = Value::Bool(false);
                Property::Enumerable(value)
            });
            heap.set_own(bool_proto, "isCool".into(), {
                let value = Value::Bool(true);
                Property::Enumerable(value)
            });
        }

        let bool_ = Value::Bool(true);
        assert!(heap.get_own(bool_, "isNumber".into()).is_none());
        assert!(heap.get_own(bool_, "isCool".into()).is_none());
        assert_eq!(
            heap.get_chained(bool_, IndexOrKey::Key("isNumber")),
            Some(Property::Enumerable(Value::Bool(false)))
        );
        assert_eq!(
            heap.get_chained(bool_, IndexOrKey::Key("isCool")),
            Some(Property::Enumerable(Value::Bool(true)))
        );
    }

    #[test]
    fn test_array() {
        let mut heap = Heap::new();
        {
            let array_proto = Value::Object(heap.array_proto());
            heap.set_own(array_proto, "specialArrayProperty".into(), {
                let value = Value::Number(999.0);
                Property::Enumerable(value)
            });
        }

        let arr = heap.new_array(vec![
            Value::Number(9.0),
            Value::Number(6.0),
            Value::Number(3.0),
        ]);
        let arr = Value::Object(arr);
        assert!(heap.get_own(arr, "specialArrayProperty".into()).is_none());
        assert_eq!(
            heap.get_chained(arr, IndexOrKey::Key("specialArrayProperty")),
            Some(Property::Enumerable(Value::Number(999.0)))
        );
        assert_eq!(heap.get_chained(arr, IndexOrKey::Key("isCool")), None);
    }
}

/// A JavaScript string.
///
/// Immutable.
///
/// Actually, following JavaScript semantics, this type represents a slice view
/// into a buffer of characters that may be shared between other views
/// (instances of JSString). Cloning a JSString cheaply produces a second
/// equivalent instance of JSString without copying the underlying buffer.
#[derive(Clone)]
pub struct JSString {
    // The string is internally represented as UTF-16.
    full: Rc<Vec<u16>>,
    start: u32,
    end: u32,
}

impl JSString {
    pub fn empty() -> Self {
        JSString {
            full: Rc::new(Vec::new()),
            start: 0,
            end: 0,
        }
    }

    pub(crate) fn new(buf: Vec<u16>) -> Self {
        let end = buf.len().try_into().unwrap();
        Self {
            full: Rc::new(buf),
            start: 0,
            end,
        }
    }

    pub fn new_from_str(s: &str) -> Self {
        let full: Rc<Vec<u16>> = Rc::new(s.encode_utf16().collect());
        let start = 0;
        let end = full.len().try_into().unwrap();
        JSString { full, start, end }
    }

    pub fn view(&self) -> &[u16] {
        &self.full[self.start as usize..self.end as usize]
    }

    /// Returns a new JSString that offers a narrower view into the same string
    /// buffer than what `self` offers.
    ///
    /// The returned window covers the [ofs_start, ofs_end) interval (right-open
    /// interval!), where 0 is the start of `self` (regardless of the buffer).
    /// The interval is implicitly truncated if it's "too far on the right".
    /// This implies that invalid indices can return a shorter substring than
    /// expected, or even the empty string.
    pub fn substring(&self, ofs_start: u32, ofs_end: u32) -> Self {
        let len_u32 = self.end - self.start;
        let ofs_start = ofs_start.min(len_u32);
        let ofs_end = ofs_end.min(len_u32).max(ofs_start);

        let start = self.start + ofs_start;
        let end = self.start + ofs_end;
        debug_assert!(end <= self.full.len().try_into().unwrap());

        JSString {
            full: Rc::clone(&self.full),
            start,
            end,
        }
    }

    pub(crate) fn to_string(&self) -> String {
        String::from_utf16_lossy(self.view())
    }
}

impl std::fmt::Debug for JSString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start = self.start as usize;
        let end = self.end as usize;
        if (start, end) != (0, self.full.len()) {
            write!(f, "(partial) ")?;
        }

        let view = self.view();
        if view.len() > 100 {
            let as_str = String::from_utf16_lossy(&view[0..100]);
            write!(f, "{:?} ... (total {} chars)", as_str, view.len())
        } else {
            let as_str = String::from_utf16_lossy(view);
            write!(f, "{:?}", as_str)
        }
    }
}
impl PartialEq for JSString {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || self.view() == other.view()
    }
}
impl Eq for JSString {}

#[cfg(test)]
mod tests_js_string {
    use super::JSString;

    #[test]
    fn test_basic_sanity() {
        let my_str = "asdlol123";
        let jss = JSString::new_from_str(&my_str);
        let check: Vec<_> = my_str.encode_utf16().collect();
        assert_eq!(jss.view(), check);
    }

    #[test]
    fn test_substring() {
        let my_str = "asdlol123";
        let jss = JSString::new_from_str(&my_str);
        let check: Vec<_> = "lol123".encode_utf16().collect();
        assert_eq!(jss.substring(3, 9).view(), check);
    }

    #[test]
    fn test_double_substring() {
        let my_str = "asdlol123";
        let jss0 = JSString::new_from_str(&my_str);
        assert_eq!(jss0.view().len(), 9);
        let jss1 = jss0.substring(3, 9); // "lol123"
        assert_eq!(jss1.view().len(), 6);
        let jss2 = jss1.substring(2, 5); // "l12"
        assert_eq!(jss2.view().len(), 3);

        let check: Vec<_> = "l12".encode_utf16().collect();
        assert_eq!(jss2.view(), check);
    }

    #[test]
    fn test_substring_out_of_range_end() {
        let my_str = "asdlol123";
        let jss = JSString::new_from_str(&my_str);
        let jss1 = jss.substring(3, 10);
        assert_eq!(jss1.to_string(), "lol123");
    }

    #[test]
    fn test_substring_out_of_range_inv() {
        let my_str = "asdlol123";
        let jss = JSString::new_from_str(&my_str);
        let jss1 = jss.substring(7, 5);
        assert_eq!(jss1.view().len(), 0);
    }
}
