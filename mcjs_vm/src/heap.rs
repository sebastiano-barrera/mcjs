#![allow(unused_variables)]

use std::{
    borrow::{Borrow, Cow},
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::interpreter::{Closure, Value};

/// Common interface implemented by types that are sufficently "object-like" to be exposed
/// as objects in JavaScript code.   This includes all objects: ordinary and exotic,
/// immediate and heap-stored.
///
/// This interface only allows access to the object's "own" properties, without using the
/// prototype.  For that, you need use a `Heap` and its methods.
// TODO Will this design work even though this trait is not object-safe?
pub trait Object {
    fn get_own_property(&self, key: &str) -> Option<Value>;
    fn set_own_property(&mut self, key: String, value: Value);
    fn delete_own_property(&mut self, key: &str);

    // I know, inefficient, but so, *so* simple, and enum_dispatch-able.
    // TODO: make it slightly better, return a Cow<'static, str>
    fn own_properties<'a>(&'a self) -> Vec<String>;

    fn len(&self) -> usize;
    fn get_element(&self, index: usize) -> Option<Value>;
    fn set_element(&mut self, index: usize, value: Value);
    fn delete_element(&mut self, index: usize);

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

pub trait ObjectExt {
    fn get_own_element_or_property<S: Deref<Target = str>>(
        &self,
        index_or_key: IndexOrKey<S>,
    ) -> Option<Value>;
    fn set_own_element_or_property<S: Deref<Target = str>>(
        &mut self,
        index_or_key: IndexOrKey<S>,
        value: Value,
    );
    fn delete_own_element_or_property<S: Deref<Target = str>>(
        &mut self,
        index_or_key: IndexOrKey<S>,
    );
}

impl<O: ?Sized + Object> ObjectExt for O {
    fn get_own_element_or_property<S: Deref<Target = str>>(
        &self,
        index_or_key: IndexOrKey<S>,
    ) -> Option<Value> {
        match index_or_key {
            IndexOrKey::Index(ndx) => self.get_element(ndx),
            IndexOrKey::Key(key) => self.get_own_property(key.as_ref()),
        }
    }
    fn set_own_element_or_property<S: Deref<Target = str>>(
        &mut self,
        index_or_key: IndexOrKey<S>,
        value: Value,
    ) {
        match index_or_key {
            IndexOrKey::Index(ndx) => self.set_element(ndx, value),
            IndexOrKey::Key(key) => self.set_own_property(key.to_owned(), value),
        }
    }
    fn delete_own_element_or_property<S: Deref<Target = str>>(
        &mut self,
        index_or_key: IndexOrKey<S>,
    ) {
        match index_or_key {
            IndexOrKey::Index(ndx) => self.delete_element(ndx),
            IndexOrKey::Key(key) => self.delete_own_property(&key),
        }
    }
}

pub enum IndexOrKey<S: Deref<Target = str>> {
    Index(usize),
    Key(S),
}

//
// Ordinary objects
//

slotmap::new_key_type! { pub struct ObjectId; }

pub struct Heap {
    objects: slotmap::SlotMap<ObjectId, RefCell<HeapObject>>,

    number_proto: ObjectId,
    string_proto: ObjectId,
    func_proto: ObjectId,
    array_proto: ObjectId,
    bool_proto: ObjectId,
}

impl Heap {
    pub(crate) fn new() -> Heap {
        let mut objects = slotmap::SlotMap::with_key();
        let number_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));
        let string_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));
        let func_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));
        let bool_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));
        let array_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));

        Heap {
            objects,
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
    pub fn string_proto(&self) -> ObjectId {
        self.string_proto
    }
    pub fn func_proto(&self) -> ObjectId {
        self.func_proto
    }
    pub fn array_proto(&self) -> ObjectId {
        self.array_proto
    }
    pub fn bool_proto(&self) -> ObjectId {
        self.bool_proto
    }

    pub(crate) fn new_ordinary_object(&mut self, properties: HashMap<String, Value>) -> ObjectId {
        self.objects
            .insert(RefCell::new(OrdObject::new(properties).into()))
    }
    pub(crate) fn new_array(&mut self, elements: Vec<Value>) -> ObjectId {
        let mut obj = OrdObject::new_array(elements);
        obj.proto_id = Some(self.array_proto);
        self.objects.insert(RefCell::new(obj.into()))
    }
    pub(crate) fn new_function(
        &mut self,
        closure: Closure,
        properties: HashMap<String, Value>,
    ) -> ObjectId {
        let cobj = ClosureObject {
            closure,
            properties,
        };
        self.objects.insert(RefCell::new(cobj.into()))
    }
    pub(crate) fn new_string(&mut self, string: String) -> ObjectId {
        self.objects
            .insert(RefCell::new(StringObject(string).into()))
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
            match proto.as_object().proto(self) {
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

    pub fn get(&self, oid: ObjectId) -> Option<Ref<HeapObject>> {
        self.objects.get(oid).map(|refcell| refcell.borrow())
    }
    pub fn get_mut(&self, oid: ObjectId) -> Option<RefMut<HeapObject>> {
        self.objects.get(oid).map(|refcell| refcell.borrow_mut())
    }

    pub fn get_property_chained<O: ?Sized + Object>(&self, obj: &O, key: &str) -> Option<Value> {
        let value = obj.get_own_property(key);

        if value.is_some() {
            value
        } else if let Some(proto_id) = obj.proto(self) {
            let proto = self.get(proto_id)?;
            self.get_property_chained(proto.deref().as_object(), key)
        } else {
            None
        }
    }
}

pub enum HeapObject {
    OrdObject(OrdObject),
    StringObject(StringObject),
    ClosureObject(ClosureObject),
}
impl HeapObject {
    pub fn as_object(&self) -> &dyn Object {
        match self {
            HeapObject::OrdObject(oo) => oo as &dyn Object,
            HeapObject::StringObject(so) => so as &dyn Object,
            HeapObject::ClosureObject(co) => co as &dyn Object,
        }
    }

    pub fn as_object_mut(&mut self) -> &mut dyn Object {
        match self {
            HeapObject::OrdObject(oo) => oo as &mut dyn Object,
            HeapObject::StringObject(so) => so as &mut dyn Object,
            HeapObject::ClosureObject(co) => co as &mut dyn Object,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            HeapObject::StringObject(so) => Some(so.0.as_str()),
            _ => None,
        }
    }

    pub fn as_closure(&self) -> Option<&Closure> {
        match self {
            HeapObject::ClosureObject(ClosureObject { closure, .. }) => Some(closure),
            _ => None,
        }
    }
}
impl From<OrdObject> for HeapObject {
    fn from(inner: OrdObject) -> Self {
        HeapObject::OrdObject(inner)
    }
}
impl From<StringObject> for HeapObject {
    fn from(inner: StringObject) -> Self {
        HeapObject::StringObject(inner)
    }
}
impl From<ClosureObject> for HeapObject {
    fn from(inner: ClosureObject) -> Self {
        HeapObject::ClosureObject(inner)
    }
}

/// An ordinary object, i.e. one that you can create in JavaScript with the `{a: 1, b: 2}`
/// syntax.
///
/// It stores key-value pairs where keys ("properties") are strings, and any JS value can
/// be stored as value.
#[derive(Debug, Clone)]
pub struct OrdObject {
    proto_id: Option<ObjectId>,
    properties: HashMap<String, Value>,

    // TODO Move this to a specific Array object type?
    array_part: Option<Vec<Value>>,
}

impl OrdObject {
    fn new(properties: HashMap<String, Value>) -> Self {
        OrdObject {
            proto_id: None,
            properties,
            array_part: None,
        }
    }

    fn new_array(elements: Vec<Value>) -> Self {
        OrdObject {
            proto_id: None,
            properties: HashMap::new(),
            array_part: Some(elements),
        }
    }

    pub(crate) fn is_array(&self) -> bool {
        self.array_part.is_some()
    }
}

impl Object for OrdObject {
    fn type_of(&self) -> Typeof {
        Typeof::Object
    }

    fn get_own_property(&self, key: &str) -> Option<Value> {
        if key == "__proto__" {
            // TODO Delete
            return self.proto_id.map(Value::Object);
        }
        self.properties.get(key).copied()
    }
    fn set_own_property(&mut self, key: String, value: Value) {
        if key == "__proto__" {
            // TODO Delete
            if let Value::Object(new_proto_id) = value {
                self.proto_id = Some(new_proto_id);
            } else {
                // Apparently this is simply a nop in V8?
            }
        } else {
            self.properties.insert(key, value);
        }
    }
    fn delete_own_property(&mut self, key: &str) {
        self.properties.remove(key);
    }

    fn own_properties(&self) -> Vec<String> {
        self.properties.keys().cloned().collect()
    }

    fn set_element(&mut self, ndx: usize, value: Value) {
        match &mut self.array_part {
            Some(arrp) => {
                if arrp.len() < ndx + 1 {
                    arrp.resize(ndx + 1, Value::Undefined);
                }
                arrp[ndx] = value;
            }
            None => {
                let ndx_str = format!("{}", ndx);
                self.set_own_property(ndx_str, value)
            }
        }
    }
    fn get_element(&self, ndx: usize) -> Option<Value> {
        match &self.array_part {
            Some(arrp) => arrp.get(ndx).cloned(),
            None => {
                let ndx_str = format!("{}", ndx);
                self.get_own_property(&ndx_str)
            }
        }
    }
    fn delete_element(&mut self, ndx: usize) {
        match &mut self.array_part {
            Some(arrp) => {
                // TODO Shorten the array if the tail is all Undefined's?
                arrp[ndx] = Value::Undefined;
            }
            None => {
                let ndx_str = ndx.to_string();
                self.set_own_property(ndx_str, Value::Undefined);
            }
        }
    }

    fn len(&self) -> usize {
        self.array_part.as_ref().map(|arrp| arrp.len()).unwrap_or(0)
    }

    fn proto(&self, heap: &Heap) -> Option<ObjectId> {
        self.proto_id
    }

    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        self.proto_id = proto_id;
    }
}

//
// Closures
//

pub struct ClosureObject {
    closure: Closure,
    properties: HashMap<String, Value>,
}

impl ClosureObject {
    pub fn closure(&self) -> &Closure {
        &self.closure
    }
}

impl Object for ClosureObject {
    fn get_own_property(&self, key: &str) -> Option<Value> {
        self.properties.get(key).copied()
    }

    fn set_own_property(&mut self, key: String, value: Value) {
        self.properties.insert(key, value);
    }

    fn delete_own_property(&mut self, key: &str) {
        self.properties.remove(key);
    }

    fn own_properties<'a>(&'a self) -> Vec<String> {
        Vec::new()
    }

    fn len(&self) -> usize {
        0
    }

    fn get_element(&self, index: usize) -> Option<Value> {
        None
    }

    fn set_element(&mut self, index: usize, value: Value) {}

    fn delete_element(&mut self, index: usize) {}

    fn type_of(&self) -> Typeof {
        Typeof::Function
    }

    fn proto(&self, heap: &Heap) -> Option<ObjectId> {
        Some(heap.func_proto())
    }

    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        // Nop
    }
}

pub struct StringObject(String);
impl StringObject {
    pub fn string(&self) -> &String {
        &self.0
    }
    pub fn string_mut(&mut self) -> &mut String {
        &mut self.0
    }
}
impl Object for StringObject {
    fn get_own_property(&self, key: &str) -> Option<Value> {
        match key {
            "length" => Some(Value::Number(self.0.len() as f64)),
            _ => None,
        }
    }

    fn set_own_property(&mut self, key: String, value: Value) {
        // Nop
    }

    fn delete_own_property(&mut self, key: &str) {
        // Nop
    }

    fn own_properties<'s>(&'s self) -> Vec<String> {
        vec!["length".to_string()]
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    // TODO Right now, this would need to allocate a string on the heap, to represent a
    // 1-char-long substring of self.0.  Such a waste! Instead, because it should be
    // possible to have some sort of StrObject representing a *read-only* string or portion of
    // it. Maybe it could even replace the current StringObject
    fn get_element(&self, index: usize) -> Option<Value> {
        None
    }

    fn set_element(&mut self, index: usize, value: Value) {
        // Nop
    }

    fn delete_element(&mut self, index: usize) {
        // Nop
    }

    fn type_of(&self) -> Typeof {
        Typeof::String
    }

    fn proto(&self, heap: &Heap) -> Option<ObjectId> {
        Some(heap.string_proto())
    }

    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        // Nop
    }
}

pub struct NumberObject(pub f64);
impl Object for NumberObject {
    fn get_own_property(&self, key: &str) -> Option<Value> {
        None
    }

    fn set_own_property(&mut self, key: String, value: Value) {
        // Nop
    }

    fn delete_own_property(&mut self, key: &str) {
        // Nop
    }

    fn own_properties<'a>(&'a self) -> Vec<String> {
        Vec::new()
    }

    fn len(&self) -> usize {
        0
    }

    fn get_element(&self, index: usize) -> Option<Value> {
        None
    }

    fn set_element(&mut self, index: usize, value: Value) {
        // Nop
    }

    fn delete_element(&mut self, index: usize) {
        // Nop
    }

    fn type_of(&self) -> Typeof {
        Typeof::Number
    }

    fn proto(&self, heap: &Heap) -> Option<ObjectId> {
        Some(heap.number_proto())
    }

    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        // Nop
    }
}

pub struct BoolObject(pub bool);
impl Object for BoolObject {
    fn get_own_property(&self, key: &str) -> Option<Value> {
        None
    }

    fn set_own_property(&mut self, key: String, value: Value) {
        // Nop
    }

    fn delete_own_property(&mut self, key: &str) {
        // Nop
    }

    fn own_properties<'a>(&'a self) -> Vec<String> {
        Vec::new()
    }

    fn len(&self) -> usize {
        0
    }

    fn get_element(&self, index: usize) -> Option<Value> {
        None
    }

    fn set_element(&mut self, index: usize, value: Value) {
        // Nop
    }

    fn delete_element(&mut self, index: usize) {
        // Nop
    }

    fn type_of(&self) -> Typeof {
        Typeof::Boolean
    }

    fn proto(&self, heap: &Heap) -> Option<ObjectId> {
        Some(heap.bool_proto)
    }

    fn set_proto(&mut self, proto_id: Option<ObjectId>) {
        // Nop
    }
}

// ValueObjectRef and ValueObjectMut  must exist, in order to carry either
// an immediate object (number, boolean) inline, or a Ref<HeapObject> (the
// HeapObject contains an OrdObject, or a StringObject, etc.).  This way you
// get a uniform "Object" interface, but you can also keep the runtime-checked
// borrowed ref of a heap-allocated object, and release it correctly when it's
// time.
pub enum ValueObjectRef<'h> {
    NumberObject(NumberObject),
    BoolObject(BoolObject),
    HeapObject(Ref<'h, HeapObject>),
}
impl<'h> ValueObjectRef<'h> {
    pub fn as_object(&self) -> &dyn Object {
        match self {
            ValueObjectRef::NumberObject(no) => no as &dyn Object,
            ValueObjectRef::BoolObject(bo) => bo as &dyn Object,
            ValueObjectRef::HeapObject(ho) => ho.as_object(),
        }
    }

    pub fn as_str(self) -> Result<Ref<'h, str>, Self> {
        // This is a bit contorted because Ref::filter_map consumes self, and may return a part of
        // it, or return self back whole as it was.  Then we have to pack it back in its initial
        // form.
        match self {
            Self::HeapObject(horef) => {
                Ref::filter_map(horef, |hobj| hobj.as_str()).map_err(|hobj| Self::HeapObject(hobj))
            }
            _ => Err(self),
        }
    }
}
impl<'h> From<NumberObject> for ValueObjectRef<'h> {
    fn from(value: NumberObject) -> Self {
        Self::NumberObject(value)
    }
}
impl<'h> From<BoolObject> for ValueObjectRef<'h> {
    fn from(value: BoolObject) -> Self {
        Self::BoolObject(value)
    }
}
impl<'h> From<Ref<'h, HeapObject>> for ValueObjectRef<'h> {
    fn from(value: Ref<'h, HeapObject>) -> Self {
        Self::HeapObject(value)
    }
}

pub enum ValueObjectMut<'h> {
    NumberObject(NumberObject),
    BoolObject(BoolObject),
    HeapObject(RefMut<'h, HeapObject>),
}
impl<'h> ValueObjectMut<'h> {
    pub fn as_object_mut(&mut self) -> &mut dyn Object {
        match self {
            ValueObjectMut::NumberObject(no) => no as &mut dyn Object,
            ValueObjectMut::BoolObject(bo) => bo as &mut dyn Object,
            ValueObjectMut::HeapObject(ho) => ho.as_object_mut(),
        }
    }
}
impl<'h> From<NumberObject> for ValueObjectMut<'h> {
    fn from(value: NumberObject) -> Self {
        Self::NumberObject(value)
    }
}
impl<'h> From<BoolObject> for ValueObjectMut<'h> {
    fn from(value: BoolObject) -> Self {
        Self::BoolObject(value)
    }
}
impl<'h> From<RefMut<'h, HeapObject>> for ValueObjectMut<'h> {
    fn from(value: RefMut<'h, HeapObject>) -> Self {
        Self::HeapObject(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_properties() {
        let heap = Heap::new();
        {
            let mut number_proto = heap.get_mut(heap.number_proto()).unwrap();
            let number_proto = number_proto.deref_mut().as_object_mut();
            number_proto.set_own_property("isNumber".to_string(), Value::Bool(true));
            number_proto.set_own_property("isCool".to_string(), Value::Bool(true));
        }

        let num = NumberObject(123.0);
        assert!(num.get_own_property("isNumber").is_none());
        assert!(num.get_own_property("isCool").is_none());
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
            let mut bool_proto = heap.get_mut(heap.bool_proto()).unwrap();
            let bool_proto = bool_proto.deref_mut().as_object_mut();
            bool_proto.set_own_property("isNumber".to_string(), Value::Bool(false));
            bool_proto.set_own_property("isCool".to_string(), Value::Bool(true));
        }

        let num = BoolObject(true);
        assert!(num.get_own_property("isNumber").is_none());
        assert!(num.get_own_property("isCool").is_none());
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
            let mut array_proto = heap.get_mut(heap.array_proto()).unwrap();
            let array_proto = array_proto.deref_mut().as_object_mut();
            array_proto.set_own_property("specialArrayProperty".to_string(), Value::Number(999.0));
        }

        let arr = heap.new_array(vec![
            Value::Number(9.0),
            Value::Number(6.0),
            Value::Number(3.0),
        ]);
        let arr = heap.get(arr).unwrap();
        let arr = arr.as_object();
        assert!(arr.get_own_property("specialArrayProperty").is_none());
        assert_eq!(
            heap.get_property_chained(arr, "specialArrayProperty"),
            Some(Value::Number(999.0))
        );
        assert_eq!(heap.get_property_chained(arr, "isCool"), None);
    }
}
