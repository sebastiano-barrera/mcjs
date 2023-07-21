#![allow(unused_variables)]

use std::{
    borrow::{Borrow, Cow},
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use enum_dispatch::enum_dispatch;

use crate::interpreter::{Closure, Value};

/// Common interface implemented by types that are sufficently "object-like" to be exposed
/// as objects in JavaScript code.
///
/// This is used to wrap non-object data (e.g. strings, arrays, numbers, booleans, ...) in
/// an object-like shell.
///
/// JS objects behave as if they had a 'dict-like' part and an 'array' part.  While still
/// maintaining comparable behavior from the caller's point of view, Arrays and Objects
/// may store data in different internal structures.
#[enum_dispatch]
pub trait Object {
    fn get_property(&self, key: &str) -> Option<Value>;
    fn set_property(&mut self, key: String, value: Value);
    fn delete_property(&mut self, key: &str);

    fn properties<'a>(&'a self) -> ObjectProperties<'a>;

    fn len(&self) -> usize;
    fn get_element(&self, index: usize) -> Option<Value>;
    fn set_element(&mut self, index: usize, value: Value);
    fn delete_element(&mut self, index: usize);

    fn type_of(&self) -> Typeof;

    fn proto(&self) -> Option<ObjectId>;
}

pub trait ObjectExt {
    fn get_element_or_property<S: Deref<Target = str>>(
        &self,
        index_or_key: IndexOrKey<S>,
    ) -> Option<Value>;
    fn set_element_or_property<S: Deref<Target = str>>(
        &mut self,
        index_or_key: IndexOrKey<S>,
        value: Value,
    );
    fn delete_element_or_property<S: Deref<Target = str>>(&mut self, index_or_key: IndexOrKey<S>);
}

impl<O: ?Sized + Object> ObjectExt for O {
    fn get_element_or_property<S: Deref<Target = str>>(
        &self,
        index_or_key: IndexOrKey<S>,
    ) -> Option<Value> {
        match index_or_key {
            IndexOrKey::Index(ndx) => self.get_element(ndx),
            IndexOrKey::Key(key) => self.get_property(key.as_ref()),
        }
    }
    fn set_element_or_property<S: Deref<Target = str>>(
        &mut self,
        index_or_key: IndexOrKey<S>,
        value: Value,
    ) {
        match index_or_key {
            IndexOrKey::Index(ndx) => self.set_element(ndx, value),
            IndexOrKey::Key(key) => self.set_property(key.to_owned(), value),
        }
    }
    fn delete_element_or_property<S: Deref<Target = str>>(&mut self, index_or_key: IndexOrKey<S>) {
        match index_or_key {
            IndexOrKey::Index(ndx) => self.delete_element(ndx),
            IndexOrKey::Key(key) => self.delete_property(&key),
        }
    }
}

type ObjectProperties<'a> = Box<dyn 'a + ExactSizeIterator<Item = &'a str>>;

pub enum IndexOrKey<S: Deref<Target = str>> {
    Index(usize),
    Key(S),
}

#[derive(Clone, Copy)]
pub enum Typeof {
    Object,
    Function,
    String,
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
}

impl Heap {
    pub(crate) fn new() -> Heap {
        let mut objects = slotmap::SlotMap::with_key();
        let number_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));
        let string_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));
        let func_proto = objects.insert(RefCell::new(OrdObject::new(HashMap::new()).into()));

        Heap {
            objects,
            number_proto,
            string_proto,
            func_proto,
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

    pub(crate) fn new_ordinary_object(&mut self, properties: HashMap<String, Value>) -> ObjectId {
        // TODO .into() should work here.  Why doesn't it?
        self.objects
            .insert(RefCell::new(OrdObject::new(properties).into()))
    }
    pub(crate) fn new_array(&mut self, elements: Vec<Value>) -> ObjectId {
        self.objects
            .insert(RefCell::new(OrdObject::new_array(elements).into()))
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
        let mut cur_proto = match obj.proto() {
            Some(proto_id) if proto_id == sup_oid => return true,
            Some(proto_id) => self.get(proto_id),
            None => return false,
        };

        while let Some(proto) = cur_proto {
            match obj.proto() {
                Some(proto_id) if proto_id == sup_oid => return true,
                Some(proto_id) => {
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

    pub fn get_property_chained(&self, oid: ObjectId, key: &str) -> Option<Value> {
        let obj = self.objects.get(oid).unwrap().borrow();
        let value = obj.get_property(key);

        if value.is_some() {
            value
        } else if let Some(proto_id) = obj.proto() {
            self.get_property_chained(proto_id, key)
        } else {
            None
        }
    }
}

#[enum_dispatch(Object)]
pub enum HeapObject {
    OrdObject,
    StringObject,
    ClosureObject,
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

    // I just didn't want to make a whole ArrayHeap, ArrayId, etc.
    // Important: OrdObjects are either created with or without an array part.  They MUST NOT gain
    // or lose one during their lifecycle.
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

    fn get_property(&self, key: &str) -> Option<Value> {
        if key == "__proto__" {
            return self.proto_id.map(Value::Object);
        }
        self.properties.get(key).copied()
    }
    fn set_property(&mut self, key: String, value: Value) {
        if key == "__proto__" {
            if let Value::Object(new_proto_id) = value {
                self.proto_id = Some(new_proto_id);
            } else {
                // Apparently this is simply a nop in V8?
            }
        } else {
            self.properties.insert(key, value);
        }
    }
    fn delete_property(&mut self, key: &str) {
        self.properties.remove(key);
    }
    fn properties(&self) -> ObjectProperties {
        Box::new(self.properties.keys().map(|s| s.as_str()))
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
                self.set_property(ndx_str, value)
            }
        }
    }
    fn get_element(&self, ndx: usize) -> Option<Value> {
        match &self.array_part {
            Some(arrp) => arrp.get(ndx).cloned(),
            None => {
                let ndx_str = format!("{}", ndx);
                self.get_property(&ndx_str)
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
                self.set_property(ndx_str, Value::Undefined);
            }
        }
    }

    fn len(&self) -> usize {
        self.array_part.as_ref().map(|arrp| arrp.len()).unwrap_or(0)
    }

    fn proto(&self) -> Option<ObjectId> {
        self.proto_id
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
    fn get_property(&self, key: &str) -> Option<Value> {
        self.properties.get(key).copied()
    }

    fn set_property(&mut self, key: String, value: Value) {
        self.properties.insert(key, value);
    }

    fn delete_property(&mut self, key: &str) {
        self.properties.remove(key);
    }

    fn properties<'a>(&'a self) -> ObjectProperties<'a> {
        Box::new(std::iter::empty())
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
        Typeof::Object
    }

    fn proto(&self) -> Option<ObjectId> {
        todo!()
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
    fn get_property(&self, key: &str) -> Option<Value> {
        todo!()
    }

    fn set_property(&mut self, key: String, value: Value) {
        todo!()
    }

    fn delete_property(&mut self, key: &str) {
        todo!()
    }

    fn properties<'s>(&'s self) -> ObjectProperties<'s> {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn get_element(&self, index: usize) -> Option<Value> {
        todo!()
    }

    fn set_element(&mut self, index: usize, value: Value) {
        todo!()
    }

    fn delete_element(&mut self, index: usize) {
        todo!()
    }

    fn type_of(&self) -> Typeof {
        todo!()
    }

    fn proto(&self) -> Option<ObjectId> {
        todo!()
    }
}

pub struct NumberObject(pub f64);
impl Object for NumberObject {
    fn get_property(&self, key: &str) -> Option<Value> {
        todo!()
    }

    fn set_property(&mut self, key: String, value: Value) {
        todo!()
    }

    fn delete_property(&mut self, key: &str) {
        todo!()
    }

    fn properties<'a>(&'a self) -> ObjectProperties<'a> {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn get_element(&self, index: usize) -> Option<Value> {
        todo!()
    }

    fn set_element(&mut self, index: usize, value: Value) {
        todo!()
    }

    fn delete_element(&mut self, index: usize) {
        todo!()
    }

    fn type_of(&self) -> Typeof {
        todo!()
    }

    fn proto(&self) -> Option<ObjectId> {
        todo!()
    }
}

pub struct BoolObject(pub bool);
impl Object for BoolObject {
    fn get_property(&self, key: &str) -> Option<Value> {
        todo!()
    }

    fn set_property(&mut self, key: String, value: Value) {
        todo!()
    }

    fn delete_property(&mut self, key: &str) {
        todo!()
    }

    fn properties<'a>(&'a self) -> ObjectProperties<'a> {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn get_element(&self, index: usize) -> Option<Value> {
        todo!()
    }

    fn set_element(&mut self, index: usize, value: Value) {
        todo!()
    }

    fn delete_element(&mut self, index: usize) {
        todo!()
    }

    fn type_of(&self) -> Typeof {
        todo!()
    }

    fn proto(&self) -> Option<ObjectId> {
        todo!()
    }
}

//
// ValueObject
//

/// A representation of a generic Value as an Object.
pub enum ValueObjectRef<'p> {
    NumberObject(NumberObject),
    BoolObject(BoolObject),
    HeapObject(Ref<'p, HeapObject>),
}

impl<'p> ValueObjectRef<'p> {
    pub fn as_object(&self) -> &dyn Object {
        match self {
            ValueObjectRef::NumberObject(no) => no as &dyn Object,
            ValueObjectRef::BoolObject(bo) => bo as &dyn Object,
            ValueObjectRef::HeapObject(ho) => ho.deref(),
        }
    }

    pub fn as_string(self) -> Option<Ref<'p, String>> {
        match self {
            ValueObjectRef::HeapObject(ho) => string_of_object(ho).ok(),
            _ => None,
        }
    }
}

pub fn string_of_object(ho: Ref<HeapObject>) -> Result<Ref<String>, Ref<HeapObject>> {
    Ref::filter_map(ho, |ho| match ho {
        HeapObject::StringObject(sobj) => Some(sobj.string()),
        _ => None,
    })
}

/// Mutable variant of ValueObjectRef
pub enum ValueObjectMut<'p> {
    NumberObject(NumberObject),
    BoolObject(BoolObject),
    HeapObject(RefMut<'p, HeapObject>),
}

impl<'p> ValueObjectMut<'p> {
    pub fn as_object_mut(&mut self) -> &mut dyn Object {
        match self {
            ValueObjectMut::NumberObject(no) => no as &mut dyn Object,
            ValueObjectMut::BoolObject(bo) => bo as &mut dyn Object,
            ValueObjectMut::HeapObject(ho) => ho.deref_mut(),
        }
    }
}
