use std::collections::HashMap;

use crate::interpreter::{Closure, Value};

pub struct ObjectHeap {
    objects: slotmap::SlotMap<ObjectId, Object>,
}

impl ObjectHeap {
    pub(crate) fn new() -> ObjectHeap {
        ObjectHeap {
            objects: slotmap::SlotMap::with_key(),
        }
    }

    pub(crate) fn new_object(&mut self) -> ObjectId {
        self.objects.insert(Object::new())
    }
    pub(crate) fn new_function(&mut self, closure: Closure) -> ObjectId {
        self.objects.insert(Object::new_function(closure))
    }
    pub(crate) fn new_string(&mut self, string: String) -> ObjectId {
        self.objects.insert(Object::new_string(string))
    }

    pub fn is_instance_of(&self, oid: ObjectId, sup_oid: ObjectId) -> bool {
        let mut cur_oid = Some(oid);
        while let Some(oid) = cur_oid {
            let proto_id = self.objects.get(oid).unwrap().proto_id;
            if proto_id == Some(sup_oid) {
                return true;
            }
            cur_oid = proto_id;
        }

        false
    }

    pub fn is_array(&self, oid: ObjectId) -> Option<bool> {
        let obj = self.objects.get(oid)?;
        Some(!obj.array_items.is_empty())
    }

    pub fn get_closure(&self, oid: ObjectId) -> Option<&Closure> {
        let obj = self.objects.get(oid).unwrap();
        obj.closure.as_ref()
    }

    pub(crate) fn set_property(&mut self, oid: ObjectId, key: ObjectKey, value: Value) {
        let obj = self.objects.get_mut(oid).unwrap();

        match key {
            ObjectKey::ArrayIndex(ndx) => {
                obj.set_arr_element(ndx, value);
            }
            ObjectKey::Property(pkey) => {
                let PropertyKey::String(key_str) = &pkey;
                if key_str == "__proto__" {
                    if let Value::Object(new_proto_id) = value {
                        obj.proto_id = Some(new_proto_id);
                    } else {
                        // Apparently this is simply a nop in V8?
                    }

                    return;
                }

                obj.set_property(pkey, value);
            }
        }
    }

    pub fn get_property(&self, oid: ObjectId, key: &ObjectKey) -> Option<Value> {
        let obj = self.objects.get(oid).unwrap();

        let value = match key {
            // TODO(performance) Right now this .cloned() is inefficient due to the fact that
            // we may be copying a whole string.
            ObjectKey::ArrayIndex(ndx) => obj.get_arr_element(*ndx).cloned(),
            ObjectKey::Property(pkey) => {
                let PropertyKey::String(key_str) = &pkey;
                if key_str == "__proto__" {
                    return obj.proto_id.map(Value::Object);
                }

                // TODO(performance) Right now this .cloned() is inefficient due to the fact that
                // we may be copying a whole string.
                obj.get_property(pkey).cloned()
            }
        };

        if value.is_some() {
            value
        } else if let Some(proto_id) = obj.proto_id {
            self.get_property(proto_id, key)
        } else {
            None
        }
    }

    pub(crate) fn delete_property(&mut self, oid: ObjectId, key: &ObjectKey) -> Option<Value> {
        let obj = self.objects.get_mut(oid).unwrap();

        match key {
            // TODO(performance) Right now this .cloned() is inefficient due to the fact that
            // we may be copying a whole string.
            ObjectKey::ArrayIndex(ndx) => obj.delete_arr_element(*ndx),
            ObjectKey::Property(pkey) => {
                let PropertyKey::String(key_str) = &pkey;
                if key_str == "__proto__" {
                    return None;
                }
                obj.delete_property(pkey)
            }
        }
    }

    // TODO(performance) This stuff works, but is terribly inefficient.
    //
    // - Allocating a whole ass object just to use its array part puts pressure on the garbage
    // collector
    // - Object keys have to be boxed in order to be put in the array
    // - Object keys then have to be *unboxed* (!) in order to be used after getting them out
    //   of
    // the array
    //
    // There has to be a better way, but it probably involves writing my own HashMap, and I'm
    // not about that right now
    pub(crate) fn get_keys_as_array(&mut self, oid: ObjectId) -> ObjectId {
        let obj = self.objects.get(oid).unwrap();

        let mut ret = Object::new();
        ret.array_items = obj
            .get_keys()
            .into_iter()
            .map(|key| match key {
                ObjectKey::ArrayIndex(ndx) => Value::Number(ndx as f64),
                ObjectKey::Property(PropertyKey::String(name)) => {
                    let oid = self.new_string(name.clone());
                    Value::Object(oid)
                }
            })
            .collect();

        self.objects.insert(ret)
    }

    pub fn array_len(&self, oid: ObjectId) -> usize {
        let obj = self.objects.get(oid).unwrap();
        obj.array_items.len()
    }

    pub fn array_nth(&self, oid: ObjectId, ndx: usize) -> Option<Value> {
        let obj = self.objects.get(oid).unwrap();
        obj.array_items.get(ndx).cloned()
    }

    pub(crate) fn array_push(&mut self, oid: ObjectId, value: Value) {
        let obj = self.objects.get_mut(oid).unwrap();
        obj.array_items.push(value);
    }

    pub fn get_typeof(&self, oid: ObjectId) -> Typeof {
        let obj = self.objects.get(oid).unwrap();
        obj.type_of()
    }

    pub fn get_string(&self, oid: ObjectId) -> Option<&str> {
        let obj = self.objects.get(oid).unwrap();
        obj.as_str()
    }
}

slotmap::new_key_type! { pub struct ObjectId; }

#[derive(Clone, Copy)]
pub enum Typeof {
    Object,
    Function,
    String,
}

#[derive(Debug, Clone)]
struct Object {
    proto_id: Option<ObjectId>,
    properties: HashMap<PropertyKey, Value>,
    array_items: Vec<Value>,
    closure: Option<Closure>,
    string_payload: Option<String>,
}

impl Object {
    fn new() -> Self {
        Object {
            proto_id: None,
            properties: HashMap::new(),
            array_items: Vec::new(),
            closure: None,
            string_payload: None,
        }
    }

    fn new_function(closure: Closure) -> Self {
        Object {
            proto_id: None,
            properties: HashMap::new(),
            array_items: Vec::new(),
            closure: Some(closure),
            string_payload: None,
        }
    }

    fn new_string(string: String) -> Object {
        Object {
            proto_id: None,
            properties: HashMap::new(),
            array_items: Vec::new(),
            closure: None,
            string_payload: Some(string),
        }
    }

    fn type_of(&self) -> Typeof {
        if self.string_payload.is_some() {
            Typeof::String
        } else if self.closure.is_some() {
            Typeof::Function
        } else {
            Typeof::Object
        }
    }

    fn as_str(&self) -> Option<&str> {
        self.string_payload.as_ref().map(|s| s.as_str())
    }

    fn get_property(&self, key: &PropertyKey) -> Option<&Value> {
        self.properties.get(key)
    }
    fn set_property(&mut self, key: PropertyKey, value: Value) {
        self.properties.insert(key, value);
    }

    fn set_arr_element(&mut self, ndx: usize, value: Value) {
        if ndx >= self.array_items.len() {
            self.array_items.resize(ndx + 1, Value::Undefined)
        }
        self.array_items[ndx] = value;
    }
    fn get_arr_element(&self, ndx: usize) -> Option<&Value> {
        self.array_items.get(ndx)
    }

    fn get_keys(&self) -> Vec<ObjectKey> {
        (0..self.array_items.len())
            .into_iter()
            .map(|ndx| ObjectKey::ArrayIndex(ndx))
            .chain(
                self.properties
                    .keys()
                    .map(|pkey| ObjectKey::Property(pkey.clone())),
            )
            .collect()
    }

    fn delete_arr_element(&mut self, ndx: usize) -> Option<Value> {
        if ndx < self.array_items.len() {
            Some(self.array_items.remove(ndx))
        } else {
            None
        }
    }

    fn delete_property(&mut self, pkey: &PropertyKey) -> Option<Value> {
        self.properties.remove(pkey)
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ObjectKey {
    ArrayIndex(usize),
    Property(PropertyKey),
}
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum PropertyKey {
    String(String),
}
impl From<String> for ObjectKey {
    fn from(value: String) -> Self {
        ObjectKey::Property(PropertyKey::String(value))
    }
}
