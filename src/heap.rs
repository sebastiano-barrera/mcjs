use std::collections::HashMap;

use crate::interpreter::{Closure, Value};

pub(crate) struct ObjectHeap {
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

    pub(crate) fn get_closure(&self, oid: ObjectId) -> Option<&Closure> {
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

    pub(crate) fn get_property(&self, oid: ObjectId, key: &ObjectKey) -> Option<Value> {
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
                ObjectKey::Property(PropertyKey::String(name)) => Value::String(name.into()),
            })
            .collect();

        self.objects.insert(ret)
    }

    pub(crate) fn array_len(&self, oid: ObjectId) -> usize {
        let obj = self.objects.get(oid).unwrap();
        obj.array_items.len()
    }

    pub(crate) fn array_nth(&self, oid: ObjectId, ndx: usize) -> Option<Value> {
        let obj = self.objects.get(oid).unwrap();
        obj.array_items.get(ndx).cloned()
    }
}

slotmap::new_key_type! { pub struct ObjectId; }

#[derive(Debug, Clone)]
struct Object {
    proto_id: Option<ObjectId>,
    properties: HashMap<PropertyKey, Value>,
    array_items: Vec<Value>,
    closure: Option<Closure>,
}

impl Object {
    fn new() -> Self {
        Object {
            proto_id: None,
            properties: HashMap::new(),
            array_items: Vec::new(),
            closure: None,
        }
    }

    fn new_function(closure: Closure) -> Self {
        Object {
            proto_id: None,
            properties: HashMap::new(),
            array_items: Vec::new(),
            closure: Some(closure),
        }
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
pub(crate) enum ObjectKey {
    ArrayIndex(usize),
    Property(PropertyKey),
}
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub(crate) enum PropertyKey {
    String(String),
}
impl ObjectKey {
    pub(crate) fn from_value(value: &Value) -> Option<ObjectKey> {
        match value {
            Value::Number(n) if *n >= 0.0 => {
                let n_trunc = n.trunc();
                if *n == n_trunc {
                    let ndx = n_trunc as usize;
                    Some(ObjectKey::ArrayIndex(ndx))
                } else {
                    None
                }
            }
            Value::String(s) => {
                let pkey = PropertyKey::String(s.to_string());
                Some(ObjectKey::Property(pkey))
            }
            _ => None,
        }
    }
}
impl From<String> for ObjectKey {
    fn from(value: String) -> Self {
        ObjectKey::Property(PropertyKey::String(value))
    }
}
