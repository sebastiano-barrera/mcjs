use std::collections::HashMap;

use crate::interpreter::Value;

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

    pub(crate) fn set_property(&mut self, oid: ObjectId, key: ObjectKey, value: Value) {
        let obj = self.objects.get_mut(oid).unwrap();
        obj.set(key, value);
    }

    pub(crate) fn get_property(&self, oid: ObjectId, key: &ObjectKey) -> Option<&Value> {
        let obj = self.objects.get(oid).unwrap();
        obj.get(key)
    }
}

slotmap::new_key_type! { pub struct ObjectId; }

#[derive(Debug, PartialEq, Clone)]
pub struct Object(HashMap<ObjectKey, Value>);

impl Object {
    fn new() -> Self {
        Object(HashMap::new())
    }

    fn get(&self, key: &ObjectKey) -> Option<&Value> {
        self.0.get(key)
    }

    fn set(&mut self, key: ObjectKey, value: Value) {
        self.0.insert(key, value);
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ObjectKey {
    String(String),
}
impl ObjectKey {
    pub(crate) fn from_value(value: &Value) -> Option<ObjectKey> {
        if let Value::String(s) = value {
            Some(ObjectKey::String(s.to_string()))
        } else {
            None
        }
    }
}
impl<S: ToString> From<S> for ObjectKey {
    fn from(value: S) -> Self {
        ObjectKey::String(value.to_string())
    }
}
