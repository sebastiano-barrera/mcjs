use slotmap::SlotMap;
use std::{cell::Cell, marker::PhantomData};

pub(crate) use crate::interpreter::Value;

slotmap::new_key_type! {
    struct FrameId;
}

/// A graph of activation frames (with a terrible name; I'm avoiding the word
/// "stack" as it pretty quickly becomes a graph, as soon as a closure is
/// created).
pub(crate) struct FrameGraph {
    frames: SlotMap<FrameId, Frame>,
}

impl FrameGraph {
    pub(crate) fn new() -> Self {
        FrameGraph {
            frames: SlotMap::with_key(),
        }
    }

    pub(crate) fn new_frame(
        &mut self,
        n_local_vars: usize,
        lexical_parent: Option<FrameHandle>,
    ) -> FrameHandle {
        let variables = vec![Value::Undefined; n_local_vars].into_boxed_slice();
        let lexical_parent = lexical_parent.map(|handle| handle.0);
        let key = self.frames.insert(Frame {
            lexical_parent,
            variables,
        });

        FrameHandle(key)
    }

    pub(crate) fn get(&self, handle: &FrameHandle) -> Option<&Frame> {
        self.frames.get(handle.0)
    }

    pub(crate) fn get_mut(&mut self, handle: &FrameHandle) -> Option<&mut Frame> {
        self.frames.get_mut(handle.0)
    }

    pub(crate) fn get_lexical_scope(
        &self,
        handle: &FrameHandle,
        offset: crate::interpreter::FrameOffset,
    ) -> Option<FrameHandle> {
        let mut key = handle.0;

        for _ in 0..offset.0 {
            let frame = self.frames.get(key)?;
            key = frame.lexical_parent.clone()?;
        }

        Some(FrameHandle(key))
    }

    pub(crate) fn get_var(&self, handle: &FrameHandle, var_ndx: u16) -> Option<Value> {
        let frame = self.get(handle)?;
        frame.variables.get(var_ndx as usize).cloned()
    }

    pub(crate) fn set_var(&mut self, handle: &FrameHandle, var_ndx: u16, value: Value) {
        let frame = self.get_mut(&handle).unwrap();
        *frame.variables.get_mut(var_ndx as usize).unwrap() = value;
    }
}

pub(crate) struct Frame {
    lexical_parent: Option<FrameId>,
    variables: Box<[Value]>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct FrameHandle(FrameId);
