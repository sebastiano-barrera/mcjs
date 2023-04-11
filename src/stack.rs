use slotmap::SlotMap;
use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

pub(crate) use crate::interpreter::Value;
use crate::bytecode::{FnId, VarIndex};

slotmap::new_key_type! {
    pub(crate) struct FrameId;
}

/// A graph of activation frames (with a terrible name; I'm avoiding the word
/// "stack" as it pretty quickly becomes a graph, as soon as a closure is
/// created).
pub(crate) struct FrameGraph {
    frames: SlotMap<FrameId, Frame>,
}

pub(crate) struct Frame {
    fnid: FnId,
    lexical_parent: Option<FrameId>,
    variables: Box<[Value]>,
}

impl FrameGraph {
    pub(crate) fn new() -> Self {
        FrameGraph {
            frames: SlotMap::with_key(),
        }
    }

    pub(crate) fn new_frame(
        &mut self,
        fnid: FnId,
        n_local_vars: usize,
        lexical_parent: Option<FrameId>,
    ) -> FrameId {
        let variables = vec![Value::Undefined; n_local_vars].into_boxed_slice();
        self.frames.insert(Frame {
            fnid,
            lexical_parent,
            variables,
        })
    }

    pub(crate) fn get(&self, fid: FrameId) -> Option<&Frame> {
        self.frames.get(fid)
    }

    pub(crate) fn get_mut(&mut self, fid: FrameId) -> Option<&mut Frame> {
        self.frames.get_mut(fid)
    }

    pub(crate) fn get_var(&self, fid: FrameId, var_ndx: VarIndex) -> Option<Value> {
        self.get(fid)?.variables.get(var_ndx.0 as usize).cloned()
    }

    pub(crate) fn set_var(&mut self, fid: FrameId, var_ndx: VarIndex, value: Value) {
        let frame = self.get_mut(fid).unwrap();
        *frame.variables.get_mut(var_ndx.0 as usize).unwrap() = value;
    }

    pub(crate) fn get_lexical_scope(&self, mut fid: FrameId, fnid: FnId) -> Option<FrameId> {
        loop {
            let frame = self.get(fid).unwrap();
            if frame.fnid == fnid {
                return Some(fid);
            }
            fid = frame.lexical_parent?;
        }
    }
}
