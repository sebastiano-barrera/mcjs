use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
};

use crate::{
    interpreter::{self, StaticVarId, VarIndex},
    stack::FrameId,
};

use super::builder::{BoxedValue, ValueId, ValueType};

/// Stores a model (a "proxy" of sorts) of every stack frame and variable "seen" by the JIT.
///
/// This includes captured frames (by closure).  It also tracks which
/// assignments were "seen" by the JIT (this allows detecting variables that
/// should be resolved by trace parameter rather than by instruction ID or
/// constant folding).
pub(super) struct VarsState {
    stack_model: Vec<FrameId>,

    frame_models: HashMap<FrameId, Frame>,

    /// Associates RUNTIME variable identifiers (current frame id, VarIndex)
    /// from the interpreter to ValueIds in the SSA trace we're building.
    ///
    /// Only values for variables whose assignment has been "seen" by the JIT.
    /// This allows variables to be introduced as trace parameters when the JIT
    /// starts later than the start of the function's body.
    vid_of_varid: HashMap<DynamicVarId, ValueId>,
    // TODO(opt) More compact data structures?
    first_write_of_varid: HashMap<DynamicVarId, ValueId>,
    overwritten_vars: RefCell<HashSet<DynamicVarId>>,
    read_before_overwrite: RefCell<HashSet<DynamicVarId>>,
}

/// A "model" of the interpreter's stack, represented in terms that the JIT
/// cares about.
struct Frame {
    args: Vec<ValueId>,
    vid_of_iid: HashMap<interpreter::IID, ValueId>,
    n_vars: usize,
}

type DynamicVarId = (FrameId, VarIndex);

impl VarsState {
    pub(super) fn new() -> Self {
        VarsState {
            stack_model: Vec::new(),
            frame_models: HashMap::new(),
            vid_of_varid: HashMap::new(),
            first_write_of_varid: HashMap::new(),
            overwritten_vars: RefCell::new(HashSet::new()),
            read_before_overwrite: RefCell::new(HashSet::new()),
        }
    }

    pub(super) fn stack_depth(&self) -> usize {
        self.stack_model.len()
    }

    // TODO(big feat) Garbage collection?

    pub(super) fn push_frame(&mut self, frame_id: FrameId, args: Vec<ValueId>, n_vars: usize) {
        let frame_model = Frame {
            args,
            n_vars,
            vid_of_iid: HashMap::new(),
        };
        self.frame_models.insert(frame_id, frame_model);
        self.stack_model.push(frame_id);
    }

    pub(super) fn pop_frame(&mut self) {
        self.stack_model.pop().unwrap();
    }

    fn cur_frame_id(&self) -> FrameId {
        *self
            .stack_model
            .last()
            .expect("JIT bug: no frame model on stack (trace ended?)")
    }
    fn cur_frame(&self) -> &Frame {
        self.frame_models.get(&self.cur_frame_id()).unwrap()
    }
    fn cur_frame_mut(&mut self) -> &mut Frame {
        self.frame_models.get_mut(&self.cur_frame_id()).unwrap()
    }

    pub(super) fn get_result(&self, iid: interpreter::IID) -> Option<&ValueId> {
        self.cur_frame().vid_of_iid.get(&iid)
    }
    pub(super) fn set_result(&mut self, iid: interpreter::IID, value: ValueId) {
        self.cur_frame_mut().vid_of_iid.insert(iid, value);
    }

    pub(super) fn get_var(&self, frame_id: FrameId, var_ndx: VarIndex) -> Option<ValueId> {
        let ws = self.overwritten_vars.borrow();
        if !ws.contains(&(frame_id, var_ndx)) {
            let mut rbws = self.read_before_overwrite.borrow_mut();
            rbws.insert((frame_id, var_ndx));
        }

        self.vid_of_varid.get(&(frame_id, var_ndx)).copied()
    }
    pub(super) fn set_var(&mut self, frame_id: FrameId, var_ndx: VarIndex, value: ValueId) {
        // TODO(opt) skip this test if we're not in #[cfg(test)]?
        if let Some(frame) = self.frame_models.get(&frame_id) {
            assert!(usize::from(var_ndx.0) < frame.n_vars);
        } else {
            // The frame was not seen by the JIT before. If I'm not getting
            // confused, this means that we're capturing this variable from one
            // of the outer frames.   We still store it to "short-cut" reads to
            // it later on in the trace.
        }

        self.vid_of_varid
            .entry((frame_id, var_ndx))
            .or_insert(value);

        let key = (frame_id, var_ndx);
        if self.first_write_of_varid.contains_key(&key) {
            // Not the first write
            self.overwritten_vars.get_mut().insert(key);
        } else {
            self.first_write_of_varid.insert(key, value);
        }
    }

    pub(super) fn was_frame_seen(&self, frame_id: FrameId) -> bool {
        self.frame_models.contains_key(&frame_id)
    }

    pub(super) fn get_arg(&self, arg_ndx: usize) -> &ValueId {
        self.cur_frame()
            .args
            .get(arg_ndx)
            .expect("bytecode_compiler bug: unbound arg var")
    }

    pub(super) fn get_reads_before_overwritten(&self) -> HashSet<(FrameId, VarIndex)> {
        let rbw = self.read_before_overwrite.borrow();
        let ws = self.overwritten_vars.borrow();
        rbw.intersection(&ws).copied().collect()
    }

    pub(super) fn get_first_write(&self, frame_id: FrameId, var_ndx: VarIndex) -> Option<ValueId> {
        self.first_write_of_varid.get(&(frame_id, var_ndx)).copied()
    }
}
