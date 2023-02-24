use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
};

use crate::{
    interpreter::{self, VarIndex},
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
    pub(super) stack_model: Vec<FrameId>,
    pub(super) frame_models: HashMap<FrameId, FrameTracker>,
}

impl VarsState {
    pub(super) fn new() -> Self {
        VarsState {
            stack_model: Vec::new(),
            frame_models: HashMap::new(),
        }
    }

    pub(super) fn stack_depth(&self) -> usize {
        self.stack_model.len()
    }

    // TODO(big feat) Garbage collection?

    pub(super) fn push_frame(&mut self, frame_id: FrameId, args: Vec<ValueId>, n_vars: usize) {
        let frame_model = FrameTracker::new(args, n_vars);
        self.frame_models.insert(frame_id, frame_model);
        self.stack_model.push(frame_id);
    }

    pub(super) fn pop_frame(&mut self) {
        self.stack_model.pop().unwrap();
    }

    pub(super) fn cur_frame_id(&self) -> FrameId {
        *self
            .stack_model
            .last()
            .expect("JIT bug: no frame model on stack (trace ended?)")
    }
    pub(super) fn cur_frame(&self) -> &FrameTracker {
        self.get(self.cur_frame_id())
    }
    pub(super) fn cur_frame_mut(&mut self) -> &mut FrameTracker {
        self.get_mut(self.cur_frame_id())
    }

    pub(super) fn get(&self, frame_id: FrameId) -> &FrameTracker {
        self.frame_models.get(&frame_id).unwrap()
    }
    pub(super) fn get_mut(&mut self, frame_id: FrameId) -> &mut FrameTracker {
        self.frame_models.get_mut(&frame_id).unwrap()
    }

    pub(crate) fn get_reads_before_writes<'a>(&'a self) -> HashSet<(FrameId, VarIndex)> {
        // TODO(opt) Can we avoid a whole new data structure for this? Maybe it's OK
        // anyway, since this is only done once per trace
        let mut ret = HashSet::new();

        for (frm_id, frm) in self.frame_models.iter() {
            for var_ndx in frm.get_read_before_overwritten().iter() {
                ret.insert((*frm_id, *var_ndx));
            }
        }

        ret
    }
}

/// A "model" of the interpreter's stack, represented in terms that the JIT
/// cares about.
pub(super) struct FrameTracker {
    args: Vec<ValueId>,
    /// Associates RUNTIME variable identifiers (current frame id, VarIndex)
    /// from the interpreter to ValueIds in the SSA trace we're building.
    ///
    /// The stored value is optional, and present (Some) *only* when the
    /// assignment has been "seen" by the JIT.  This allows variables to be
    /// introduced as trace parameters when the JIT starts later than the start
    /// of the function's body.
    vid_of_varid: Box<[Option<ValueId>]>,
    // TODO(opt) More compact data structures
    first_write_of_varid: Box<[Option<ValueId>]>,
    overwritten_vars: RefCell<HashSet<VarIndex>>,
    read_before_overwrite: RefCell<HashSet<VarIndex>>,

    vid_of_iid: HashMap<interpreter::IID, ValueId>,
    unboxes: HashMap<ValueId, (ValueType, ValueId)>,
}

impl FrameTracker {
    fn new(args: Vec<ValueId>, n_vars: usize) -> Self {
        let vid_of_varid = vec![None; n_vars].into_boxed_slice();
        let first_write_of_varid = vec![None; n_vars].into_boxed_slice();
        FrameTracker {
            args,
            vid_of_varid,
            first_write_of_varid,
            overwritten_vars: RefCell::new(HashSet::new()),
            read_before_overwrite: RefCell::new(HashSet::new()),
            vid_of_iid: HashMap::new(),
            unboxes: HashMap::new(),
        }
    }


    pub(super) fn get_arg(&self, arg_ndx: usize) -> &ValueId {
        self.args
            .get(arg_ndx)
            .expect("bytecode_compiler bug: unbound arg var")
    }

    pub(super) fn get_result(&self, iid: interpreter::IID) -> Option<&ValueId> {
        self.vid_of_iid.get(&iid)
    }
    pub(super) fn set_result(&mut self, iid: interpreter::IID, value: ValueId) {
        self.vid_of_iid.insert(iid, value);
    }

    pub(super) fn get_var(&self, var_ndx: VarIndex) -> Option<&ValueId> {
        let ws = self.overwritten_vars.borrow();
        if !ws.contains(&var_ndx) {
            let mut rbws = self.read_before_overwrite.borrow_mut();
            rbws.insert(var_ndx.clone());
        }
        self.vid_of_varid
            .get(var_ndx.0 as usize)
            .expect("bug: invalid variable index for this frame")
            .as_ref()
    }

    pub(super) fn set_var(&mut self, var_ndx: VarIndex, value: ValueId) {
        let slot = self
            .vid_of_varid
            .get_mut(var_ndx.0 as usize)
            .expect("bug: invalid variable index for this frame");
        *slot = Some(value);

        let fw = &mut self.first_write_of_varid[var_ndx.0 as usize];
        if fw.is_none() {
            *fw = Some(value);
        } else {
            self.overwritten_vars.get_mut().insert(var_ndx);
        }
    }

    pub(super) fn get_unbox(&self, value_id: ValueId) -> Option<(ValueType, ValueId)> {
        self.unboxes.get(&value_id).copied()
    }
    pub(super) fn set_unbox(&mut self, value_id: ValueId, ty: ValueType, value: ValueId) {
        let prev = self.unboxes.insert(value_id, (ty, value));
        assert!(prev.is_none(), "JIT bug: multiple unboxes");
    }

    pub(super) fn get_read_before_overwritten(&self) -> HashSet<VarIndex> {
        let rbw = self.read_before_overwrite.borrow();
        let ws = self.overwritten_vars.borrow();
        rbw.intersection(&ws).copied().collect()
    }

    pub(crate) fn get_first_write(&self, var_ndx: VarIndex) -> Option<ValueId> {
        self.first_write_of_varid
            .get(var_ndx.0 as usize)
            .cloned()
            .flatten()
    }
}
