use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
};

use crate::{bytecode, interpreter};

use super::builder::{BoxedValue, ValueId, ValueType};

/// Stores a model (a "proxy" of sorts) of every stack frame and variable "seen" by the
/// JIT.
///
/// This includes captured frames (by closure).  It also tracks which
/// assignments were "seen" by the JIT (this allows detecting variables that
/// should be resolved by trace parameter rather than by instruction ID or
/// constant folding).
pub(super) struct VarsState {
    stack_model: Vec<Frame>,

    // These are only used for determining the PHI instuctions to be added when building a looping
    // trace.  As such, the only variables that they track are the ones from the "outermost stack
    // frame", i.e. from the call context where the trace originates (not from any calls made after
    // the trace starts).
    read_before_any_write: Box<[bool]>,
    overwritten: Box<[bool]>,
    first_write: Box<[Option<ValueId>]>,
}

/// A "model" of the interpreter's stack, represented in terms that the JIT
/// cares about.
struct Frame {
    args: Box<[ValueId]>,
    /// Associates variable identifiers (= IID of the instruction that initializes it)
    /// from the interpreter to ValueIds in the SSA trace we're building.
    ///
    /// Only stores values for variables/instruction results whose assignment has been
    /// "seen" by the JIT. This allows variables to be introduced as trace parameters
    /// when the JIT starts later than the start of the function's body.
    vid_of_iid: Box<[Option<ValueId>]>,
}

impl Frame {
    fn new(args: Box<[ValueId]>, n_values: usize) -> Self {
        let vid_of_iid = vec![None; n_values].into_boxed_slice();
        Frame { args, vid_of_iid }
    }
}

impl VarsState {
    pub(super) fn new(n_values_outermost_frame: usize) -> Self {
        VarsState {
            stack_model: Vec::new(),
            overwritten: vec![false; n_values_outermost_frame].into_boxed_slice(),
            read_before_any_write: vec![false; n_values_outermost_frame].into_boxed_slice(),
            first_write: vec![None; n_values_outermost_frame].into_boxed_slice(),
        }
    }

    pub(super) fn stack_depth(&self) -> usize {
        self.stack_model.len()
    }

    // TODO(big feat) Garbage collection?

    pub(super) fn push_frame(&mut self, args: Box<[ValueId]>, n_values: usize) {
        self.stack_model.push(Frame::new(args, n_values));
    }

    pub(super) fn pop_frame(&mut self) {
        self.stack_model.pop().unwrap();
    }

    fn cur_frame(&self) -> &Frame {
        self.stack_model.last().unwrap()
    }
    fn cur_frame_mut(&mut self) -> &mut Frame {
        self.stack_model.last_mut().unwrap()
    }

    pub(super) fn get_var(&mut self, var_iid: bytecode::IID) -> Option<ValueId> {
        assert!(self.stack_model.len() > 0);

        let var_iid = var_iid.0 as usize;

        if self.stack_model.len() == 1 {
            if !self.overwritten[var_iid] {
                self.read_before_any_write[var_iid] = true;
            }
        }

        self.cur_frame().vid_of_iid[var_iid]
    }

    //pub(super) fn get_upvalue(&self) {}
    //pub(super) fn set_upvalue(&mut self) {}

    pub(super) fn init_var(&mut self, iid: bytecode::IID, value: ValueId) {
        assert!(self.stack_model.len() > 0);

        let iid = iid.0 as usize;
        let frame = self.cur_frame_mut();
        assert!(frame.vid_of_iid[iid].is_none());
        frame.vid_of_iid[iid] = Some(value);

        if self.stack_model.len() == 1 && self.first_write[iid].is_none() {
            self.first_write[iid] = Some(value);
        }
    }

    pub(super) fn set_var(&mut self, iid: bytecode::IID, value: ValueId) {
        assert!(self.stack_model.len() > 0);

        let iid = iid.0 as usize;
        let frame = self.cur_frame_mut();
        assert!(frame.vid_of_iid[iid].is_some());
        frame.vid_of_iid[iid] = Some(value);

        if self.stack_model.len() == 1 {
            self.overwritten[iid] = true;
        }
    }

    pub(super) fn get_arg(&self, arg_ndx: usize) -> &ValueId {
        self.cur_frame()
            .args
            .get(arg_ndx)
            .expect("bytecode_compiler bug: unbound arg var")
    }

    pub(super) fn get_reads_before_overwritten(&self) -> Box<[bytecode::IID]> {
        let n_values = self.stack_model[0].vid_of_iid.len();
        assert_eq!(self.overwritten.len(), n_values);
        assert_eq!(self.overwritten.len(), self.read_before_any_write.len());

        (0..n_values)
            .filter(|&i| self.read_before_any_write[i] && self.overwritten[i])
            .map(|i| bytecode::IID(u32::try_from(i).unwrap()))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub(super) fn get_first_write(&self, iid: bytecode::IID) -> Option<ValueId> {
        self.first_write[iid.0 as usize]
    }

    pub(super) fn was_var_seen(&self, iid: bytecode::IID) -> bool {
        self.stack_model[0].vid_of_iid[iid.0 as usize].is_some()
    }
}
