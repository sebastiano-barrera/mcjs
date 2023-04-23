use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
};

use crate::{
    bytecode::{self, LocalVarIndex},
    interpreter,
};

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
    // trace.  As such, the only variables that they track are the ones from the call context where
    // the trace originates (i.e. not from any calls made after the trace starts).
    // TODO(opt) More compact data structures?
    first_write_of_varid: HashMap<LocalVarIndex, ValueId>,
    overwritten_vars: RefCell<HashSet<LocalVarIndex>>,
    read_before_overwrite: RefCell<HashSet<LocalVarIndex>>,
}

/// A "model" of the interpreter's stack, represented in terms that the JIT
/// cares about.
struct Frame {
    args: Vec<ValueId>,
    vid_of_iid: HashMap<bytecode::IID, ValueId>,
    n_vars: usize,
    /// Associates RUNTIME variable identifiers (LocalVarIndex, local to this stack frame)
    /// from the interpreter to ValueIds in the SSA trace we're building.
    ///
    /// Only stores values for variables whose assignment has been "seen" by the JIT. This
    /// allows variables to be introduced as trace parameters when the JIT starts
    /// later than the start of the function's body.
    vid_of_varid: HashMap<LocalVarIndex, ValueId>,
}

impl VarsState {
    pub(super) fn new() -> Self {
        VarsState {
            stack_model: Vec::new(),
            first_write_of_varid: HashMap::new(),
            overwritten_vars: RefCell::new(HashSet::new()),
            read_before_overwrite: RefCell::new(HashSet::new()),
        }
    }

    pub(super) fn stack_depth(&self) -> usize {
        self.stack_model.len()
    }

    // TODO(big feat) Garbage collection?

    pub(super) fn push_frame(&mut self, args: Vec<ValueId>, n_vars: usize) {
        self.stack_model.push(Frame {
            args,
            n_vars,
            vid_of_iid: HashMap::new(),
            vid_of_varid: HashMap::new(),
        });
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

    pub(super) fn get_result(&self, iid: bytecode::IID) -> Option<&ValueId> {
        self.cur_frame().vid_of_iid.get(&iid)
    }
    pub(super) fn set_result(&mut self, iid: bytecode::IID, value: ValueId) {
        self.cur_frame_mut().vid_of_iid.insert(iid, value);
    }

    pub(super) fn get_var(&self, var: &bytecode::Var) -> Option<ValueId> {
        assert!(self.stack_model.len() > 0);
        match var {
            bytecode::Var::Local(var_ndx) => {
                if self.stack_model.len() == 1 {
                    let ws = self.overwritten_vars.borrow();
                    if !ws.contains(var_ndx) {
                        let mut rbws = self.read_before_overwrite.borrow_mut();
                        rbws.insert(*var_ndx);
                    }
                }

                self.cur_frame().vid_of_varid.get(&var_ndx).copied()
            }
            bytecode::Var::Upvalue(_) => todo!("get_var(upvalue)"),
        }
    }
    pub(super) fn set_var(&mut self, var: &bytecode::Var, value: ValueId) {
        use std::collections::hash_map::Entry;

        assert!(self.stack_model.len() > 0);

        match var {
            bytecode::Var::Local(var_ndx) => {
                let frame = self.cur_frame_mut();
                // TODO(opt) skip this test if we're not in #[cfg(test)]?
                assert!(usize::from(*var_ndx) < frame.n_vars);
                frame.vid_of_varid.entry(*var_ndx).or_insert(value);

                if self.stack_model.len() == 1 {
                    if let Entry::Vacant(e) = self.first_write_of_varid.entry(*var_ndx) {
                        e.insert(value);
                    } else {
                        // Not the first write
                        self.overwritten_vars.get_mut().insert(*var_ndx);
                    }
                }
            }
            bytecode::Var::Upvalue(_) => todo!("set_var(upvalue)"),
        }
    }

    pub(super) fn get_arg(&self, arg_ndx: usize) -> &ValueId {
        self.cur_frame()
            .args
            .get(arg_ndx)
            .expect("bytecode_compiler bug: unbound arg var")
    }

    pub(super) fn get_reads_before_overwritten(&self) -> HashSet<LocalVarIndex> {
        let rbw = self.read_before_overwrite.borrow();
        let ws = self.overwritten_vars.borrow();
        rbw.intersection(&ws).copied().collect()
    }

    pub(super) fn get_first_write(&self, var_ndx: LocalVarIndex) -> Option<ValueId> {
        self.first_write_of_varid.get(&var_ndx).copied()
    }

    pub(crate) fn was_var_seen(&self, var: &bytecode::Var) -> bool {
        match var {
            bytecode::Var::Local(var_ndx) => self.cur_frame().vid_of_varid.contains_key(&var_ndx),
            bytecode::Var::Upvalue(_) => todo!("was_var_seen(upvalue)"),
        }
    }
}
