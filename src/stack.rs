use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

use crate::bytecode;
use crate::interpreter::{self, UpvalueId, Value};

/// The interpreter's stack.
///
/// Mostly stores local variables.
pub(crate) struct InterpreterData {
    upv_alloc: slotmap::SlotMap<UpvalueId, Value>,

    meta: Vec<CallMeta>,

    args: Stack<Value>,
    results: Stack<Value>,
    local_upvalues: Stack<Option<UpvalueId>>,
    captured_upvalues: Stack<UpvalueId>,
}

pub(crate) struct CallBuilder<'a> {
    stack: &'a mut InterpreterData,
}

#[derive(Clone)]
pub(crate) struct CallMeta {
    pub fnid: bytecode::FnId,
    pub n_instrs: u32,
    pub n_captured_upvalues: u16,
    pub n_args: u16,
    pub call_iid: Option<bytecode::IID>,
}

impl InterpreterData {
    /// TODO(small feat) Better value?
    const INIT_CAPACITY: usize = 4096;

    pub(crate) fn new() -> Self {
        InterpreterData {
            upv_alloc: slotmap::SlotMap::with_key(),
            meta: Vec::with_capacity(Self::INIT_CAPACITY),
            args: Stack::with_capacity(Self::INIT_CAPACITY),
            results: Stack::with_capacity(Self::INIT_CAPACITY),
            local_upvalues: Stack::with_capacity(Self::INIT_CAPACITY),
            captured_upvalues: Stack::with_capacity(Self::INIT_CAPACITY),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.meta.len()
    }

    pub(crate) fn push(&mut self, call_meta: CallMeta) {
        self.args
            .push_and_lock(call_meta.n_args as _, Value::Undefined);
        self.captured_upvalues
            .push_and_lock(call_meta.n_captured_upvalues as _, UpvalueId::default());
        self.results
            .push_and_lock(call_meta.n_instrs as _, Value::Undefined);
        self.local_upvalues
            .push_and_lock(call_meta.n_instrs as _, None);

        self.meta.push(call_meta);
    }

    pub(crate) fn pop(&mut self) {
        use crate::util::shorten_by;
        let call_meta = self.meta.pop().unwrap();

        self.args.pop_n(call_meta.n_args as usize);
        self.captured_upvalues
            .pop_n(call_meta.n_captured_upvalues as usize);
        self.local_upvalues.pop_n(call_meta.n_instrs as usize);
        self.results.pop_n(call_meta.n_instrs as usize);

        if let Some(new_cur_meta) = self.meta.last() {
            self.args.set_n_readable(Some(new_cur_meta.n_args as usize));
            self.captured_upvalues
                .set_n_readable(Some(new_cur_meta.n_captured_upvalues as usize));
            self.local_upvalues
                .set_n_readable(Some(new_cur_meta.n_instrs as usize));
            self.results
                .set_n_readable(Some(new_cur_meta.n_instrs as usize));
        } else {
            self.args.set_n_readable(Some(0));
            self.captured_upvalues.set_n_readable(Some(0));
            self.local_upvalues.set_n_readable(Some(0));
            self.results.set_n_readable(Some(0));
        }
    }

    pub(crate) fn cur_meta(&self) -> &CallMeta {
        self.meta.last().as_ref().unwrap()
    }

    pub(crate) fn call_iid(&self) -> Option<bytecode::IID> {
        self.meta.last().and_then(|meta| meta.call_iid)
    }

    pub(crate) fn get_result(&self, iid: bytecode::IID) -> &Value {
        let ndx = iid.0 as usize;
        if let Some(upv_id) = self.local_upvalues.get(ndx) {
            self.upv_alloc
                .get(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack")
        } else {
            self.results.get(ndx)
        }
    }
    pub(crate) fn set_result(&mut self, iid: bytecode::IID, value: Value) {
        let ndx = iid.0 as usize;
        if let Some(upv_id) = self.local_upvalues.get(ndx) {
            let slot = self
                .upv_alloc
                .get_mut(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack");
            *slot = value;
        } else {
            *self.results.get_mut(ndx) = value;
        }
    }

    pub(crate) fn get_arg(&self, arg_ndx: usize) -> &Value {
        self.args.get(arg_ndx)
    }
    pub(crate) fn set_arg(&mut self, arg_ndx: usize, value: Value) {
        *self.args.get_mut(arg_ndx) = value;
    }

    pub(crate) fn ensure_in_upvalue(&mut self, var: bytecode::IID) -> UpvalueId {
        self.move_to_upvalue(var);
        self.get_upvalue(var)
            .expect("bug: var is not associated to an upvalue")
    }
    fn get_upvalue(&self, var: bytecode::IID) -> Option<UpvalueId> {
        *self.local_upvalues.get(var.0 as usize)
    }
    fn move_to_upvalue(&mut self, var: bytecode::IID) {
        assert!(var.0 < self.cur_meta().n_instrs);

        let varndx = var.0 as usize;
        if self.local_upvalues.get(varndx).is_some() {
            return;
        }

        let mut cur_value = Value::Undefined;
        std::mem::swap(self.results.get_mut(varndx), &mut cur_value);

        let upv_id = self.upv_alloc.insert(cur_value);
        *self.local_upvalues.get_mut(varndx) = Some(upv_id);
    }

    pub(crate) fn capture_to_var(
        &mut self,
        capture_ndx: bytecode::CaptureIndex,
        var: bytecode::IID,
    ) {
        let upvalue_id = self.captured_upvalues.get(capture_ndx as usize);
        let prev = self
            .local_upvalues
            .get_mut(var.0 as usize)
            .replace(*upvalue_id);
        assert!(prev.is_none());
    }

    pub(crate) fn set_capture(&mut self, capndx: usize, capture: UpvalueId) {
        *self.captured_upvalues.get_mut(capndx) = capture;
    }
}

struct Stack<T> {
    items: Vec<T>,
    n_readable_slots: Option<usize>,
}

impl<T: Clone> Stack<T> {
    fn with_capacity(cap: usize) -> Self {
        let items = Vec::with_capacity(cap);
        Stack {
            items,
            n_readable_slots: None,
        }
    }

    fn push_and_lock(&mut self, count: usize, value: T) {
        self.push_n(count, value);
        self.set_n_readable(Some(count));
    }
    fn push_n(&mut self, count: usize, value: T) {
        self.items.resize(self.items.len() + count, value);
    }
    fn push(&mut self, value: T) {
        self.items.push(value);
    }

    fn pop_n(&mut self, count: usize) {
        self.items.truncate(self.items.len() - count);
    }

    fn get(&self, ndx: usize) -> &T {
        if let Some(n_readable) = self.n_readable_slots {
            assert!(
                ndx < n_readable,
                "requested index {}, but only {} slots are readable",
                ndx,
                n_readable
            );
        }

        let ndx = self.items.len() - 1 - ndx;
        &self.items[ndx]
    }
    fn get_mut(&mut self, ndx: usize) -> &mut T {
        if let Some(n_readable) = self.n_readable_slots {
            assert!(ndx < n_readable);
        }

        let ndx = self.items.len() - 1 - ndx;
        &mut self.items[ndx]
    }

    fn set_n_readable(&mut self, count: Option<usize>) {
        self.n_readable_slots = count;
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
