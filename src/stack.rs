use slotmap::SlotMap;
use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

use crate::bytecode;
use crate::interpreter::{Value, self};

/// The interpreter's stack.
///
/// Mostly stores local variables.
pub(crate) struct Stack {
    meta: Vec<CallMeta>,

    args: Vec<Value>,
    results: Vec<Value>,
    locals: Vec<Value>,
    upvalue_ids: Vec<interpreter::UpvalueId>,
}

pub(crate) struct CallBuilder<'a> {
    stack: &'a mut Stack,
}

pub(crate) struct CallMeta {
    pub fnid: bytecode::FnId,
    pub n_instrs: u16,
    pub n_upvalues: u16,
    pub n_locals: u16,
    pub n_args: u16,
    pub parent_iid: bytecode::IID,
}

impl Stack {
    /// TODO(small feat) Better value?
    const INIT_CAPACITY: usize = 4096;

    pub(crate) fn new() -> Self {
        Stack {
            meta: Vec::with_capacity(Self::INIT_CAPACITY),
            args: Vec::with_capacity(Self::INIT_CAPACITY),
            results: Vec::with_capacity(Self::INIT_CAPACITY),
            locals: Vec::with_capacity(Self::INIT_CAPACITY),
            upvalue_ids: Vec::with_capacity(Self::INIT_CAPACITY),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.meta.len()
    }

    pub(crate) fn push(&mut self, call_meta: CallMeta) {
        for _ in 0..call_meta.n_args {
            self.args.push(Value::Undefined);
        }

        for _ in 0..call_meta.n_upvalues {
            // TODO This OK?
            self.upvalue_ids.push(0u16);
        }

        for _ in 0..call_meta.n_locals {
            self.locals.push(Value::Undefined);
        }

        for _ in 0..call_meta.n_instrs {
            self.results.push(Value::Undefined);
        }

        self.meta.push(call_meta);
    }

    pub(crate) fn pop(&mut self) {
        use crate::util::shorten_by;
        let call_meta = self.meta.pop().unwrap();

        shorten_by(&mut self.args, call_meta.n_args as usize);
        shorten_by(&mut self.upvalue_ids, call_meta.n_upvalues as usize);
        shorten_by(&mut self.locals, call_meta.n_locals as usize);
        shorten_by(&mut self.results, call_meta.n_instrs as usize);
    }

    pub(crate) fn parent_iid(&self) -> Option<bytecode::IID> {
        self.meta.last().map(|meta| meta.parent_iid)
    }

    pub(crate) fn get_result(&self, iid: bytecode::IID) -> &Value {
        &self.results[self.results.len() - iid.0 as usize - 1]
    }

    pub(crate) fn set_result(&mut self, iid: bytecode::IID, value: Value) {
        let ndx = self.results.len() - iid.0 as usize - 1;
        self.results[ndx] = value;
    }

    pub(crate) fn get_local(&self, var_ndx: u16) -> &Value {
        &self.locals[self.locals.len() - var_ndx as usize - 1]
    }

    pub(crate) fn set_local(&mut self, var_ndx: u16, value: Value) {
        let ndx = self.locals.len() - var_ndx as usize - 1;
        self.locals[ndx] = value;
    }

    pub(crate) fn get_upvalue(&self, upvalue_id: u16) -> Value {
        todo!("get_upvalue")
    }

    pub(crate) fn set_upvalue(&self, upvalue_id: u16, value: Value) {
        todo!("set_upvalue")
    }

    pub(crate) fn get_arg(&self, arg_ndx: usize) -> &Value {
        &self.args[self.args.len() - arg_ndx as usize - 1]
    }

    pub(crate) fn set_arg(&mut self, arg_ndx: usize, value: Value) {
        let ndx = self.args.len() - arg_ndx as usize - 1;
        self.args[ndx] = value;
    }
}
