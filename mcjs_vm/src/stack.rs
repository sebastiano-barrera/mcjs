use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

use crate::interpreter::{self, UpvalueId, Value};
use crate::stack_access::Stack;
use crate::{bytecode, stack_access};

type Heap = slotmap::SlotMap<UpvalueId, Value>;

/// The interpreter's stack.
///
/// Mostly stores local variables.
pub struct InterpreterData {
    upv_alloc: Heap,
    stack: Stack,
    n_frames: usize,
}

const STACK_SIZE: usize = 256 * 1024;

#[derive(Clone)]
pub(crate) struct CallMeta {
    pub fnid: bytecode::FnId,
    pub n_instrs: u32,
    pub n_captured_upvalues: u16,
    pub n_args: u8,
    pub this: Value,
    pub return_value_reg: Option<bytecode::VReg>,
    pub return_to_iid: Option<bytecode::IID>,
}

impl InterpreterData {
    /// TODO(small feat) Better value?
    const INIT_CAPACITY: usize = 4096;

    pub(crate) fn new() -> Self {
        InterpreterData {
            upv_alloc: slotmap::SlotMap::with_key(),
            stack: Stack::new(Box::new([0u8; STACK_SIZE])),
            n_frames: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n_frames
    }

    pub(crate) fn push(&mut self, call_meta: CallMeta) {
        let frame_hdr = stack_access::FrameHeader {
            #[cfg(test)]
            magic: stack_access::FrameHeader::MAGIC,
            n_instrs: call_meta.n_instrs,
            n_args: call_meta.n_args,
            n_captures: call_meta.n_captured_upvalues,
            this: call_meta.this,
            return_value_vreg: call_meta.return_value_reg,
            return_to_iid: call_meta.return_to_iid,
            fn_id: call_meta.fnid,
        };

        #[cfg(test)]
        eprintln!(
            "  (allocated frame for {} args, {} captures, {} instrs)",
            call_meta.n_args, call_meta.n_captured_upvalues, call_meta.n_instrs,
        );

        self.stack.push_frame(frame_hdr);
        self.n_frames += 1;
    }

    pub(crate) fn pop(&mut self) {
        self.stack.pop_frame();
        self.n_frames -= 1;
    }

    pub fn top(&self) -> Frame {
        Frame {
            inner: self.stack.top_frame(),
            upv_alloc: &self.upv_alloc,
        }
    }

    pub fn top_mut(&mut self) -> FrameMut {
        FrameMut {
            inner: self.stack.top_frame_mut(),
            upv_alloc: &mut self.upv_alloc,
        }
    }

    /// Returns the sequence of stack frames in the form of an iterator, ordered top to
    /// bottom.
    pub fn frames(&self) -> impl ExactSizeIterator<Item = Frame> {
        self.stack.frames().into_iter().map(|ll_frame| Frame {
            inner: ll_frame,
            upv_alloc: &self.upv_alloc,
        })
    }

    pub(crate) fn capture_to_var(
        &mut self,
        capture_ndx: bytecode::CaptureIndex,
        vreg: bytecode::VReg,
    ) {
        let upv_id = self.top().get_capture(capture_ndx);
        self.top_mut().inner.vars_mut()[vreg.0 as usize] = stack_access::Slot::Upvalue(upv_id);
    }
}

pub struct Frame<'a> {
    inner: stack_access::FrameView<'a>,
    upv_alloc: &'a Heap,
}
pub struct FrameMut<'a> {
    inner: stack_access::FrameViewMut<'a>,
    upv_alloc: &'a mut Heap,
}

impl<'a> Frame<'a> {
    pub fn header(&self) -> &'a stack_access::FrameHeader {
        self.inner.header()
    }

    pub fn get_result(&self, vreg: bytecode::VReg) -> Value {
        let slot = &self.inner.vars()[vreg.0 as usize];
        slot_value(slot, &self.upv_alloc)
    }
    pub fn results<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Value> {
        (0..self.header().n_instrs).map(|i| {
            let vreg = bytecode::VReg(i.try_into().unwrap());
            self.get_result(vreg)
        })
    }

    pub fn get_arg(&self, argndx: bytecode::ArgIndex) -> Option<Value> {
        self.inner
            .args()
            .get(argndx.0 as usize)
            .map(|slot| slot_value(slot, &self.upv_alloc))
    }
    pub fn args<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Option<Value>> {
        (0..self.header().n_args).map(|i| {
            let argndx = bytecode::ArgIndex(i.try_into().unwrap());
            self.get_arg(argndx)
        })
    }

    pub fn get_capture(&self, capture_ndx: bytecode::CaptureIndex) -> UpvalueId {
        let slot = &self.inner.captures()[capture_ndx.0 as usize];
        match slot {
            stack_access::Slot::Inline(_) => unreachable!(),
            stack_access::Slot::Upvalue(upv_id) => *upv_id,
        }
    }
    pub fn captures<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = UpvalueId> {
        (0..self.header().n_captures).map(|i| {
            let capndx = bytecode::CaptureIndex(i.try_into().unwrap());
            self.get_capture(capndx)
        })
    }

    pub fn deref_upvalue(&self, upv_id: UpvalueId) -> Option<Value> {
        self.upv_alloc.get(upv_id).copied()
    }
}

impl<'a> FrameMut<'a> {
    pub(crate) fn set_result(mut self, vreg: bytecode::VReg, value: Value) {
        let slot = &mut self.inner.vars_mut()[vreg.0 as usize];
        set_slot_value(slot, value, &mut self.upv_alloc);
    }

    pub(crate) fn set_arg(mut self, argndx: bytecode::ArgIndex, value: Value) {
        let slot = &mut self.inner.args_mut()[argndx.0 as usize];
        set_slot_value(slot, value, &mut self.upv_alloc);
    }

    pub(crate) fn ensure_in_upvalue(self, var: bytecode::VReg) -> UpvalueId {
        let varndx = var.0 as usize;
        let slot = &mut self.inner.vars_mut()[varndx as usize];

        match slot {
            stack_access::Slot::Inline(value) => {
                let upv_id = self.upv_alloc.insert(*value);
                *slot = stack_access::Slot::Upvalue(upv_id);
                upv_id
            }
            stack_access::Slot::Upvalue(upv_id) => *upv_id,
        }
    }

    pub(crate) fn set_capture(self, capture_ndx: bytecode::CaptureIndex, capture: UpvalueId) {
        self.inner.captures_mut()[capture_ndx.0 as usize] = stack_access::Slot::Upvalue(capture);
    }
}

fn slot_value(slot: &stack_access::Slot, upv_alloc: &Heap) -> Value {
    match slot {
        stack_access::Slot::Inline(value) => *value,
        stack_access::Slot::Upvalue(upv_id) => *upv_alloc
            .get(*upv_id)
            .expect("gc bug: value deleted but still referenced by stack"),
    }
}
fn set_slot_value(slot: &mut stack_access::Slot, value: Value, upv_alloc: &mut Heap) {
    match slot {
        stack_access::Slot::Inline(slot_value) => {
            *slot_value = value;
        }
        stack_access::Slot::Upvalue(upv_id) => {
            let heap_value = upv_alloc
                .get_mut(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack");
            *heap_value = value;
        }
    };
}
