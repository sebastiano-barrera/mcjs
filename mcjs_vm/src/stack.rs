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

    pub(crate) fn push(&mut self, call_meta: CallMeta, captures: &[UpvalueId]) {
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

        self.stack.push_frame(frame_hdr, captures);
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

    pub(crate) fn load_upvalue(&self, upv_id: UpvalueId) -> Value {
        let slot = self.upv_alloc.get(upv_id).unwrap();
        *slot
    }

    pub(crate) fn store_upvalue(&mut self, upv_id: UpvalueId, value: Value) {
        let slot = self.upv_alloc.get_mut(upv_id).unwrap();
        *slot = value;
    }

    pub(crate) fn new_upvalue(&mut self, value: Value) -> UpvalueId {
        self.upv_alloc.insert(value)
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
        self.inner.vars()[vreg.0 as usize]
    }
    pub fn results<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Value> {
        (0..self.header().n_instrs).map(|i| {
            let vreg = bytecode::VReg(i.try_into().unwrap());
            self.get_result(vreg)
        })
    }

    pub fn get_arg(&self, argndx: bytecode::ArgIndex) -> Option<Value> {
        self.inner.args().get(argndx.0 as usize).copied()
    }
    pub fn args<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Option<Value>> {
        (0..self.header().n_args).map(|i| {
            let argndx = bytecode::ArgIndex(i.try_into().unwrap());
            self.get_arg(argndx)
        })
    }

    pub fn get_capture(&self, capture_ndx: bytecode::CaptureIndex) -> Option<UpvalueId> {
        self.inner.captures().get(capture_ndx.0 as usize).copied()
    }
    pub fn captures<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = UpvalueId> {
        (0..self.header().n_captures).map(|i| {
            let capndx = bytecode::CaptureIndex(i.try_into().unwrap());
            self.get_capture(capndx).unwrap()
        })
    }
}

impl<'a> FrameMut<'a> {
    pub(crate) fn set_result(mut self, vreg: bytecode::VReg, value: Value) {
        let slot = &mut self.inner.vars_mut()[vreg.0 as usize];
        *slot = value;
    }

    pub(crate) fn set_arg(mut self, argndx: bytecode::ArgIndex, value: Value) {
        let slot = &mut self.inner.args_mut()[argndx.0 as usize];
        *slot = value;
    }

    pub(crate) fn set_capture(self, capture_ndx: bytecode::CaptureIndex, upv_id: UpvalueId) {
        self.inner.captures_mut()[capture_ndx.0 as usize] = upv_id;
    }
}

fn read_capture(upv_alloc: &slotmap::SlotMap<UpvalueId, Value>, upv_id: &UpvalueId) -> Value {
    *upv_alloc
        .get(*upv_id)
        .expect("gc bug: value deleted but still referenced by stack")
}

fn write_capture(
    upv_alloc: &mut slotmap::SlotMap<UpvalueId, Value>,
    upv_id: &mut UpvalueId,
    value: Value,
) {
    let heap_value = upv_alloc
        .get_mut(*upv_id)
        .expect("gc bug: value deleted but still referenced by stack");
    *heap_value = value;
}
