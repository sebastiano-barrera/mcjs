use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

use crate::bytecode::{self, VReg, ARGS_COUNT_MAX, IID};
use crate::interpreter::{self, UpvalueId, Value};

/// The interpreter's stack.
///
/// Mostly stores local variables.
pub struct InterpreterData {
    upv_alloc: Heap,
    headers: Vec<FrameHeader>,
    values: Vec<Slot>,
}

type Heap = slotmap::SlotMap<UpvalueId, Value>;

#[derive(Clone, Copy)]
pub(crate) enum Slot {
    Inline(Value),
    Upvalue(UpvalueId),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameHeader {
    pub regs_offset: u32,
    pub regs_count: u32,
    pub fn_id: bytecode::FnId,
    pub this: Value,
    pub return_target: Option<(IID, VReg)>,
}

#[derive(Clone)]
pub(crate) struct CallMeta<'a> {
    pub fnid: bytecode::FnId,
    pub n_regs: u32,
    pub captures: &'a [UpvalueId],
    pub this: Value,
}

impl InterpreterData {
    /// TODO(small feat) Better value?
    const INIT_CAPACITY: usize = 256;

    pub(crate) fn new() -> Self {
        InterpreterData {
            upv_alloc: slotmap::SlotMap::with_key(),
            headers: Vec::with_capacity(Self::INIT_CAPACITY),
            values: Vec::with_capacity(Self::INIT_CAPACITY),
        }
    }

    pub fn len(&self) -> usize {
        self.headers.len()
    }

    pub(crate) fn push(&mut self, call_meta: CallMeta) {
        let frame_hdr = FrameHeader {
            regs_offset: self.values.len().try_into().unwrap(),
            regs_count: call_meta.n_regs + ARGS_COUNT_MAX as u32,
            fn_id: call_meta.fnid,
            this: call_meta.this,
            return_target: None,
        };

        #[cfg(test)]
        eprintln!("  (allocated frame: {:?})", frame_hdr);

        self.headers.push(frame_hdr);
        for _ in 0..frame_hdr.regs_count {
            self.values.push(Slot::Inline(Value::Undefined));
        }

        self.check_invariants();
    }

    fn check_invariants(&self) {
        // TODO Turn into debug_asserts?

        for (ndx, hdr) in self.headers.iter().enumerate() {
            assert!(hdr.regs_count >= ARGS_COUNT_MAX as u32);
            if ndx > 0 {
                let prev = &self.headers[ndx - 1];
                assert_eq!(hdr.regs_offset, prev.regs_offset + prev.regs_count);
            }
        }

        let n_values_total: usize = self.headers.iter().map(|hdr| hdr.regs_count as usize).sum();
        debug_assert_eq!(n_values_total, self.values.len());
    }

    pub(crate) fn pop(&mut self) {
        let frame_hdr = self.headers.pop().unwrap();
        self.values
            .truncate(self.values.len() - frame_hdr.regs_count as usize);
        self.check_invariants();
    }

    pub(crate) fn nth_frame(&self, ndx: usize) -> Frame {
        let header = &self.headers[ndx];
        let regs_offset = header.regs_offset as usize;
        let regs_count = header.regs_count as usize;
        let values = &self.values[regs_offset..regs_offset + regs_count];
        Frame {
            header,
            values,
            upv_alloc: &self.upv_alloc,
        }
    }

    fn nth_frame_mut(&mut self, ndx: usize) -> FrameMut {
        let header = &mut self.headers[ndx];
        let regs_offset = header.regs_offset as usize;
        let regs_count = header.regs_count as usize;
        let values = &mut self.values[regs_offset..regs_offset + regs_count];
        FrameMut {
            header,
            values,
            upv_alloc: &mut self.upv_alloc,
        }
    }

    pub fn top(&self) -> Frame {
        self.nth_frame(self.headers.len() - 1)
    }
    pub fn top_mut(&mut self) -> FrameMut {
        self.nth_frame_mut(self.headers.len() - 1)
    }

    /// Returns the sequence of stack frames in the form of an iterator, ordered top to
    /// bottom.
    pub fn frames(&self) -> impl ExactSizeIterator<Item = Frame> {
        let n_frames = self.headers.len();
        (0..n_frames).rev().map(|ndx| self.nth_frame(ndx))
    }

    pub(crate) fn capture_to_var(
        &mut self,
        capture_ndx: bytecode::CaptureIndex,
        vreg: bytecode::VReg,
    ) {
        let upv_id = self.top().get_capture(capture_ndx);
        self.top_mut().values[vreg.0 as usize] = Slot::Upvalue(upv_id);
    }
}

pub struct Frame<'a> {
    header: &'a FrameHeader,
    values: &'a [Slot],
    upv_alloc: &'a Heap,
}
pub struct FrameMut<'a> {
    header: &'a mut FrameHeader,
    values: &'a mut [Slot],
    upv_alloc: &'a mut Heap,
}

impl<'a> Frame<'a> {
    pub fn header(&self) -> &'a FrameHeader {
        self.header
    }

    pub fn get_result(&self, vreg: bytecode::VReg) -> Value {
        let slot = &self.values[vreg.0 as usize];
        match slot {
            Slot::Inline(value) => *value,
            Slot::Upvalue(upv_id) => *self
                .upv_alloc
                .get(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack"),
        }
    }
    pub fn results<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Value> {
        (0..self.values.len()).map(|i| {
            let vreg = bytecode::VReg(i.try_into().unwrap());
            self.get_result(vreg)
        })
    }

    pub fn get_arg(&self, argndx: bytecode::ArgIndex) -> Option<Value> {
        // TODO to support more than ARGS_MAX_COUNT arguments, the plan is to
        // have an external array of 'extra arguments', managed as part of the
        // FrameHeader
        assert!(
            argndx.0 < ARGS_COUNT_MAX,
            "not yet implemented: call with >= {}",
            ARGS_COUNT_MAX
        );
        Some(self.get_result(bytecode::VReg(argndx.0)))
    }
    pub fn args<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Option<Value>> {
        (0..ARGS_COUNT_MAX).map(|i| self.get_arg(bytecode::ArgIndex(i)))
    }

    pub fn get_capture(&self, capture_ndx: bytecode::CaptureIndex) -> UpvalueId {
        // another/better approach: add N slots to the header and use them as cache for the closure's captures.
        todo!(
            "get_capture [get closure ID from header; get closure from closure heap; get capture]"
        )
    }
    pub fn captures<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = UpvalueId> {
        todo!(
            "get_capture [get closure ID from header; get closure from closure heap; get capture]"
        );
        std::iter::empty()
    }

    pub fn deref_upvalue(&self, upv_id: UpvalueId) -> Option<Value> {
        self.upv_alloc.get(upv_id).copied()
    }

    pub fn return_target(&self) -> Option<(IID, VReg)> {
        self.header.return_target
    }
}

impl<'a> FrameMut<'a> {
    pub(crate) fn set_result(&mut self, vreg: bytecode::VReg, value: Value) {
        let slot = &mut self.values[vreg.0 as usize];
        match slot {
            Slot::Inline(slot_value) => {
                *slot_value = value;
            }
            Slot::Upvalue(upv_id) => {
                let heap_value = self
                    .upv_alloc
                    .get_mut(*upv_id)
                    .expect("gc bug: value deleted but still referenced by stack");
                *heap_value = value;
            }
        };
    }

    pub(crate) fn set_arg(&mut self, argndx: bytecode::ArgIndex, value: Value) {
        // Check comment in `Frame::get_arg`
        assert!(
            argndx.0 < ARGS_COUNT_MAX,
            "not yet implemented: call with >= {}",
            ARGS_COUNT_MAX
        );
        self.set_result(bytecode::VReg(argndx.0), value)
    }

    pub(crate) fn ensure_in_upvalue(&mut self, var: bytecode::VReg) -> UpvalueId {
        let slot = &mut self.values[var.0 as usize];
        match slot {
            Slot::Inline(value) => {
                let upv_id = self.upv_alloc.insert(*value);
                *slot = Slot::Upvalue(upv_id);
                upv_id
            }
            Slot::Upvalue(upv_id) => *upv_id,
        }
    }

    pub(crate) fn set_return_target(&mut self, iid: IID, reg: VReg) {
        self.header.return_target = Some((iid, reg));
    }

    pub(crate) fn take_return_target(&mut self) -> (IID, VReg) {
        self.header.return_target.take().unwrap()
    }
}
