use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

use crate::bytecode::{self, VReg, IID};
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

const ARGS_MAX_COUNT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameHeader {
    pub regs_offset: u32,
    pub regs_count: u32,
    pub fn_id: bytecode::FnId,
    pub this: Value,
    pub return_value_vreg: Option<VReg>,
    pub return_to_iid: Option<IID>,
}

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
        assert!(
            call_meta.n_args as usize <= ARGS_MAX_COUNT,
            "calls with more than {} arguments are not implemented yet",
            ARGS_MAX_COUNT
        );
        let frame_hdr = FrameHeader {
            regs_offset: self.values.len().try_into().unwrap(),
            regs_count: call_meta.n_instrs + call_meta.n_args as u32,
            this: call_meta.this,
            return_value_vreg: call_meta.return_value_reg,
            return_to_iid: call_meta.return_to_iid,
            fn_id: call_meta.fnid,
        };

        #[cfg(test)]
        eprintln!(
            "  (allocated frame for {} args, {} captures, {} instrs -> {} values)",
            call_meta.n_args,
            call_meta.n_captured_upvalues,
            call_meta.n_instrs,
            frame_hdr.regs_count
        );

        self.headers.push(frame_hdr);
        for _ in 0..frame_hdr.regs_count {
            self.values.push(Slot::Inline(Value::Undefined));
        }

        self.check_invariants();
    }

    fn check_invariants(&self) {
        // TODO Turn into debug_asserts?

        for (ndx, hdr) in self.headers.iter().enumerate() {
            assert!(hdr.regs_count as usize >= ARGS_MAX_COUNT);
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

    fn nth_frame(&self, ndx: usize) -> Frame {
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
        self.nth_frame(0)
    }
    pub fn top_mut(&mut self) -> FrameMut {
        self.nth_frame_mut(0)
    }

    /// Returns the sequence of stack frames in the form of an iterator, ordered top to
    /// bottom.
    pub fn frames(&self) -> impl ExactSizeIterator<Item = Frame> {
        let n_frames = self.headers.len();
        (0..n_frames).map(|ndx| self.nth_frame(ndx))
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
        todo!(
            "delete this function: arguments are now located in the first {} registers",
            ARGS_MAX_COUNT
        )
    }
    pub fn args<'s>(&'s self) -> impl 's + ExactSizeIterator<Item = Option<Value>> {
        todo!(
            "delete this function: arguments are now located in the first {} registers",
            ARGS_MAX_COUNT
        );
        std::iter::empty()
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
}

impl<'a> FrameMut<'a> {
    pub(crate) fn set_result(mut self, vreg: bytecode::VReg, value: Value) {
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

    pub(crate) fn set_arg(mut self, argndx: bytecode::ArgIndex, value: Value) {
        todo!(
            "delete this function: arguments are now located in the first {} registers",
            ARGS_MAX_COUNT
        )
    }

    pub(crate) fn ensure_in_upvalue(self, var: bytecode::VReg) -> UpvalueId {
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

    pub(crate) fn set_capture(self, capture_ndx: bytecode::CaptureIndex, capture: UpvalueId) {
        todo!("move this operation to InterpreterData::push()?")
    }
}
