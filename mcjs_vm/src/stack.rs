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

    pub fn header(&self) -> &stack_access::FrameHeader {
        self.stack.top_header()
    }

    pub fn get_result(&self, vreg: bytecode::VReg) -> Value {
        let slot = &self.stack.vars()[vreg.0 as usize];
        slot_value(slot, &self.upv_alloc)
    }
    pub(crate) fn set_result(&mut self, vreg: bytecode::VReg, value: Value) {
        let slot = &mut self.stack.vars_mut()[vreg.0 as usize];
        set_slot_value(slot, value, &mut self.upv_alloc);
    }

    pub fn get_arg(&self, argndx: bytecode::ArgIndex) -> Option<Value> {
        self.stack
            .args()
            .get(argndx.0 as usize)
            .map(|slot| slot_value(slot, &self.upv_alloc))
    }
    pub(crate) fn set_arg(&mut self, argndx: bytecode::ArgIndex, value: Value) {
        let slot = &mut self.stack.args_mut()[argndx.0 as usize];
        set_slot_value(slot, value, &mut self.upv_alloc);
    }

    pub(crate) fn ensure_in_upvalue(&mut self, var: bytecode::VReg) -> UpvalueId {
        let varndx = var.0 as usize;
        let slot = &mut self.stack.vars_mut()[varndx as usize];

        match slot {
            stack_access::Slot::Inline(value) => {
                let upv_id = self.upv_alloc.insert(*value);
                *slot = stack_access::Slot::Upvalue(upv_id);
                upv_id
            }
            stack_access::Slot::Upvalue(upv_id) => *upv_id,
        }
    }

    pub(crate) fn capture_to_var(
        &mut self,
        capture_ndx: bytecode::CaptureIndex,
        vreg: bytecode::VReg,
    ) {
        let slot = self.stack.captures()[capture_ndx.0 as usize];
        self.stack.vars_mut()[vreg.0 as usize] = slot;
    }

    pub(crate) fn set_capture(&mut self, capture_ndx: bytecode::CaptureIndex, capture: UpvalueId) {
        self.stack.captures_mut()[capture_ndx.0 as usize] = stack_access::Slot::Upvalue(capture);
    }
    pub fn get_capture(&self, capture_ndx: bytecode::CaptureIndex) -> UpvalueId {
        let slot = &self.stack.captures()[capture_ndx.0 as usize];
        match slot {
            stack_access::Slot::Inline(_) => unreachable!(),
            stack_access::Slot::Upvalue(upv_id) => *upv_id,
        }
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
