use std::{cell::Cell, marker::PhantomData, sync::atomic::AtomicUsize};

use crate::interpreter::{self, UpvalueId, Value};
use crate::stack_access::FrameMetrics;
use crate::{bytecode, stack_access};

type Heap = slotmap::SlotMap<UpvalueId, Value>;

/// The interpreter's stack.
///
/// Mostly stores local variables.
pub(crate) struct InterpreterData {
    upv_alloc: Heap,
    stack_buffer: [u8; STACK_SIZE],
    metrics: FrameMetrics,
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

            stack_buffer: [0u8; STACK_SIZE],
            metrics: FrameMetrics { top: STACK_SIZE },

            n_frames: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.n_frames
    }

    pub(crate) fn push(&mut self, call_meta: CallMeta) {
        let frame_hdr = stack_access::FrameHeader {
            n_instrs: call_meta.n_instrs,
            n_args: call_meta.n_args,
            n_captures: call_meta.n_captured_upvalues,
            this: call_meta.this,
            return_value_vreg: call_meta.return_value_reg,
            return_to_iid: call_meta.return_to_iid,
            fn_id: call_meta.fnid,
        };

        let frame_sz = frame_hdr.expected_frame_size();
        eprintln!(
            "  (allocated frame of size {} for {} args, {} captures, {} instrs)",
            frame_sz, call_meta.n_args, call_meta.n_captured_upvalues, call_meta.n_instrs,
        );
        assert!(
            frame_sz < self.metrics.top,
            "interpreter's stack overflowed"
        );
        self.metrics.top -= frame_sz;

        // NOTE: Safety: Initializing the frame completely is *vital*
        self.metrics.header().put(&mut self.stack_buffer, frame_hdr);
        for i in 0..call_meta.n_args {
            self.set_arg(i.into(), Value::Undefined);
        }
        for i in 0..call_meta.n_instrs {
            self.set_result(
                bytecode::VReg(i.try_into().expect("too many instructions")),
                Value::Undefined,
            );
        }

        self.n_frames += 1;
    }

    pub(crate) fn pop(&mut self) {
        use crate::util::shorten_by;
        let cur_frame_sz = self.metrics.frame_size(&self.stack_buffer);
        self.metrics.top += cur_frame_sz;
        self.n_frames -= 1;
    }

    pub(crate) fn fnid(&self) -> bytecode::FnId {
        let frame_hdr = self.metrics.header().get(&self.stack_buffer);
        frame_hdr.fn_id
    }

    pub(crate) fn caller_retval_reg(&self) -> Option<bytecode::VReg> {
        let frame_hdr = self.metrics.header().get(&self.stack_buffer);
        frame_hdr.return_value_vreg
    }

    pub(crate) fn return_to_iid(&self) -> Option<bytecode::IID> {
        let frame_hdr = self.metrics.header().get(&self.stack_buffer);
        frame_hdr.return_to_iid
    }

    fn slot_value<'a, 's: 'a>(&'s self, slot: &'a stack_access::Slot) -> &'a Value {
        match slot {
            stack_access::Slot::Inline(value) => value,
            stack_access::Slot::Upvalue(upv_id) => &self
                .upv_alloc
                .get(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack"),
        }
    }
    fn set_slot_value(&mut self, offset: stack_access::Offset<stack_access::Slot>, value: Value) {
        let slot = offset.get(&mut self.stack_buffer);
        match slot {
            stack_access::Slot::Inline(_) => {
                offset.put(&mut self.stack_buffer, stack_access::Slot::Inline(value));
            }
            stack_access::Slot::Upvalue(upv_id) => {
                let heap_value = self
                    .upv_alloc
                    .get_mut(*upv_id)
                    .expect("gc bug: value deleted but still referenced by stack");
                *heap_value = value;
            }
        };
    }

    pub(crate) fn get_result(&self, vreg: bytecode::VReg) -> Option<&Value> {
        let ndx = vreg.0 as usize;
        let slot = self
            .metrics
            .result_slot(ndx, &self.stack_buffer)?
            .get(&self.stack_buffer);
        Some(self.slot_value(slot))
    }
    pub(crate) fn set_result(&mut self, vreg: bytecode::VReg, value: Value) {
        let ndx = vreg.0 as usize;
        let offset = self.metrics.result_slot(ndx, &self.stack_buffer).unwrap();
        self.set_slot_value(offset, value);
    }
    pub(crate) fn get_arg(&self, arg_ndx: usize) -> Option<&Value> {
        let slot = self
            .metrics
            .arg_slot(arg_ndx, &self.stack_buffer)?
            .get(&self.stack_buffer);
        Some(self.slot_value(slot))
    }
    pub(crate) fn set_arg(&mut self, arg_ndx: usize, value: Value) {
        let offset = self.metrics.arg_slot(arg_ndx, &self.stack_buffer).unwrap();
        self.set_slot_value(offset, value);
    }

    pub(crate) fn get_this(&self) -> &Value {
        let header = self.metrics.header().get(&self.stack_buffer);
        &header.this
    }

    pub(crate) fn ensure_in_upvalue(&mut self, var: bytecode::VReg) -> UpvalueId {
        let varndx = var.0 as usize;
        let offset = self
            .metrics
            .result_slot(varndx, &self.stack_buffer)
            .unwrap();
        let slot = offset.get(&mut self.stack_buffer);
        match slot {
            stack_access::Slot::Inline(value) => {
                let upv_id = self.upv_alloc.insert(value.clone());
                offset.put(&mut self.stack_buffer, stack_access::Slot::Upvalue(upv_id));
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
        let upvalue_id = *self
            .metrics
            .capture_slot(capture_ndx.0 as usize, &self.stack_buffer)
            .get(&self.stack_buffer);

        let value = stack_access::Slot::Upvalue(upvalue_id);
        self.metrics
            .result_slot(vreg.0 as usize, &self.stack_buffer)
            .unwrap()
            .put(&mut self.stack_buffer, value);
    }

    pub(crate) fn set_capture(&mut self, capture_ndx: usize, capture: UpvalueId) {
        self.metrics
            .capture_slot(capture_ndx, &self.stack_buffer)
            .put(&mut self.stack_buffer, capture);
    }
}

fn slot_value<'a>(slot: &'a stack_access::Slot, upv_alloc: &'a Heap) -> &'a Value {
    match slot {
        stack_access::Slot::Inline(value) => value,
        stack_access::Slot::Upvalue(upv_id) => upv_alloc
            .get(*upv_id)
            .expect("gc bug: value deleted but still referenced by stack"),
    }
}

fn set_slot_value(slot: &mut stack_access::Slot, upv_alloc: &mut Heap, value: Value) {
    let value_slot = match slot {
        stack_access::Slot::Inline(slot_value) => slot_value,
        stack_access::Slot::Upvalue(upv_id) => upv_alloc
            .get_mut(*upv_id)
            .expect("gc bug: value deleted but still referenced by stack"),
    };

    *value_slot = value;
}
