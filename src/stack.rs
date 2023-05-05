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

const STACK_SIZE: usize = 16 * 1024;

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
            call_iid: call_meta.call_iid,
            fn_id: call_meta.fnid,
        };

        let frame_sz = frame_hdr.expected_frame_size();
        self.metrics.top -= frame_sz;
        *self.metrics.header().get_mut(&mut self.stack_buffer) = frame_hdr;
        self.n_frames += 1;

        #[cfg(test)]
        eprintln!(
            "pushed frame, {} bytes. top = {}; hdr = {:?}",
            frame_sz,
            self.metrics.top,
            self.metrics.header().get(&self.stack_buffer)
        );
    }

    pub(crate) fn pop(&mut self) {
        use crate::util::shorten_by;
        let cur_frame_sz = self.metrics.frame_size(&self.stack_buffer);
        self.metrics.top += cur_frame_sz;
        self.n_frames -= 1;

        #[cfg(test)]
        if self.n_frames == 0 {
            eprintln!(
                "popped frame, {} bytes. top = {}, no more stack!",
                cur_frame_sz, self.metrics.top
            );
        } else {
            let hdr = self.metrics.header().get(&self.stack_buffer);
            eprintln!(
                "popped frame, {} bytes. top = {}, hdr = {:?}",
                cur_frame_sz, self.metrics.top, hdr
            );
        }
    }

    pub(crate) fn fnid(&self) -> bytecode::FnId {
        let frame_hdr = self.metrics.header().get(&self.stack_buffer);
        frame_hdr.fn_id
    }

    pub(crate) fn call_iid(&self) -> Option<bytecode::IID> {
        let frame_hdr = self.metrics.header().get(&self.stack_buffer);
        frame_hdr.call_iid
    }

    pub(crate) fn get_result(&self, iid: bytecode::IID) -> &Value {
        let ndx = iid.0 as usize;
        let slot = self
            .metrics
            .result_slot(ndx, &self.stack_buffer)
            .get(&self.stack_buffer);
        slot_value(slot, &self.upv_alloc)
    }
    pub(crate) fn set_result(&mut self, iid: bytecode::IID, value: Value) {
        let ndx = iid.0 as usize;
        let slot = self
            .metrics
            .result_slot(ndx, &self.stack_buffer)
            .get_mut(&mut self.stack_buffer);
        set_slot_value(slot, &mut self.upv_alloc, value);
    }

    pub(crate) fn get_arg(&self, arg_ndx: usize) -> &Value {
        let slot = self
            .metrics
            .arg_slot(arg_ndx, &self.stack_buffer)
            .get(&self.stack_buffer);
        slot_value(slot, &self.upv_alloc)
    }
    pub(crate) fn set_arg(&mut self, arg_ndx: usize, value: Value) {
        let slot = self
            .metrics
            .arg_slot(arg_ndx, &self.stack_buffer)
            .get_mut(&mut self.stack_buffer);
        set_slot_value(slot, &mut self.upv_alloc, value);
    }

    pub(crate) fn ensure_in_upvalue(&mut self, var: bytecode::IID) -> UpvalueId {
        let varndx = var.0 as usize;
        let slot = self
            .metrics
            .result_slot(varndx, &self.stack_buffer)
            .get_mut(&mut self.stack_buffer);
        match slot {
            stack_access::Slot::Inline(value) => {
                // This is just to give this shuffle well-defined behavior.  Hopefully it gets
                // optimized out.
                let value = std::mem::replace(value, Value::Undefined);
                let upv_id = self.upv_alloc.insert(value);
                *slot = stack_access::Slot::Upvalue(upv_id);
                upv_id
            }
            stack_access::Slot::Upvalue(upv_id) => *upv_id,
        }
    }

    pub(crate) fn capture_to_var(
        &mut self,
        capture_ndx: bytecode::CaptureIndex,
        var: bytecode::IID,
    ) {
        let upvalue_id = *self
            .metrics
            .capture_slot(capture_ndx.into(), &self.stack_buffer)
            .get(&self.stack_buffer);

        let slot = self
            .metrics
            .result_slot(var.0 as usize, &self.stack_buffer)
            .get_mut(&mut self.stack_buffer);
        *slot = stack_access::Slot::Upvalue(upvalue_id);
    }

    pub(crate) fn set_capture(&mut self, capture_ndx: usize, capture: UpvalueId) {
        let capture_slot = self
            .metrics
            .capture_slot(capture_ndx, &self.stack_buffer)
            .get_mut(&mut self.stack_buffer);
        *capture_slot = capture;
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
        stack_access::Slot::Inline(slot_value) => {
            eprintln!(".. set inline result = {:?}", value);
            slot_value
        }
        stack_access::Slot::Upvalue(upv_id) => {
            eprintln!(".. set upvalue {:?} = {:?}", upv_id, value);
            upv_alloc
                .get_mut(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack")
        }
    };

    *value_slot = value;
}
