//! [1] "this" substituion: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Strict_mode#no_this_substitution
use std::cell::RefCell;
use std::rc::Rc;

use crate::bytecode::{self, VReg, ARGS_COUNT_MAX};
use crate::interpreter::{JSClosure, Value};

/// The interpreter's stack.
///
/// Mostly stores local variables.
pub struct InterpreterData {
    upv_alloc: Upvalues,
    headers: Vec<FrameHeader>,
    values: Vec<Slot>,

    /// The value of `this` to be used in non-strict mode when "this" substituion [1]
    /// happens.  It should always be the same as the realm's `globalThis`/`window`.
    default_this: Value,

    /// The "current exception", i.e. the exception being handled (in the `catch` block
    /// that is currently running).
    cur_exc: Option<Value>,

    // TODO Make tests work without this hack
    #[cfg(any(test, feature = "debugger"))]
    pub sink: Vec<Value>,

    /// This flag is set right before the Interpreter is suspended due to a breakpoint of
    /// any kind.
    ///
    /// This is necessary to track because, the "successor" Interpreter that resumes
    /// executing with this InterpreterData needs to 'skip' the breakpoint in order not to
    /// enter an infinite loop of always suspending on the same breakpoint!
    #[cfg(feature = "debugger")]
    resuming_from_breakpoint: bool,
}

// Throughout this module, `Option<Value>` is stored instead of `Value`.  The `None` case
// represents an uninitialized storage location corresponding to a variable still in the
// temporal dead zone.   See [1] for the source-level semantics in JavaScript.
//
// [1] https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/let#temporal_dead_zone_tdz

slotmap::new_key_type! { pub struct UpvalueId; }

type Upvalues = slotmap::SlotMap<UpvalueId, Option<Value>>;

#[derive(Clone, Copy, PartialEq)]
enum Slot {
    Inline(Option<Value>),
    Upvalue(UpvalueId),
}

#[cfg(feature = "debugger")]
#[derive(Debug)]
pub enum SlotDebug {
    // slightly nicer to have uninitialized value show up explicitly as "Uninitialized" instead of
    // "Inline(None)"
    Uninitialized,
    Inline(Value),
    Upvalue(UpvalueId),
}
#[cfg(feature = "debugger")]
impl From<Slot> for SlotDebug {
    fn from(value: Slot) -> Self {
        match value {
            Slot::Inline(None) => SlotDebug::Uninitialized,
            Slot::Inline(Some(x)) => SlotDebug::Inline(x),
            Slot::Upvalue(x) => SlotDebug::Upvalue(x),
        }
    }
}

pub struct FrameHeader {
    regs_offset: u32,
    regs_count: u32,
    pub fnid: bytecode::FnId,
    pub iid: bytecode::IID,
    pub this: Value,

    /// Where to store the return value in *this* frame (the caller's frame).
    pub return_target: Option<VReg>,

    /// Exception handlers, ordered by scope, *local to this stack frame*
    /// Cheap (not allocated) until we enter a 'try' block (PushExcHandler)
    // TODO Move to a shared vector, like for registers?
    pub exc_handlers: Vec<bytecode::IID>,

    /// True iff this frame represents a call to a module's root function.
    ///
    /// In this case, the return value can be cached for future imports of the same
    /// module. (None of the other mechanisms related to the return value changes.)
    // TODO Move this to closure
    pub is_module_root_fn: bool,

    pub closure: Rc<JSClosure>,
}

impl InterpreterData {
    /// TODO(small feat) Better value?
    const INIT_CAPACITY: usize = 256;

    pub(crate) fn new() -> Self {
        InterpreterData {
            upv_alloc: slotmap::SlotMap::with_key(),
            headers: Vec::with_capacity(Self::INIT_CAPACITY),
            values: Vec::with_capacity(Self::INIT_CAPACITY),
            default_this: Value::Undefined,

            cur_exc: None,

            #[cfg(any(test, feature = "debugger"))]
            sink: Vec::new(),

            #[cfg(feature = "debugger")]
            resuming_from_breakpoint: false,
        }
    }

    pub fn set_default_this(&mut self, value: Value) {
        self.default_this = value;
    }

    pub fn len(&self) -> usize {
        self.headers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push onto the stack a frame for a simple call to the given function, with no
    /// arguments and no captures.
    ///
    /// A closure is created on-the-fly for purpose of this one call.
    ///
    /// `this` is always used (there is not going to be any 'forced this' on the closure,
    /// nor any "this" substitution kicking in).
    pub(crate) fn push_direct(
        &mut self,
        fnid: bytecode::FnId,
        func: &bytecode::Function,
        this: Value,
    ) {
        let closure = Rc::new(JSClosure {
            fnid,
            upvalues: Vec::new(),
            forced_this: Some(this),
            generator_snapshot: RefCell::new(None),
        });

        let frame_hdr = FrameHeader {
            regs_offset: self.values.len().try_into().unwrap(),
            regs_count: func.n_regs() as u32 + ARGS_COUNT_MAX as u32,
            fnid,
            iid: bytecode::IID(0),
            this,
            return_target: None,
            exc_handlers: Vec::new(),
            is_module_root_fn: false,
            closure,
        };

        for _ in 0..frame_hdr.regs_count {
            self.values.push(Slot::Inline(None));
        }
        self.headers.push(frame_hdr);

        self.check_invariants();
    }

    /// Push a stack frame for a call to the given closure.
    ///
    /// The first (bytecode::ARGS_COUNT_MAX) registers (destined for call arguments) are
    /// all initialized `undefined`.
    ///
    /// In the new stack frame, `local_this` is used as the value for `this` unless the
    /// closure has a `forced_this` value (result of `Function.prototype.bind`) or "this"
    /// substitution happens [1].
    pub(crate) fn push_call(
        &mut self,
        closure: Rc<JSClosure>,
        local_this: Value,
        loader: &crate::Loader,
    ) {
        let callee_func = loader.get_function(closure.fnid).unwrap();

        let is_strict = callee_func.is_strict_mode();
        let this = match (closure.forced_this, is_strict, local_this) {
            (Some(value), _, _) => value,
            // "this" substitution (see [1]).  If I understand this correctly, we don't
            // need to box anything right now.  We just pass the value, and the callee
            // will box it when needed.
            (_, false, Value::Null | Value::Undefined) => self.default_this,
            (_, _, _) => local_this,
        };

        let gen_snap = closure.generator_snapshot.borrow_mut().take();

        if let Some(gen_snap) = gen_snap {
            // resume the suspended of the generator
            let frame_hdr = FrameHeader {
                regs_offset: self.values.len().try_into().unwrap(),
                regs_count: callee_func.n_regs() as u32 + ARGS_COUNT_MAX as u32,
                fnid: closure.fnid,
                iid: gen_snap.iid,
                this: gen_snap.this,
                return_target: gen_snap.return_target,
                closure,
                exc_handlers: gen_snap.exc_handlers,
                is_module_root_fn: gen_snap.is_module_root_fn,
            };

            assert_eq!(frame_hdr.regs_count as usize, gen_snap.slots.len());
            self.values.extend(gen_snap.slots);
            self.headers.push(frame_hdr);
        } else {
            // brand new call (or just not a generator)
            let frame_hdr = FrameHeader {
                regs_offset: self.values.len().try_into().unwrap(),
                regs_count: callee_func.n_regs() as u32 + ARGS_COUNT_MAX as u32,
                fnid: closure.fnid,
                iid: bytecode::IID(0),
                this,
                return_target: None,
                closure,
                exc_handlers: Vec::new(),
                is_module_root_fn: false,
            };
            for _ in 0..frame_hdr.regs_count {
                self.values.push(Slot::Inline(None));
            }
            self.headers.push(frame_hdr);
        };

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

    /// Pop a frame from the stack.
    ///
    /// Panics if the stack is currently empty.  (Also contains some asserts that check
    /// invariants.)
    pub(crate) fn pop(&mut self) {
        let frame_hdr = self.headers.pop().unwrap();
        self.values
            .truncate(self.values.len() - frame_hdr.regs_count as usize);
        self.check_invariants();
    }

    pub fn nth_frame(&self, ndx: usize) -> Frame {
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

    #[cfg(test)]
    pub(crate) fn push_to_sink(&mut self, value: Value) {
        self.sink.push(value)
    }

    #[cfg(any(test, feature = "debugger"))]
    pub(crate) fn sink(&self) -> &[Value] {
        &self.sink
    }

    pub(crate) fn get_cur_exc(&self) -> Option<Value> {
        self.cur_exc
    }
    pub(crate) fn set_cur_exc(&mut self, value: Value) {
        self.cur_exc = Some(value);
    }
}

#[cfg(feature = "debugger")]
impl InterpreterData {
    /// Set the "resuming from breakpoint" flag to true.
    pub(crate) fn set_resuming_from_breakpoint(&mut self) {
        self.resuming_from_breakpoint = true;
    }

    /// Return the "resuming from breakpoint" flag and set it to false immediately
    /// afterwards.
    pub(crate) fn take_resuming_from_breakpoint(&mut self) -> bool {
        std::mem::replace(&mut self.resuming_from_breakpoint, false)
    }
}

pub struct Frame<'a> {
    header: &'a FrameHeader,
    values: &'a [Slot],
    upv_alloc: &'a Upvalues,
}
pub struct FrameMut<'a> {
    header: &'a mut FrameHeader,
    values: &'a mut [Slot],
    upv_alloc: &'a mut Upvalues,
}

impl<'a> Frame<'a> {
    pub fn header(&self) -> &'a FrameHeader {
        self.header
    }

    #[cfg(feature = "debugger")]
    pub fn get_slot(&self, vreg: bytecode::VReg) -> SlotDebug {
        self.values[vreg.0 as usize].into()
    }

    /// Return the value of the given virtual register.
    ///
    /// Returns `None` iff the value is still uninitialized (i.e. in the temporal dead
    /// zone).
    pub fn get_result(&self, vreg: bytecode::VReg) -> Option<Value> {
        let slot = &self.values[vreg.0 as usize];
        match slot {
            Slot::Inline(value) => *value,
            Slot::Upvalue(upv_id) => *self
                .upv_alloc
                .get(*upv_id)
                .expect("gc bug: value deleted but still referenced by stack"),
        }
    }
    /// Returns an iterator for the value of each virtual register.  See `get_result`.
    pub fn results(&self) -> impl '_ + ExactSizeIterator<Item = Option<Value>> {
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
            "not yet implemented: call with >= {} arguments",
            ARGS_COUNT_MAX
        );
        self.get_result(bytecode::VReg(argndx.0))
    }
    pub fn args(&self) -> impl '_ + ExactSizeIterator<Item = Option<Value>> {
        (0..ARGS_COUNT_MAX).map(|i| self.get_arg(bytecode::ArgIndex(i)))
    }

    pub fn get_capture(&self, capture_ndx: bytecode::CaptureIndex) -> UpvalueId {
        // another/better approach: add N slots to the header and use them as cache for the
        // closure's captures.
        self.header
            .closure
            .upvalues
            .get(capture_ndx.0 as usize)
            .copied()
            .expect("capture index is out of range")
    }
    pub fn captures(&self) -> impl '_ + ExactSizeIterator<Item = UpvalueId> {
        self.header.closure.upvalues.iter().copied()
    }

    /// Get the upvalue with the given ID.
    ///
    /// The return value has (unfortunately) two levels of `Option` with distinct meaning:
    /// - the outer Option is None iff the given `upv_id` is invalid.
    /// - the inner Option is None iff the upvalue exists but contains an uninitialized
    ///   value (i.e. in the temporal dead zone).
    pub fn deref_upvalue(&self, upv_id: UpvalueId) -> Option<Option<Value>> {
        self.upv_alloc.get(upv_id).copied()
    }

    pub fn return_target_reg(&self) -> Option<VReg> {
        self.header.return_target
    }
}

impl<'a> FrameMut<'a> {
    pub(crate) fn set_result(&mut self, vreg: bytecode::VReg, value: Value) {
        let slot = &mut self.values[vreg.0 as usize];
        match slot {
            Slot::Inline(slot_value) => {
                *slot_value = Some(value);
            }
            Slot::Upvalue(upv_id) => {
                let heap_value = self
                    .upv_alloc
                    .get_mut(*upv_id)
                    .expect("gc bug: value deleted but still referenced by stack");
                *heap_value = Some(value);
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

    pub(crate) fn ensure_inline(&mut self, var: bytecode::VReg) {
        let slot = &mut self.values[var.0 as usize];
        match slot {
            Slot::Inline(_) => {}
            Slot::Upvalue(upv_id) => {
                let value = *self.upv_alloc.get(*upv_id).unwrap();
                *slot = Slot::Inline(value);
            }
        }
    }

    pub(crate) fn set_return_target(&mut self, reg: VReg) {
        self.header.return_target = Some(reg);
    }
    pub(crate) fn take_return_target(&mut self) -> Option<VReg> {
        self.header.return_target.take()
    }

    pub(crate) fn set_resume_iid(&mut self, iid: crate::IID) {
        self.header.iid = iid;
    }

    pub(crate) fn pop_exc_handler(&mut self) -> Option<bytecode::IID> {
        self.header.exc_handlers.pop()
    }
    pub(crate) fn push_exc_handler(&mut self, iid: bytecode::IID) {
        self.header.exc_handlers.push(iid)
    }

    pub(crate) fn set_is_module_root_fn(&mut self) {
        self.header.is_module_root_fn = true;
    }

    pub(super) fn save_snapshot(&mut self, resume_iid: bytecode::IID) {
        let hdr = &mut self.header;

        let snapshot = FrameSnapshot {
            // Note that this (likely) points to a different IID than the header!
            iid: resume_iid,
            this: hdr.this,
            return_target: hdr.return_target,
            exc_handlers: hdr.exc_handlers.clone(),
            is_module_root_fn: hdr.is_module_root_fn,
            // pity for this allocation
            slots: self.values.to_vec(),
        };

        let mut generator_snapshot = hdr.closure.generator_snapshot.borrow_mut();
        *generator_snapshot = Some(snapshot);
    }
}

// Right now I think it's fine to compare and clone FrameSnapshot objects
#[derive(PartialEq, Clone)]
pub struct FrameSnapshot {
    // restoring a frame from a snapshot starts from the JSClosure object that contains
    // the FrameSnapshot.  a FrameSnapshot stores whatever is required that isn't already
    // stored there.
    //
    // see `FrameHeader` for a description of the fields
    iid: bytecode::IID,
    this: Value,
    return_target: Option<VReg>,
    exc_handlers: Vec<bytecode::IID>,
    is_module_root_fn: bool,

    // only the slots corresponding to this exact frame.  the restored FrameHeader hdr
    // will have `hdr.regs_count == slots.len()`.
    slots: Vec<Slot>,
}
