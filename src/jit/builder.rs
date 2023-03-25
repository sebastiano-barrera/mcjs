use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::jit::codegen::index_of_reg;
use crate::{
    interpreter::{self, FnId, VarIndex, IID},
    stack::{self, FrameId},
};

use dynasm::dynasm;
use strum_macros::EnumIter;

use super::{regalloc, tracking, Trace};

// This is going to be changed at some point
pub(super) type BoxedValue = interpreter::Value;

#[derive(Debug, Clone)]
pub struct TypeError {
    desired_type: ValueType,
}

#[derive(Debug, Clone)]
pub struct InconsistentUnbox {
    desired_type: ValueType,
}

#[derive(Debug, Clone)]
enum Error {
    Type(TypeError),
    Unsupported(Cow<'static, str>),
    InconsistentUnbox(InconsistentUnbox),

    /// The compiler has encountered code that has been deemed inconvenient/
    /// not worth it to compile (for the gained performance or implementation
    /// complexity)
    ///
    /// This is an *internal* signal, not for external consumption.
    UntraceableByDesign,
}
impl From<TypeError> for Error {
    fn from(type_err: TypeError) -> Self {
        Error::Type(type_err)
    }
}

#[derive(PartialEq, Clone, Debug)]
pub(super) struct Cmp {
    pub(super) ty: ValueType,
    pub(super) op: CmpOp,
    pub(super) a: ValueId,
    pub(super) b: ValueId,
}

impl Cmp {
    fn invert(&self) -> Self {
        let inverted_op = match self.op {
            CmpOp::GE => CmpOp::LT,
            CmpOp::GT => CmpOp::LE,
            CmpOp::LT => CmpOp::GE,
            CmpOp::LE => CmpOp::GT,
            CmpOp::EQ => CmpOp::NE,
            CmpOp::NE => CmpOp::EQ,
        };
        Cmp {
            op: inverted_op,
            ..self.clone()
        }
    }
}

// TODO(cleanup) Move this to the super module
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub(super) u32);

impl std::fmt::Debug for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Type of a value at runtime, in the trace.
///
/// Note that this type is used both by the trace builder and by the trace
/// wrapper (at runtime!) when passing the snapshot to the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
#[repr(u8)]
pub(super) enum ValueType {
    Boxed,
    Bool,
    Num,
    Str,
    Obj,
    Null,
    Undefined,
    Function,
}

impl ValueType {
    pub(super) fn of(value: &BoxedValue) -> Self {
        match value {
            interpreter::Value::Number(_) => ValueType::Num,
            interpreter::Value::String(_) => ValueType::Str,
            interpreter::Value::Bool(_) => ValueType::Bool,
            interpreter::Value::Null => ValueType::Null,
            interpreter::Value::Undefined => ValueType::Undefined,
            interpreter::Value::SelfFunction => ValueType::Function,
            interpreter::Value::NativeFunction(_) => ValueType::Function,
            interpreter::Value::Object(_) => ValueType::Obj,
            interpreter::Value::Closure(_) => ValueType::Function,
        }
    }

    fn js_typeof(&self) -> Option<&'static str> {
        let ret_str = match self {
            ValueType::Bool => "boolean",
            ValueType::Num => "number",
            ValueType::Str => "string",
            ValueType::Obj => "object",
            // TODO(cleanup) This is actually an error in our type system.  null is really a value of the 'object' type
            ValueType::Null => "object",
            ValueType::Undefined => "undefined",
            ValueType::Function => "function",
            ValueType::Boxed => return None,
        };

        Some(ret_str)
    }
}

pub struct TraceBuilder {
    vars: tracking::VarsState,

    // "Parking spot" used to tranfer the arguments from caller to callee
    // frame, and the return value in the other direction
    args_buf: Option<Vec<ValueId>>,
    return_value: Option<ValueId>,

    loop_head: Option<interpreter::GlobalIID>,
    loop_entered: bool,

    instrs: Vec<Instr>,

    snapshot_map: SnapshotMap,

    // Temporary buffer used to set snapshot updates "coordinates" into trace-
    // exiting instructions (such as AssertTrue and Unbox).  These will result
    // in snapshot update instructions in the native code.
    exit_snapshot_changes: Vec<Option<ValueId>>,

    state: TraceBuilderState,
    exit_function_expected: bool,
}

// Tells which interpreter value to write into or read from each slot of the
// snapshot.
//
// Items that are `None` are "exit-only" slots: they're written by the
// trace on (some of) its exits, and the interpreter can simply ignore
// those when "initializing" the trace.
//
// (See jit::codegen for an explanation of what the "snapshot" is and how
// it works.)
pub(crate) struct SnapshotMap {
    items: Vec<SnapshotMapItem>,
}

pub(crate) struct SnapshotMapItem {
    pub(crate) write_on_entry: bool,
    pub(crate) write_on_exit: bool,
    pub(crate) operand: interpreter::Operand,
}

#[cfg(test)]
impl From<Vec<SnapshotMapItem>> for SnapshotMap {
    fn from(items: Vec<SnapshotMapItem>) -> Self {
        SnapshotMap { items }
    }
}

impl SnapshotMap {
    pub(super) fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub(super) fn len(&self) -> u16 {
        self.items.len().try_into().unwrap()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &SnapshotMapItem> {
        self.items.iter()
    }

    pub(super) fn get(&self, ndx: usize) -> &SnapshotMapItem {
        &self.items[ndx]
    }

    pub(super) fn find(&mut self, operand: &interpreter::Operand) -> Option<usize> {
        self.items
            .iter()
            .enumerate()
            .find(|(_, item)| &item.operand == operand)
            .map(|(ndx, _)| ndx)
    }

    fn find_or_insert(&mut self, operand: &interpreter::Operand) -> usize {
        self.find(operand).unwrap_or_else(|| {
            // TODO(small feat) on building, assert that no item has both write_on_ flags set to false
            self.items.push(SnapshotMapItem {
                operand: operand.clone(),
                write_on_entry: false,
                write_on_exit: false,
            });
            self.items.len() - 1
        })
    }

    fn set_entry(&mut self, operand: &interpreter::Operand) -> u16 {
        eprintln!("[..tb snapshot: set written-on-entry: {:?}]", operand);
        let ndx = self.find_or_insert(operand);
        self.items[ndx].write_on_entry = true;
        ndx as u16
    }

    fn set_exit(&mut self, operand: &interpreter::Operand) -> u16 {
        eprintln!("[..tb snapshot: set written-on-exit: {:?}]", operand);
        let ndx = self.find_or_insert(operand);
        self.items[ndx].write_on_exit = true;
        ndx as u16
    }
}

impl std::fmt::Debug for SnapshotMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{snapshot ")?;
        for item in self.items.iter() {
            write!(
                f,
                "{}{:?}{}, ",
                if item.write_on_entry { '>' } else { ' ' },
                item.operand,
                if item.write_on_exit { '>' } else { ' ' },
            )?;
        }
        write!(f, "}}")
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct VarId {
    // Signed, unlike interpreter::VarId.  Negative values refer to values
    // defined in inner scopes (i.e. functions currently being JIT-compiled), while
    // positive values refer to variables captured from the trace's entry point
    // function/loop.
    stack_depth: i32,
    var_ndx: u16,
}

#[derive(Debug)]
pub(super) struct TraceParam {
    operand: interpreter::Operand,
}

#[derive(Debug, PartialEq, Eq)]
enum TraceBuilderState {
    Tracing,
    Failed,
    Finished,
}

pub enum CloseMode {
    Loop(interpreter::GlobalIID),
    FunctionExit,
}

impl TraceBuilder {
    pub(crate) fn start(fid: stack::FrameId, n_local_vars: usize, close_mode: CloseMode) -> Self {
        let mut tb = TraceBuilder {
            return_value: None,
            args_buf: None,
            vars: tracking::VarsState::new(),
            instrs: Vec::new(),
            snapshot_map: SnapshotMap::new(),
            exit_snapshot_changes: Vec::new(),
            loop_head: match close_mode {
                CloseMode::Loop(giid) => Some(giid),
                CloseMode::FunctionExit => None,
            },
            loop_entered: false,
            state: TraceBuilderState::Tracing,
            exit_function_expected: false,
        };

        let args = Vec::new();
        tb.vars.push_frame(fid, args, n_local_vars);

        tb
    }

    // TODO(cleanup) Inline into all callers
    fn stack_depth(&self) -> usize {
        self.vars.stack_depth()
    }

    fn resolve_interpreter_operand<F>(
        &mut self,
        intrp_operand: &interpreter::Operand,
        fnid_to_frameid: F,
    ) -> Result<ValueId, Error>
    where
        F: Fn(FnId) -> FrameId,
    {
        match intrp_operand {
            // constants from the interpreter bytecode are simply preserved
            interpreter::Operand::Value(value) => Ok(self.emit(Instr::Const(value.clone()))?),

            // variables from the interpreter (identified by the ID of the
            // instruction that produced them) are mapped to the JIT instruction
            // that last assigned to it (the JIT trace is SSA)
            interpreter::Operand::IID(iid) => {
                let stack_depth = self.stack_depth();
                match self.vars.get_result(*iid).cloned() {
                    Some(vid_for_var) => {
                        print_indent(stack_depth);
                        eprintln!("[..tb {:?} is {:?}]", iid, vid_for_var);
                        Ok(vid_for_var)
                    }
                    None => {
                        print_indent(stack_depth);
                        eprintln!(
                            "[..tb {:?} is unresolved => considered trace parameter, adding guard]",
                            intrp_operand
                        );
                        // TODO(cleanup) Test that this is really necessary. If so, describe it here
                        assert_eq!(1, stack_depth);

                        let param = self.add_entry_snapshot_var(intrp_operand)?;
                        self.vars.set_result(*iid, param);
                        Ok(param)
                    }
                }
            }

            interpreter::Operand::Var(intrp_varid) => {
                let frame_id = fnid_to_frameid(intrp_varid.fnid);
                let var_ndx = intrp_varid.var_ndx;

                let slot = self.vars.get_var(frame_id, var_ndx);
                if let Some(value) = slot {
                    return Ok(value);
                }

                let vid = self.add_entry_snapshot_var(intrp_operand)?;
                self.vars.set_var(frame_id, intrp_varid.var_ndx, vid);
                Ok(vid)
            }
        }
    }

    fn ensure_type(&mut self, vid: ValueId, desired_type: ValueType) -> Result<ValueId, Error> {
        let type_error = Error::Type(TypeError { desired_type });

        let src_ins = self.instrs.get_mut(vid.0 as usize).unwrap();
        let input_type_field = match src_ins {
            Instr::GetSnapshotItem { ty, .. } => Some(ty),
            Instr::GetArg { ty, .. } => Some(ty),
            _ => None,
        };

        if let Some(input_type_field) = input_type_field {
            // Source instruction has an implicit unbox phase
            if *input_type_field == ValueType::Boxed {
                if desired_type != ValueType::Boxed {
                    print_indent(self.vars.stack_depth());
                    eprintln!("[..tb emit unbox: v{:<4} as {:?}]", vid.0, desired_type);
                    *input_type_field = desired_type;
                }

                return Ok(vid);
            }
        }

        // We're now either in a non-unboxing instruction, or in an unboxing
        // instruction that was already set to a non-box type

        let input_type = src_ins
            .result_type()
            .unwrap_or_else(|| panic!("JIT bug: src instr has no result: {:?}", src_ins));

        // There should never be an instruction that has Boxed as its *fixed*
        // result type.  Rather, it should have an implicit unbox phase and a
        // 'result type' field that allows us to control the result type of the
        // unboxing.
        assert_ne!(input_type, ValueType::Boxed);
        let converted_operand = match (input_type, desired_type) {
            (a, b) if a == b => Some(vid),

            (_, ValueType::Boxed) => Some(self.emit(Instr::Box(vid))?),

            (_, ValueType::Null) => Some(self.emit(Instr::Const(BoxedValue::Null))?),
            (_, ValueType::Undefined) => Some(self.emit(Instr::Const(BoxedValue::Undefined))?),
            (_, ValueType::Function) => None,

            (ValueType::Bool, ValueType::Num) => {
                let if_true = self.emit(Instr::Const(BoxedValue::Number(1.0)))?;
                let if_false = self.emit(Instr::Const(BoxedValue::Number(0.0)))?;
                Some(self.emit(Instr::Choose {
                    ty: ValueType::Num,
                    cond: vid,
                    if_true,
                    if_false,
                })?)
            }
            (ValueType::Bool, ValueType::Str) => {
                let if_true = self.emit(Instr::Const("true".into()))?;
                let if_false = self.emit(Instr::Const("false".into()))?;
                Some(self.emit(Instr::Choose {
                    ty: ValueType::Str,
                    cond: vid,
                    if_true,
                    if_false,
                })?)
            }

            (ValueType::Num, ValueType::Bool) => {
                let cmp = Cmp {
                    ty: ValueType::Num,
                    op: CmpOp::EQ,
                    a: vid,
                    b: self.emit(Instr::Const(BoxedValue::Number(0.0)))?,
                };
                Some(self.emit(cmp.into())?)
            }
            (ValueType::Num, ValueType::Str) => Some(self.emit(Instr::Num2Str(vid))?),

            (ValueType::Str, ValueType::Bool) => {
                let cmp = Cmp {
                    ty: ValueType::Str,
                    op: CmpOp::EQ,
                    a: vid,
                    // TODO(opt) this string allocation could be avoided
                    b: self.emit(Instr::Const("".into()))?,
                };
                Some(self.emit(cmp.into())?)
            }

            // TODO(big feat) Convert string to number
            (ValueType::Str, ValueType::Num) => None,

            (ValueType::Null, ValueType::Bool) => {
                Some(self.emit(Instr::Const(BoxedValue::Bool(false)))?)
            }
            (ValueType::Null, ValueType::Num) => {
                Some(self.emit(Instr::Const(BoxedValue::Number(0.0)))?)
            }

            // TODO(opt) this string allocation could be avoided
            (ValueType::Null, ValueType::Str) => Some(self.emit(Instr::Const("null".into()))?),

            (ValueType::Undefined, ValueType::Bool) => {
                Some(self.emit(Instr::Const(BoxedValue::Bool(false)))?)
            }
            (ValueType::Undefined, ValueType::Num) => {
                Some(self.emit(Instr::Const(BoxedValue::Number(0.0)))?)
            }
            (ValueType::Undefined, ValueType::Str) => {
                Some(self.emit(Instr::Const("undefined".into()))?)
            }

            (ValueType::Function, _) => None,

            other_conv => unreachable!("{:?}", other_conv),
        };

        if let Some(converted_operand) = converted_operand {
            assert_eq!(
                Some(desired_type),
                self.instrs
                    .get(converted_operand.0 as usize)
                    .unwrap()
                    .result_type()
            );
        }

        converted_operand.ok_or(type_error)
    }

    fn resolve_operand_as<F>(
        &mut self,
        interp_oper: &interpreter::Operand,
        fnid_to_frameid: F,
        desired_type: ValueType,
    ) -> Result<ValueId, Error>
    where
        F: Fn(FnId) -> FrameId,
    {
        let oper = self.resolve_interpreter_operand(interp_oper, fnid_to_frameid)?;
        self.ensure_type(oper, desired_type)
    }

    pub(crate) fn interpreter_step(&mut self, step: &InterpreterStep) {
        match self.state {
            TraceBuilderState::Tracing => {
                if self.loop_entered && Some(step.global_iid()) == self.loop_head {
                    eprintln!("[..tb trace finished <- loop closed]");
                    self.state = TraceBuilderState::Finished;
                    return;
                }

                if let Err(error) = self.trace_step(step) {
                    print_indent(self.stack_depth());
                    eprintln!("[..tb trace failed: {:?}]", error);
                    self.state = TraceBuilderState::Failed;
                }

                if !self.loop_entered && Some(step.global_iid()) == self.loop_head {
                    self.loop_entered = true;
                }
            }
            TraceBuilderState::Failed => {}
            TraceBuilderState::Finished => {}
        }
    }

    fn trace_step(&mut self, step: &InterpreterStep) -> Result<(), Error> {
        assert!(self.state == TraceBuilderState::Tracing);

        if self.exit_function_expected {
            panic!("JIT bug: should have called exit_function() !");
        }

        let instr = step.cur_instr();

        print_indent(self.stack_depth());
        eprintln!("[..tb step]");

        let result = match instr {
            interpreter::Instr::Nop => None,
            interpreter::Instr::Const(value) => Some(self.emit(Instr::Const(value.clone()))?),
            interpreter::Instr::Not(oper) => {
                let oper = self.resolve_operand_as(oper, step.fnid_to_frameid, ValueType::Bool)?;
                Some(self.emit(Instr::Not(oper))?)
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = self.resolve_operand_as(a, step.fnid_to_frameid, ValueType::Num)?;
                let b = self.resolve_operand_as(b, step.fnid_to_frameid, ValueType::Num)?;
                Some(self.emit(Instr::Arith { op: *op, a, b })?)
            }
            interpreter::Instr::Cmp { op, a, b } => {
                let a_rt_type = {
                    // TODO(opt) fix: This causes allocations with strings
                    let rt_value = (step.get_operand)(a);
                    let rt_ty = ValueType::of(&rt_value);
                    assert_ne!(rt_ty, ValueType::Boxed);
                    rt_ty
                };
                let b_rt_type = {
                    // TODO(opt) fix: This causes allocations with strings
                    let rt_value = (step.get_operand)(b);
                    let rt_ty = ValueType::of(&rt_value);
                    assert_ne!(rt_ty, ValueType::Boxed);
                    rt_ty
                };

                let a = self.resolve_interpreter_operand(a, step.fnid_to_frameid)?;
                let b = self.resolve_interpreter_operand(b, step.fnid_to_frameid)?;

                let instr = if let (Instr::Const(_), Instr::Const(_)) =
                    (self.get_instr(a), self.get_instr(b))
                {
                    let interpreter_result = &step.values_buf[step.iid.0 as usize];
                    Instr::Const(interpreter_result.clone())
                } else {
                    // We always unbox before Cmp.
                    // This is because  it's smarter to check the boxed
                    // type now (at trace building time), and have the trace
                    // rely on it (as an unbox assert).  This creates a more
                    // specialized trace that doesn't waste time and code
                    // checking a box type (which is likely to pass!)
                    let a = self.ensure_type(a, a_rt_type)?;
                    let b = self.ensure_type(b, b_rt_type)?;

                    let cmp = match (a_rt_type, b_rt_type) {
                        (ValueType::Num, ValueType::Num) => Cmp {
                            ty: ValueType::Num,
                            op: *op,
                            a,
                            b,
                        },
                        (ValueType::Str, ValueType::Str) => Cmp {
                            ty: ValueType::Str,
                            op: *op,
                            a,
                            b,
                        },
                        _ => {
                            let msg =
                                format!("unsupported operand type for comparison: {:?}", a_rt_type);
                            return Err(Error::Unsupported(msg.into()));
                        }
                    };

                    cmp.into()
                };

                Some(self.emit(instr)?)
            }
            interpreter::Instr::JmpIf { cond, .. } => {
                let cond = self.resolve_operand_as(cond, step.fnid_to_frameid, ValueType::Bool)?;

                match self.get_instr(cond) {
                    Instr::Const(_) => None,
                    Instr::Cmp(cmp) => {
                        let branch_taken = step.next_iid.0 != (step.iid.0 + 1);

                        print_indent(self.stack_depth());
                        eprintln!(
                            "[..tb jmpif: branch {}taken ({:?} -> {:?})]",
                            if branch_taken { "" } else { "not " },
                            step.iid,
                            step.next_iid,
                        );

                        let cmp = if branch_taken {
                            cmp.clone()
                        } else {
                            cmp.invert()
                        };

                        let pre_snap_update = self.take_exit_snapshot()?;
                        Some(self.emit(Instr::ExitUnless {
                            cond: cmp,
                            pre_snap_update,
                        })?)
                    }

                    _ => {
                        panic!("JIT bug: jmpif's cond operand should resolve to a Cmp instruction")
                    }
                }
            }
            interpreter::Instr::Jmp(_) => {
                // unconditional jump.  Nothing to do, let's just follow the interpreter to the next instruction
                None
            }
            interpreter::Instr::Set { var_id, value } => {
                let value = self.resolve_interpreter_operand(value, step.fnid_to_frameid)?;
                let frame_id = (step.fnid_to_frameid)(var_id.fnid);

                print_indent(self.stack_depth());
                eprintln!(
                    "[..tb map var: {:?}:{} = {:?}]",
                    frame_id, var_id.var_ndx.0, value
                );

                let is_outside_trace = !self.vars.was_frame_seen(frame_id);
                self.vars.set_var(frame_id, var_id.var_ndx, value);

                if is_outside_trace {
                    self.set_exit_snapshot_var(*var_id, value)?;
                }

                None
            }
            interpreter::Instr::PushSink(value) => {
                let value = self.resolve_interpreter_operand(value, step.fnid_to_frameid)?;
                self.emit(Instr::PushSink(value))?;
                None
            }
            interpreter::Instr::Return(value) => {
                self.exit_function_expected = true;

                let value = if self.stack_depth() == 1 {
                    self.resolve_operand_as(value, step.fnid_to_frameid, ValueType::Boxed)?
                } else {
                    self.resolve_interpreter_operand(value, step.fnid_to_frameid)?
                };

                self.return_value = Some(value);
                None
            }
            interpreter::Instr::GetArg(index) => {
                let value = if self.stack_depth() == 1 {
                    let post_snap_update = self.take_exit_snapshot()?;
                    self.emit(Instr::GetArg {
                        ndx: *index,
                        ty: ValueType::Boxed,
                        post_snap_update,
                    })?
                } else {
                    *self.vars.get_arg(*index)
                };
                Some(value)
            }
            interpreter::Instr::Call { .. } => {
                // We reach this point *after* the interpreter has completed the call, i.e.:
                //   - intepreter gets instruction Call
                //     - interpreter calls TraceBuilder::enter_function
                //     - interpreter runs callee
                //       - interpreter calls TraceBuilder::exit_function
                //     - interpreter collects return value
                //   - interpreter calls TraceBuilder::trace_step(call instruction)
                //
                // All that's left to do for us is to pick the return value
                // from the JIT (not the interpreter), so that we can continue
                // constant folding etc.

                let ret_val = self
                    .return_value
                    .take()
                    .unwrap_or(self.emit(Instr::Const(BoxedValue::Undefined))?);
                Some(ret_val)
            }

            interpreter::Instr::ObjNew => Some(self.emit(Instr::ObjNew)?),
            interpreter::Instr::ObjSet { obj, key, value } => {
                let obj = self.resolve_operand_as(obj, step.fnid_to_frameid, ValueType::Obj)?;
                let key = self.resolve_operand_as(key, step.fnid_to_frameid, ValueType::Str)?;
                let value =
                    self.resolve_operand_as(value, step.fnid_to_frameid, ValueType::Boxed)?;

                self.emit(Instr::ObjSet { obj, key, value })?;
                None
            }
            interpreter::Instr::ObjGet { obj, key } => {
                let obj = self.resolve_operand_as(obj, step.fnid_to_frameid, ValueType::Obj)?;
                let key = self.resolve_operand_as(key, step.fnid_to_frameid, ValueType::Str)?;

                Some(self.emit(Instr::ObjGet { obj, key })?)
            }
            interpreter::Instr::ArrayNew => {
                todo!("(big feat) array new")
            }
            interpreter::Instr::ArrayPush(_arr, _elem) => {
                todo!("(big feat) array push")
            }
            interpreter::Instr::TypeOf(arg) => {
                let arg = self.resolve_interpreter_operand(arg, step.fnid_to_frameid)?;
                Some(self.emit(Instr::TypeOf(arg))?)
            }

            interpreter::Instr::BoolOp { op, a, b } => {
                let a = self.resolve_operand_as(a, step.fnid_to_frameid, ValueType::Bool)?;
                let b = self.resolve_operand_as(b, step.fnid_to_frameid, ValueType::Bool)?;
                Some(self.emit(Instr::BoolOp { op: *op, a, b })?)
            }
            interpreter::Instr::ClosureNew { .. } => {
                // This instruction really does nothing.  Its real value is in
                // assigning the value ID to this "virtual closure" that is being
                // created. It's a "virtual closure" because it can't escape the
                // trace (it's UntraceableByDesign); it can only be inlined.
                Some(self.emit(Instr::ClosureNew)?)
            }

            interpreter::Instr::UnaryMinus(_) => {
                todo!("(small feat) jit::builder: UnaryMinus")
            }

            interpreter::Instr::StartTrace { .. } => {
                // Nothing to do.  The logic for starting and stopping the TraceBuilder is mostly in the Interpreter.
                // The TraceBuilder is created in the 'Tracing' state.
                None
            }
        };

        // Map IID to the result operand
        if let Some(result) = result {
            self.map_iid(step.iid, result);
        }

        Ok(())
    }

    pub(crate) fn enter_function(&mut self, fid: FrameId, n_vars: usize) {
        if self.state != TraceBuilderState::Tracing {
            return;
        }

        let args = self
            .args_buf
            .take()
            .expect("enter_function called without calling set_args first");
        self.vars.push_frame(fid, args, n_vars);

        print_indent(self.stack_depth());
        eprintln!("[..tb call (stack depth -> {})]", self.stack_depth());
    }

    /// Tell the JIT about the arguments passed to a function call that is
    /// about to happen.
    ///
    /// It is mandatory that this method is called exactly once before each
    /// call to `Self::enter_function`.
    ///
    /// If the TraceBuilder is not "active" (e.g. the trace is finished or
    /// failed, or it is still waiting for the right start instruction), then
    /// this function does nothing (if the arguments are needed later by the
    /// trace, they will be acquired  as trace parameters).
    pub(crate) fn set_args<F>(&mut self, args: &[interpreter::Operand], fnid_to_frameid: &F)
    where
        F: Fn(FnId) -> FrameId,
    {
        if self.state != TraceBuilderState::Tracing {
            return;
        }

        assert!(
            self.args_buf.is_none(),
            "set_args called twice before enter_function"
        );

        let args_values = args
            .iter()
            .flat_map(|arg| self.resolve_interpreter_operand(arg, fnid_to_frameid))
            .collect();
        self.args_buf = Some(args_values);
    }

    pub(crate) fn exit_function(&mut self) -> bool {
        if self.state == TraceBuilderState::Tracing {
            // self.exit_function_expected might be false, if the source bytecode
            // does not have an explicit Return instruction. In this case, we do
            // an implicit Return(undefined)

            print_indent(self.stack_depth());
            eprintln!("[..tb exit function]");

            self.vars.pop_frame();

            if self.vars.stack_depth() == 0 {
                print_indent(self.stack_depth());
                eprintln!("[..tb trace ended]");
                self.state = TraceBuilderState::Finished;
            }

            self.exit_function_expected = false;
        }

        self.state == TraceBuilderState::Finished
    }

    fn map_iid(&mut self, iid: IID, jit_vid: ValueId) {
        print_indent(self.stack_depth());
        eprintln!("[..tb map {:?} -> {:?}]", iid, jit_vid);
        self.vars.set_result(iid, jit_vid);
    }

    fn emit(&mut self, instr: Instr) -> Result<ValueId, Error> {
        match instr {
            Instr::Not(vid) => match self.get_instr(vid) {
                Instr::Const(BoxedValue::Bool(value)) => {
                    Ok(self.emit(Instr::Const(BoxedValue::Bool(!value)))?)
                }
                Instr::Cmp(cmp) => Ok(self.emit(Instr::Cmp(cmp.invert()))?),
                _ => self.write(instr),
            },

            Instr::Arith { op, ref a, ref b } => {
                let (a, b) = (self.get_instr(*a), self.get_instr(*b));

                if let (Instr::Const(BoxedValue::Number(a)), Instr::Const(BoxedValue::Number(b))) =
                    (a, b)
                {
                    let res = match op {
                        ArithOp::Add => a + b,
                        ArithOp::Sub => a - b,
                        ArithOp::Mul => a * b,
                        ArithOp::Div => a / b,
                    };
                    let res = self.emit(Instr::Const(BoxedValue::Number(res)))?;
                    Ok(res)
                } else {
                    self.write(instr)
                }
            }

            // TODO(small feat) Re-enable this feature
            // Instr::Arith {
            //     op: ArithOp::Add,
            //     a: Operand::Imm(BoxedValue::Number(a_const)),
            //     b,
            // } if a_const == 0.0 => Ok(b),
            // Instr::Arith {
            //     op: ArithOp::Add,
            //     a,
            //     b: Operand::Imm(BoxedValue::Number(b_const)),
            // } if b_const == 0.0 => Ok(a),
            // Instr::Arith {
            //     op: ArithOp::Sub,
            //     a: Operand::Imm(BoxedValue::Number(a_const)),
            //     b,
            // } if a_const == 0.0 => Ok(b),
            // Instr::Arith {
            //     op: ArithOp::Sub,
            //     a,
            //     b: Operand::Imm(BoxedValue::Number(b_const)),
            // } if b_const == 0.0 => Ok(a),

            // Instr::Arith {
            //     op: ArithOp::Mul,
            //     a: Operand::Imm(BoxedValue::Number(a_const)),
            //     b,
            // } if a_const == 1.0 => Ok(b),
            // Instr::Arith {
            //     op: ArithOp::Mul,
            //     a,
            //     b: Operand::Imm(BoxedValue::Number(b_const)),
            // } if b_const == 1.0 => Ok(a),

            // Instr::Arith {
            //     op: ArithOp::Div,
            //     a,
            //     b: Operand::Imm(BoxedValue::Number(b_const)),
            // } if b_const == 1.0 => Ok(a),

            // Instr::BoolOp {
            //     op: BoolOp::And,
            //     a: Operand::Imm(BoxedValue::Bool(false)),
            //     b: _,
            // } => Ok(Operand::Imm(BoxedValue::Bool(false))),
            // Instr::BoolOp {
            //     op: BoolOp::And,
            //     a: _,
            //     b: Operand::Imm(BoxedValue::Bool(false)),
            // } => Ok(Operand::Imm(BoxedValue::Bool(false))),
            // Instr::BoolOp {
            //     op: BoolOp::And,
            //     a: Operand::Imm(BoxedValue::Bool(true)),
            //     b: Operand::Imm(BoxedValue::Bool(true)),
            // } => Ok(Operand::Imm(BoxedValue::Bool(true))),

            // Instr::BoolOp {
            //     op: BoolOp::Or,
            //     a: Operand::Imm(BoxedValue::Bool(true)),
            //     b: _,
            // } => Ok(Operand::Imm(BoxedValue::Bool(true))),
            // Instr::BoolOp {
            //     op: BoolOp::Or,
            //     a: _,
            //     b: Operand::Imm(BoxedValue::Bool(true)),
            // } => Ok(Operand::Imm(BoxedValue::Bool(true))),
            // Instr::BoolOp {
            //     op: BoolOp::Or,
            //     a: Operand::Imm(BoxedValue::Bool(false)),
            //     b: Operand::Imm(BoxedValue::Bool(false)),
            // } => Ok(Operand::Imm(BoxedValue::Bool(false))),
            Instr::TypeOf(ref vid) => {
                let target_instr = self.instrs.get(vid.0 as usize).unwrap();
                let ty = target_instr.result_type().and_then(|ty| ty.js_typeof());

                let instr = match ty {
                    Some(ret) => Instr::Const(ret.into()),
                    None => instr,
                };

                self.write(instr)
            }

            _ => self.write(instr),
        }
    }

    fn write(&mut self, instr: Instr) -> Result<ValueId, Error> {
        let vid = ValueId(self.instrs.len() as u32);
        print_indent(self.stack_depth());
        eprintln!("[..tb emit: v{:<4} {:?}]", vid.0, instr);
        self.instrs.push(instr);
        Ok(vid)
    }

    fn take_exit_snapshot(&mut self) -> Result<Vec<Option<ValueId>>, Error> {
        let post_snap_update = std::mem::take(&mut self.exit_snapshot_changes);
        for vid in post_snap_update.iter().flatten() {
            let instr = self.get_instr(*vid);
            if let Instr::ClosureNew = instr {
                // At this point, we know there is at least one case where a
                // closure that was *created* within the trace is being yielded
                // back to the interpreter (for example for storage or later call)
                //
                // This compiler, at least now, won't support this case.  It
                // would be very complicated to make the closure valid in the
                // interpreter's context. For example, you would have to determine
                // which stack frames have to be created, and create them (so that
                // the interpreter will be able to access captured variables).
                // This is a job better left to the interpreter.
                return Err(Error::UntraceableByDesign);
            }
        }
        Ok(post_snap_update)
    }

    fn add_entry_snapshot_var(&mut self, operand: &interpreter::Operand) -> Result<ValueId, Error> {
        print_indent(self.stack_depth());
        eprintln!("[..tb add entry snap var {:?}]", operand);

        let ndx = self.snapshot_map.set_entry(operand);

        let post_snap_update = self.take_exit_snapshot()?;
        self.emit(Instr::GetSnapshotItem {
            ndx,
            ty: ValueType::Boxed,
            post_snap_update,
        })
    }

    fn set_exit_snapshot_var(
        &mut self,
        intrp_varid: interpreter::StaticVarId,
        value_id: ValueId,
    ) -> Result<(), Error> {
        print_indent(self.stack_depth());
        eprintln!("[..tb add exit snap var {value_id:?} -> {intrp_varid:?}]");

        let ndx = self
            .snapshot_map
            .set_exit(&interpreter::Operand::Var(intrp_varid)) as usize;

        if self.exit_snapshot_changes.len() <= ndx {
            self.exit_snapshot_changes.resize(ndx + 1, None);
        }
        self.exit_snapshot_changes[ndx] = Some(value_id);

        Ok(())
    }

    pub(crate) fn build(mut self) -> Option<Trace> {
        if let TraceBuilderState::Finished = self.state {
            let is_loop = self.loop_head.is_some();
            let snapshot_final_update = self.take_exit_snapshot().ok()?;
            let mut instrs = self.instrs;

            if is_loop {
                let phis = compute_phis(&self.vars);
                for (old, new) in phis.iter() {
                    instrs.push(Instr::Phi(*old, *new));
                }
            }

            let constraints = place_reg_constraints(&instrs);
            let reg_classes = get_reg_classes(&instrs);
            // TODO change Instr so that operands are explicitly stored in a separate Vec<OperandsSet> (so no need for extract_operands)
            let operands = extract_operands(&instrs);
            assert_eq!(instrs.len(), reg_classes.len());
            assert_eq!(instrs.len(), operands.len());
            let hreg_alloc: regalloc::Allocation =
                regalloc::allocate_registers(&reg_classes, &operands, &constraints, 6, 8);

            let is_enabled = {
                let mut enb: Vec<_> = instrs
                    .iter()
                    .map(|instr| instr.has_side_effects())
                    .collect();
                for asmt in &hreg_alloc.asmts {
                    enb[asmt.vid.0 as usize] = true;
                }
                enb
            };

            Some(Trace {
                instrs,
                is_loop,
                is_enabled,
                hreg_alloc,
                snapshot_map: self.snapshot_map,
                snapshot_final_update,
            })
        } else {
            None
        }
    }

    fn get_instr(&self, vid: ValueId) -> &Instr {
        let ndx = vid.0 as usize;
        self.instrs.get(ndx).unwrap()
    }
}

fn place_reg_constraints(instrs: &[Instr]) -> HashMap<regalloc::ConstraintPt, regalloc::HardReg> {
    // TODO(arch) Ideally, this would be independent from the target
    // architecture. Better to move this to jit::codegen?

    use super::codegen;
    use super::regalloc::{ConstraintPt, HardReg, RegClass};
    use dynasmrt::x64::{Rq, Rx};

    let mut constraints = HashMap::new();
    for (ndx, instr) in instrs.iter().enumerate() {
        let pos = ValueId(ndx as _);
        if let Instr::PushSink(arg) = instr {
            let src_instr = &instrs[arg.0 as usize];
            let reg_class = src_instr
                .result_type()
                .map(reg_class_of_type)
                .expect("bug: malformed code: PushSink arg's source instruction has no result?!");

            let hreg = match reg_class {
                RegClass::General => HardReg {
                    class: RegClass::General,
                    index: index_of_reg(&codegen::GP_REGS, &Rq::RDI),
                },
                RegClass::Numeric => HardReg {
                    class: RegClass::Numeric,
                    index: index_of_reg(&codegen::NUM_REGS, &Rx::XMM0),
                },
            };

            constraints.insert(ConstraintPt { pos, vid: *arg }, hreg);
        }
    }
    constraints
}

fn extract_operands(instrs: &[Instr]) -> Vec<OperandsSet> {
    instrs.iter().map(|i| i.operands()).collect()
}

fn get_reg_classes(instrs: &[Instr]) -> Vec<Option<regalloc::RegClass>> {
    instrs
        .iter()
        .map(|i| i.result_type().map(reg_class_of_type))
        .collect()
}

// TODO(cleanup) move elsewhere?
pub(super) fn reg_class_of_type(ty: ValueType) -> regalloc::RegClass {
    match ty {
        ValueType::Num => regalloc::RegClass::Numeric,
        _ => regalloc::RegClass::General,
    }
}

fn compute_phis(vars_state: &tracking::VarsState) -> HashMap<ValueId, ValueId> {
    let mut phis = HashMap::new();
    let rbw = vars_state.get_reads_before_overwritten();
    for &(frame_id, var_ndx) in rbw.iter() {
        let first_vid = vars_state
            .get_first_write(frame_id, var_ndx)
            .expect("bug in VarsState: returned variable without a first write");
        let last_vid = vars_state
            .get_var(frame_id, var_ndx)
            .expect("bug in VarsState: returned variable without a current write");
        assert_ne!(first_vid, last_vid);

        let prev = phis.insert(first_vid, last_vid);
        assert!(
            prev.is_none(),
            "JIT bug: tried to place multiple phis for the same variable"
        );
    }
    phis
}

fn print_indent(stack_depth: usize) {
    for _ in 0..stack_depth {
        eprint!(" · ");
    }
}

pub struct InterpreterStep<'a> {
    pub(crate) values_buf: &'a Vec<interpreter::Value>,
    pub(crate) fnid: interpreter::FnId,
    pub(crate) func: &'a interpreter::Function,
    pub(crate) iid: interpreter::IID,
    pub(crate) next_iid: interpreter::IID,
    pub(crate) fnid_to_frameid: &'a dyn Fn(FnId) -> FrameId,
    pub(crate) get_operand: &'a dyn Fn(&interpreter::Operand) -> BoxedValue,
}

impl<'a> InterpreterStep<'a> {
    fn cur_instr(&self) -> &interpreter::Instr {
        &self.func.instrs()[self.iid.0 as usize]
    }

    fn global_iid(&self) -> interpreter::GlobalIID {
        interpreter::GlobalIID {
            fnid: self.fnid,
            iid: self.iid,
        }
    }
}

// TODO(cleanup) Move to jit/mod.rs
#[derive(PartialEq, Debug)]
pub(super) enum Instr {
    Const(BoxedValue),
    Not(ValueId),
    Arith {
        op: ArithOp,
        a: ValueId,
        b: ValueId,
    },
    Cmp(Cmp),
    BoolOp {
        op: BoolOp,
        a: ValueId,
        b: ValueId,
    },
    Choose {
        ty: ValueType,
        cond: ValueId,
        if_true: ValueId,
        if_false: ValueId,
    },

    // ASSERTS: on failure, they exit the trace with a "failure", i.e.
    // a snapshot rollback (i.e. the interpreter restores an earlier
    // snapshot, the last committed one)
    GetSnapshotItem {
        ndx: u16,
        ty: ValueType,
        // TODO(performance) replace this with something that doesn't require a dedicated allocation?
        post_snap_update: SnapshotUpdate,
    },
    GetArg {
        ndx: usize, // TODO(small feat) Change to u16
        ty: ValueType,
        post_snap_update: SnapshotUpdate,
    },

    // EXITS: on failure, they exit the trace with a "success", i.e.
    // every effect on local variables that happened before this
    // instruction becomes visible by the interpreter.
    ExitUnless {
        cond: Cmp,
        pre_snap_update: SnapshotUpdate,
    },

    ObjNew,
    ObjSet {
        obj: ValueId,
        key: ValueId,
        value: ValueId,
    },
    ObjGet {
        obj: ValueId,
        key: ValueId,
    },

    TypeOf(ValueId),
    Num2Str(ValueId),
    Box(ValueId),

    ClosureNew,

    PushSink(ValueId),
    Return(ValueId),
    Phi(ValueId, ValueId),
}

pub(super) type SnapshotUpdate = Vec<Option<ValueId>>;

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;
type BoolOp = interpreter::BoolOp;

impl From<Cmp> for Instr {
    fn from(cmp: Cmp) -> Self {
        Instr::Cmp(cmp)
    }
}

const MAX_OPERANDS_PER_INSTR: usize = 4;
pub(super) struct OperandsSet([Option<ValueId>; MAX_OPERANDS_PER_INSTR]);

impl OperandsSet {
    pub(super) fn none() -> Self {
        Default::default()
    }
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = &Option<ValueId>> {
        self.0.iter()
    }
}

impl Default for OperandsSet {
    fn default() -> Self {
        OperandsSet([None; MAX_OPERANDS_PER_INSTR])
    }
}
impl From<&[ValueId]> for OperandsSet {
    fn from(vids: &[ValueId]) -> Self {
        assert!(vids.len() <= MAX_OPERANDS_PER_INSTR);
        let mut arr = [None; MAX_OPERANDS_PER_INSTR];
        for i in 0..vids.len() {
            arr[i] = Some(vids[i]);
        }
        OperandsSet(arr)
    }
}

impl Instr {
    pub(super) fn result_type(&self) -> Option<ValueType> {
        match self {
            Instr::Not(_) => Some(ValueType::Bool),
            Instr::Arith { .. } => Some(ValueType::Num),
            Instr::Cmp { .. } => Some(ValueType::Bool),
            Instr::BoolOp { .. } => Some(ValueType::Bool),
            Instr::ExitUnless { .. } => None,
            Instr::PushSink(_) => None,
            Instr::Return(_) => None,
            Instr::GetArg { ty, .. } => Some(*ty),
            Instr::Choose { ty, .. } => Some(*ty),
            Instr::Num2Str(_) => Some(ValueType::Str),
            Instr::ObjNew => Some(ValueType::Obj),
            Instr::ObjSet { .. } => None,
            Instr::ObjGet { .. } => Some(ValueType::Boxed),
            Instr::TypeOf(_) => Some(ValueType::Str),
            Instr::ClosureNew => Some(ValueType::Function),
            Instr::Const(val) => Some(ValueType::of(val)),
            Instr::Phi(_, _) => None,
            Instr::GetSnapshotItem { ty, .. } => Some(*ty),
            Instr::Box(_) => Some(ValueType::Boxed),
        }
    }

    pub(super) fn operands(&self) -> OperandsSet {
        use std::iter::once;
        match self {
            Instr::Not(arg) => [*arg].as_slice().into(),
            Instr::Arith { a, b, .. } => [*a, *b].as_slice().into(),
            Instr::Cmp(Cmp { a, b, .. }) => [*a, *b].as_slice().into(),
            Instr::BoolOp { a, b, .. } => [*a, *b].as_slice().into(),
            Instr::Choose {
                cond,
                if_true,
                if_false,
                ..
            } => [*cond, *if_true, *if_false].as_slice().into(),
            Instr::ExitUnless {
                cond: Cmp { a, b, .. },
                ..
            } => [*a, *b].as_slice().into(),
            Instr::Num2Str(arg) => [*arg].as_slice().into(),
            Instr::PushSink(arg) => [*arg].as_slice().into(),
            Instr::Return(arg) => [*arg].as_slice().into(),

            Instr::GetArg { .. } => [].as_slice().into(),

            Instr::ObjNew => [].as_slice().into(),
            Instr::ObjSet { obj, key, value } => [*obj, *key, *value].as_slice().into(),
            Instr::ObjGet { obj, key } => [*obj, *key].as_slice().into(),
            Instr::TypeOf(arg) => [*arg].as_slice().into(),
            Instr::ClosureNew => [].as_slice().into(),
            Instr::Const(_) => [].as_slice().into(),
            Instr::Phi(_tgt, new_value) => [*new_value].as_slice().into(),
            Instr::GetSnapshotItem { .. } => [].as_slice().into(),
            Instr::Box(arg) => [*arg].as_slice().into(),
        }
    }

    pub(crate) fn has_side_effects(&self) -> bool {
        match self {
            Instr::GetArg { .. } => false,
            Instr::Const(_) => false,
            Instr::Not(_) => false,
            Instr::Arith { .. } => false,
            Instr::Cmp(_) => false,
            Instr::BoolOp { .. } => false,
            Instr::Choose { .. } => false,
            Instr::ExitUnless { .. } => true,
            Instr::Num2Str(_) => false,
            Instr::ObjNew => false,
            Instr::ObjSet { .. } => false,
            Instr::ObjGet { .. } => false,
            Instr::TypeOf(_) => false,
            Instr::ClosureNew => false,
            Instr::PushSink(_) => true,
            Instr::Return(_) => false,
            Instr::Phi(_, _) => true,
            Instr::GetSnapshotItem { .. } => false,
            Instr::Box(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{interpreter::StaticVarId, jit};

    #[test]
    fn test_exit_unless_operands() {
        // Instr::ExitUnless {
        // cond: Cmp { a, b, .. },
        // ..
        // } => [*a, *b].as_slice().into(),
        let instr = Instr::ExitUnless {
            cond: Cmp {
                ty: ValueType::Num,
                op: CmpOp::LE,
                a: ValueId(1),
                b: ValueId(3),
            },
            pre_snap_update: Vec::new(),
        };

        let operands: Vec<_> = instr.operands().iter().flatten().copied().collect();
        assert_eq!(&operands, &[ValueId(1), ValueId(3)]);
    }

    struct Output {
        sink: Vec<interpreter::Value>,
        vm: interpreter::VM,
    }
    impl Output {
        fn get_trace(&self, trace_id: &str) -> Option<&jit::Trace> {
            let (trace, _) = self.vm.get_trace(trace_id)?;
            Some(trace)
        }
    }

    fn run_and_compile(vm: &mut crate::VM, code: &str) {
        let base_flags = Default::default();

        {
            let flags = interpreter::InterpreterFlags {
                jit_mode: interpreter::JitMode::Compile,
                ..base_flags
            };
            vm.run_script(code.to_string(), flags)
                .expect("first run (jit compilation) failed");
        }

        assert!(
            vm.trace_ids().len() > 0,
            "at least 1 trace was expected to be built"
        );
    }

    fn quick_jit(code: &str) -> Output {
        let mut vm = interpreter::VM::new();

        run_and_compile(&mut vm, code);
        vm.take_sink(); // ... and discard it

        {
            let flags = interpreter::InterpreterFlags {
                jit_mode: interpreter::JitMode::UseTraces,
                ..Default::default()
            };
            vm.run_script(code.to_string(), flags)
                .expect("second run (trace run) failed");
        }

        Output {
            sink: vm.take_sink(),
            vm,
        }
    }

    fn get_pushsink_consts(trace: &Trace) -> Vec<&BoxedValue> {
        let pushsink_operands: Vec<_> = trace
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::PushSink(operand) => Some(operand),
                _ => None,
            })
            .filter_map(|vid| match trace.get_instr(*vid) {
                Some(Instr::Const(value)) => Some(value),
                _ => None,
            })
            .collect();
        pushsink_operands
    }

    #[ignore]
    #[test]
    fn test_tracing_simple_constant_folding() {
        let output = quick_jit(
            "
            __start_trace('the-trace');
            const x = 123;
            let y = 'a';
            if (x < 256) {
                y = 'b';
            }
            sink(y);
            ",
        );

        eprint!("trace = ");
        let trace = output.get_trace("the-trace").unwrap();
        trace.dump();

        let pushsink_operands = get_pushsink_consts(&trace);
        assert_eq!(pushsink_operands.len(), 1);
        assert!(PartialEq::eq(pushsink_operands[0], &"b".into()));
    }

    #[test]
    fn test_tracing_one_func() {
        let output = quick_jit(
            "
            function foo(mode, a, b) { 
                __start_trace('the-trace');
                if (mode === 'sum')
                    return a + b;
                else if (mode === 'product')
                    return a * b;
                else
                    return 'something else';
            }

            sink(foo('product', 9, 8));
            ",
        );

        eprint!("trace = ");
        let (trace, thunk) = output.vm.get_trace("the-trace").unwrap();
        trace.dump();
        thunk.dump();

        // TODO: Run the trace (continue writing this test when traces can be run)
        // TODO(cleanup) Run the trace (continue writing this test when traces can be run)
    }

    #[ignore]
    #[test]
    fn test_while_loop() {
        // TODO(bug) loops are broken without loop invariant code hoisting.
        //     The typical trace has a bunch of GetSnapshotItem instructions
        //     at the beginning, and a few phis that overwrite one ore more of
        //     those variables for the next iteration of the loop.  But until
        //     loop invariant code hoisting is implemented, those GetSnapshotItem
        //     instructions are repeated at every iteration of the loop, undoing
        //     the effect of the PHIs, making every loop infinite.
        let code = "
            function sum_range(n) {
                let i = 0;
                let ret = 0;

                __start_trace_loop('the-trace');
                while (i <= n) {
                    ret += i;
                    sink(ret);
                    i++;
                }
                return ret;
            }

            sink(sum_range(5));
        ";
        let output = quick_jit(code);

        eprint!("trace = ");
        let trace = output.get_trace("the-trace").unwrap();
        trace.dump();

        let native_thunk = jit::codegen::to_native(&trace);
        native_thunk.dump();

        assert_eq!(
            &output.sink,
            &[
                BoxedValue::Number(0.0),
                BoxedValue::Number(1.0),
                BoxedValue::Number(3.0),
                BoxedValue::Number(6.0),
                BoxedValue::Number(10.0),
                BoxedValue::Number(15.0),
                // sum_range's return value
                BoxedValue::Number(15.0),
            ]
        );
    }

    #[ignore]
    #[test]
    fn test_varid_resolution_with_inlining() {
        let output = quick_jit(
            "
            const far_away = 'asd';
            function f() {
                __start_trace('the-trace');

                const a_little_closer = 'lol';
                function g() {
                    function h() {
                        sink(far_away);
                        sink(a_little_closer);
                    }
                    h();
                }
                g();
            }

            f();
            ",
        );

        let trace = output.get_trace("the-trace").unwrap();
        trace.dump();

        let sink_operands = get_pushsink_consts(&trace);
        assert_eq!(sink_operands, [&"asd".into(), &"lol".into(),]);
    }

    #[ignore]
    #[test]
    fn test_closures() {
        let output = quick_jit(
            "
            function iota(factor) {
                let i = 0;
                return function() {
                    i += 1;
                    return i * factor;
                };
            }

            __start_trace('t');
            const next1 = iota(11);
            const next2 = iota(7);

            sink(next1()); // 11

            sink(next2()); // 7

            sink(next1()); // 22
            sink(next1()); // 33

            sink(next2()); // 14
            sink(next2()); // 21
            sink(next2()); // 28
            ",
        );

        let trace = output.get_trace("t").unwrap();
        trace.dump();

        let native_thunk = jit::codegen::to_native(&trace);
        native_thunk.dump();

        let sink_operands = get_pushsink_consts(&trace);
        assert_eq!(
            sink_operands,
            [
                &BoxedValue::Number(11.0),
                &BoxedValue::Number(7.0),
                &BoxedValue::Number(22.0),
                &BoxedValue::Number(33.0),
                &BoxedValue::Number(14.0),
                &BoxedValue::Number(21.0),
                &BoxedValue::Number(28.0),
            ]
        );
    }

    #[test]
    fn test_loop_unrolling() {
        let output = quick_jit(
            "
            function sum_range(n) {
                let i = 0;
                let ret = 0;
                while (i <= n) {
                    ret += i;
                    sink(ret);
                    i++;
                }
                return ret;
            }

            __start_trace('t');
            sink(sum_range(5));
            ",
        );

        let (trace, thunk) = output.vm.get_trace("t").unwrap();
        eprint!("trace = ");
        trace.dump();

        assert_eq!(thunk.n_runs(), 1);

        assert_eq!(
            &output.sink,
            &[
                BoxedValue::Number(0.0),
                BoxedValue::Number(1.0),
                BoxedValue::Number(3.0),
                BoxedValue::Number(6.0),
                BoxedValue::Number(10.0),
                BoxedValue::Number(15.0),
                // last sink item: sum_range's return value
                BoxedValue::Number(15.0),
            ]
        );
    }

    #[ignore]
    #[test]
    fn test_loop_unrolling_jit() {
        let mut vm = interpreter::VM::new();

        let code = &"
            function sum_range(n) {
                __start_trace('t');
                let i = 0;
                let ret = 0;
                while (i <= n) {
                    ret += i;
                    sink(ret);
                    i++;
                }
                return ret;
            }

            sink(sum_range(5));
            ";
        run_and_compile(&mut vm, code);

        eprint!("trace = ");
        let (trace, thunk) = vm.get_trace("t").unwrap();
        trace.dump();

        let mut snapshot = vec![BoxedValue::Number(5.0)];
        assert_eq!(trace.snapshot_map.len() as usize, snapshot.len());

        thunk.dump();
        eprintln!("=== run");
        thunk.run(&mut snapshot);

        let sink = jit::codegen::take_test_sink();
        assert_eq!(
            &sink,
            &[
                BoxedValue::Number(0.0),
                BoxedValue::Number(1.0),
                BoxedValue::Number(3.0),
                BoxedValue::Number(6.0),
                BoxedValue::Number(10.0),
                BoxedValue::Number(15.0),
                // last sink item: sum_range's return value
                BoxedValue::Number(15.0),
            ]
        );
    }
}
