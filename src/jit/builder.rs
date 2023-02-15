use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::{
    interpreter::{self, FnId, VarIndex, IID},
    regalloc,
    stack::{self, FrameId},
};

use dynasm::dynasm;
use strum_macros::EnumIter;

use super::{tracking, Trace};

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
pub enum Error {
    Type(TypeError),
    Unsupported(String),
    InconsistentUnbox(InconsistentUnbox),
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

// TODO Move this to the super module
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub(super) u32);

impl std::fmt::Debug for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
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
    fn of(value: &BoxedValue) -> Self {
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
            // TODO This is actually an error in our type system.  null is really a value of the 'object' type
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

    instrs: Vec<Instr>,
    parameters: Vec<TraceParam>,

    state: TraceBuilderState,
    start_mode: TraceStart,
    exit_function_expected: bool,
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
    WaitingStart,
    Tracing,
    Failed,
    Finished,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TraceStart {
    Function,
    FirstLoop,
}

impl TraceBuilder {
    pub fn new(start_mode: TraceStart) -> Self {
        TraceBuilder {
            return_value: None,
            args_buf: None,
            vars: tracking::VarsState::new(),
            instrs: Vec::new(),
            parameters: Vec::new(),
            start_mode,
            loop_head: None,
            state: TraceBuilderState::WaitingStart,
            exit_function_expected: false,
        }
    }

    // TODO Inline into all callers
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
                match self.vars.cur_frame_mut().get_result(*iid).cloned() {
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
                        // TODO Test that this is really necessary. If so, describe it here
                        assert_eq!(1, stack_depth);

                        let param = self.add_parameter(intrp_operand.clone())?;
                        self.vars.cur_frame_mut().set_result(*iid, param.clone());
                        Ok(param)
                    }
                }
            }

            interpreter::Operand::Var(intrp_varid) => {
                let frame_id = fnid_to_frameid(intrp_varid.fnid);

                let slot = self.vars.get(frame_id).get_var(intrp_varid.var_ndx);
                if let Some(value) = slot {
                    return Ok(value.clone());
                }

                let param = self.add_parameter(intrp_operand.clone())?;
                self.vars
                    .get_mut(frame_id)
                    .set_var(intrp_varid.var_ndx, param.clone());
                Ok(param)
            }
        }
    }

    fn value_type(&mut self, vid: &ValueId) -> Option<ValueType> {
        self.instrs.get(vid.0 as usize).unwrap().result_type()
    }

    fn ensure_type(&mut self, vid: ValueId, desired_type: ValueType) -> Result<ValueId, Error> {
        let type_error = Error::Type(TypeError { desired_type });
        let input_type = self.value_type(&vid).ok_or(type_error.clone())?;

        let converted_operand = match (input_type, desired_type) {
            (a, b) if a == b => Some(vid),

            (_, ValueType::Null) => Some(self.emit(Instr::Const(BoxedValue::Null))?),
            (_, ValueType::Undefined) => Some(self.emit(Instr::Const(BoxedValue::Undefined))?),
            (_, ValueType::Function) => None,

            (ValueType::Boxed, desired_type) => {
                assert_ne!(ValueType::Boxed, desired_type);
                Some(self.emit(Instr::Unbox(desired_type, vid))?)
            }

            (input_type, ValueType::Boxed) => {
                assert_ne!(ValueType::Boxed, input_type);
                Some(self.emit(Instr::Box(vid))?)
            }

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
                    // TODO this string allocation could be avoided
                    b: self.emit(Instr::Const("".into()))?,
                };
                Some(self.emit(cmp.into())?)
            }

            // TODO Convert string to number
            (ValueType::Str, ValueType::Num) => None,

            (ValueType::Null, ValueType::Bool) => {
                Some(self.emit(Instr::Const(BoxedValue::Bool(false)))?)
            }
            (ValueType::Null, ValueType::Num) => {
                Some(self.emit(Instr::Const(BoxedValue::Number(0.0)))?)
            }

            // TODO this string allocation could be avoided
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

            _ => unreachable!(),
        };

        if let Some(converted_operand) = &converted_operand {
            assert_eq!(Some(desired_type), self.value_type(converted_operand));
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
            TraceBuilderState::WaitingStart => {
                if self.check_start(step) {
                    self.state = TraceBuilderState::Tracing;

                    let fid = (step.fnid_to_frameid)(step.fnid);
                    let n_vars = step.func.n_slots().into();
                    let args = Vec::new();
                    self.vars.push_frame(fid, args, n_vars);

                    // For the first instruction, self.loop_head stays None.
                    // Then (if the trace starts at a loop), we set
                    // self.loop_head, so that we can detect that we've closed
                    // the loop and can finish the trace.
                    self.interpreter_step(step);
                    if self.start_mode == TraceStart::FirstLoop {
                        self.loop_head = Some(step.global_iid());
                    } else {
                        assert!(self.loop_head.is_none());
                    }
                }
            }
            TraceBuilderState::Tracing => {
                if Some(step.global_iid()) == self.loop_head {
                    self.state = TraceBuilderState::Finished;
                } else if let Err(error) = self.trace_step(step) {
                    print_indent(self.stack_depth());
                    eprintln!("[..tb trace failed: {:?}]", error);
                    self.state = TraceBuilderState::Failed;
                }
            }
            TraceBuilderState::Failed => {}
            TraceBuilderState::Finished => {}
        }
    }

    fn check_start(&self, step: &InterpreterStep) -> bool {
        print_indent(self.stack_depth());
        match self.start_mode {
            // Trace starts immediately
            TraceStart::Function => {
                eprintln!("[..tb trace starts immediately]");
                true
            }
            TraceStart::FirstLoop => {
                if step.func.is_loop_head(step.iid) {
                    eprintln!("[..tb trace starts now, on a loop head]");
                    true
                } else {
                    eprintln!("[..tb waiting on a loop head...]");
                    false
                }
            }
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
                let oper =
                    self.resolve_operand_as(&oper, &step.fnid_to_frameid, ValueType::Bool)?;
                Some(self.emit(Instr::Not(oper))?)
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = self.resolve_operand_as(&a, &step.fnid_to_frameid, ValueType::Num)?;
                let b = self.resolve_operand_as(&b, &step.fnid_to_frameid, ValueType::Num)?;
                Some(self.emit(Instr::Arith { op: *op, a, b })?)
            }
            interpreter::Instr::Cmp { op, a, b } => {
                let a = self.resolve_interpreter_operand(&a, &step.fnid_to_frameid)?;
                let b = self.resolve_interpreter_operand(&b, &step.fnid_to_frameid)?;

                let instr = if let (Instr::Const(_), Instr::Const(_)) =
                    (self.get_instr(a), self.get_instr(b))
                {
                    let interpreter_result = &step.values_buf[step.iid.0 as usize];
                    Instr::Const(interpreter_result.clone())
                } else {
                    let a_type = self.get_instr(a).result_type().unwrap();
                    let b_type = self.get_instr(b).result_type().unwrap();

                    let cmp = match (a_type, b_type) {
                        (ValueType::Boxed, unboxed_type) if unboxed_type != ValueType::Boxed => {
                            Cmp {
                                ty: unboxed_type,
                                op: *op,
                                a: self.emit(Instr::Unbox(unboxed_type, a))?,
                                b,
                            }
                        }
                        (unboxed_type, ValueType::Boxed) if unboxed_type != ValueType::Boxed => {
                            Cmp {
                                ty: unboxed_type,
                                op: *op,
                                a,
                                b: self.emit(Instr::Unbox(unboxed_type, b))?,
                            }
                        }
                        _ => {
                            let b = self.ensure_type(b, a_type)?;
                            Cmp {
                                ty: a_type,
                                op: *op,
                                a,
                                b,
                            }
                        }
                    };

                    cmp.into()
                };

                Some(self.emit(instr)?)
            }
            interpreter::Instr::JmpIf { cond, .. } => {
                let cond =
                    self.resolve_operand_as(&cond, &step.fnid_to_frameid, ValueType::Bool)?;

                match self.get_instr(cond) {
                    Instr::Const(_) => None,
                    _ => {
                        let branch_taken = step.next_iid.0 != (step.iid.0 + 1);

                        print_indent(self.stack_depth());
                        eprintln!(
                            "[..tb jmpif: branch {}taken ({:?} -> {:?})]",
                            if branch_taken { "" } else { "not " },
                            step.iid,
                            step.next_iid,
                        );

                        let cond = if branch_taken {
                            cond
                        } else {
                            self.emit(Instr::Not(cond))?
                        };
                        Some(self.emit(Instr::AssertTrue { cond })?)
                    }
                }
            }
            interpreter::Instr::Jmp(_) => {
                // unconditional jump.  Nothing to do, let's just follow the interpreter to the next instruction
                None
            }
            interpreter::Instr::Set { var_id, value } => {
                let value = self.resolve_interpreter_operand(value, &step.fnid_to_frameid)?;
                let frame_id = (step.fnid_to_frameid)(var_id.fnid);

                print_indent(self.stack_depth());
                eprintln!(
                    "[..tb map var: {:?}:{} = {:?}]",
                    frame_id, var_id.var_ndx.0, value
                );

                self.vars.get_mut(frame_id).set_var(var_id.var_ndx, value);

                None
            }
            interpreter::Instr::PushSink(value) => {
                let value = self.resolve_interpreter_operand(&value, &step.fnid_to_frameid)?;
                self.emit(Instr::PushSink(value))?;
                None
            }
            interpreter::Instr::Return(value) => {
                self.exit_function_expected = true;

                let value = if self.stack_depth() == 1 {
                    self.resolve_operand_as(&value, &step.fnid_to_frameid, ValueType::Boxed)?
                } else {
                    self.resolve_interpreter_operand(&value, &step.fnid_to_frameid)?
                };

                self.return_value = Some(value);
                None
            }
            interpreter::Instr::GetArg(index) => {
                let value = if self.stack_depth() == 1 {
                    self.emit(Instr::GetArg(*index))?
                } else {
                    self.vars.cur_frame_mut().get_arg(*index).clone()
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
                let obj = self.resolve_operand_as(obj, &step.fnid_to_frameid, ValueType::Obj)?;
                let key = self.resolve_operand_as(key, &step.fnid_to_frameid, ValueType::Str)?;
                let value =
                    self.resolve_operand_as(value, &step.fnid_to_frameid, ValueType::Boxed)?;

                self.emit(Instr::ObjSet { obj, key, value })?;
                None
            }
            interpreter::Instr::ObjGet { obj, key } => {
                let obj = self.resolve_operand_as(obj, &step.fnid_to_frameid, ValueType::Obj)?;
                let key = self.resolve_operand_as(key, &step.fnid_to_frameid, ValueType::Str)?;

                Some(self.emit(Instr::ObjGet { obj, key })?)
            }
            interpreter::Instr::ArrayNew => {
                todo!("array new")
            }
            interpreter::Instr::ArrayPush(_arr, _elem) => {
                todo!("array push")
            }
            interpreter::Instr::TypeOf(arg) => {
                let arg = self.resolve_interpreter_operand(arg, &step.fnid_to_frameid)?;
                Some(self.emit(Instr::TypeOf(arg))?)
            }

            interpreter::Instr::BoolOp { op, a, b } => {
                let a = self.resolve_operand_as(a, &step.fnid_to_frameid, ValueType::Bool)?;
                let b = self.resolve_operand_as(b, &step.fnid_to_frameid, ValueType::Bool)?;
                Some(self.emit(Instr::BoolOp { op: *op, a, b })?)
            }
            interpreter::Instr::ClosureNew { .. } => Some(self.emit(Instr::ClosureNew)?),
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
            .map(|arg| self.resolve_interpreter_operand(arg, fnid_to_frameid))
            .flatten()
            .collect();
        self.args_buf = Some(args_values);
    }

    pub(crate) fn exit_function(&mut self) {
        if self.state != TraceBuilderState::Tracing {
            return;
        }
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

    fn map_iid(&mut self, iid: IID, jit_vid: ValueId) {
        print_indent(self.stack_depth());
        eprintln!("[..tb map {:?} -> {:?}]", iid, jit_vid);
        self.vars.cur_frame_mut().set_result(iid, jit_vid);
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

            // TODO TODO Re-enable this feature
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
            Instr::Unbox(value_type, ref arg_vid) => {
                // TODO move some of this stuff in FrameTracker
                if let Some((prev_unbox_type, unboxed)) = self.vars.cur_frame().get_unbox(*arg_vid)
                {
                    if prev_unbox_type == value_type {
                        print_indent(self.stack_depth());
                        eprintln!(
                            "[..tb reuse unbox v{:<4} as {:?} is at {:?}]",
                            arg_vid.0, value_type, unboxed
                        );
                        Ok(unboxed.clone())
                    } else {
                        Err(Error::InconsistentUnbox(InconsistentUnbox {
                            desired_type: value_type.clone(),
                        }))
                    }
                } else {
                    print_indent(self.stack_depth());
                    eprintln!("[..tb emit unbox: v{:<4} as {:?}]", arg_vid.0, value_type);
                    let new_vid = ValueId(self.instrs.len() as u32);

                    self.vars
                        .cur_frame_mut()
                        .set_unbox(*arg_vid, value_type, new_vid);
                    self.write(instr)?;
                    Ok(new_vid.into())
                }
            }

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
        Ok(vid.into())
    }

    fn add_parameter(&mut self, operand: interpreter::Operand) -> Result<ValueId, Error> {
        let ndx = self.parameters.len() as u16;
        self.parameters.push(TraceParam { operand });
        self.emit(Instr::TraceParam(ndx))
    }

    pub(crate) fn build(mut self) -> Option<Trace> {
        if let TraceBuilderState::Finished = self.state {
            let mut phis = HashMap::new();
            let is_loop = self.loop_head.is_some();

            if is_loop {
                let rbw = self.vars.get_reads_before_writes();
                for (frame_id, var_ndx) in rbw.iter() {
                    let frame = &self.vars.get(*frame_id);
                    let first_vid = frame
                        .get_first_write(*var_ndx)
                        .expect("bug in VarsState: returned variable without a first write");
                    let last_vid = frame
                        .get_var(*var_ndx)
                        .expect("bug in VarsState: returned variable without a current write");
                    assert_ne!(first_vid, *last_vid);

                    let phi_instr = self.write(Instr::Phi(first_vid, *last_vid)).unwrap();
                    let prev = phis.insert(first_vid, phi_instr);
                    assert!(
                        prev.is_none(),
                        "JIT bug: tried to place multiple phis for the same variable"
                    );
                }
            }

            let hregs = regalloc::allocate_registers(self.instrs.as_slice(), 6);

            Some(Trace {
                instrs: self.instrs,
                phis,
                is_loop,
                hreg_alloc: hregs,
                parameters: self.parameters,
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

fn print_indent(stack_depth: usize) {
    for _ in 0..stack_depth {
        eprint!("      | ");
    }
}

pub struct InterpreterStep<'a> {
    pub(crate) values_buf: &'a Vec<interpreter::Value>,
    pub(crate) fnid: interpreter::FnId,
    pub(crate) func: &'a interpreter::Function,
    pub(crate) iid: interpreter::IID,
    pub(crate) next_iid: interpreter::IID,
    pub(crate) fnid_to_frameid: &'a dyn Fn(FnId) -> FrameId,
}

impl<'a> InterpreterStep<'a> {
    fn cur_instr(&self) -> &interpreter::Instr {
        &self.func.instrs()[self.iid.0 as usize]
    }

    fn global_iid(&self) -> interpreter::GlobalIID {
        interpreter::GlobalIID {
            fn_id: self.fnid,
            iid: self.iid,
        }
    }
}

// TODO Move to jit/mod.rs
#[derive(PartialEq, Debug)]
pub(super) enum Instr {
    TraceParam(u16),
    GetArg(usize),
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
    AssertTrue {
        cond: ValueId,
    },
    AssertEqConst {
        x: ValueId,
        expected: BoxedValue,
    },

    Unbox(ValueType, ValueId),
    Box(ValueId),
    Num2Str(ValueId),

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

    ClosureNew,

    PushSink(ValueId),
    Return(ValueId),
    Phi(ValueId, ValueId),
}

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;
type BoolOp = interpreter::BoolOp;

impl From<Cmp> for Instr {
    fn from(cmp: Cmp) -> Self {
        Instr::Cmp(cmp)
    }
}

impl Instr {
    pub(super) fn result_type(&self) -> Option<ValueType> {
        match self {
            Instr::Not(_) => Some(ValueType::Bool),
            Instr::Arith { .. } => Some(ValueType::Num),
            Instr::Cmp { .. } => Some(ValueType::Bool),
            Instr::BoolOp { .. } => Some(ValueType::Bool),
            Instr::AssertTrue { .. } => None,
            Instr::AssertEqConst { .. } => None,
            Instr::PushSink(_) => None,
            Instr::Return(_) => None,
            Instr::GetArg(_) => Some(ValueType::Boxed),
            Instr::TraceParam(_) => Some(ValueType::Boxed),
            Instr::Choose { ty, .. } => Some(ty.clone()),
            Instr::Unbox(ty, _) => Some(ty.clone()),
            Instr::Box(_) => Some(ValueType::Boxed),
            Instr::Num2Str(_) => Some(ValueType::Str),
            Instr::ObjNew => Some(ValueType::Obj),
            Instr::ObjSet { .. } => None,
            Instr::ObjGet { .. } => Some(ValueType::Boxed),
            Instr::TypeOf(_) => Some(ValueType::Str),
            Instr::ClosureNew => Some(ValueType::Function),
            Instr::Const(val) => Some(ValueType::of(val)),
            Instr::Phi(_, _) => None,
        }
    }

    fn operands<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = &'a ValueId>> {
        use std::iter::once;
        match self {
            Instr::Not(arg) => Box::new(once(arg)),
            Instr::Arith { a, b, .. } => Box::new([a, b].into_iter()),
            Instr::Cmp(Cmp { a, b, .. }) => Box::new([a, b].into_iter()),
            Instr::BoolOp { a, b, .. } => Box::new([a, b].into_iter()),
            Instr::Choose {
                cond,
                if_true,
                if_false,
                ..
            } => Box::new([cond, if_true, if_false].into_iter()),
            Instr::AssertTrue { cond } => Box::new([cond].into_iter()),
            Instr::AssertEqConst { x, .. } => Box::new([x].into_iter()),
            Instr::Unbox(_, arg) => Box::new([arg].into_iter()),
            Instr::Box(arg) => Box::new([arg].into_iter()),
            Instr::Num2Str(arg) => Box::new([arg].into_iter()),
            Instr::PushSink(arg) => Box::new([arg].into_iter()),
            Instr::Return(arg) => Box::new([arg].into_iter()),

            Instr::GetArg(_) => Box::new(std::iter::empty()),
            Instr::TraceParam(_) => Box::new(std::iter::empty()),

            Instr::ObjNew => Box::new(std::iter::empty()),
            Instr::ObjSet { obj, key, value } => Box::new([obj, key, value].into_iter()),
            Instr::ObjGet { obj, key } => Box::new([obj, key].into_iter()),
            Instr::TypeOf(arg) => Box::new([arg].into_iter()),
            Instr::ClosureNew => Box::new(std::iter::empty()),
            Instr::Const(_) => Box::new(std::iter::empty()),
            Instr::Phi(_tgt, new_value) => Box::new(std::iter::once(new_value)),
        }
    }

    pub(crate) fn has_side_effects(&self) -> bool {
        match self {
            Instr::TraceParam(_) => false,
            Instr::GetArg(_) => false,
            Instr::Const(_) => false,
            Instr::Not(_) => false,
            Instr::Arith { .. } => false,
            Instr::Cmp(_) => false,
            Instr::BoolOp { .. } => false,
            Instr::Choose { .. } => false,
            Instr::AssertTrue { .. } => true,
            Instr::AssertEqConst { .. } => true,
            Instr::Unbox(_, _) => false,
            Instr::Box(_) => false,
            Instr::Num2Str(_) => false,
            Instr::ObjNew => false,
            Instr::ObjSet { .. } => false,
            Instr::ObjGet { .. } => false,
            Instr::TypeOf(_) => false,
            Instr::ClosureNew => false,
            Instr::PushSink(_) => true,
            Instr::Return(_) => false,
            Instr::Phi(_, _) => true,
        }
    }
}

impl regalloc::Instruction for Instr {
    fn inputs(&self) -> Vec<regalloc::SoftReg> {
        self.operands()
            .map(|ValueId(index)| regalloc::SoftReg(*index))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit;

    struct Output {
        sink: Vec<interpreter::Value>,
        trace: Option<jit::Trace>,
    }

    fn quick_jit_func(code: &str, start_stack_depth: u32) -> Output {
        quick_jit(start_stack_depth, jit::TraceStart::Function, code)
    }
    fn quick_jit_loop(code: &str, start_stack_depth: u32) -> Output {
        quick_jit(start_stack_depth, jit::TraceStart::FirstLoop, code)
    }
    fn quick_jit(start_stack_depth: u32, whence: TraceStart, code: &str) -> Output {
        let mut vm = interpreter::VM::new();
        let flags = interpreter::InterpreterFlags {
            tracer_flags: Some(interpreter::TracerFlags {
                start_depth: start_stack_depth,
                whence,
            }),
            ..Default::default()
        };
        vm.run_script(code.to_string(), flags).unwrap();
        Output {
            sink: vm.take_sink(),
            trace: vm.take_trace(),
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

    #[test]
    fn test_tracing_simple_constant_folding() {
        let output = quick_jit_func(
            "
            const x = 123;
            let y = 'a';
            if (x < 256) {
                y = 'b';
            }
            sink(y);
            ",
            0,
        );

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        let pushsink_operands = get_pushsink_consts(&trace);
        assert_eq!(pushsink_operands.len(), 1);
        assert!(PartialEq::eq(pushsink_operands[0], &"b".into()));
    }

    #[ignore]
    #[test]
    fn test_tracing_one_func() {
        let output = quick_jit_func(
            "
            function foo(mode, a, b) { 
                if (mode === 'sum')
                    return a + b;
                else if (mode === 'product')
                    return a * b;
                else
                    return 'something else';
            }

            sink(foo('product', 9, 8));
            ",
            1,
        );

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        // TODO: Run the trace (continue writing this test when traces can be run)
        assert!(false);
    }

    #[test]
    fn test_while_loop() {
        let code = "
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

            sink(sum_range(5));
        ";
        let output = quick_jit_loop(code, 1);

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        let native_thunk = jit::codegen::to_native(&trace);
        native_thunk.dump();

        todo!("Run the trace (continue writing this test when traces can be run)")
    }

    #[test]
    fn test_varid_resolution_with_inlining() {
        let output = quick_jit_func(
            "
            const far_away = 'asd';
            function f() {
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
            2,
        );

        let trace = output.trace.unwrap();
        trace.dump();

        let sink_operands = get_pushsink_consts(&trace);
        assert_eq!(sink_operands, [&"asd".into(), &"lol".into(),]);
    }

    #[test]
    fn test_closures() {
        let output = quick_jit_func(
            "
            function iota(factor) {
                let i = 0;
                return function() {
                    i += 1;
                    return i * factor;
                };
            }

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
            1,
        );

        let trace = output.trace.unwrap();
        trace.dump();

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
        let output = quick_jit_loop(
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

            sink(sum_range(5));
            ",
            1,
        );

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        todo!()
    }
}
