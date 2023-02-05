use std::collections::HashMap;
use std::fmt::Debug;

use crate::{
    interpreter::{self, IID},
    regalloc,
};

use dynasm::dynasm;

// This is going to be changed at some point
type BoxedValue = interpreter::Value;

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
struct Cmp {
    ty: ValueType,
    op: CmpOp,
    a: Operand,
    b: Operand,
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

#[derive(PartialEq, Clone)]
enum Operand {
    ValueId(ValueId),
    Imm(BoxedValue),
    Cmp(Box<Cmp>),
}

impl Operand {
    fn value_ids(&self) -> Box<dyn Iterator<Item = ValueId>> {
        match self {
            Operand::ValueId(vid) => Box::new(std::iter::once(vid.clone())),
            Operand::Imm(_) => Box::new(std::iter::empty()),
            Operand::Cmp(boxed_cmp) => {
                let Cmp { ty: _, op: _, a, b } = boxed_cmp.as_ref();
                Box::new(a.value_ids().chain(b.value_ids()))
            }
        }
    }
}

impl From<ValueId> for Operand {
    fn from(vid: ValueId) -> Self {
        Operand::ValueId(vid)
    }
}

impl From<Cmp> for Operand {
    fn from(cmp: Cmp) -> Self {
        Operand::Cmp(Box::new(cmp))
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::ValueId(vid) => vid.fmt(f),
            Operand::Imm(value) => value.fmt(f),
            Operand::Cmp(cmp) => cmp.fmt(f),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(u32);

impl std::fmt::Debug for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueType {
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
            interpreter::Value::LocalFn(_) => ValueType::Function,
            interpreter::Value::NativeFunction(_) => ValueType::Function,
            interpreter::Value::Object(_) => ValueType::Obj,
        }
    }

    fn js_typeof(&self) -> Option<Operand> {
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

        Some(Operand::Imm(BoxedValue::String(ret_str.into())))
    }
}

pub struct TraceBuilder {
    // This map associates VarId's from the interpreter to ValueIds in
    // the SSA trace we're building.
    //
    // Important: VarId's need to be "resolved" to an outer frame to the extent possible.
    //
    // For example, with the following JS code:
    // ```js
    // // some variable `far_away` is defined here as var[0:22]
    // function f() {
    //     let x = 1; // e.g. var[0:5]  in f
    //     function g() {
    //         function h() {
    //             use(x);  // x is var[2:5] in h's b.ytecode
    //             use(far_away); // far_away is var[3:22]
    //         }
    //     }
    // }
    // ```
    //
    // In the bytecode for function `h`, variable `x` is represented with VarId
    // var[2:5]. But since tracing JIT inlines all function calls, it's important for
    // the JIT compiler to detect that var[2:5] in h is the same as var[0:5] in f, and
    // should resolve it to the same JIT IR register.  To do this, we track the stack
    // depth of the calls we're inlining, and use it to convert VarId's to the lowest
    // stack depth possible. Note that variables captures outside the trace's entry
    // point (function or loop) will still have a stack depth component that is > 0
    // (i.e. `far_away` is resolved to var[1:22]).
    operand_of_varid: HashMap<VarId, Operand>,
    operand_of_iid: HashMap<(u32, interpreter::IID), Operand>,
    args_stack: Vec<Vec<Operand>>,
    unboxes: HashMap<ValueId, (ValueType, ValueId)>,

    // Current stack depth relative to the trace's entry point (function or loop)
    stack_depth: u32,

    instrs: Vec<Instr>,
    parameters: Vec<TraceParam>,

    failed: bool,
    trace_ended: bool,
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
struct TraceParam {
    operand: interpreter::Operand,
}

impl TraceBuilder {
    pub fn new() -> Self {
        TraceBuilder {
            operand_of_iid: HashMap::new(),
            operand_of_varid: HashMap::new(),
            unboxes: HashMap::new(),
            args_stack: Vec::new(),
            stack_depth: 0,
            instrs: Vec::new(),
            parameters: Vec::new(),
            failed: false,
            trace_ended: false,
            exit_function_expected: false,
        }
    }

    fn resolve_interpreter_operand(
        &mut self,
        operand: &interpreter::Operand,
        value_buf: &[interpreter::Value],
    ) -> Result<Operand, Error> {
        match operand {
            // constants from the interpreter bytecode are simply preserved
            interpreter::Operand::Value(value) => Ok(Operand::Imm(value.clone())),

            // variables from the interpreter (identified by the ID of the
            // instruction that produced them) are mapped to the JIT instruction
            // that last assigned to it (the JIT trace is SSA)
            interpreter::Operand::IID(iid) => {
                let key = (self.stack_depth, *iid);

                if let Some(operand_for_var) = self.operand_of_iid.get(&key) {
                    eprintln!("TB: {:?} is {:?}", iid, operand_for_var);
                    Ok(operand_for_var.clone())
                } else {
                    eprintln!(
                        "TB: {:?} is unresolved => considered trace parameter, adding guard",
                        operand
                    );

                    let observed_value = &value_buf[iid.0 as usize];

                    let param_ndx = self.add_parameter(operand.clone());
                    let param = self.emit(Instr::TraceParam(param_ndx))?;
                    // TODO Question: Is this really necessary?
                    self.emit(Instr::AssertEqConst {
                        x: param.clone(),
                        expected: observed_value.clone(),
                    })?;

                    self.operand_of_iid.insert(key, param.clone());
                    Ok(param)
                }
            }

            interpreter::Operand::Var(intrp_varid) => {
                // Resolve stack_fid to the outermost function/loop covered by
                // the trace (see comment for operand_of_varid)
                let resolved_varid = self.resolve_interpreter_varid(intrp_varid);
                eprintln!("resolving {resolved_varid:?}");

                match self.operand_of_varid.get(&resolved_varid) {
                    Some(oper) => Ok(oper.clone()),
                    None => {
                        let param_ndx =
                            self.add_parameter(interpreter::Operand::Var(intrp_varid.clone()));
                        let oper = self.emit(Instr::TraceParam(param_ndx))?;
                        self.operand_of_varid.insert(resolved_varid, oper.clone());
                        Ok(oper)
                    }
                }
            }
        }
    }

    fn resolve_interpreter_varid(&mut self, intrp_var_id: &interpreter::VarId) -> VarId {
        VarId {
            stack_depth: intrp_var_id.stack_fid.0 as i32 - self.stack_depth as i32,
            var_ndx: intrp_var_id.var_ndx.0,
        }
    }

    fn operand_type(&mut self, operand: &Operand) -> Option<ValueType> {
        match operand {
            Operand::ValueId(vid) => {
                let instr = self.instrs.get(vid.0 as usize).unwrap();
                instr.result_type()
            }
            Operand::Imm(value) => Some(ValueType::of(value)),
            Operand::Cmp(_) => Some(ValueType::Bool),
        }
    }

    fn ensure_type(&mut self, operand: Operand, desired_type: ValueType) -> Result<Operand, Error> {
        let type_error = Error::Type(TypeError { desired_type });
        let input_type = self.operand_type(&operand).ok_or(type_error.clone())?;

        let converted_operand = match (input_type, desired_type) {
            (a, b) if a == b => Some(operand),

            (_, ValueType::Null) => Some(Operand::Imm(BoxedValue::Null)),
            (_, ValueType::Undefined) => Some(Operand::Imm(BoxedValue::Undefined)),
            (_, ValueType::Function) => None,

            (ValueType::Boxed, desired_type) => {
                assert_ne!(ValueType::Boxed, desired_type);
                Some(self.emit(Instr::Unbox(desired_type, operand))?)
            }

            (input_type, ValueType::Boxed) => {
                assert_ne!(ValueType::Boxed, input_type);
                Some(self.emit(Instr::Box(operand))?)
            }

            (ValueType::Bool, ValueType::Num) => Some(self.emit(Instr::Choose {
                ty: ValueType::Num,
                cond: operand,
                if_true: Operand::Imm(BoxedValue::Number(1.0)),
                if_false: Operand::Imm(BoxedValue::Number(0.0)),
            })?),
            (ValueType::Bool, ValueType::Str) => Some(self.emit(Instr::Choose {
                ty: ValueType::Str,
                cond: operand,
                // TODO the String allocation could be avoided
                if_true: Operand::Imm("true".into()),
                if_false: Operand::Imm("false".into()),
            })?),

            (ValueType::Num, ValueType::Bool) => Some(
                Cmp {
                    ty: ValueType::Num,
                    op: CmpOp::EQ,
                    a: operand,
                    b: Operand::Imm(BoxedValue::Number(0.0)),
                }
                .into(),
            ),
            (ValueType::Num, ValueType::Str) => Some(self.emit(Instr::Num2Str(operand))?),

            (ValueType::Str, ValueType::Bool) => Some(
                Cmp {
                    ty: ValueType::Str,
                    op: CmpOp::EQ,
                    a: operand,
                    // TODO this string allocation could be avoided
                    b: Operand::Imm("".into()),
                }
                .into(),
            ),

            // TODO Convert string to number
            (ValueType::Str, ValueType::Num) => None,

            (ValueType::Null, ValueType::Bool) => Some(Operand::Imm(BoxedValue::Bool(false))),
            (ValueType::Null, ValueType::Num) => Some(Operand::Imm(BoxedValue::Number(0.0))),

            // TODO this string allocation could be avoided
            (ValueType::Null, ValueType::Str) => Some(Operand::Imm("null".into())),

            (ValueType::Undefined, ValueType::Bool) => Some(Operand::Imm(BoxedValue::Bool(false))),
            (ValueType::Undefined, ValueType::Num) => Some(Operand::Imm(BoxedValue::Number(0.0))),
            (ValueType::Undefined, ValueType::Str) => Some(Operand::Imm("undefined".into())),

            (ValueType::Function, _) => None,

            _ => unreachable!(),
        };

        if let Some(converted_operand) = &converted_operand {
            assert_eq!(Some(desired_type), self.operand_type(converted_operand));
        }

        converted_operand.ok_or(type_error)
    }

    fn resolve_operand_as(
        &mut self,
        interp_oper: &interpreter::Operand,
        values_buf: &[interpreter::Value],
        desired_type: ValueType,
    ) -> Result<Operand, Error> {
        let oper = self.resolve_interpreter_operand(interp_oper, values_buf)?;
        self.ensure_type(oper, desired_type)
    }

    pub(crate) fn interpreter_step(&mut self, step: &InterpreterStep) {
        if self.trace_ended {
            return;
        }
        if let Err(error) = self.try_interpreter_step(step) {
            eprintln!("TB: trace failed: {:?}", error);
            // loops not supported yet!
            self.failed = true;
            self.trace_ended = true;
        }
    }

    fn try_interpreter_step(&mut self, step: &InterpreterStep) -> Result<(), Error> {
        if self.trace_ended {
            return Ok(());
        }

        if self.exit_function_expected {
            panic!("JIT bug: should have called exit_function() !");
        }

        let instr = &step.func.instrs()[step.iid.0 as usize];
        eprintln!("TB: step");

        // TODO Disable loops. Loops are temporarily allowed, though they
        // produce several types of incorrect traces, because it makes development easier.
        if false && step.func.is_loop_head(step.iid) {
            return Err(Error::Unsupported("loops".to_string()));
        }

        let interpreter_result = &step.values_buf[step.iid.0 as usize];

        let result = match instr {
            interpreter::Instr::Nop => None,
            interpreter::Instr::Const(value) => Some(Operand::Imm(value.clone())),
            interpreter::Instr::Not(oper) => {
                let oper = self.resolve_operand_as(&oper, &step.values_buf, ValueType::Bool)?;
                Some(self.emit(Instr::Not(oper))?)
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = self.resolve_operand_as(&a, &step.values_buf, ValueType::Num)?;
                let b = self.resolve_operand_as(&b, &step.values_buf, ValueType::Num)?;
                Some(self.emit(Instr::Arith { op: *op, a, b })?)
            }
            interpreter::Instr::Cmp { op, a, b } => {
                let a = self.resolve_interpreter_operand(&a, &step.values_buf)?;
                let b = self.resolve_interpreter_operand(&b, &step.values_buf)?;

                let result = match (a, b) {
                    (Operand::Imm(_), Operand::Imm(_)) => Operand::Imm(interpreter_result.clone()),
                    (Operand::Cmp(_), _) | (_, Operand::Cmp(_)) => {
                        panic!("invalid operand of type Cmp for instr Cmp")
                    }
                    (a, b) => {
                        let a_type = self.operand_type(&a).unwrap();
                        let b_type = self.operand_type(&b).unwrap();
                        let cmp = match (a_type, b_type) {
                            (ValueType::Boxed, unboxed_type) => Cmp {
                                ty: unboxed_type,
                                op: *op,
                                a: self.emit(Instr::Unbox(unboxed_type, a))?,
                                b,
                            },
                            (unboxed_type, ValueType::Boxed) => Cmp {
                                ty: unboxed_type,
                                op: *op,
                                a,
                                b: self.emit(Instr::Unbox(unboxed_type, b))?,
                            },
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
                    }
                };

                Some(result)
            }
            interpreter::Instr::JmpIf { cond, .. } => {
                let cond = self.resolve_operand_as(&cond, &step.values_buf, ValueType::Bool)?;
                match cond {
                    // treat this the same as an unconditional jump
                    Operand::Imm(_) => None,
                    Operand::Cmp(_) | Operand::ValueId(_) => {
                        let branch_taken = step.next_iid.0 != (step.iid.0 + 1);
                        eprintln!(
                            "TB: jmpif: branch {}taken ({:?} -> {:?})",
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
                let resolved_varid = self.resolve_interpreter_varid(var_id);
                let value = self.resolve_interpreter_operand(value, &step.values_buf)?;
                self.operand_of_varid.insert(resolved_varid, value);
                None
            }
            interpreter::Instr::PushSink(value) => {
                let value = self.resolve_interpreter_operand(&value, &step.values_buf)?;
                self.emit(Instr::PushSink(value))?;
                None
            }
            interpreter::Instr::Return(value) => {
                self.exit_function_expected = true;

                let value = if self.stack_depth == 0 {
                    self.resolve_operand_as(&value, &step.values_buf, ValueType::Boxed)?
                } else {
                    self.resolve_interpreter_operand(&value, &step.values_buf)?
                };
                Some(value)
            }
            interpreter::Instr::GetArg(index) => {
                let value = if self.stack_depth == 0 {
                    self.emit(Instr::GetArg(*index))?
                } else {
                    self.args_stack
                        .last_mut()
                        .expect("JIT bug: no arg vec on stack")
                        .get(*index)
                        .expect("bytecode_compiler bug: unbound arg var")
                        .clone()
                };
                Some(value)
            }
            interpreter::Instr::Call { args, .. } => {
                // TODO Something to do for the return value?
                None
            }

            interpreter::Instr::ObjNew => Some(self.emit(Instr::ObjNew)?),
            interpreter::Instr::ObjSet { obj, key, value } => {
                let obj = self.resolve_operand_as(obj, &step.values_buf, ValueType::Obj)?;
                let key = self.resolve_operand_as(key, &step.values_buf, ValueType::Str)?;
                let value = self.resolve_operand_as(value, &step.values_buf, ValueType::Boxed)?;

                self.emit(Instr::ObjSet { obj, key, value })?;
                None
            }
            interpreter::Instr::ObjGet { obj, key } => {
                let obj = self.resolve_operand_as(obj, &step.values_buf, ValueType::Obj)?;
                let key = self.resolve_operand_as(key, &step.values_buf, ValueType::Str)?;

                Some(self.emit(Instr::ObjGet { obj, key })?)
            }
            interpreter::Instr::ArrayNew => {
                todo!("array new")
            }
            interpreter::Instr::ArrayPush(_arr, _elem) => {
                todo!("array push")
            }
            interpreter::Instr::TypeOf(arg) => {
                let arg = self.resolve_interpreter_operand(arg, &step.values_buf)?;
                Some(self.emit(Instr::TypeOf(arg))?)
            }

            interpreter::Instr::BoolOp { op, a, b } => {
                let a = self.resolve_operand_as(a, &step.values_buf, ValueType::Bool)?;
                let b = self.resolve_operand_as(b, &step.values_buf, ValueType::Bool)?;
                Some(self.emit(Instr::BoolOp { op: *op, a, b })?)
            }
        };

        // Map IID to the result operand
        if let Some(result) = result {
            self.map_iid(step.iid, result);
        }

        Ok(())
    }

    pub(crate) fn enter_function(
        &mut self,
        args: &[interpreter::Operand],
        values_buf: &[interpreter::Value],
    ) {
        // TODO Use try_collect when it becomes stable
        let mut arg_values = Vec::new();
        for (ndx, arg) in args.iter().enumerate() {
            let arg = match self.resolve_interpreter_operand(arg, &values_buf) {
                Ok(arg) => arg,
                Err(err) => {
                    eprintln!("TB: trace failed, couldn't resolve argument #{ndx}: {err:?}");
                    self.failed = true;
                    self.trace_ended = true;
                    return;
                }
            };
            arg_values.push(arg);
        }
        self.args_stack.push(arg_values);
        self.stack_depth += 1;
        eprintln!("TB: call (stack depth -> {})", self.stack_depth);
    }

    pub(crate) fn exit_function(&mut self) {
        if self.stack_depth == 0 {
            eprintln!("TB: trace ended");
            self.trace_ended = true;
        } else {
            let depth_to_remove = -(self.stack_depth as i32);
            // TODO Change to HashMap::drain_filter once it becomes stable
            let to_remove: Vec<_> = self
                .operand_of_varid
                .keys()
                .filter(|var_id| var_id.stack_depth == depth_to_remove)
                .copied()
                .collect();
            for key in to_remove.iter() {
                self.operand_of_varid.remove(&key);
            }
            self.stack_depth -= 1;
            self.args_stack.pop();
            eprintln!("TB: returned from function");
        }

        self.exit_function_expected = false;
    }

    fn map_iid(&mut self, iid: IID, jit_operand: Operand) {
        eprintln!("TB: map {:?} -> {:?}", iid, jit_operand);
        self.operand_of_iid
            .insert((self.stack_depth, iid), jit_operand);
    }

    fn emit(&mut self, instr: Instr) -> Result<Operand, Error> {
        match instr {
            Instr::Not(Operand::Imm(BoxedValue::Bool(value))) => {
                Ok(Operand::Imm(BoxedValue::Bool(!value)))
            }
            Instr::Not(Operand::Cmp(cmp)) => Ok(Operand::Cmp(Box::new(cmp.invert()))),

            Instr::Arith {
                op,
                a: Operand::Imm(BoxedValue::Number(a)),
                b: Operand::Imm(BoxedValue::Number(b)),
            } => Ok(Operand::Imm(BoxedValue::Number(match op {
                ArithOp::Add => a + b,
                ArithOp::Sub => a - b,
                ArithOp::Mul => a * b,
                ArithOp::Div => a / b,
            }))),

            Instr::Arith {
                op: ArithOp::Add,
                a: Operand::Imm(BoxedValue::Number(a_const)),
                b,
            } if a_const == 0.0 => Ok(b),
            Instr::Arith {
                op: ArithOp::Add,
                a,
                b: Operand::Imm(BoxedValue::Number(b_const)),
            } if b_const == 0.0 => Ok(a),
            Instr::Arith {
                op: ArithOp::Sub,
                a: Operand::Imm(BoxedValue::Number(a_const)),
                b,
            } if a_const == 0.0 => Ok(b),
            Instr::Arith {
                op: ArithOp::Sub,
                a,
                b: Operand::Imm(BoxedValue::Number(b_const)),
            } if b_const == 0.0 => Ok(a),

            Instr::Arith {
                op: ArithOp::Mul,
                a: Operand::Imm(BoxedValue::Number(a_const)),
                b,
            } if a_const == 1.0 => Ok(b),
            Instr::Arith {
                op: ArithOp::Mul,
                a,
                b: Operand::Imm(BoxedValue::Number(b_const)),
            } if b_const == 1.0 => Ok(a),

            Instr::Arith {
                op: ArithOp::Div,
                a,
                b: Operand::Imm(BoxedValue::Number(b_const)),
            } if b_const == 1.0 => Ok(a),

            Instr::BoolOp {
                op: BoolOp::And,
                a: Operand::Imm(BoxedValue::Bool(false)),
                b: _,
            } => Ok(Operand::Imm(BoxedValue::Bool(false))),
            Instr::BoolOp {
                op: BoolOp::And,
                a: _,
                b: Operand::Imm(BoxedValue::Bool(false)),
            } => Ok(Operand::Imm(BoxedValue::Bool(false))),
            Instr::BoolOp {
                op: BoolOp::And,
                a: Operand::Imm(BoxedValue::Bool(true)),
                b: Operand::Imm(BoxedValue::Bool(true)),
            } => Ok(Operand::Imm(BoxedValue::Bool(true))),

            Instr::BoolOp {
                op: BoolOp::Or,
                a: Operand::Imm(BoxedValue::Bool(true)),
                b: _,
            } => Ok(Operand::Imm(BoxedValue::Bool(true))),
            Instr::BoolOp {
                op: BoolOp::Or,
                a: _,
                b: Operand::Imm(BoxedValue::Bool(true)),
            } => Ok(Operand::Imm(BoxedValue::Bool(true))),
            Instr::BoolOp {
                op: BoolOp::Or,
                a: Operand::Imm(BoxedValue::Bool(false)),
                b: Operand::Imm(BoxedValue::Bool(false)),
            } => Ok(Operand::Imm(BoxedValue::Bool(false))),

            Instr::Unbox(value_type, ref arg) => match arg {
                Operand::Imm(_) => panic!("invalid operand for Unbox: immediate"),
                Operand::Cmp(_) => panic!("invalid operand for Unbox: cmp"),
                Operand::ValueId(arg_vid) => {
                    if let Some((prev_unbox_type, unboxed)) = self.unboxes.get(&arg_vid) {
                        if *prev_unbox_type == value_type {
                            eprintln!(
                                "TB: reuse unbox v{:<4} as {:?} is at {:?}",
                                arg_vid.0, value_type, unboxed
                            );
                            Ok(Operand::ValueId(unboxed.clone()))
                        } else {
                            Err(Error::InconsistentUnbox(InconsistentUnbox {
                                desired_type: value_type.clone(),
                            }))
                        }
                    } else {
                        eprintln!("TB: emit unbox: v{:<4} as {:?}", arg_vid.0, value_type);
                        let new_vid = ValueId(self.instrs.len() as u32);
                        self.unboxes.insert(*arg_vid, (value_type, new_vid));
                        self.instrs.push(instr);
                        Ok(new_vid.into())
                    }
                }
            },

            Instr::TypeOf(ref arg) => {
                let ty = match arg {
                    Operand::ValueId(vid) => {
                        let target_instr = self.instrs.get(vid.0 as usize).unwrap();
                        target_instr.result_type().and_then(|ty| ty.js_typeof())
                    }
                    Operand::Imm(value) => ValueType::of(value).js_typeof(),
                    Operand::Cmp(_) => panic!("JIT bug: invalid operand of type Cmp for TypeOf"),
                };

                match ty {
                    Some(ret) => Ok(ret),
                    None => self.write(instr),
                }
            }

            _ => self.write(instr),
        }
    }

    fn write(&mut self, instr: Instr) -> Result<Operand, Error> {
        let vid = ValueId(self.instrs.len() as u32);
        eprintln!("TB: emit: v{:<4} {:?}", vid.0, instr);
        self.instrs.push(instr);
        Ok(vid.into())
    }

    pub(crate) fn build(self) -> Option<Trace> {
        if self.failed {
            None
        } else {
            let hregs = regalloc::allocate_registers(self.instrs.as_slice(), 6);

            Some(Trace {
                instrs: self.instrs,
                hreg_alloc: hregs,
                parameters: self.parameters,
            })
        }
    }

    pub(crate) fn add_parameter(&mut self, operand: interpreter::Operand) -> u16 {
        let ndx = self.parameters.len() as u16;
        self.parameters.push(TraceParam { operand });
        ndx
    }
}

pub struct InterpreterStep<'a> {
    pub(crate) values_buf: &'a Vec<interpreter::Value>,
    pub(crate) fnid: interpreter::FnId,
    pub(crate) func: &'a interpreter::Function,
    pub(crate) iid: interpreter::IID,
    pub(crate) next_iid: interpreter::IID,
}

#[derive(PartialEq, Debug)]
enum Instr {
    TraceParam(u16),
    GetArg(usize),
    Not(Operand),
    Arith {
        op: ArithOp,
        a: Operand,
        b: Operand,
    },
    Cmp {
        ty: ValueType,
        op: CmpOp,
        a: Operand,
        b: Operand,
    },
    BoolOp {
        op: BoolOp,
        a: Operand,
        b: Operand,
    },
    Choose {
        ty: ValueType,
        cond: Operand,
        if_true: Operand,
        if_false: Operand,
    },
    AssertTrue {
        cond: Operand,
    },
    AssertEqConst {
        x: Operand,
        expected: BoxedValue,
    },

    Unbox(ValueType, Operand),
    Box(Operand),
    Num2Str(Operand),

    ObjNew,
    ObjSet {
        obj: Operand,
        key: Operand,
        value: Operand,
    },
    ObjGet {
        obj: Operand,
        key: Operand,
    },

    TypeOf(Operand),

    PushSink(Operand),
    Return(Operand),
}

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;
type BoolOp = interpreter::BoolOp;

impl Instr {
    fn result_type(&self) -> Option<ValueType> {
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
        }
    }

    fn operands<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = &'a Operand>> {
        use std::iter::once;
        match self {
            Instr::Not(arg) => Box::new(once(arg)),
            Instr::Arith { a, b, .. } => Box::new([a, b].into_iter()),
            Instr::Cmp { a, b, .. } => Box::new([a, b].into_iter()),
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
        }
    }
}

impl regalloc::Instruction for Instr {
    fn inputs(&self) -> Vec<regalloc::SoftReg> {
        self.operands()
            .flat_map(|operand| operand.value_ids())
            .map(|ValueId(index)| regalloc::SoftReg(index))
            .collect()
    }
}

// A JIT trace, in SSA representation.
pub struct Trace {
    instrs: Vec<Instr>,
    hreg_alloc: regalloc::Allocation,
    parameters: Vec<TraceParam>,
}

impl Trace {
    pub fn dump(&self) {
        use std::borrow::Cow;

        eprintln!(" === trace");
        eprintln!(" {} parameters", self.parameters.len());
        for (ndx, param) in self.parameters.iter().enumerate() {
            eprintln!("    param[{}] = {:?}", ndx, param);
        }
        for (ndx, instr) in self.instrs.iter().enumerate() {
            let hreg = self.hreg_alloc.hreg_of_instr(ndx);

            let hreg = hreg
                .map(|x| Cow::Owned(format!("{:?}", x)))
                .unwrap_or_else(|| Cow::Borrowed("???"));
            eprintln!(" {:4?} {:5} {:?}", ndx, hreg, instr);
        }
    }

    pub(crate) fn compile(&self) -> NativeThunk {
        use dynasmrt::{DynasmApi, DynasmLabelApi};
        let mut asm = dynasmrt::x64::Assembler::new().unwrap();

        dynasm!(asm
        ; .arch x64
        ; entry:
        ; mov eax, 123
        ; ret
        );

        let entry_offset = asm.labels().resolve_local("entry").unwrap();

        let buf = asm.finalize().unwrap();

        NativeThunk {
            buf,
            entry_offset,
            sink: Vec::new(),
        }
    }
}

pub struct NativeThunk {
    buf: dynasmrt::ExecutableBuffer,
    entry_offset: dynasmrt::AssemblyOffset,
    sink: Vec<BoxedValue>,
}

impl NativeThunk {
    pub(crate) fn run(&self) -> u64 {
        let ptr = self.buf.ptr(self.entry_offset);
        let thunk: extern "C" fn() -> u64 = unsafe { std::mem::transmute(ptr) };
        thunk()
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

    fn quick_run(code: &str, start_stack_depth: u32) -> Output {
        use crate::bytecode_compiler;

        let mut vm = interpreter::VM::new();
        let flags = interpreter::InterpreterFlags {
            tracer_flags: Some(interpreter::TracerFlags {
                start_depth: start_stack_depth,
            }),
            ..Default::default()
        };
        vm.run_script(code.to_string(), flags).unwrap();
        Output {
            sink: vm.take_sink(),
            trace: vm.take_trace(),
        }
    }

    #[ignore]
    #[test]
    fn test_tracing_simple_constant_folding() {
        let output = quick_run(
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

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();
        assert_eq!(
            &trace.instrs[..],
            &[Instr::PushSink(Operand::Imm("b".to_string().into()))]
        );
    }

    #[ignore]
    #[test]
    fn test_tracing_one_func() {
        let output = quick_run(
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

        assert!(false);
    }

    #[ignore]
    #[test]
    fn test_while() {
        let output = quick_run(
            "
            function sum_range(n) {
                let i = 0;
                let ret = 0;
                while (i <= n) {
                    ret += i;
                    i++;
                }
                return ret;
            }

            function sum_range_down(n) {
                let ret = 0;
                while (n > 0) {
                    ret += n;
                    n--;
                }
                return ret;
            }
            
            sink(sum_range_down(5));
            ",
            0,
        );

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        assert!(false);
    }

    #[test]
    fn test_var_mapping() {
        let output = quick_run(
            "
            function iota(n) {
                function foo(value) { sink(value); }
                let i = 0;
                while (i < n) {
                    foo(i * 11);
                    i++;
                }
            }

            iota(10);
            ",
            1,
        );

        let trace = output.trace.as_ref().unwrap();

        let mut counts = HashMap::new();
        for param in trace.parameters.iter() {
            if let TraceParam {
                operand: interpreter::Operand::Var(var_id),
            } = param
            {
                *counts.entry(*var_id).or_insert(0) += 1;
            }
        }

        assert!(counts.len() > 0);
        for count in counts.values() {
            assert_eq!(1, *count);
        }
    }
}
