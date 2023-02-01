use std::collections::HashMap;
use std::fmt::Debug;

use crate::interpreter::{self, IID};

use dynasm::dynasm;

// This is going to be changed at some point
type BoxedValue = interpreter::Value;

#[derive(Debug)]
struct TypeError {
    desired_type: ValueType,
}

#[derive(Debug)]
enum Error {
    Type(TypeError),
    Unsupported(String),
}
impl From<TypeError> for Error {
    fn from(type_err: TypeError) -> Self {
        Error::Type(type_err)
    }
}

#[derive(PartialEq, Clone)]
enum Operand {
    ValueId(ValueId),
    Imm(BoxedValue),
}

impl From<ValueId> for Operand {
    fn from(vid: ValueId) -> Self {
        Operand::ValueId(vid)
    }
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::ValueId(vid) => vid.fmt(f),
            Operand::Imm(value) => value.fmt(f),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
        }
    }
}

pub struct TraceBuilder {
    // This map associates Instruction IDs from the interpreter to ValueIds in
    // the SSA trace we're building. Instruction IDs that are absent from this
    // map are considered constant, and the corresponding value can be fetched
    // directly from the interpreter's value buffer. (The case where it is
    // "yet to be processed" is automatically excluded by the structure of the
    // interpreter bytecode.)
    //
    // Key is (stack depth, IID)
    // TODO change this name...
    var_iid: HashMap<(u32, interpreter::IID), Operand>,
    stack_depth: u32,

    instrs: Vec<Instr>,
    parameters: Vec<TraceParam>,

    failed: bool,
    trace_ended: bool,
}

struct TraceParam {
    iid: interpreter::IID,
}

impl TraceBuilder {
    pub fn new() -> Self {
        TraceBuilder {
            var_iid: HashMap::new(),
            stack_depth: 0,
            instrs: Vec::new(),
            parameters: Vec::new(),
            failed: false,
            trace_ended: false,
        }
    }

    fn resolve_interpreter_operand(
        &mut self,
        operand: &interpreter::Operand,
        value_buf: &[interpreter::Value],
    ) -> Operand {
        match operand {
            // constants from the interpreter bytecode are simply preserved
            interpreter::Operand::Value(value) => Operand::Imm(value.clone()),

            // variables from the interpreter (identified by the ID of the
            // instruction that produced them) are mapped to the JIT instruction
            // that last assigned to it (the JIT trace is SSA)
            interpreter::Operand::IID(iid) => {
                let key = (self.stack_depth, *iid);

                if let Some(operand_for_var) = self.var_iid.get(&key) {
                    eprintln!("TB: {:?} is {:?}", iid, operand_for_var);
                    operand_for_var.clone()
                } else {
                    eprintln!(
                        "TB: {:?} is unresolved => considered parameter, adding guard",
                        operand
                    );
                    let param_ndx = self.add_parameter(*iid);
                    let vid = self.emit(Instr::TraceParam(param_ndx));
                    let observed_value = &value_buf[iid.0 as usize];
                    self.emit(Instr::AssertEqConst {
                        x: Operand::ValueId(vid),
                        expected: observed_value.clone(),
                    });

                    let operand_for_var = Operand::ValueId(vid);
                    self.var_iid.insert(key, operand_for_var.clone());
                    operand_for_var
                }
            }
        }
    }

    fn operand_type(&mut self, operand: &Operand) -> Option<ValueType> {
        match operand {
            Operand::ValueId(vid) => {
                let instr = self.instrs.get(vid.0 as usize).unwrap();
                instr.result_type()
            }
            Operand::Imm(value) => Some(ValueType::of(value)),
        }
    }

    fn ensure_type(&mut self, operand: Operand, desired_type: ValueType) -> Option<Operand> {
        let input_type = self.operand_type(&operand)?;

        let converted_operand = match (input_type, desired_type) {
            (a, b) if a == b => Some(operand),

            (_, ValueType::Null) => Some(Operand::Imm(BoxedValue::Null)),
            (_, ValueType::Undefined) => Some(Operand::Imm(BoxedValue::Undefined)),
            (_, ValueType::Function) => None,

            (ValueType::Boxed, desired_type) => {
                assert_ne!(ValueType::Boxed, desired_type);
                Some(self.emit(Instr::Unbox(desired_type, operand)).into())
            }

            (input_type, ValueType::Boxed) => {
                assert_ne!(ValueType::Boxed, input_type);
                Some(self.emit(Instr::Box(operand)).into())
            }

            (ValueType::Bool, ValueType::Num) => Some(
                self.emit(Instr::Choose {
                    ty: ValueType::Num,
                    cond: operand,
                    if_true: Operand::Imm(BoxedValue::Number(1.0)),
                    if_false: Operand::Imm(BoxedValue::Number(0.0)),
                })
                .into(),
            ),
            (ValueType::Bool, ValueType::Str) => Some(
                self.emit(Instr::Choose {
                    ty: ValueType::Str,
                    cond: operand,
                    // TODO the String allocation could be avoided
                    if_true: Operand::Imm(BoxedValue::String("true".to_string())),
                    if_false: Operand::Imm(BoxedValue::String("false".to_string())),
                })
                .into(),
            ),

            (ValueType::Num, ValueType::Bool) => Some(
                self.emit(Instr::Cmp {
                    ty: ValueType::Num,
                    op: CmpOp::EQ,
                    a: operand,
                    b: Operand::Imm(BoxedValue::Number(0.0)),
                })
                .into(),
            ),
            (ValueType::Num, ValueType::Str) => Some(self.emit(Instr::Num2Str(operand)).into()),

            (ValueType::Str, ValueType::Bool) => Some(
                self.emit(Instr::Cmp {
                    ty: ValueType::Str,
                    op: CmpOp::EQ,
                    a: operand,
                    // TODO this string allocation could be avoided
                    b: Operand::Imm(BoxedValue::String("".to_string())),
                })
                .into(),
            ),

            // TODO Convert string to number
            (ValueType::Str, ValueType::Num) => None,

            (ValueType::Null, ValueType::Bool) => Some(Operand::Imm(BoxedValue::Bool(false))),
            (ValueType::Null, ValueType::Num) => Some(Operand::Imm(BoxedValue::Number(0.0))),

            // TODO this string allocation could be avoided
            (ValueType::Null, ValueType::Str) => {
                Some(Operand::Imm(BoxedValue::String("null".to_string())))
            }

            (ValueType::Undefined, ValueType::Bool) => Some(Operand::Imm(BoxedValue::Bool(false))),
            (ValueType::Undefined, ValueType::Num) => Some(Operand::Imm(BoxedValue::Number(0.0))),
            (ValueType::Undefined, ValueType::Str) => {
                Some(Operand::Imm(BoxedValue::String("undefined".to_string())))
            }

            (ValueType::Function, _) => None,

            _ => unreachable!(),
        };

        if let Some(converted_operand) = &converted_operand {
            assert_eq!(Some(desired_type), self.operand_type(converted_operand));
        }

        converted_operand
    }

    fn resolve_operand_as(
        &mut self,
        interp_oper: &interpreter::Operand,
        values_buf: &[interpreter::Value],
        desired_type: ValueType,
    ) -> Result<Operand, TypeError> {
        let oper = self.resolve_interpreter_operand(interp_oper, values_buf);
        self.ensure_type(oper, desired_type)
            .ok_or(TypeError { desired_type })
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
            interpreter::Instr::Const(value) => {
                // This is constant folding: the interpreter's IID is associated directly
                // to the constant, not to an instruction from the trace.
                Some(Operand::Imm(value.clone()))
            }
            interpreter::Instr::Not(oper) => {
                // if the operand resolves to a constant, we can just skip it
                // (functions as constant folding)
                let oper = self.resolve_operand_as(&oper, &step.values_buf, ValueType::Bool)?;
                match oper {
                    Operand::ValueId(vid) => Some(self.emit(Instr::Not(vid)).into()),
                    Operand::Imm(BoxedValue::Bool(value)) => {
                        Some(Operand::Imm(BoxedValue::Bool(!value)))
                    }
                    _ => unreachable!(),
                }
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = self.resolve_operand_as(&a, &step.values_buf, ValueType::Num)?;
                let b = self.resolve_operand_as(&b, &step.values_buf, ValueType::Num)?;
                match (a, b) {
                    (Operand::Imm(_), Operand::Imm(_)) => {
                        Some(Operand::Imm(interpreter_result.clone()))
                    }
                    (a, b) => Some(self.emit(Instr::Arith { op: *op, a, b }).into()),
                }
            }
            interpreter::Instr::Cmp { op, a, b } => {
                let a = self.resolve_operand_as(&a, &step.values_buf, ValueType::Num)?;
                let b = self.resolve_operand_as(&b, &step.values_buf, ValueType::Num)?;
                Some(match (a, b) {
                    (Operand::Imm(_), Operand::Imm(_)) => Operand::Imm(interpreter_result.clone()),
                    (a, b) => self
                        .emit(Instr::Cmp {
                            ty: ValueType::Num,
                            op: *op,
                            a,
                            b,
                        })
                        .into(),
                })
            }
            interpreter::Instr::JmpIf { cond, .. } => {
                let cond = self.resolve_operand_as(&cond, &step.values_buf, ValueType::Bool)?;
                match cond {
                    // treat this the same as an unconditional jump
                    Operand::Imm(_cond) => None,
                    Operand::ValueId(vid) => {
                        let branch_taken = step.next_iid.0 != (step.iid.0 + 1);
                        eprintln!(
                            "TB: jmpif: branch {}taken ({:?} -> {:?})",
                            if branch_taken { "" } else { "not " },
                            step.iid,
                            step.next_iid,
                        );
                        let vid = if branch_taken {
                            vid
                        } else {
                            self.emit(Instr::Not(vid))
                        };
                        let cond = Operand::ValueId(vid);
                        Some(self.emit(Instr::AssertTrue { cond }).into())
                    }
                }
            }
            interpreter::Instr::Jmp(_) => {
                // unconditional jump.  Nothing to do, let's just follow the interpreter to the next instruction
                None
            }
            interpreter::Instr::Set { var_id, value } => {
                let value = self.resolve_interpreter_operand(value, &step.values_buf);
                self.map_iid(*var_id, value);
                None
            }
            interpreter::Instr::PushSink(value) => {
                let value = self.resolve_interpreter_operand(&value, &step.values_buf);
                self.emit(Instr::PushSink(value));
                None
            }
            interpreter::Instr::Return(value) => {
                let value = self.resolve_operand_as(&value, &step.values_buf, ValueType::Boxed)?;
                if self.stack_depth == 0 {
                    self.trace_ended = true;
                } else {
                    self.stack_depth -= 1;
                }
                Some(self.emit(Instr::Return(value)).into())
            }
            interpreter::Instr::GetArg(index) => Some(self.emit(Instr::GetArg(*index)).into()),
            interpreter::Instr::Call { .. } => {
                eprintln!("TB: warning: following call no matter what (not correct in all cases!)");
                self.stack_depth += 1;
                None
            }
        };

        // Map IID to the result operand
        if let Some(result) = result {
            self.map_iid(step.iid, result);
        }

        Ok(())
    }

    fn map_iid(&mut self, iid: IID, jit_operand: Operand) {
        eprintln!("TB: map {:?} -> {:?}", iid, jit_operand);
        self.var_iid.insert((self.stack_depth, iid), jit_operand);
    }

    fn emit(&mut self, instr: Instr) -> ValueId {
        let vid = ValueId(self.instrs.len() as u32);
        eprintln!("TB: emit: v{:<4} {:?}", vid.0, instr);
        self.instrs.push(instr);
        vid
    }

    pub(crate) fn build(self) -> Option<Trace> {
        if self.failed {
            None
        } else {
            Some(Trace {
                instrs: self.instrs,
            })
        }
    }

    pub(crate) fn add_parameter(&mut self, iid: IID) -> u16 {
        let ndx = self.parameters.len() as u16;
        self.parameters.push(TraceParam { iid });
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
    GetArg(usize),
    TraceParam(u16),
    Not(ValueId),
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

    PushSink(Operand),
    Return(Operand),
}

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;

impl Instr {
    fn result_type(&self) -> Option<ValueType> {
        match self {
            Instr::Not(_) => Some(ValueType::Bool),
            Instr::Arith { .. } => Some(ValueType::Num),
            Instr::Cmp { .. } => Some(ValueType::Bool),
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
        }
    }
}

// A JIT trace, in SSA representation.
pub struct Trace {
    instrs: Vec<Instr>,
}

impl Trace {
    #[cfg(test)]
    pub(crate) fn dump(&self) {
        eprintln!();
        for (ndx, instr) in self.instrs.iter().enumerate() {
            eprintln!(" {:4?} {:?}", ndx, instr);
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

    fn quick_compile(code: &str) -> interpreter::Module {
        use crate::bytecode_compiler;

        let module =
            bytecode_compiler::compile_file("input.js".to_string(), code.to_string()).unwrap();
        module.dump();
        module
    }

    #[test]
    fn test_tracing_simple_constant_folding() {
        let module = quick_compile(
            "
            const x = 123;
            let y = 'a';
            if (x < 256) {
                y = 'b';
            }
            sink(y);
            ",
        );

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        let output = interpreter::interpret_and_trace(&module, 0).unwrap();

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();
        assert_eq!(
            &trace.instrs[..],
            &[Instr::PushSink(Operand::Imm("b".to_string().into()))]
        );
    }

    #[test]
    fn test_tracing_one_func() {
        let module = quick_compile(
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
        );

        let output = interpreter::interpret_and_trace(&module, 1).unwrap();

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        assert!(false);
    }

    #[test]
    fn test_while() {
        let module = quick_compile(
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
            
            sink(sum_range_down(2));
            ",
        );

        let output = interpreter::interpret_and_trace(&module, 0).unwrap();

        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        assert!(false);
    }
}
