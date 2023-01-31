use std::collections::HashMap;
use std::fmt::Debug;

use crate::interpreter::{self, IID};

use dynasm::dynasm;

// This is going to be changed at some point
type BoxedValue = interpreter::Value;

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

struct TypeError;

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
        value_buf: &Vec<interpreter::Value>,
    ) -> Operand {
        match operand {
            // constants from the interpreter bytecode are simply preserved
            interpreter::Operand::Value(value) => Operand::Imm(value.clone()),

            // variables from the interpreter (identified by the ID of the
            // instruction that produced them) are mapped to the corresponding
            // JIT instruction ID
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

    pub(crate) fn interpreter_step(&mut self, step: &InterpreterStep) {
        if self.trace_ended {
            return;
        }

        let instr = &step.func.instrs()[step.iid.0 as usize];
        eprintln!("TB: step");

        if false && step.func.is_loop_head(step.iid) {
            eprintln!("TB: instr is loop head. aborting.");
            // loops not supported yet!
            self.failed = true;
            self.trace_ended = true;
            return;
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
                match self.resolve_interpreter_operand(&oper, &step.values_buf) {
                    Operand::ValueId(vid) => Some(self.emit(Instr::Not(vid)).into()),
                    imm @ Operand::Imm(_) => Some(imm),
                }
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = self.resolve_interpreter_operand(&a, &step.values_buf);
                let b = self.resolve_interpreter_operand(&b, &step.values_buf);
                match (a, b) {
                    (Operand::Imm(_), Operand::Imm(_)) => {
                        Some(Operand::Imm(interpreter_result.clone()))
                    }
                    (a, b) => Some(self.emit(Instr::Arith { op: *op, a, b }).into()),
                }
            }
            interpreter::Instr::Cmp { op, a, b } => {
                let a = self.resolve_interpreter_operand(&a, &step.values_buf);
                let b = self.resolve_interpreter_operand(&b, &step.values_buf);
                match (a, b) {
                    (Operand::Imm(_), Operand::Imm(_)) => {
                        Some(Operand::Imm(interpreter_result.clone()))
                    }
                    (a, b) => Some(self.emit(Instr::CmpNum { op: *op, a, b }).into()),
                }
            }
            interpreter::Instr::JmpIf { cond, .. } => {
                let cond = self.resolve_interpreter_operand(&cond, &step.values_buf);
                match cond {
                    // treat this the same as an unconditional jump
                    Operand::Imm(_cond) => None,
                    Operand::ValueId(vid) => {
                        let branch_taken = step.next_iid.0 != step.iid.0;
                        let vid = if branch_taken {
                            self.emit(Instr::Not(vid))
                        } else {
                            vid
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
                let value = self.resolve_interpreter_operand(&value, &step.values_buf);
                if self.stack_depth == 0 {
                    self.trace_ended = true;
                } else {
                    self.stack_depth -= 1;
                }
                Some(self.emit(Instr::Return(value)).into())
            }
            interpreter::Instr::GetArg(index) => Some(self.emit(Instr::GetArg(*index)).into()),
            interpreter::Instr::Call { .. } => {
                eprintln!("TB: warning: following call no matter what");
                self.stack_depth += 1;
                None
            }
        };

        // Map IID to the result operand
        if let Some(result) = result {
            self.map_iid(step.iid, result);
        }
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
    Arith { op: ArithOp, a: Operand, b: Operand },
    CmpNum { op: CmpOp, a: Operand, b: Operand },
    AssertTrue { cond: Operand },
    AssertEqConst { x: Operand, expected: BoxedValue },
    PushSink(Operand),
    Return(Operand),
}

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;

enum ValueType {
    Boxed,
    Bool,
    Num,
    Str,
    None,
}

impl Instr {
    fn result_type(&self) -> ValueType {
        match self {
            Instr::Not(_) => ValueType::Bool,
            Instr::Arith { .. } => ValueType::Num,
            Instr::CmpNum { .. } => ValueType::Bool,
            Instr::AssertTrue { .. } => ValueType::None,
            Instr::AssertEqConst { .. } => ValueType::None,
            Instr::PushSink(_) => ValueType::None,
            Instr::Return(_) => ValueType::None,
            Instr::GetArg(_) => ValueType::Boxed,
            Instr::TraceParam(_) => ValueType::Boxed,
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
}
