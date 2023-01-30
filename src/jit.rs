use std::collections::HashMap;

use crate::interpreter::{self, IID};

use dynasm::dynasm;

// This is going to be changed at some point
type Value = interpreter::Value;

#[derive(Debug)]
enum Operand {
    ValueId(ValueId),
    Const(Value),
}

#[derive(Clone, Copy, Debug)]
pub struct ValueId(u32);

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
    var_iid: HashMap<(u32, interpreter::IID), ValueId>,
    stack_depth: u32,

    instrs: Vec<Instr>,

    failed: bool,
    trace_ended: bool,
}

impl TraceBuilder {
    pub fn new() -> Self {
        TraceBuilder {
            var_iid: HashMap::new(),
            stack_depth: 0,
            instrs: Vec::new(),
            failed: false,
            trace_ended: false,
        }
    }

    fn resolve_interpreter_operand(
        &self,
        operand: &interpreter::Operand,
        value_buf: &Vec<interpreter::Value>,
    ) -> Operand {
        match operand {
            interpreter::Operand::Value(value) => Operand::Const(value.clone()),
            interpreter::Operand::IID(iid) => {
                if let Some(value_id) = self.var_iid.get(&(self.stack_depth, *iid)) {
                    Operand::ValueId(*value_id)
                } else {
                    Operand::Const(value_buf[iid.0 as usize].clone())
                }
            }
        }
    }

    pub(crate) fn interpreter_step(&mut self, step: &InterpreterStep) {
        if self.trace_ended {
            return;
        }

        let instr = &step.func.instrs()[step.iid.0 as usize];
        eprintln!("TB: step: {:?}", instr);

        if step.func.is_loop_head(step.iid) {
            eprintln!("TB: instr is loop head. aborting.");
            // loops not supported yet!
            self.failed = true;
            self.trace_ended = true;
            return;
        }

        match instr {
            interpreter::Instr::Nop => {}
            interpreter::Instr::Const(_) => {
                // Nothing to do. In later instructions, we can fetch the value
                // from the interpreter's value buffer.  This is equivalent to
                // constant folding.
            }
            interpreter::Instr::Not(oper) => {
                // if the operand resolves to a constant, we can just skip it
                // (functions as constant folding)
                if let Operand::ValueId(vid) =
                    self.resolve_interpreter_operand(&oper, &step.values_buf)
                {
                    self.instrs.push(Instr::Not(vid));
                }
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = self.resolve_interpreter_operand(&a, &step.values_buf);
                let b = self.resolve_interpreter_operand(&b, &step.values_buf);
                match (a, b) {
                    (Operand::Const(_), Operand::Const(_)) => {
                        // constant folding
                    }
                    (a, b) => {
                        self.emit(Instr::Arith { op: *op, a, b });
                    }
                }
            }
            interpreter::Instr::Cmp { op, a, b } => {
                let a = self.resolve_interpreter_operand(&a, &step.values_buf);
                let b = self.resolve_interpreter_operand(&b, &step.values_buf);
                match (a, b) {
                    (Operand::Const(_), Operand::Const(_)) => {
                        // constant folding
                    }
                    (a, b) => {
                        self.emit(Instr::Cmp { op: *op, a, b });
                    }
                }
            }
            interpreter::Instr::JmpIf { cond, dest } => {
                let cond = self.resolve_interpreter_operand(&cond, &step.values_buf);
                match cond {
                    Operand::Const(_) => {
                        // constant folding
                    }
                    Operand::ValueId(vid) => {
                        self.emit(Instr::ExitUnless {
                            cond: vid,
                            dest: *dest,
                        });
                    }
                }
            }
            interpreter::Instr::Jmp(_) => {
                // unconditional jump.  Nothing to do, let's just follow the interpreter to the next instruction
            }
            interpreter::Instr::Set { var_id, value } => {
                let value = self.resolve_interpreter_operand(&value, &step.values_buf);
                match value {
                    Operand::ValueId(vid) => {
                        self.var_iid.insert((self.stack_depth, *var_id), vid);
                    }
                    Operand::Const(_) => {
                        // constant folding
                    }
                };
            }
            interpreter::Instr::PushSink(value) => {
                let value = self.resolve_interpreter_operand(&value, &step.values_buf);
                self.emit(Instr::PushSink(value));
            }
            interpreter::Instr::Return(value) => {
                let value = self.resolve_interpreter_operand(&value, &step.values_buf);
                self.emit(Instr::Return(value));
                self.trace_ended = true;
            }
            interpreter::Instr::GetArg(_) => todo!(),
            interpreter::Instr::Call { callee, args } => todo!(),
        }
    }

    fn emit(&mut self, instr: Instr) -> ValueId {
        let vid = ValueId(self.instrs.len() as u32);
        eprintln!("TB: emit: {:?}", instr);
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
}

pub struct InterpreterStep<'a> {
    pub(crate) values_buf: &'a Vec<interpreter::Value>,
    pub(crate) fnid: interpreter::FnId,
    pub(crate) func: &'a interpreter::Function,
    pub(crate) iid: interpreter::IID,
    pub(crate) next_iid: interpreter::IID,
}

#[derive(Debug)]
enum Instr {
    Not(ValueId),
    Arith {
        op: ArithOp,
        a: Operand,
        b: Operand,
    },
    Cmp {
        op: CmpOp,
        a: Operand,
        b: Operand,
    },
    ExitUnless {
        cond: ValueId,
        dest: interpreter::IID,
    },
    PushSink(Operand),
    Return(Operand),
}

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;

// Number(f64),
// String(String),
// Bool(bool),
// Null,
// Undefined,
// SelfFunction,
// LocalFn(FnId),

// A JIT trace, in SSA representation.
pub struct Trace {
    instrs: Vec<Instr>,
}

impl Trace {
    #[cfg(test)]
    pub(crate) fn dump(&self) {
        eprintln!("\n");
        for (ndx, instr) in self.instrs.iter().enumerate() {
            eprintln!("{:4} {:?}\n", ndx, instr);
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

        NativeThunk { buf, entry_offset }
    }
}

pub struct NativeThunk {
    buf: dynasmrt::ExecutableBuffer,
    entry_offset: dynasmrt::AssemblyOffset,
}

impl NativeThunk {
    pub(crate) fn run(&self) -> u64 {
        let ptr = self.buf.ptr(self.entry_offset);
        let thunk: extern "C" fn() -> u64 = unsafe { std::mem::transmute(ptr) };
        thunk()
    }
}
