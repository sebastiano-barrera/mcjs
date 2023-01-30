use std::collections::HashMap;
use std::fmt::Debug;

use crate::interpreter::{self, IID};

use dynasm::dynasm;

// This is going to be changed at some point
type BoxedValue = interpreter::Value;

trait TypeTag: 'static + Sized + Debug {
    type Value: Debug;
    fn check_type(interpreter_value: &interpreter::Value) -> Option<Self::Value>;
}

#[derive(Debug)]
struct TNum;
impl TypeTag for TNum {
    type Value = f64;

    fn check_type(interpreter_value: &interpreter::Value) -> Option<Self::Value> {
        if let interpreter::Value::Number(num) = interpreter_value {
            Some(*num)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct TBool;
impl TypeTag for TBool {
    type Value = bool;

    fn check_type(interpreter_value: &interpreter::Value) -> Option<Self::Value> {
        if let interpreter::Value::Bool(bool_val) = interpreter_value {
            Some(*bool_val)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct TStr;
impl TypeTag for TStr {
    type Value = String;

    fn check_type(interpreter_value: &interpreter::Value) -> Option<Self::Value> {
        if let interpreter::Value::String(s) = interpreter_value {
            Some(s.clone())
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct TBoxed;
impl TypeTag for TBoxed {
    type Value = BoxedValue;

    fn check_type(interpreter_value: &interpreter::Value) -> Option<Self::Value> {
        Some(interpreter_value.clone())
    }
}

#[derive(Debug)]
struct TNone;
impl TypeTag for TNone {
    type Value = ();

    fn check_type(_: &interpreter::Value) -> Option<()> {
        None
    }
}

#[derive(Debug)]
enum Operand<Tag: TypeTag> {
    ValueId(ValueId),
    Const(Tag::Value),
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

struct TypeError;

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

    fn resolve_interpreter_operand<ReqTag: TypeTag>(
        &self,
        operand: &interpreter::Operand,
        value_buf: &Vec<interpreter::Value>,
    ) -> Option<Operand<ReqTag>> {
        match operand {
            interpreter::Operand::Value(value) => {
                ReqTag::check_type(value).map(|x| Operand::Const(x))
            }

            interpreter::Operand::IID(iid) => {
                if let Some(value_id) = self.var_iid.get(&(self.stack_depth, *iid)) {
                    let instr = &self.instrs[value_id.0 as usize];
                    if instr.result_type_is::<ReqTag>() {
                        Some(Operand::ValueId(*value_id))
                    } else {
                        None
                    }
                } else {
                    ReqTag::check_type(&value_buf[iid.0 as usize]).map(|x| Operand::Const(x))
                }
            }
        }
    }

    pub(crate) fn interpreter_step(&mut self, step: &InterpreterStep) {
        if self.trace_ended {
            return;
        }

        macro_rules! resolve {
            ($type_tag:ty, $operand:expr) => {
                match self.resolve_interpreter_operand::<$type_tag>($operand, &step.values_buf) {
                    None => {
                        // Type error => fail the trace
                        self.failed = true;
                        self.trace_ended = true;
                        return;
                    }
                    Some(operand) => operand,
                }
            };
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
                if let Operand::ValueId(vid) = resolve!(TBool, &oper) {
                    self.instrs.push(Instr::Not(vid));
                }
            }
            interpreter::Instr::Arith { op, a, b } => {
                let a = resolve!(TNum, &a);
                let b = resolve!(TNum, &b);
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
                let a = resolve!(TNum, &a);
                let b = resolve!(TNum, &b);
                match (a, b) {
                    (Operand::Const(_), Operand::Const(_)) => {
                        // constant folding
                    }
                    (a, b) => {
                        self.emit(Instr::CmpNum { op: *op, a, b });
                    }
                }
            }
            interpreter::Instr::JmpIf { cond, dest } => {
                let cond = resolve!(TBool, &cond);
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
                if let interpreter::Operand::IID(vid) = value {
                    if let Some(value_id) = self.var_iid.get(&(self.stack_depth, *vid)) {
                        self.var_iid
                            .insert((self.stack_depth, *var_id), value_id.clone());
                    }
                }
            }
            interpreter::Instr::PushSink(value) => {
                let value = resolve!(TBoxed, &value);
                self.emit(Instr::PushSink(value));
            }
            interpreter::Instr::Return(value) => {
                let value = resolve!(TBoxed, &value);
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
        a: Operand<TNum>,
        b: Operand<TNum>,
    },
    CmpNum {
        op: CmpOp,
        a: Operand<TNum>,
        b: Operand<TNum>,
    },
    ExitUnless {
        cond: ValueId,
        dest: interpreter::IID,
    },
    PushSink(Operand<TBoxed>),
    Return(Operand<TBoxed>),
}

type ArithOp = interpreter::ArithOp;
type CmpOp = interpreter::CmpOp;

impl Instr {
    fn result_type_is<ReqTag: TypeTag>(&self) -> bool {
        std::any::TypeId::of::<ReqTag>() == self.type_tag()
    }

    fn type_tag(&self) -> std::any::TypeId {
        match self {
            Instr::Not(_) => std::any::TypeId::of::<TBool>(),
            Instr::Arith { op, a, b } => std::any::TypeId::of::<TNum>(),
            Instr::CmpNum { op, a, b } => std::any::TypeId::of::<TBool>(),
            Instr::ExitUnless { cond, dest } => std::any::TypeId::of::<TNone>(),
            Instr::PushSink(_) => std::any::TypeId::of::<TNone>(),
            Instr::Return(_) => std::any::TypeId::of::<TNone>(),
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
