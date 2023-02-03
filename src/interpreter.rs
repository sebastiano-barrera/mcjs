use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
};

pub use crate::common::Error;
use crate::{
    common::Result,
    jit::{self, InterpreterStep},
};

/// A value that can be input, output, or processed by the program.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
    Undefined,
    SelfFunction,
    LocalFn(FnId),
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u32);

impl std::fmt::Debug for IID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

#[derive(Debug)]
pub enum Instr {
    Nop,
    Const(Value),
    Not(Operand),
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
    JmpIf {
        cond: Operand,
        dest: IID,
    },
    Jmp(IID),
    Set {
        var_id: IID,
        value: Operand,
    },
    PushSink(Operand),
    Return(Operand),
    GetArg(usize),
    Call {
        callee: Operand,
        // smallvec?
        args: Vec<Operand>,
    },
}

#[allow(clippy::upper_case_acronyms)]
pub enum Operand {
    Value(Value),
    IID(IID),
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(val) => val.fmt(f),
            Self::IID(iid) => iid.fmt(f),
        }
    }
}
impl From<IID> for Operand {
    fn from(iid: IID) -> Self {
        Operand::IID(iid)
    }
}
impl From<Value> for Operand {
    fn from(value: Value) -> Self {
        Operand::Value(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp {
    GE,
    GT,
    LT,
    LE,
    EQ,
    NE,
}

pub struct VM {
    modules: HashMap<String, Module>,
    sink: Vec<Value>,
    trace_builder: Option<jit::TraceBuilder>,
}

impl VM {
    pub fn new() -> Self {
        VM {
            modules: HashMap::new(),
            sink: Vec::new(),
            trace_builder: None,
        }
    }

    pub fn take_sink(&mut self) -> Vec<Value> {
        let mut swap_area = Vec::new();
        std::mem::swap(&mut swap_area, &mut self.sink);
        swap_area
    }

    pub fn take_trace(&mut self) -> Option<jit::Trace> {
        self.trace_builder.take().and_then(|tb| tb.build())
    }

    pub fn run_script(&mut self, script_text: String, flags: InterpreterFlags) -> Result<()> {
        let module =
            crate::bytecode_compiler::compile_file("<input>".to_string(), script_text).unwrap();
        #[cfg(test)]
        {
            module.dump();
        }

        if flags.tracer_flags.is_some() {
            self.trace_builder = Some(jit::TraceBuilder::new());
        }
        self.run_module_fn(&module, FnId::ROOT_FN, &[], flags)?;
        Ok(())
    }

    fn run_module_fn(
        &mut self,
        module: &Module,
        fnid: FnId,
        args: &[Value],
        flags: InterpreterFlags,
    ) -> Result<Value> {
        let indent = "     | ".repeat(flags.indent_level as usize);

        let func = module.fns.get(&fnid).unwrap();
        let instrs = &func.instrs;

        let mut values_buf = vec![Value::Undefined; instrs.len()];

        let get_operand = |results: &Vec<Value>, operand: &Operand| match operand {
            Operand::Value(value) => value.clone(),
            Operand::IID(IID(ndx)) => {
                let value = results[*ndx as usize].clone();
                eprintln!("{}     ' {:?} = {:?}", indent, operand, value);
                value
            }
        };

        let mut ndx = 0;
        let mut return_value = None;
        while return_value.is_none() {
            if ndx >= instrs.len() {
                break;
            }
            let instr = &instrs[ndx];
            let mut next_ndx = ndx + 1;

            eprintln!("{}i{:<4} {:?}", indent, ndx, instr);

            match instr {
                Instr::Const(value) => {
                    values_buf[ndx] = value.clone();
                }
                Instr::Arith { op, a, b } => {
                    let a = get_operand(&mut values_buf, a);
                    let b = get_operand(&mut values_buf, b);

                    if let (Value::Number(a), Value::Number(b)) = (&a, &b) {
                        values_buf[ndx] = Value::Number(match op {
                            ArithOp::Add => a + b,
                            ArithOp::Sub => a - b,
                            ArithOp::Mul => a * b,
                            ArithOp::Div => a / b,
                        });
                    } else {
                        panic!("invalid operands for arith op: {:?}; {:?}", a, b);
                    }
                }
                Instr::PushSink(operand) => {
                    let value = get_operand(&mut values_buf, operand);
                    self.sink.push(value);
                }
                Instr::Cmp { op, a, b } => {
                    let a = get_operand(&mut values_buf, a);
                    let b = get_operand(&mut values_buf, b);

                    match (&a, &b) {
                        (Value::Number(a), Value::Number(b)) => {
                            values_buf[ndx] = Value::Bool(match op {
                                CmpOp::GE => a >= b,
                                CmpOp::GT => a > b,
                                CmpOp::LT => a < b,
                                CmpOp::LE => a <= b,
                                CmpOp::EQ => a == b,
                                CmpOp::NE => a != b,
                            });
                        }
                        (Value::String(a), Value::String(b)) => {
                            let ordering = a.cmp(b);
                            values_buf[ndx] = Value::Bool(matches!(
                                (op, ordering),
                                (CmpOp::GE, Ordering::Greater)
                                    | (CmpOp::GE, Ordering::Equal)
                                    | (CmpOp::GT, Ordering::Greater)
                                    | (CmpOp::LT, Ordering::Less)
                                    | (CmpOp::LE, Ordering::Less)
                                    | (CmpOp::LE, Ordering::Equal)
                                    | (CmpOp::EQ, Ordering::Equal)
                                    | (CmpOp::NE, Ordering::Greater)
                                    | (CmpOp::NE, Ordering::Less)
                            ));
                        }

                        _ => {
                            values_buf[ndx] = Value::Bool(false);
                            // panic!("invalid operands for cmp op: {:?}; {:?}", a, b);
                        }
                    }
                }
                Instr::JmpIf { cond, dest } => {
                    let cond_value = get_operand(&mut values_buf, cond);
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0 as usize;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => panic!("invalid if condition (not boolean): {:?}", other),
                    }
                }
                Instr::Set { var_id, value } => {
                    let value = get_operand(&values_buf, value);
                    let var_ndx = var_id.0 as usize;
                    values_buf[var_ndx] = value;
                }
                Instr::Nop => {}
                Instr::Not(value) => {
                    let value = get_operand(&values_buf, value);
                    values_buf[ndx] = match value {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        _ => {
                            Value::Bool(false)

                            // panic!("invalid operand for `not` (not boolean): {:?}", other);
                        }
                    };
                }
                Instr::Jmp(dest) => {
                    next_ndx = dest.0 as usize;
                }
                Instr::Return(value) => {
                    return_value = Some(get_operand(&values_buf, value));
                }
                Instr::GetArg(arg_ndx) => {
                    values_buf[ndx] = args[*arg_ndx].clone();
                }

                Instr::Call { callee, args } => {
                    let callee = get_operand(&values_buf, callee);
                    if let Value::LocalFn(callee_fnid) = callee {
                        let arg_values: Vec<Value> = args
                            .iter()
                            .map(|oper| get_operand(&values_buf, oper))
                            .collect();

                        let ret_val = self.run_module_fn(
                            module,
                            callee_fnid,
                            &arg_values[..],
                            flags.for_call(),
                        )?;

                        values_buf[ndx] = ret_val;
                    } else {
                        panic!("invalid callee (not a function): {:?}", callee);
                    }
                }
            }

            if let Some(trace_builder) = &mut self.trace_builder {
                if flags.tracer_flags.as_ref().unwrap().start_depth == 0 {
                    trace_builder.interpreter_step(&InterpreterStep {
                        values_buf: &values_buf,
                        fnid,
                        func: &func,
                        iid: IID(ndx as u32),
                        next_iid: IID(next_ndx as u32),
                    });
                }
            }

            ndx = next_ndx;
        }

        Ok(return_value.unwrap_or(Value::Undefined))
    }
}

#[derive(Clone)]
pub struct InterpreterFlags {
    pub indent_level: u8,
    pub tracer_flags: Option<TracerFlags>,
}
impl InterpreterFlags {
    fn for_call(&self) -> InterpreterFlags {
        let mut ret = self.clone();
        ret.indent_level += 1;
        if let Some(tf) = &mut ret.tracer_flags {
            if tf.start_depth > 0 {
                tf.start_depth -= 1;
            }
        }
        return ret;
    }
}

#[derive(Clone)]
pub struct TracerFlags {
    pub start_depth: u32,
}

impl Default for InterpreterFlags {
    fn default() -> Self {
        Self {
            indent_level: 0,
            tracer_flags: None,
        }
    }
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub struct FnId(pub u32);
impl FnId {
    pub const ROOT_FN: FnId = FnId(0);
}

pub struct Module {
    fns: HashMap<FnId, Function>,
}

impl Module {
    pub fn new(fns: HashMap<FnId, Function>) -> Self {
        Module { fns }
    }

    pub(crate) fn dump(&self) {
        eprintln!("=== module");
        for (fnid, func) in self.fns.iter() {
            eprintln!("fn #{}():", fnid.0);
            for (ndx, instr) in func.instrs.iter().enumerate() {
                eprintln!(
                    "  {:4}{:4}: {:?}",
                    if func.loop_heads.contains(&IID(ndx as u32)) {
                        ">>"
                    } else {
                        ""
                    },
                    ndx,
                    instr
                );
            }
        }
        eprintln!("---");
    }
}

pub struct Function {
    instrs: Box<[Instr]>,
    loop_heads: HashSet<IID>,
}
impl Function {
    pub(crate) fn new(instrs: Box<[Instr]>) -> Function {
        let loop_heads = find_loop_heads(&instrs[..]);
        Function { instrs, loop_heads }
    }

    pub(crate) fn instrs(&self) -> &[Instr] {
        self.instrs.as_ref()
    }

    pub(crate) fn is_loop_head(&self, iid: IID) -> bool {
        self.loop_heads.contains(&iid)
    }
}

fn find_loop_heads(instrs: &[Instr]) -> HashSet<IID> {
    let mut heads = HashSet::new();
    for (ndx, inst) in instrs.iter().enumerate() {
        match inst {
            Instr::Jmp(IID(dest_ndx))
            | Instr::JmpIf {
                dest: IID(dest_ndx),
                ..
            } if *dest_ndx as usize <= ndx => {
                heads.insert(IID(*dest_ndx));
            }
            _ => {}
        }
    }

    heads
}
