use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    path::PathBuf,
};

use crate::common::{Error, Result};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u32);

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
            Self::IID(iid) => write!(f, "v{}", iid.0),
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

#[derive(Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug)]
pub enum CmpOp {
    GE,
    GT,
    LT,
    LE,
    EQ,
    NE,
}

pub struct Config {
    pub include_paths: Vec<PathBuf>,
}

pub struct VM {
    cfg: Config,
    module: Module,
    sink: RefCell<Vec<Value>>,
}

impl VM {
    pub fn new(module: Module, cfg: Config) -> Self {
        VM {
            cfg,
            module,
            sink: RefCell::new(Vec::new()),
        }
    }

    #[cfg(disabled)]
    pub fn require(&mut self, require_path: &str) -> Result<()> {
        let filename = format!("{}.js", require_path);
        let include_paths = &self.cfg.include_paths;
        let file_path = include_paths
            .iter()
            .map(|include_path| include_path.join(&filename))
            .find(|path| path.is_file())
            .ok_or(Error::FileNotFound)?;

        let mut program_text = String::new();
        std::fs::File::open(file_path)?.read_to_string(&mut program_text)?;

        parser::parse_file(filename, program_text);
        // eprintln!("Parse tree:\n{:?}", parse_tree);
        // Ok(())
    }

    pub fn take_sink(&mut self) -> Vec<Value> {
        let mut cur_sink = self.sink.borrow_mut();
        let mut old_sink = Vec::new();
        std::mem::swap(&mut old_sink, &mut cur_sink);
        old_sink
    }

    pub fn interpret(&mut self) -> Result<Value> {
        self.interpret_fn(FnId::ROOT_FN, &[])
    }

    fn interpret_fn(&self, fnid: FnId, args: &[Value]) -> Result<Value> {
        let func = self.module.fns.get(&fnid).expect("invalid function ID");
        let instrs = &func.instrs;

        let mut results = vec![Value::Null; instrs.len()];

        let get_operand = |results: &Vec<Value>, operand: &Operand| {
            let value = match operand {
                Operand::Value(value) => value.clone(),
                Operand::IID(IID(ndx)) => results[*ndx as usize].clone(),
            };
            eprintln!("   get {:?} -> {:?}", operand, value);
            value
        };

        let mut ndx = 0;
        loop {
            if ndx >= instrs.len() {
                break;
            }
            let instr = &instrs[ndx];
            let mut next_ndx = ndx + 1;

            eprintln!("{:4} {:?}", ndx, instr);

            match instr {
                Instr::Const(value) => {
                    results[ndx] = value.clone();
                }
                Instr::Arith { op, a, b } => {
                    let a = get_operand(&mut results, a);
                    let b = get_operand(&mut results, b);

                    if let (Value::Number(a), Value::Number(b)) = (&a, &b) {
                        results[ndx] = Value::Number(match op {
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
                    let value = get_operand(&mut results, operand);
                    self.sink.borrow_mut().push(value);
                }
                Instr::Cmp { op, a, b } => {
                    let a = get_operand(&mut results, a);
                    let b = get_operand(&mut results, b);

                    if let (Value::Number(a), Value::Number(b)) = (&a, &b) {
                        results[ndx] = Value::Bool(match op {
                            CmpOp::GE => a >= b,
                            CmpOp::GT => a > b,
                            CmpOp::LT => a < b,
                            CmpOp::LE => a <= b,
                            CmpOp::EQ => a == b,
                            CmpOp::NE => a != b,
                        });
                    } else {
                        panic!("invalid operands for cmp op: {:?}; {:?}", a, b);
                    }
                }
                Instr::JmpIf { cond, dest } => {
                    let cond_value = get_operand(&mut results, cond);
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0 as usize;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => panic!("invalid if condition (not boolean): {:?}", other),
                    }
                }
                Instr::Set { var_id, value } => {
                    let value = get_operand(&results, value);
                    let var_ndx = var_id.0 as usize;
                    results[var_ndx] = value;
                }
                Instr::Nop => {}
                Instr::Not(value) => {
                    let value = get_operand(&results, value);
                    results[ndx] = match value {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        other => panic!("invalid operand for `not` (not boolean): {:?}", other),
                    };
                }
                Instr::Jmp(dest) => {
                    next_ndx = dest.0 as usize;
                }
                Instr::Return(value) => return Ok(get_operand(&results, value)),
                Instr::GetArg(arg_ndx) => {
                    results[ndx] = args[*arg_ndx].clone();
                }

                Instr::Call { callee, args } => {
                    let callee = get_operand(&results, callee);
                    if let Value::LocalFn(callee_fnid) = callee {
                        let arg_values: Vec<Value> = args
                            .iter()
                            .map(|oper| get_operand(&results, oper))
                            .collect();
                        let ret_val = self.interpret_fn(callee_fnid, &arg_values[..])?;
                        results[ndx] = ret_val;
                    } else {
                        panic!("invalid callee (not a function): {:?}", callee);
                    }
                }
            }

            ndx = next_ndx;
        }

        Ok(Value::Undefined)
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
}

pub struct Function {
    instrs: Vec<Instr>,
}
impl Function {
    pub(crate) fn new(instrs: Vec<Instr>) -> Function {
        Function { instrs }
    }
}
