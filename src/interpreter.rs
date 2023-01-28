use std::{collections::VecDeque, path::PathBuf};

use crate::common::{Error, Result};

/// A value that can be input, output, or processed by the program.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
    Undefined,
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

// Instruction ID. Can identify an instruction, or its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IID(pub u32);

pub enum Instr {
    Nop,
    Const(Value),
    Not(Operand),
    Arith { op: ArithOp, a: Operand, b: Operand },
    Cmp { op: CmpOp, a: Operand, b: Operand },
    JmpIf { cond: Operand, dest: IID },
    Jmp(IID),
    Set { var_id: IID, value: Operand },
    PushSink(Operand),
}

pub enum Operand {
    Value(Value),
    IID(IID),
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

pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

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
    sink: Vec<Value>,
}

impl VM {
    pub fn new(module: Module, cfg: Config) -> Self {
        VM {
            cfg,
            module,
            sink: Vec::new(),
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
        let mut old_sink = Vec::new();
        std::mem::swap(&mut old_sink, &mut self.sink);
        old_sink
    }

    pub fn interpret(&mut self) -> Result<()> {
        let instrs = &self.module.instrs;

        let mut results = vec![Value::Null; instrs.len()];

        let get_operand = |results: &Vec<Value>, operand: &Operand| match operand {
            Operand::Value(value) => value.clone(),
            Operand::IID(IID(ndx)) => results[*ndx as usize].clone(),
        };

        let mut ndx = 0;
        loop {
            if ndx >= instrs.len() {
                break;
            }
            let instr = &instrs[ndx];
            let mut next_ndx = ndx + 1;

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
                    self.sink.push(value);
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
            }

            ndx = next_ndx;
        }

        Ok(())
    }
}

pub struct Module {
    instrs: Vec<Instr>,
}

impl Module {
    pub fn new(instrs: Vec<Instr>) -> Self {
        Module { instrs }
    }
}
