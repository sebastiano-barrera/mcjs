use std::{
    cell::RefCell,
    cmp::Ordering,
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

pub struct VM<'a> {
    module: &'a Module,
    sink: Vec<Value>,
}

pub struct Output {
    pub sink: Vec<Value>,
}

pub fn interpret(module: &Module) -> Result<Output> {
    interpret_with_flags(module, Default::default())
}

pub fn interpret_and_trace(module: &Module) -> Result<Output> {
    let flags = InterpreterFlags {
        create_trace: true,
        ..Default::default()
    };
    interpret_with_flags(module, flags)
}

fn interpret_with_flags(module: &Module, flags: InterpreterFlags) -> Result<Output> {
    let mut vm = VM {
        module,
        sink: Vec::new(),
    };
    vm.interpret_fn(FnId::ROOT_FN, &[], flags)?;
    Ok(Output { sink: vm.sink })
}

impl<'a> VM<'a> {
    fn interpret_fn(
        &mut self,
        fnid: FnId,
        args: &[Value],
        flags: InterpreterFlags,
    ) -> Result<Value> {
        let indent = "     | ".repeat(flags.indent_level as usize);

        let func = self.module.fns.get(&fnid).expect("invalid function ID");
        let instrs = &func.instrs;

        let mut results = vec![Value::Null; instrs.len()];

        let get_operand = |results: &Vec<Value>, operand: &Operand| match operand {
            Operand::Value(value) => value.clone(),
            Operand::IID(IID(ndx)) => {
                let value = results[*ndx as usize].clone();
                eprintln!("{}     ' {:?} = {:?}", indent, operand, value);
                value
            }
        };

        let mut ndx = 0;
        loop {
            if ndx >= instrs.len() {
                break;
            }
            let instr = &instrs[ndx];
            let mut next_ndx = ndx + 1;

            eprintln!("{}{:4} {:?}", indent, ndx, instr);

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

                    match (&a, &b) {
                        (Value::Number(a), Value::Number(b)) => {
                            results[ndx] = Value::Bool(match op {
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
                            results[ndx] = Value::Bool(matches!(
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
                            panic!("invalid operands for cmp op: {:?}; {:?}", a, b);
                        }
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
                        let flags = InterpreterFlags {
                            indent_level: flags.indent_level + 1,
                            ..flags
                        };
                        let ret_val = self.interpret_fn(callee_fnid, &arg_values[..], flags)?;
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

#[derive(Clone, Copy)]
struct InterpreterFlags {
    indent_level: u8,
    create_trace: bool,
}

impl Default for InterpreterFlags {
    fn default() -> Self {
        Self {
            indent_level: 0,
            create_trace: false,
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
}

pub struct Function {
    instrs: Vec<Instr>,
}
impl Function {
    pub(crate) fn new(instrs: Vec<Instr>) -> Function {
        Function { instrs }
    }
}
