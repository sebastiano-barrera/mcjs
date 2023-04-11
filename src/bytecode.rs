use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u32);

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub struct FnId(pub u32);
impl FnId {
    pub const ROOT_FN: FnId = FnId(0);
}

/// NOTE cross-module function calls are unsupported yet
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalIID {
    pub fnid: FnId,
    pub iid: IID,
}

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
    UnaryMinus(Operand),
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
    BoolOp {
        op: BoolOp,
        a: Operand,
        b: Operand,
    },
    JmpIf {
        cond: Operand,
        dest: IID,
    },
    Jmp(IID),
    Set {
        var_id: StaticVarId,
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

    ClosureNew {
        fnid: FnId,
    },

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

    // TODO(big feat) Temporary; should be replaced by objects, just like all other "classes"
    ArrayNew,
    ArrayPush(Operand, Operand),

    TypeOf(Operand),

    StartTrace {
        trace_id: String,
        wait_loop: bool,
    },
}

impl Instr {
    const MAX_OPERANDS: usize = 4;
}

type Operands = crate::util::LimVec<{ Instr::MAX_OPERANDS }, Operand>;

/// A value literal that is allowed to appear in the bytecode.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
    Undefined,
    // TODO(cleanup) Delete, Closure supersedes this
    SelfFunction,
    // TODO(cleanup) Delete, Closure supersedes this
    NativeFunction(u32),
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

type Vars = crate::util::LimVec<{ Instr::MAX_OPERANDS }, StaticVarId>;

impl Instr {
    pub fn read_vars<'a>(&'a self) -> Vars {
        let operands = self.read_operands();
        Vars::from_iter(operands.iter().filter_map(|oper| match oper {
            Operand::Var(var_id) => Some(*var_id),
            _ => None,
        }))
    }

    fn read_operands<'a>(&'a self) -> Operands {
        match *self {
            Instr::Nop => Default::default(),
            Instr::Const(_) => Default::default(),
            Instr::Not(oper) => Operands::from_iter([oper].into_iter()),
            Instr::UnaryMinus(oper) => Operands::from_iter([oper].into_iter()),
            Instr::Arith { op: _, a, b } => Operands::from_iter([a, b].into_iter()),
            Instr::Cmp { op: _, a, b } => Operands::from_iter([a, b].into_iter()),
            Instr::BoolOp { op: _, a, b } => Operands::from_iter([a, b].into_iter()),
            Instr::JmpIf { cond, .. } => Operands::from_iter([cond].into_iter()),
            Instr::Jmp(_) => Default::default(),
            Instr::Set { var_id: _, value } => Operands::from_iter([value].into_iter()),
            Instr::PushSink(arg) => Operands::from_iter([arg].into_iter()),
            Instr::Return(arg) => Operands::from_iter([arg].into_iter()),
            Instr::GetArg(_) => Default::default(),
            Instr::Call {
                callee: _,
                ref args,
            } => Operands::from_iter(args.iter().cloned()),
            Instr::ObjNew => Default::default(),
            Instr::ObjSet { obj, key, value } => Operands::from_iter([obj, key, value].into_iter()),
            Instr::ObjGet { obj, key } => Operands::from_iter([obj, key].into_iter()),
            Instr::ArrayNew => Default::default(),
            Instr::ArrayPush(arr, value) => Operands::from_iter([arr, value].into_iter()),
            Instr::TypeOf(arg) => Operands::from_iter([arg].into_iter()),
            Instr::ClosureNew { .. } => Default::default(),
            Instr::StartTrace { .. } => Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarIndex(pub(crate) u16);

/// Static identifier for a variable, used in the bytecode as operand.
///
/// Designates a variable in the bytecode, not at runtime.  If the function
/// containing the variable declaration is called multiple times (e.g. via
/// recursion, or by producing multiple closures from the same function
/// literal), this identifier is the same every time, but the *runtime*
/// identifier will change.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticVarId {
    pub(crate) fnid: FnId,
    pub(crate) var_ndx: VarIndex,
}

impl std::fmt::Debug for StaticVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "var[{}:{}]", self.fnid.0, self.var_ndx.0)
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, PartialEq)]
pub enum Operand {
    // Evaluates to the value of the indicated variable.
    Var(StaticVarId),
    // Evaluates to the result of the indicated instruction.
    IID(IID),
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IID(iid) => iid.fmt(f),
            Self::Var(varid) => varid.fmt(f),
        }
    }
}
impl From<IID> for Operand {
    fn from(iid: IID) -> Self {
        Operand::IID(iid)
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoolOp {
    And,
    Or,
}

pub struct Module {
    fns: HashMap<FnId, Function>,
}

impl Module {
    pub fn new(fns: HashMap<FnId, Function>) -> Self {
        Module { fns }
    }

    pub fn get_function(&self, fnid: FnId) -> Option<&Function> {
        self.fns.get(&fnid)
    }

    pub(crate) fn dump(&self) {
        eprintln!("=== module");
        for (fnid, func) in self.fns.iter() {
            eprintln!("fn #{} [{} vars]:", fnid.0, func.n_slots);
            for (ndx, instr) in func.instrs.iter().enumerate() {
                let lh = func.loop_heads.get(&IID(ndx as u32));
                eprintln!(
                    "  {:4}{:4}: {:?}",
                    if lh.is_some() { ">>" } else { "" },
                    ndx,
                    instr,
                );
                if let Some(lh) = lh {
                    eprint!("            (phis: ");
                    for var in &lh.interloop_vars {
                        eprint!("{:?}, ", var);
                    }
                    eprintln!(")");
                }
            }
        }
        eprintln!("---");
    }
}

pub struct Function {
    instrs: Box<[Instr]>,
    loop_heads: HashMap<IID, LoopInfo>,
    n_slots: u16,
}
pub struct LoopInfo {
    // Variables that change in value during each cycle, in such a way that
    // each cycle sees the value in  the previous cycle.  Phi instructions are
    // added based on this set.
    interloop_vars: HashSet<StaticVarId>,
}
impl Function {
    pub(crate) fn new(instrs: Box<[Instr]>, n_slots: u16) -> Function {
        let loop_heads = find_loop_heads(&instrs[..]);
        Function {
            instrs,
            loop_heads,
            n_slots,
        }
    }

    pub(crate) fn instrs(&self) -> &[Instr] {
        self.instrs.as_ref()
    }

    pub(crate) fn is_loop_head(&self, iid: IID) -> bool {
        self.loop_heads.contains_key(&iid)
    }

    pub(crate) fn n_slots(&self) -> u16 {
        self.n_slots
    }
}

fn find_loop_heads(instrs: &[Instr]) -> HashMap<IID, LoopInfo> {
    // The set of interloop variables is the set of variables where, within a
    // loop, at least one read happens before a write.
    let mut heads = HashMap::new();

    // It ain't linear, but it does the job (plus I don't think
    // there should be so many nesting levels for loops within the
    // same function...)
    for (end_ndx, inst) in instrs.iter().enumerate() {
        match inst {
            Instr::Jmp(dest) | Instr::JmpIf { dest, .. } if dest.0 as usize <= end_ndx => {
                // Loop goes from end_ndx to dest

                let dest_ndx = dest.0 as usize;
                let mut interloop_vars = HashSet::new();
                let mut reads: HashSet<StaticVarId> = HashSet::new();
                for inst in &instrs[dest_ndx..end_ndx] {
                    reads.extend(inst.read_vars().iter());

                    if let Instr::Set { var_id, .. } = inst {
                        if reads.remove(var_id) {
                            interloop_vars.insert(*var_id);
                        }
                    }
                }

                heads.insert(*dest, LoopInfo { interloop_vars });
            }
            _ => {}
        }
    }

    heads
}
