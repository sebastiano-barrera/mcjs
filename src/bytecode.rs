use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u32);

impl std::fmt::Display for IID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

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

pub type ConstIndex = u16;
pub type CaptureIndex = u16;

#[derive(Debug)]
pub enum Instr {
    Nop,
    Const(Value),

    SetVar {
        var: IID,
        value: IID,
    },
    GetCapture(CaptureIndex),

    Not(IID),
    UnaryMinus(IID),
    Arith {
        op: ArithOp,
        a: IID,
        b: IID,
    },
    Cmp {
        op: CmpOp,
        a: IID,
        b: IID,
    },
    BoolOp {
        op: BoolOp,
        a: IID,
        b: IID,
    },
    JmpIf {
        cond: IID,
        dest: IID,
    },
    Jmp(IID),
    PushSink(IID),
    Return(IID),
    GetArg(usize),
    Call {
        callee: IID,
        // smallvec?
        args: Vec<IID>,
    },

    ClosureNew {
        fnid: FnId,
    },
    ClosureAddCapture(IID),
    GetNativeFn(NativeFnId),

    ObjNew,
    ObjSet {
        obj: IID,
        key: IID,
        value: IID,
    },
    ObjGet {
        obj: IID,
        key: IID,
    },

    // TODO(big feat) Temporary; should be replaced by objects, just like all other "classes"
    ArrayNew,
    ArrayPush(IID, IID),

    TypeOf(IID),
}

impl Instr {
    const MAX_OPERANDS: usize = 4;
}

type Operands = crate::util::LimVec<{ Instr::MAX_OPERANDS }, IID>;

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
}

pub type NativeFnId = u32;

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

type Vars = crate::util::LimVec<{ Instr::MAX_OPERANDS }, IID>;

impl Instr {
    fn read_operands<'a>(&'a self) -> Operands {
        match *self {
            Instr::Not(oper) => Operands::from_iter([oper].into_iter()),
            Instr::UnaryMinus(oper) => Operands::from_iter([oper].into_iter()),
            Instr::Arith { op: _, a, b } => Operands::from_iter([a, b].into_iter()),
            Instr::Cmp { op: _, a, b } => Operands::from_iter([a, b].into_iter()),
            Instr::BoolOp { op: _, a, b } => Operands::from_iter([a, b].into_iter()),
            Instr::JmpIf { cond, .. } => Operands::from_iter([cond].into_iter()),
            Instr::SetVar { var: _, value } => Operands::from_iter([value].into_iter()),
            Instr::PushSink(arg) => Operands::from_iter([arg].into_iter()),
            Instr::Return(arg) => Operands::from_iter([arg].into_iter()),
            Instr::Call {
                callee: _,
                ref args,
            } => Operands::from_iter(args.iter().cloned()),
            Instr::ObjSet { obj, key, value } => Operands::from_iter([obj, key, value].into_iter()),
            Instr::ObjGet { obj, key } => Operands::from_iter([obj, key].into_iter()),
            Instr::ArrayPush(arr, value) => Operands::from_iter([arr, value].into_iter()),
            Instr::TypeOf(arg) => Operands::from_iter([arg].into_iter()),
            Instr::ClosureAddCapture(value) => Operands::from_iter([value].into_iter()),

            Instr::GetNativeFn(_) => Default::default(),
            Instr::ClosureNew { .. } => Default::default(),
            Instr::ArrayNew => Default::default(),
            Instr::ObjNew => Default::default(),
            Instr::GetArg(_) => Default::default(),
            Instr::GetCapture(_) => Default::default(),
            Instr::Jmp(_) => Default::default(),
            Instr::Const(_) => Default::default(),
            Instr::Nop => Default::default(),
        }
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
            eprintln!("fn #{}:", fnid.0);
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
    // TODO(performance) following elision of Operand, better data structures
    loop_heads: HashMap<IID, LoopInfo>,
    trace_anchors: HashMap<IID, TraceAnchor>,
}
pub struct TraceAnchor {
    pub trace_id: String,
}
pub struct LoopInfo {
    // Variables that change in value during each cycle, in such a way that
    // each cycle sees the value in  the previous cycle.  Phi instructions are
    // added based on this set.
    interloop_vars: HashSet<IID>,
}
impl Function {
    pub(crate) fn new(instrs: Box<[Instr]>, trace_anchors: HashMap<IID, TraceAnchor>) -> Function {
        let loop_heads = find_loop_heads(&instrs[..]);
        Function {
            instrs,
            loop_heads,
            trace_anchors,
        }
    }

    pub(crate) fn instrs(&self) -> &[Instr] {
        self.instrs.as_ref()
    }

    pub(crate) fn is_loop_head(&self, iid: IID) -> bool {
        self.loop_heads.contains_key(&iid)
    }

    pub(crate) fn get_trace_anchor(&self, iid: IID) -> Option<&TraceAnchor> {
        self.trace_anchors.get(&iid)
    }

    pub(crate) fn trace_start_id(&self, iid: IID) -> Option<&str> {
        self.trace_anchors
            .get(&iid)
            .map(|tanch| tanch.trace_id.as_str())
    }
}

fn find_loop_heads(instrs: &[Instr]) -> HashMap<IID, LoopInfo> {
    // The set of interloop variables is the set of variables where, within a
    // loop, at least one read happens before a write.
    let mut heads = HashMap::new();

    // TODO(small feat) This CAN be linear, can't it?
    // It ain't linear, but it does the job (plus I don't think
    // there should be so many nesting levels for loops within the
    // same function...)
    for (end_ndx, inst) in instrs.iter().enumerate() {
        match inst {
            Instr::Jmp(dest) | Instr::JmpIf { dest, .. } if dest.0 as usize <= end_ndx => {
                // Loop goes from end_ndx to dest
                let dest_ndx = dest.0 as usize;
                let mut interloop_vars = HashSet::new();
                for ndx in dest_ndx..end_ndx {
                    let inst = &instrs[ndx];
                    if let Instr::SetVar { var, .. } = inst {
                        let var_ndx = var.0 as usize;
                        assert!(var_ndx < ndx);

                        if var_ndx >= dest_ndx {
                            interloop_vars.insert(*var);
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
