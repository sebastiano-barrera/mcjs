use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use swc_atoms::JsWord;

// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u16);

impl std::fmt::Display for IID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub struct FnId(pub u32);

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
pub type ArgIndex = u8;
pub type CaptureIndex = u16;

pub type VReg = u16;

/// Global ID of a module.  Can be used, among other things, to fetch the Module object
/// from importing modules.
// 64K module ought to be enough for everyone, node_modules not withstanding
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct ModuleId(pub u16);

#[derive(Debug)]
pub enum Instr {
    Nop,
    /// Load constant in accu
    LoadConst(ConstIndex),
    LoadNull,
    LoadUndefined,
    LoadArg(u8),

    /// Store: accumulator -> register
    StoAR(VReg),
    /// Load: register -> accumulator
    LoadRA(VReg),

    /// Load closure capture in accu
    LoadCapture(CaptureIndex),

    BoolNot,
    UnaryMinus,
    Arith(ArithOp, VReg),
    Cmp(CmpOp, VReg),
    BoolOp(BoolOp, VReg),

    JmpIf {
        dest: IID,
    },
    Jmp(IID),
    PushToSink,
    Return,

    // Push the value of accu to the argument list for the next Call
    CallArg,
    Call,

    ClosureNew {
        fnid: FnId,
    },
    ClosureAddCapture(VReg),
    GetNativeFn(NativeFnId),

    ObjNew,
    ObjSet {
        obj: VReg,
        key: VReg,
        // Implicitly, value = accu
    },
    ObjGet {
        key: VReg,
    },
    ObjGetKeys,

    // TODO: Remove this bytecode (should be implemented as a method with a native impl)
    ArrayPush(VReg /* array */),
    ArrayNth(VReg),
    ArraySetNth {
        index: VReg,
        value: VReg,
    },
    ArrayLen,

    TypeOf,

    GetModule(ModuleId),
}

impl Instr {
    const MAX_OPERANDS: usize = 4;
}

type Operands = crate::util::LimVec<{ Instr::MAX_OPERANDS }, VReg>;

/// A value literal that is allowed to appear in the bytecode.
#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(f64),
    String(String),
    Bool(bool),
    Null,
    Undefined,
    // TODO(cleanup) Delete, Closure supersedes this
    SelfFunction,
}

pub type NativeFnId = u32;

impl From<String> for Literal {
    fn from(value: String) -> Self {
        Literal::String(value)
    }
}

type Vars = crate::util::LimVec<{ Instr::MAX_OPERANDS }, IID>;

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

pub struct Codebase {
    fns: HashMap<FnId, Function>,
    rootfn_of_module: HashMap<ModuleId, FnId>,
}

impl Codebase {
    pub fn new(fns: HashMap<FnId, Function>, rootfn_of_module: HashMap<ModuleId, FnId>) -> Self {
        Codebase {
            fns,
            rootfn_of_module,
        }
    }

    pub fn get_function(&self, fnid: FnId) -> Option<&Function> {
        self.fns.get(&fnid)
    }

    pub fn get_module_root_fn(&self, module_id: ModuleId) -> Option<FnId> {
        self.rootfn_of_module.get(&module_id).copied()
    }

    pub fn all_functions(&self) -> impl Iterator<Item = (FnId, &Function)> {
        self.fns.iter().map(|(fnid, func_ref)| (*fnid, func_ref))
    }

    pub fn dump(&self) {
        eprintln!("=== code base");
        for (fnid, func) in self.fns.iter() {
            eprintln!("fn #{}:", fnid.0);
            for (ndx, instr) in func.instrs.iter().enumerate() {
                let lh = func.loop_heads.get(&IID(ndx as _));
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

            let size_code = func.instrs().len() * std::mem::size_of::<Instr>();
            let size_data = func.consts().len() * std::mem::size_of::<Literal>();
            eprintln!("size of code: ............ {}", size_code);
            eprintln!("size of const data: ...... {}", size_data);
            eprintln!("total size: .............. {}", size_code + size_data);
            eprintln!("---");
        }
    }
}

pub struct Function {
    instrs: Box<[Instr]>,
    consts: Box<[Literal]>,
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
    pub(crate) fn new(
        instrs: Box<[Instr]>,
        consts: Box<[Literal]>,
        trace_anchors: HashMap<IID, TraceAnchor>,
    ) -> Function {
        #[cfg(to_be_rewritten)]
        let loop_heads = find_loop_heads(&instrs[..]);
        #[cfg(not(to_be_rewritten))]
        let loop_heads = HashMap::new();
        Function {
            instrs,
            consts,
            loop_heads,
            trace_anchors,
        }
    }

    pub(crate) fn instrs(&self) -> &[Instr] {
        self.instrs.as_ref()
    }

    pub(crate) fn consts(&self) -> &[Literal] {
        self.consts.as_ref()
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

#[cfg(to_be_rewritten)]
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
