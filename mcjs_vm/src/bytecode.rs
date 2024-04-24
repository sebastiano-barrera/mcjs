use std::collections::{HashMap, HashSet};

use serde::Serialize;
use strum::IntoStaticStr;
use swc_common::Span;

pub use swc_atoms::JsWord;

// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u16);

impl std::fmt::Display for IID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}
impl std::fmt::Debug for IID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

/// Global ID of a function.
///
/// This ID is sufficient to identify a function across the entire loaded
/// codebase, regardless of the module it belongs to.
// TODO Merge FnId and LocalFnId (maybe make the integer wider).
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct FnId(pub u32);

impl std::fmt::Debug for FnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct GlobalIID(pub FnId, pub IID);

impl std::fmt::Debug for GlobalIID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}:{:?}", self.0, self.1)
    }
}

impl GlobalIID {
    pub fn parse_string(s: &str) -> Option<Self> {
        let (fnid_s, iid_s) = s.split_once('.')?;

        let fnid = fnid_s.parse().ok()?;
        let iid = iid_s.parse().ok()?;

        Some(GlobalIID(
            FnId(fnid),
            IID(iid),
        ))
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct ConstIndex(pub u16);

#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize)]
pub struct ArgIndex(pub u8);

/// Maximum number of (inline) arguments that can be passed to a function
///
/// The first few arguments are passed directly in the first few registers of
/// the callee stack frame. These first registers are reserved for this use,
/// and they must not be used for temporary values or local variables. This
/// guarantee is provided by the bytecode compiler.
///
/// NOTE: The interpreter currently won't allow calls with more than this number
/// of arguments.  In the future (hopefully not too far later) extra arguments
/// will be passed via an array on the heap.
pub const ARGS_COUNT_MAX: u8 = 8;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize)]
pub struct CaptureIndex(pub u16);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VReg(pub u8);

impl std::fmt::Debug for ArgIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arg[{}]", self.0)
    }
}
impl std::fmt::Debug for ConstIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "const[{}]", self.0)
    }
}
impl std::fmt::Debug for CaptureIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "capture[{}]", self.0)
    }
}
impl std::fmt::Debug for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, IntoStaticStr)]
pub enum Instr {
    Nop,
    LoadConst(VReg, ConstIndex),
    LoadNull(VReg),
    LoadUndefined(VReg),
    LoadCapture(VReg, CaptureIndex),
    LoadArg(VReg, ArgIndex),
    LoadThis(VReg),

    Copy {
        dst: VReg,
        src: VReg,
    },
    GetGlobalThis(VReg),
    GetGlobal {
        dest: VReg,
        name: ConstIndex,
    },

    BoolNot {
        dest: VReg,
        arg: VReg,
    },
    UnaryMinus {
        dest: VReg,
        arg: VReg,
    },

    // either numeric addition or string concatentation (chosen at runtime)
    OpAdd(VReg, VReg, VReg),
    ArithSub(VReg, VReg, VReg),
    ArithMul(VReg, VReg, VReg),
    ArithDiv(VReg, VReg, VReg),
    ArithInc(VReg, VReg),
    ArithDec(VReg, VReg),

    CmpGE(VReg, VReg, VReg),
    CmpGT(VReg, VReg, VReg),
    CmpLT(VReg, VReg, VReg),
    CmpLE(VReg, VReg, VReg),
    CmpEQ(VReg, VReg, VReg),
    CmpNE(VReg, VReg, VReg),

    BoolOpAnd(VReg, VReg, VReg),
    BoolOpOr(VReg, VReg, VReg),
    IsInstanceOf(VReg, VReg, VReg),

    JmpIf {
        cond: VReg,
        dest: IID,
    },
    Jmp(IID),
    PushToSink(VReg),
    Return(VReg),

    // Push the value of accu to the argument list for the next Call
    Call {
        return_value: VReg,
        this: VReg,
        callee: VReg,
    },
    CallArg(VReg),

    ClosureNew {
        dest: VReg,
        fnid: FnId,
        forced_this: Option<VReg>,
    },
    ClosureAddCapture(VReg),
    Unshare(VReg),

    ObjCreateEmpty(VReg),
    /// Set an enumerable property on an object
    ObjSet {
        obj: VReg,
        key: VReg,
        value: VReg,
    },
    /// Set a non-enumerable property on an object
    ObjSetN {
        obj: VReg,
        key: VReg,
        value: VReg,
    },
    ObjGet {
        dest: VReg,
        obj: VReg,
        key: VReg,
    },
    /// Get an object's Own Enumerable properties
    ObjGetKeysOE {
        dest: VReg,
        obj: VReg,
    },
    /// Get an object's Inherited Enumerable properties
    ObjGetKeysIE {
        dest: VReg,
        obj: VReg,
    },
    /// Get an object's Own properties, both enumerable and non-enumerable
    ObjGetKeysO {
        dest: VReg,
        obj: VReg,
    },
    ObjDelete {
        dest: VReg,
        obj: VReg,
        key: VReg,
    },

    // TODO: Remove this bytecode (should be implemented as a method with a native impl)
    ArrayPush {
        arr: VReg,
        value: VReg,
    },
    ArrayNth {
        dest: VReg,
        arr: VReg,
        index: VReg,
    },
    ArraySetNth {
        arr: VReg,
        index: VReg,
        value: VReg,
    },
    ArrayLen {
        dest: VReg,
        arr: VReg,
    },

    // TODO Replace these ops with native functions (?)
    StrCreateEmpty(VReg),
    StrAppend(VReg, VReg),

    NewIterator {
        dest: VReg,
        obj: VReg,
    },
    IteratorGetCurrent {
        dest: VReg,
        iter: VReg,
    },
    IteratorAdvance {
        iter: VReg,
    },
    JmpIfIteratorFinished {
        iter: VReg,
        dest: IID,
    },

    TypeOf {
        dest: VReg,
        arg: VReg,
    },

    ImportModule(VReg, VReg),

    Throw(VReg),
    PopExcHandler,
    PushExcHandler(IID),
    GetCurrentException(VReg),

    Breakpoint,
}

#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub enum InstrDescriptor {
    Description(&'static str),
    VRegRead(VReg),
    VRegWrite(VReg),
    IID(IID),
    Const(ConstIndex),
    Capture(CaptureIndex),
    Arg(ArgIndex),
    Null,
    Undefined,
    This,
}

impl Instr {
    pub fn opcode(&self) -> &'static str {
        // use strum::IntoStaticStr
        self.into()
    }

    pub fn analyze(&self, mut an: impl FnMut(InstrDescriptor)) {
        type D = InstrDescriptor;

        #[rustfmt::skip]
        match self {
            Instr::Nop => {}
            Instr::LoadConst(dest, constndx) => { an(D::VRegWrite(*dest)); an(D::Const(*constndx)); }
            Instr::LoadNull(dest) => { an(D::VRegWrite(*dest)); an(D::Null); }
            Instr::LoadUndefined(dest) => { an(D::VRegWrite(*dest)); an(D::Undefined); }
            Instr::LoadCapture(dest, capndx) => { an(D::VRegWrite(*dest)); an(D::Capture(*capndx)); },
            Instr::LoadArg(dest, argndx) => { an(D::VRegWrite(*dest)); an(D::Arg(*argndx)); },
            Instr::LoadThis(dest) => { an(D::VRegWrite(*dest)); an(D::This); },
            Instr::GetGlobalThis (dest) => { an(D::VRegWrite(*dest)); },
            Instr::GetGlobal { dest, name } => { an(D::VRegWrite(*dest)); an(D::Const(*name)); }
            Instr::Copy { dst: dest, src: arg } 
            | Instr::BoolNot { dest, arg }
            | Instr::UnaryMinus { dest, arg } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arg)); },
            Instr::ArithInc(dest, arg) => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arg)); },
            Instr::ArithDec(dest, arg) => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arg)); },
            Instr::OpAdd(dst, a, b)
            | Instr::ArithSub(dst, a, b)
            | Instr::ArithMul(dst, a, b)
            | Instr::ArithDiv(dst, a, b)
            | Instr::CmpGE(dst, a, b)
            | Instr::CmpGT(dst, a, b)
            | Instr::CmpLT(dst, a, b)
            | Instr::CmpLE(dst, a, b)
            | Instr::CmpEQ(dst, a, b)
            | Instr::CmpNE(dst, a, b)
            | Instr::BoolOpAnd(dst, a, b)
            | Instr::BoolOpOr(dst, a, b) => { an(D::VRegWrite(*dst)); an(D::VRegRead(*a)); an(D::VRegRead(*b)); },
            Instr::IsInstanceOf(dst, obj, super_) => { an(D::VRegWrite(*dst)); an(D::VRegRead(*obj)); an(D::VRegRead(*super_)); },
            Instr::JmpIf { cond, dest } => { an(D::VRegRead(*cond)); an(D::IID(*dest)); },
            Instr::Jmp(dest) => { an(D::IID(*dest)); }
            Instr::PushToSink(arg) => { an(D::VRegRead(*arg)); },
            Instr::Return(arg) => { an(D::VRegRead(*arg)); },
            Instr::Call {
                return_value,
                this,
                callee,
            } => {
                an(D::VRegWrite(*return_value));
                an(D::Description("this"));
                an(D::VRegRead(*this));
                an(D::Description("callee"));
                an(D::VRegRead(*callee));
            }
            Instr::CallArg(arg) => { an(D::VRegRead(*arg)); },
            Instr::ClosureNew {
                dest,
                fnid: _,
                forced_this: _,
            } => { an(D::VRegWrite(*dest));  },
            Instr::ClosureAddCapture(arg) => { an(D::VRegRead(*arg)); },
            Instr::Unshare(reg) => { an(D::VRegWrite(*reg)); an(D::VRegRead(*reg)); },
            Instr::ObjCreateEmpty(dest) => { an(D::VRegWrite(*dest)); },
            Instr::ObjSet { obj, key, value }
	    | Instr::ObjSetN { obj, key, value } => { an(D::VRegRead(*obj)); an(D::VRegRead(*key)); an(D::VRegRead(*value)); }
            Instr::ObjGet { dest, obj, key } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*obj)); an(D::VRegRead(*key)); }
            Instr::ObjGetKeysOE { dest, obj }
	    | Instr::ObjGetKeysIE { dest, obj }
	    | Instr::ObjGetKeysO { dest, obj } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*obj)); }
            Instr::ObjDelete { dest, obj, key } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*obj)); an(D::VRegRead(*key)); }
            Instr::ArrayPush { arr, value } => { an(D::VRegRead(*arr)); an(D::VRegRead(*value)); }
            Instr::ArrayNth { dest, arr, index } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arr)); an(D::VRegRead(*index)); }
            Instr::ArraySetNth { arr, index, value } => { an(D::VRegRead(*arr)); an(D::VRegRead(*index)); an(D::VRegRead(*value)); }
            Instr::ArrayLen { dest, arr } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arr)); }
            Instr::StrCreateEmpty(dest) => { an(D::VRegWrite(*dest)); }
            Instr::StrAppend(recipient, src) => { an(D::VRegRead(*recipient)); an(D::VRegRead(*src)); }
            Instr::NewIterator { dest, obj } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*obj)); }
            Instr::IteratorGetCurrent { dest, iter } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*iter)); }
            Instr::IteratorAdvance { iter } => { an(D::VRegRead(*iter)); },
            Instr::JmpIfIteratorFinished { iter, dest } => { an(D::VRegRead(*iter)); an(D::IID(*dest)); }
            Instr::TypeOf { dest, arg } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arg)); }
            Instr::ImportModule(dest, mod_spec) => { an(D::VRegWrite(*dest)); an(D::VRegRead(*mod_spec)); }
            Instr::Throw(arg) => { an(D::VRegRead(*arg)); }
            Instr::Breakpoint => {},
            Instr::PopExcHandler => {},
            Instr::PushExcHandler(iid) => { an(D::IID(*iid)); },
            Instr::GetCurrentException(dest) => { an(D::VRegWrite(*dest)); },
        };
    }
}

/// A value literal that is allowed to appear in the bytecode.
#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Number(f64),
    String(String),
    JsWord(JsWord),
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

/// Correspondence between source code and bytecode.
///
/// The source code range is represented by the members `lo` and `hi`, with the
/// same semantics as the members of the same name in `swc_common::Span`.
///
/// The bytecode range is [`iid_start`, `iid_end`) (note that the left-hand side
/// is inclusive, while the right-hand side is exclusive).
///
/// Each BreakRange implicitly belongs to a specific module.  "Implicit" because
/// the module ID is not included in this struct.
#[derive(Clone, Debug)]
pub struct BreakRange {
    pub lo: swc_common::BytePos,
    pub hi: swc_common::BytePos,
    pub fnid: FnId,
    pub iid_start: IID,
    pub iid_end: IID,
}

pub struct Function {
    instrs: Box<[Instr]>,
    consts: Box<[Literal]>,
    n_regs: u8,
    // TODO(performance) following elision of Operand, better data structures
    loop_heads: HashMap<IID, LoopInfo>,
    ident_history: Vec<IdentAsmt>,
    trace_anchors: HashMap<IID, TraceAnchor>,
    is_strict_mode: bool,
    span: Span,
}
pub struct TraceAnchor {
    pub trace_id: String,
}
pub struct LoopInfo {
    // Variables that change in value during each cycle, in such a way that
    // each cycle sees the value in  the previous cycle.  Phi instructions are
    // added based on this set.
    #[cfg_attr(not(enable_jit), allow(dead_code))]
    interloop_vars: HashSet<IID>,
}

#[derive(Debug)]
pub struct IdentAsmt {
    pub iid: IID,
    pub reg: VReg,
    pub ident: JsWord,
}

pub struct FunctionBuilder {
    pub instrs: Box<[Instr]>,
    pub consts: Box<[Literal]>,
    pub n_regs: u8,
    pub ident_history: Vec<IdentAsmt>,
    pub trace_anchors: HashMap<IID, TraceAnchor>,
    pub is_strict_mode: bool,
    pub span: Span,
}

impl FunctionBuilder {
    pub(crate) fn build(self) -> Function {
        #[cfg(to_be_rewritten)]
        let loop_heads = find_loop_heads(&instrs[..]);
        #[cfg(not(to_be_rewritten))]
        let loop_heads = HashMap::new();
        Function {
            instrs: self.instrs,
            consts: self.consts,
            n_regs: self.n_regs,
            loop_heads,
            ident_history: self.ident_history,
            trace_anchors: self.trace_anchors,
            is_strict_mode: self.is_strict_mode,
            span: self.span,
        }
    }
}

impl Function {
    pub fn instrs(&self) -> &[Instr] {
        self.instrs.as_ref()
    }

    pub fn consts(&self) -> &[Literal] {
        self.consts.as_ref()
    }

    pub fn n_regs(&self) -> u8 {
        self.n_regs
    }

    pub fn ident_history(&self) -> &[IdentAsmt] {
        &self.ident_history
    }

    pub fn is_loop_head(&self, iid: IID) -> bool {
        self.loop_heads.contains_key(&iid)
    }

    pub fn get_trace_anchor(&self, iid: IID) -> Option<&TraceAnchor> {
        self.trace_anchors.get(&iid)
    }

    pub fn trace_start_id(&self, iid: IID) -> Option<&str> {
        self.trace_anchors
            .get(&iid)
            .map(|tanch| tanch.trace_id.as_str())
    }

    pub fn is_strict_mode(&self) -> bool {
        self.is_strict_mode
    }

    pub fn span(&self) -> &Span {
        &self.span
    }

    pub fn dump(&self) {
        println!("-- consts");
        for (ndx, literal) in self.consts.iter().enumerate() {
            println!(" {:4} {:?}", ndx, literal);
        }

        println!("-- instrs");
        for (ndx, instr) in self.instrs.iter().enumerate() {
            println!(" {:4} {:?}", ndx, instr);
        }

        println!();
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
