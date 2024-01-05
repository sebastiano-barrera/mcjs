use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use serde::Serialize;
use strum::IntoStaticStr;
use swc_common::{BytePos, SourceMap, Span};

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

/// Global ID of a module.  Can be used, among other things, to fetch the Module object
/// from importing modules.
// me: "64K modules ought to be enough for anyone."
// guy with knife: node_modules
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize)]
pub struct ModuleId(pub u16);

pub const SCRIPT_MODULE_ID: ModuleId = ModuleId(0);

impl From<u16> for ModuleId {
    fn from(value: u16) -> Self {
        if value == SCRIPT_MODULE_ID.0 {
            panic!(
                "invalid module ID: value {} is reserved for code in script context",
                SCRIPT_MODULE_ID.0
            );
        }
        ModuleId(value)
    }
}

/// ID of a function within a specific module. (The same ID can correspond to a
/// different function, in the context of a different modules.)
///
/// Implements Ord, so it's possible to quickly and cheaply check if a set of
/// LocalFnIds are disjoint from another.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord, Serialize)]
pub struct LocalFnId(pub u16);

/// Global ID of a function, composing a module ID and a local function ID.  ///
/// This ID is sufficient to identify a function across the entire loaded codebase,
/// regardless of the module it belongs to.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize)]
pub struct FnId(pub ModuleId, pub LocalFnId);

impl std::fmt::Debug for FnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "m{}:f{}", self.0 .0, self.1 .0)
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
        let (mod_id_s, s) = s.split_once('.')?;
        let (lfnid_s, iid_s) = s.split_once('.')?;

        let mod_id: u16 = mod_id_s.parse().ok()?;
        let lfnid: u16 = lfnid_s.parse().ok()?;
        let iid: u16 = iid_s.parse().ok()?;

        Some(GlobalIID(
            FnId(ModuleId(mod_id), LocalFnId(lfnid)),
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
        fnid: LocalFnId,
        forced_this: Option<VReg>,
    },
    ClosureAddCapture(VReg),
    Unshare(VReg),

    ObjCreateEmpty(VReg),
    ObjSet {
        obj: VReg,
        key: VReg,
        value: VReg,
    },
    ObjGet {
        dest: VReg,
        obj: VReg,
        key: VReg,
    },
    ObjGetKeys {
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

impl Instr {
    const MAX_OPERANDS: usize = 4;
}

type Operands = crate::util::LimVec<{ Instr::MAX_OPERANDS }, VReg>;

pub trait InstrAnalyzer {
    fn start(&mut self, opcode_name: &'static str);
    fn read_vreg_labeled(&mut self, vreg: VReg, description: Option<&'static str>);
    fn write_vreg_labeled(&mut self, vreg: VReg, description: Option<&'static str>);
    fn read_vreg(&mut self, vreg: VReg) {
        self.read_vreg_labeled(vreg, None)
    }
    fn write_vreg(&mut self, vreg: VReg) {
        self.write_vreg_labeled(vreg, None)
    }
    fn jump_target(&mut self, iid: IID);
    fn load_const(&mut self, item: ConstIndex);
    fn load_null(&mut self);
    fn load_undefined(&mut self);
    fn load_capture(&mut self, item: CaptureIndex);
    fn load_arg(&mut self, item: ArgIndex);
    fn load_this(&mut self);
    fn end(&mut self, instr: &Instr);
}

impl Instr {
    pub fn analyze<A: InstrAnalyzer>(&self, an: &mut A) {
        use std::convert::AsRef;

        let opcode: &'static str = self.into();
        an.start(opcode);

        #[rustfmt::skip]
        match self {
            Instr::Nop => {}
            Instr::LoadConst(dest, constndx) => { an.write_vreg(*dest); an.load_const(*constndx); }
            Instr::LoadNull(dest) => { an.write_vreg(*dest); an.load_null(); }
            Instr::LoadUndefined(dest) => { an.write_vreg(*dest); an.load_undefined(); }
            Instr::LoadCapture(dest, capndx) => { an.write_vreg(*dest); an.load_capture(*capndx); },
            Instr::LoadArg(dest, argndx) => { an.write_vreg(*dest); an.load_arg(*argndx); },
            Instr::LoadThis(dest) => { an.write_vreg(*dest); an.load_this(); },
            Instr::Copy { dst, src } => { an.write_vreg(*dst); an.read_vreg(*src); },
            Instr::GetGlobalThis (dest) => { an.write_vreg(*dest); },
            Instr::BoolNot { dest, arg } => { an.write_vreg(*dest); an.read_vreg(*arg); },
            Instr::UnaryMinus { dest, arg } => { an.write_vreg(*dest); an.read_vreg(*arg); },
            Instr::OpAdd(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::ArithSub(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::ArithMul(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::ArithDiv(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::ArithInc(dest, arg) => { an.write_vreg(*dest); an.read_vreg(*arg); },
            Instr::ArithDec(dest, arg) => { an.write_vreg(*dest); an.read_vreg(*arg); },
            Instr::CmpGE(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::CmpGT(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::CmpLT(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::CmpLE(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::CmpEQ(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::CmpNE(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::BoolOpAnd(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::BoolOpOr(dst, a, b) => { an.write_vreg(*dst); an.read_vreg(*a); an.read_vreg(*b); },
            Instr::IsInstanceOf(dst, obj, super_) => { an.write_vreg(*dst); an.read_vreg(*obj); an.read_vreg(*super_); },
            Instr::JmpIf { cond, dest } => { an.read_vreg(*cond); an.jump_target(*dest); },
            Instr::Jmp(dest) => { an.jump_target(*dest); }
            Instr::PushToSink(arg) => { an.read_vreg(*arg); },
            Instr::Return(arg) => { an.read_vreg(*arg); },
            Instr::Call {
                return_value,
                this,
                callee,
            } => {
                an.write_vreg(*return_value);
                an.read_vreg_labeled(*this, Some("this"));
                an.read_vreg_labeled(*callee, Some("callee"));
            }
            Instr::CallArg(arg) => { an.read_vreg(*arg); },
            Instr::ClosureNew {
                dest,
                fnid: _,
                forced_this: _,
            } => { an.write_vreg(*dest);  },
            Instr::ClosureAddCapture(arg) => { an.read_vreg(*arg); },
            Instr::Unshare(reg) => { an.write_vreg(*reg); an.read_vreg(*reg); },
            Instr::ObjCreateEmpty(dest) => { an.write_vreg(*dest); },
            Instr::ObjSet { obj, key, value } => { an.read_vreg(*obj); an.read_vreg(*key); an.read_vreg(*value); }
            Instr::ObjGet { dest, obj, key } => { an.write_vreg(*dest); an.read_vreg(*obj); an.read_vreg(*key); }
            Instr::ObjGetKeys { dest, obj } => { an.write_vreg(*dest); an.read_vreg(*obj); }
            Instr::ObjDelete { dest, obj, key } => { an.write_vreg(*dest); an.read_vreg(*obj); an.read_vreg(*key); }
            Instr::ArrayPush { arr, value } => { an.read_vreg(*arr); an.read_vreg(*value); }
            Instr::ArrayNth { dest, arr, index } => { an.write_vreg(*dest); an.read_vreg(*arr); an.read_vreg(*index); }
            Instr::ArraySetNth { arr, index, value } => { an.read_vreg(*arr); an.read_vreg(*index); an.read_vreg(*value); }
            Instr::ArrayLen { dest, arr } => { an.write_vreg(*dest); an.read_vreg(*arr); }
            Instr::StrCreateEmpty(dest) => { an.write_vreg(*dest); }
            Instr::StrAppend(recipient, src) => { an.read_vreg(*recipient); an.read_vreg(*src); }
            Instr::NewIterator { dest, obj } => { an.write_vreg(*dest); an.read_vreg(*obj); }
            Instr::IteratorGetCurrent { dest, iter } => { an.write_vreg(*dest); an.read_vreg(*iter); }
            Instr::IteratorAdvance { iter } => { an.read_vreg(*iter); },
            Instr::JmpIfIteratorFinished { iter, dest } => { an.read_vreg(*iter); an.jump_target(*dest); }
            Instr::TypeOf { dest, arg } => { an.write_vreg(*dest); an.read_vreg(*arg); }
            Instr::ImportModule(dest, mod_spec) => { an.write_vreg(*dest); an.read_vreg(*mod_spec); }
            Instr::Throw(arg) => { an.read_vreg(*arg); }
            Instr::Breakpoint => {},
            Instr::PopExcHandler => {},
            Instr::PushExcHandler(iid) => { an.jump_target(*iid) },
            Instr::GetCurrentException(dest) => { an.write_vreg(*dest); },
        };

        an.end(self)
    }
}

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
    pub local_fnid: LocalFnId,
    pub iid_start: IID,
    pub iid_end: IID,
}

pub struct Function {
    instrs: Box<[Instr]>,
    consts: Box<[Literal]>,
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
