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
pub struct FnID(pub u32);

impl std::fmt::Debug for FnID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "f{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct GlobalIID(pub FnID, pub IID);

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

        Some(GlobalIID(FnID(fnid), IID(iid)))
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
pub const ARGS_COUNT_MAX: u16 = 8;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize)]
pub struct CaptureIndex(pub u16);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VReg(pub u16);

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
    ToNumber {
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
    JmpIfNot {
        cond: VReg,
        dest: IID,
    },
    JmpIfPrimitive {
        arg: VReg,
        dest: IID,
    },
    JmpIfNotClosure {
        arg: VReg,
        dest: IID,
    },
    JmpIfNumberNotInteger { arg: VReg, dest: IID },
    Jmp(IID),
    SaveFrameSnapshot(IID),
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
        fnid: FnID,
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

    // TODO Replace these ops with native functions (?)
    StrCreateEmpty(VReg),
    StrAppend(VReg, VReg),
    StrFromCodePoint { dest: VReg, arg: VReg },

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
            | Instr::UnaryMinus { dest, arg }
            | Instr::ToNumber { dest, arg } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arg)); },
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
            Instr::JmpIf { cond, dest }
            | Instr::JmpIfNot { cond, dest } => { an(D::VRegRead(*cond)); an(D::IID(*dest)); },
            Instr::JmpIfPrimitive { arg, dest } 
            | Instr::JmpIfNotClosure { arg, dest } 
            | Instr::JmpIfNumberNotInteger { arg, dest }=> { an(D::VRegRead(*arg)); an(D::IID(*dest)); },
            Instr::Jmp(dest)
     	    | Instr::SaveFrameSnapshot(dest) => { an(D::IID(*dest)); },
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
            Instr::ObjDelete { obj, key } => { an(D::VRegRead(*obj)); an(D::VRegRead(*key)); }
            Instr::ArrayPush { arr, value } => { an(D::VRegRead(*arr)); an(D::VRegRead(*value)); }
            Instr::ArrayNth { dest, arr, index } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arr)); an(D::VRegRead(*index)); }
            Instr::StrCreateEmpty(dest) => { an(D::VRegWrite(*dest)); }
            Instr::StrAppend(recipient, src) => { an(D::VRegRead(*recipient)); an(D::VRegRead(*src)); }
            Instr::StrFromCodePoint { dest, arg } => { an(D::VRegWrite(*dest)); an(D::VRegRead(*arg)); },
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
    // String and not JSString because `Literal` values are intended as self-contained
    // values that ared, mutually intelligible to both the "Rust" world and the JavaScript
    // world, and Rust assumes that all text be UTF-8 String's
    String(String),
    Symbol(&'static str),
    JsWord(JsWord),
    Bool(bool),
    Null,
    Undefined,
}

pub type NativeFnID = u32;

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
    pub fnid: FnID,
    pub iid_start: IID,
    pub iid_end: IID,
}

pub struct Function {
    instrs: Box<[Instr]>,
    consts: Box<[Literal]>,
    n_regs: u16,
    ident_history: Vec<IdentAsmt>,
    is_strict_mode: bool,
    span: Span,
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
    pub n_regs: u16,
    pub ident_history: Vec<IdentAsmt>,
    pub is_strict_mode: bool,
    pub span: Span,
}

impl FunctionBuilder {
    pub(crate) fn build(self) -> Function {
        Function {
            instrs: self.instrs,
            consts: self.consts,
            n_regs: self.n_regs,
            ident_history: self.ident_history,
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

    pub fn n_regs(&self) -> u16 {
        self.n_regs
    }

    pub fn ident_history(&self) -> &[IdentAsmt] {
        &self.ident_history
    }

    pub fn is_strict_mode(&self) -> bool {
        self.is_strict_mode
    }

    pub fn span(&self) -> &Span {
        &self.span
    }

    pub fn dump<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
        writeln!(out, "-- instrs ({} bytes)", std::mem::size_of_val(&*self.instrs))?;
        for (ndx, instr) in self.instrs.iter().enumerate() {
            write!(out, " {:4} {:15}  ", ndx, instr.opcode())?;
            instr.analyze(|desc| match desc {
                InstrDescriptor::Description(txt) => write!(out, "{}:", txt).unwrap(),
                InstrDescriptor::VRegRead(vreg) => write!(out, "{:?}  ", vreg).unwrap(),
                InstrDescriptor::VRegWrite(vreg) => write!(out, "{:?}<-  ", vreg).unwrap(),
                InstrDescriptor::IID(iid) => write!(out, "{:?}  ", iid).unwrap(),
                InstrDescriptor::Const(const_ndx) => {
                    let val = &self.consts[const_ndx.0 as usize];
                    write!(out, "k{:?}  ", val).unwrap();
                }
                InstrDescriptor::Capture(cap_ndx) => write!(out, "cap[{}]  ", cap_ndx.0).unwrap(),
                InstrDescriptor::Arg(arg_ndx) => write!(out, "arg[{}]  ", arg_ndx.0).unwrap(),
                InstrDescriptor::Null => write!(out, "null  ").unwrap(),
                InstrDescriptor::Undefined => write!(out, "undefined  ").unwrap(),
                InstrDescriptor::This => write!(out, "this  ").unwrap(),
            });
            writeln!(out)?;
        }
        Ok(())
    }
}
