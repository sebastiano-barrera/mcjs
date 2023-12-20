use lazy_static::lazy_static;
use swc_atoms::JsWord;

use std::{
    borrow::{BorrowMut, Cow},
    cell::{Ref, RefCell, RefMut},
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::{
    bytecode::{self, FnId, GlobalIID, Instr, VReg, IID},
    bytecode_compiler,
    common::{Context, Result},
    error,
    heap::{self, Heap, Object, ObjectExt},
    loader::{self, BreakRangeID},
    // jit::{self, InterpreterStep},
    stack,
    stack_access,
    util::Mask,
};

pub use crate::common::Error;

macro_rules! tprintln {
    ($($args:expr),*) => {
        #[cfg(test)]
        eprintln!($($args),*);
    }
}

macro_rules! tprint {
    ($($args:expr),*) => {
        #[cfg(test)]
        eprint!($($args),*);
    }
}

/// A value that can be input, output, or processed by the program at runtime.
///
/// Design notes: Value is `Copy` in an effort to make it as dumb as possible (easy to
/// copy/move/delete/etc.), as otherwise it becomes really hard to keep memory safety in
/// the interpreter's stack (which is shared with JIT-compiled code, which is inherently
/// unsafe, so we have to make some compromises).
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Value {
    Number(f64),
    Bool(bool),
    Object(heap::ObjectId),
    Null,
    Undefined,
    SelfFunction,

    // TODO(small) remove this. society has evolved past the need for values that can't be created
    // from source code
    Internal(usize),
}

macro_rules! gen_value_expect {
    ($fn_name:ident, $variant:tt, $inner_ty:ty) => {
        impl Value {
            pub(crate) fn $fn_name(&self) -> Result<$inner_ty> {
                match self {
                    Value::$variant(inner) => Ok(*inner),
                    other => Err(error!(
                        "expected a {}, got {:?}",
                        stringify!($variant),
                        other
                    )),
                }
            }
        }
    };
}

gen_value_expect!(expect_bool, Bool, bool);
gen_value_expect!(expect_num, Number, f64);
gen_value_expect!(expect_obj, Object, heap::ObjectId);

#[derive(Clone)]
pub enum Closure {
    Native(NativeFunction),
    JS(JSClosure),
}

type NativeFunction = fn(&mut Interpreter, &Value, &[Value]) -> Result<Value>;

#[derive(Clone, PartialEq)]
pub struct JSClosure {
    fnid: FnId,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueId>,
    forced_this: Option<Value>,
}

impl JSClosure {
    pub fn upvalues(&self) -> &[UpvalueId] {
        &self.upvalues
    }
}
impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Closure::Native(nf) => {
                write!(f, "<native closure {:?}>", (nf as *const _))
            }
            Closure::JS(closure) => {
                write!(f, "<closure {:?} | ", closure.fnid)?;
                for upv in &closure.upvalues {
                    write!(f, "{:?} ", upv)?;
                }
                write!(f, ">")
            }
        }
    }
}

slotmap::new_key_type! { pub struct UpvalueId; }

pub struct Options {
    pub debug_dump_module: bool,
    pub indent_level: u8,
    pub jit_mode: JitMode,
}
#[derive(Clone, Copy, Debug)]
pub enum JitMode {
    Compile,
    UseTraces,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            debug_dump_module: false,
            indent_level: 0,
            jit_mode: JitMode::UseTraces,
        }
    }
}

pub struct NotADirectoryError(PathBuf);
impl std::fmt::Debug for NotADirectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "provided path is not a directory: {}", self.0.display())
    }
}

pub struct Realm {
    heap: heap::Heap,
    module_objs: HashMap<bytecode::ModuleId, heap::ObjectId>,
    global_obj: heap::ObjectId,
}

impl Realm {
    pub fn new() -> Realm {
        let mut heap = heap::Heap::new();
        let global_obj = init_builtins(&mut heap);
        Realm {
            heap,
            module_objs: HashMap::new(),
            global_obj,
        }
    }
}

pub struct Interpreter<'a> {
    /// Exclusive reference to the Realm in which the code runs.
    ///
    /// It's important that this reference is exclusive: no one must access (much less
    /// manipulate) this Realm object until the Interpreter finishes executing.
    realm: &'a mut Realm,

    iid: bytecode::IID,

    /// The interpreter's stack.
    ///
    /// The data stored here does not survive after the Interpreter returns.  (Well, some
    /// referenced objects may survive in the heap for a while, but the GC is supposed to
    /// collect them.)
    data: stack::InterpreterData,

    #[cfg(enable_jit)]
    jitting: Option<Jitting>,

    // The loader ref must never change for the whole lifecycle of the interpreter.  What would
    // happen if the same module path suddenly corresponded to a different module? Better not to
    // know
    loader: &'a mut loader::Loader,

    sink: Vec<Value>,
    opts: Options,

    /// Instruction breakpoints
    instr_bkpts: HashMap<GlobalIID, InstrBreakpoint>,

    /// Source breakpoints, indexed by their ID.
    ///
    /// Each source breakpoint corresponds to exactly to one instruction breakpoint, which is
    /// added/deleted together with it.
    source_bkpts: HashMap<BreakRangeID, SourceBreakpoint>,

    /// This special flag can be used to cause the interpreter to suspend
    /// after N instructions. It's conceptually similar to a breakpoint that
    /// automatically follows call/return.
    ///
    /// See the type, `Fuel`.
    fuel: Fuel,
}

pub enum Fuel {
    Limited(usize),
    Unlimited,
}

// There is nothing here for now. The mere existence of an entry in Interpreter.source_bktps is
// enough (but some addtional parameters might have to be includede here later)
pub struct SourceBreakpoint;

struct InstrBreakpoint {
    src_bkpt: Option<BreakRangeID>,
}

// TODO Probably going to be changed or even replaced once I resume working on the JIT.
// TODO(performance) monomorphize get_operand?
#[cfg(enable_jit)]
pub struct InterpreterStep<'a> {
    pub fnid: bytecode::FnId,
    pub func: &'a bytecode::Function,
    pub iid: bytecode::IID,
    pub next_iid: bytecode::IID,
    pub get_operand: &'a dyn Fn(bytecode::VReg) -> Value,
}

#[cfg(enable_jit)]
impl<'a> InterpreterStep<'a> {
    fn cur_instr(&self) -> &bytecode::Instr {
        &self.func.instrs()[self.iid.0 as usize]
    }
}

pub type InterpreterResult<'a> = std::result::Result<Exit<'a>, InterpreterError<'a>>;

pub struct FinishedData {
    pub sink: Vec<Option<bytecode::Literal>>,
}

pub enum Exit<'a> {
    Finished(FinishedData),
    Suspended(Interpreter<'a>),
}
impl<'a> Exit<'a> {
    pub fn expect_finished(self) -> FinishedData {
        match self {
            Exit::Finished(fd) => fd,
            Exit::Suspended(_) => {
                panic!("interpreter was interrupted, while it was expected to finish")
            }
        }
    }
}

/// Similar to `Exit`, but only exists for internal use.
///
/// Mostly exists to avoid boolean blindness.
enum ExitInternal {
    Finished,
    Suspended,
}

// TODO Remove this InterpreterError?
// It used to be justified by the addition of a CoreDump, but the CoreDump has been
// removed since.
pub struct InterpreterError<'a> {
    pub error: Error,

    // This struct owns the failed interpreter, but we explicitly disallow doing
    // anything else with it other than examining it (read-only, via &)
    interpreter: Interpreter<'a>,
}
impl<'a> InterpreterError<'a> {
    #[cfg(any(test, feature = "debugger"))]
    pub fn probe<'s>(&'s mut self) -> debugger::Probe<'s, 'a> {
        debugger::Probe::attach(&mut self.interpreter)
    }

    pub fn restart(self) -> Interpreter<'a> {
        self.interpreter.restart()
    }
}
impl<'a> std::fmt::Debug for InterpreterError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InterpreterError: {:?}", self.error,)
    }
}

#[cfg(enable_jit)]
struct Jitting {
    fnid: FnId,
    iid: IID,
    trace_id: String,
    builder: jit::TraceBuilder,
}

impl<'a> Interpreter<'a> {
    pub fn new(realm: &'a mut Realm, loader: &'a mut loader::Loader, fnid: bytecode::FnId) -> Self {
        // Initialize the stack with a single frame, corresponding to a call to fnid with no
        // parameters
        Interpreter {
            iid: bytecode::IID(0),
            data: init_stack(loader, fnid),
            realm,
            loader,
            sink: Vec::new(),
            opts: Default::default(),
            #[cfg(enable_jit)]
            jitting: None,
            instr_bkpts: HashMap::new(),
            source_bkpts: HashMap::new(),
            fuel: Fuel::Unlimited,
        }
    }

    pub fn restart(mut self) -> Self {
        self.data = {
            let bottom_frame = self.data.frames().last().unwrap();
            let root_fnid = bottom_frame.header().fn_id;
            init_stack(self.loader, root_fnid)
        };
        self.iid = bytecode::IID(0);
        self.sink.clear();

        self
    }

    #[cfg(enable_jit)]
    pub fn take_trace(&mut self) -> Option<jit::Trace> {
        todo!("(cleanup) delete this method")
    }

    #[cfg(enable_jit)]
    pub fn get_trace(&self, trace_id: &str) -> Option<&(jit::Trace, jit::NativeThunk)> {
        self.traces.get(trace_id)
    }

    #[cfg(enable_jit)]
    pub fn trace_ids(&self) -> impl ExactSizeIterator<Item = &String> {
        self.traces.keys()
    }

    /// Run the interpreter until it either finishes, or it gets interrupted.
    ///
    /// This is the only entry point into code execution.
    ///
    /// Note that the receiver is passed by move (`mut self`, not `&mut self`).
    ///
    ///   - If the interpreter finishes execution, it is 'consumed' by this call and no
    ///     longer accessible.
    ///
    ///   - If the interpreter is interrupted (e.g. by a breakpoint), then it is returned
    ///     again via the Output::Suspended variant; then it can be run again or dropped
    ///     (destroyed).
    pub fn run(mut self) -> InterpreterResult<'a> {
        match self.run_internal() {
            Ok(ExitInternal::Finished) => Ok(Exit::Finished(self.dump_output())),
            Ok(ExitInternal::Suspended) => Ok(Exit::Suspended(self)),
            Err(common_err) => Err(InterpreterError {
                interpreter: self,
                error: common_err,
            }),
        }
    }

    /// Just `run`, but it takes self by &mut and returns a simple
    /// `mcjs_vm::common::Error`.
    ///
    /// This implementation does not consume the Interpreter like `run` does,
    /// but this allows the implementation code to be a little simpler.
    ///
    /// Returns:
    ///  - `Ok(true)` if the interpreter finished successfully.
    ///  - `Ok(false)` if the interpreter is supposed to suspend.
    ///  - `Err(err)` if the interpreter failed.
    ///
    /// In all cases, it's `run` that perfects the process of returning the
    /// control to the caller, ensuring the correct type of access to the
    /// interpreter.
    fn run_internal(&mut self) -> Result<ExitInternal> {
        while self.data.len() != 0 {
            // TODO Avoid calling get_function at each instructions
            let fnid = self.data.top().header().fn_id;

            // TODO make it so that func is "gotten" and unwrapped only when strictly necessary
            let func = self.loader.get_function(fnid).unwrap();
            let n_instrs = func.instrs().len();

            assert!(
                self.iid.0 as usize <= n_instrs,
                "can't proceed to instruction at index {} (func has {})",
                self.iid.0,
                n_instrs
            );
            if self.iid.0 as usize == n_instrs {
                if let Some(return_to_iid) = self.exit_function(None) {
                    self.iid = return_to_iid;
                    continue;
                } else {
                    return Ok(ExitInternal::Finished);
                }
            }

            let instr = func.instrs()[self.iid.0 as usize];

            self.print_indent();
            tprint!("{:<4}  {:?}", self.iid.0, instr);
            if let Instr::LoadConst(_, const_ndx) = instr {
                let lit = &func.consts()[const_ndx.0 as usize];
                tprint!(" = ({:?})", lit);
            }
            tprint!("    ");

            let mut next_ndx = self.iid.0 + 1;

            #[cfg(enable_jit)]
            if let Some(tanch) = func.get_trace_anchor(self.iid) {
                match self.flags.jit_mode {
                    JitMode::Compile => {
                        if self.jitting.is_none() {
                            let builder =
                                jit::TraceBuilder::start(n_instrs, jit::CloseMode::FunctionExit);
                            self.jitting = Some(Jitting {
                                builder,
                                fnid,
                                iid: self.iid,
                                trace_id: tanch.trace_id.clone(),
                            });
                        }
                    }
                    JitMode::UseTraces => {
                        let (trace, thunk) =
                            self.vm.get_trace(&tanch.trace_id).unwrap_or_else(|| {
                                panic!("no such trace with ID `{}`", tanch.trace_id)
                            });

                        let mut snap: Vec<_> = trace
                            .snapshot_map()
                            .iter()
                            .map(|snapitem| {
                                if snapitem.write_on_entry {
                                    self.get_operand(snapitem.operand)
                                } else {
                                    Value::Undefined
                                }
                            })
                            .collect();

                        // TODO(small feat) Update the interpreter's state from the trace
                        thunk.run(&mut snap);
                    }
                }
            }

            match &instr {
                Instr::LoadConst(dest, bytecode::ConstIndex(const_ndx)) => {
                    let literal = func.consts()[*const_ndx as usize].clone();
                    let value = literal_to_value(literal, &mut self.realm.heap);
                    self.data.top_mut().set_result(*dest, value);
                }

                Instr::ArithAdd(dest, a, b) => match self.get_operand(*a) {
                    Value::Number(_) => self.with_numbers(*dest, *a, *b, |x, y| x + y),
                    Value::Object(_) => {
                        let value = self.str_append(a, b)?;
                        self.data.top_mut().set_result(*dest, value);
                    }
                    other => return Err(error!("unsupported operator '+' for: {:?}", other)),
                },
                Instr::ArithSub(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x - y),
                Instr::ArithMul(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x * y),
                Instr::ArithDiv(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x / y),

                Instr::PushToSink(operand) => {
                    let value = self.get_operand(*operand);
                    self.sink.push(value);
                }

                Instr::CmpGE(dest, a, b) => self.compare(*dest, *a, *b, |ord| {
                    ord == ValueOrdering::Greater || ord == ValueOrdering::Equal
                }),
                Instr::CmpGT(dest, a, b) => {
                    self.compare(*dest, *a, *b, |ord| ord == ValueOrdering::Greater)
                }
                Instr::CmpLT(dest, a, b) => {
                    self.compare(*dest, *a, *b, |ord| ord == ValueOrdering::Less)
                }
                Instr::CmpLE(dest, a, b) => self.compare(*dest, *a, *b, |ord| {
                    ord == ValueOrdering::Less || ord == ValueOrdering::Equal
                }),
                Instr::CmpEQ(dest, a, b) => {
                    self.compare(*dest, *a, *b, |ord| ord == ValueOrdering::Equal)
                }
                Instr::CmpNE(dest, a, b) => {
                    self.compare(*dest, *a, *b, |ord| ord != ValueOrdering::Equal)
                }

                Instr::JmpIf { cond, dest } => {
                    let cond_value = self.get_operand(*cond);
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => {
                            return Err(error!(" invalid if condition (not boolean): {:?}", other))
                        }
                    }
                }

                Instr::Copy { dst, src } => {
                    let value = self.get_operand(*src);
                    self.data.top_mut().set_result(*dst, value);
                }
                Instr::LoadCapture(dest, cap_ndx) => {
                    self.data.capture_to_var(*cap_ndx, *dest);
                }

                Instr::Nop => {}
                Instr::BoolNot { dest, arg } => {
                    let value = self.get_operand(*arg);
                    let value = Value::Bool(!self.to_boolean(value));
                    self.data.top_mut().set_result(*dest, value);
                }
                Instr::Jmp(IID(dest_ndx)) => {
                    next_ndx = *dest_ndx;
                }
                Instr::Return(value) => {
                    if let Some(return_to_iid) = self.exit_function(Some(*value)) {
                        self.iid = return_to_iid;
                        continue;
                    } else {
                        return Ok(ExitInternal::Finished);
                    }
                }
                Instr::Call {
                    callee,
                    this,
                    return_value,
                } => {
                    let oid = self.get_operand(*callee).expect_obj()?;
                    let heap_object = self
                        .realm
                        .heap
                        .get(oid)
                        .ok_or_else(|| error!("invalid function (object is not callable)"))?;
                    let closure: &Closure =
                        if let heap::HeapObject::ClosureObject(cobj) = heap_object.deref() {
                            cobj.closure()
                        } else {
                            // TODO Generalize to other types of callable objects?
                            return Err(error!("can't call non-closure"));
                        };

                    // The arguments have to be "read" before adding the stack frame;
                    // they will no longer be accessible
                    // afterwards
                    // TODO make the above fact false, and avoid this allocation
                    let mut arg_vals: Vec<_> = func
                        .instrs()
                        .iter()
                        .skip(self.iid.0 as usize + 1)
                        .map_while(|instr| match instr {
                            Instr::CallArg(arg_reg) => Some(*arg_reg),
                            _ => None,
                        })
                        .map(|vreg| self.get_operand(vreg))
                        .collect();
                    let n_args_u16: u16 = arg_vals.len().try_into().unwrap();
                    let return_to_iid = IID(self.iid.0 + n_args_u16 + 1);

                    match closure {
                        Closure::JS(closure) => {
                            // This code was moved.  Put it back where it belongs, when you
                            // re-enable the JIT.
                            #[cfg(enable_jit)]
                            if let Some(jitting) = &mut self.jitting {
                                jitting.builder.set_args(args);
                                jitting.builder.enter_function(
                                    self.iid,
                                    *callee,
                                    n_instrs as usize,
                                );
                            }

                            let callee_func = self.loader.get_function(closure.fnid).unwrap();
                            let n_params = callee_func.n_params().0;
                            arg_vals.truncate(n_params as usize);
                            arg_vals.resize(n_params as usize, Value::Undefined);
                            assert_eq!(arg_vals.len(), n_params as usize);
                            tprintln!("     - call with {} params", n_params);
                            for (i, arg) in arg_vals.iter().enumerate() {
                                tprintln!("     - call arg[{}]: {:?}", i, arg);
                            }

                            let this = closure
                                .forced_this
                                .clone()
                                .unwrap_or_else(|| self.get_operand(*this));

                            let call_meta = stack::CallMeta {
                                fnid: closure.fnid,
                                // TODO Actually, we just need to allocate enough space for
                                // *variables*, not for instructions.  However, this is OK for now,
                                // as n_instrs is always >= n_variables.
                                n_instrs: callee_func.instrs().len().try_into().unwrap(),
                                n_captured_upvalues: closure.upvalues.len().try_into().unwrap(),
                                n_args: n_params,
                                this,
                                return_value_reg: Some(*return_value),
                                return_to_iid: Some(return_to_iid),
                            };

                            tprintln!();
                            self.print_indent();
                            tprintln!(
                                "-- fn {:?} [{} captures]",
                                call_meta.fnid.0,
                                call_meta.n_captured_upvalues
                            );

                            self.data.push(call_meta);
                            for (capndx, capture) in closure.upvalues.iter().enumerate() {
                                let capndx = bytecode::CaptureIndex(
                                    capndx.try_into().expect("too many captures!"),
                                );
                                self.print_indent();
                                tprintln!("    capture[{}] = {:?}", capndx.0, capture);
                                self.data.top_mut().set_capture(capndx, *capture);
                                assert_eq!(self.data.top().get_capture(capndx), *capture);
                            }
                            for (i, arg) in arg_vals.into_iter().enumerate() {
                                self.data.top_mut().set_arg(bytecode::ArgIndex(i as _), arg);
                            }
                            self.iid = IID(0u16);

                            // Important: we don't execute the tail part of the instruction's
                            // execution. This makes it easier
                            // to keep a consistent value in `func` and other
                            // variables, and avoid subtle bugs
                            continue;
                        }
                        Closure::Native(nf) => {
                            let nf = *nf;
                            drop(heap_object);

                            let this = self.get_operand(*this);
                            let ret_val = nf(self, &this, &arg_vals)?;
                            self.data.top_mut().set_result(*return_value, ret_val);
                            next_ndx = return_to_iid.0;
                        }
                    }
                }
                Instr::CallArg(_) => {
                    unreachable!("interpreter bug: CallArg goes through another path!")
                }

                Instr::LoadArg(dest, arg_ndx) => {
                    // TODO extra copy?
                    // TODO usize is a bit too wide
                    let value = self
                        .data
                        .top()
                        .get_arg(*arg_ndx)
                        .unwrap_or(Value::Undefined);
                    tprint!("-> {:?}", value);
                    self.data.top_mut().set_result(*dest, value);
                }

                Instr::ObjCreateEmpty(dest) => {
                    let oid = self.realm.heap.new_ordinary_object(HashMap::new());
                    self.data.top_mut().set_result(*dest, Value::Object(oid));
                }
                Instr::ObjSet { obj, key, value } => {
                    let mut obj = self.get_operand_object_mut(*obj)?;
                    let key = self.get_operand(*key);
                    let key = Self::value_to_index_or_key(&self.realm.heap, &key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self.get_operand(*value);

                    obj.as_object_mut().set_own_element_or_property(key, value);
                }
                Instr::ObjGet { dest, obj, key } => {
                    let obj = self.get_operand_object(*obj)?;
                    let key = self.get_operand(*key);
                    let key = Self::value_to_index_or_key(&self.realm.heap, &key);

                    let value = match key {
                        Some(heap::IndexOrKey::Index(ndx)) => obj.as_object().get_element(ndx),
                        Some(heap::IndexOrKey::Key(key)) => self
                            .realm
                            .heap
                            .get_property_chained(obj.as_object(), key.deref()),
                        None => None,
                    }
                    .unwrap_or(Value::Undefined);

                    tprint!("  -> {:?}", value);
                    drop(obj);
                    self.data.top_mut().set_result(*dest, value.clone());
                }
                Instr::ObjGetKeys { dest, obj } => {
                    // TODO Something more efficient?
                    let keys: Vec<String> = {
                        let obj = self.get_operand_object(*obj)?;
                        obj.as_object().own_properties()
                    };
                    let keys = keys
                        .into_iter()
                        .map(|name| {
                            let oid = self.realm.heap.new_string(name);
                            Value::Object(oid)
                        })
                        .collect();
                    let keys = Value::Object(self.realm.heap.new_array(keys));
                    self.data.top_mut().set_result(*dest, keys);
                }
                Instr::ObjDelete { dest, obj, key } => {
                    // TODO Adjust return value: true for all cases except when the property is an
                    // own non-configurable property, in which case false is returned in non-strict
                    // mode. (Source: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/delete)
                    {
                        let mut obj = self.get_operand_object_mut(*obj)?;
                        let key = self.get_operand(*key);
                        let key = Self::value_to_index_or_key(&self.realm.heap, &key)
                            .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                        obj.as_object_mut().delete_own_element_or_property(key);
                    }

                    self.data.top_mut().set_result(*dest, Value::Bool(true));
                }

                Instr::ArrayPush { arr, value } => {
                    let len = {
                        let arr = self.get_operand_object(*arr)?;
                        arr.as_object().len()
                    };

                    let value = self.get_operand(*value);
                    let mut arr = self.get_operand_object_mut(*arr)?;
                    arr.as_object_mut().set_element(len, value);
                }
                Instr::ArrayNth { dest, arr, index } => {
                    let value = {
                        let arr = self.get_operand_object(*arr)?;
                        let num = self.get_operand(*index).expect_num()?;
                        let num_trunc = num.trunc();
                        if num_trunc == num {
                            let ndx = num_trunc as usize;
                            arr.as_object().get_element(ndx).unwrap_or(Value::Undefined)
                        } else {
                            Value::Undefined
                        }
                    };
                    self.data.top_mut().set_result(*dest, value);
                }
                Instr::ArraySetNth { .. } => todo!("ArraySetNth"),
                Instr::ArrayLen { dest, arr } => {
                    let len: usize = {
                        let arr = self.get_operand_object(*arr)?;
                        arr.as_object().len()
                    };
                    self.data
                        .top_mut()
                        .set_result(*dest, Value::Number(len as f64));
                }

                Instr::TypeOf { dest, arg: value } => {
                    let value = self.get_operand(*value);
                    let result = self.js_typeof(&value);
                    tprint!("-> {:?}", result);
                    self.data.top_mut().set_result(*dest, result);
                }

                Instr::BoolOpAnd(dest, a, b) => {
                    let a: bool = self.get_operand(*a).expect_bool()?;
                    let b: bool = self.get_operand(*b).expect_bool()?;
                    let res = a && b;
                    self.data.top_mut().set_result(*dest, Value::Bool(res));
                }
                Instr::BoolOpOr(dest, a, b) => {
                    let a: bool = self.get_operand(*a).expect_bool()?;
                    let b: bool = self.get_operand(*b).expect_bool()?;
                    let res = a || b;
                    self.data.top_mut().set_result(*dest, Value::Bool(res));
                }

                Instr::ClosureNew {
                    dest,
                    fnid,
                    forced_this,
                } => {
                    let mut upvalues = Vec::new();
                    while let Instr::ClosureAddCapture(cap) = func.instrs()[next_ndx as usize] {
                        let upv_id = self.data.top_mut().ensure_in_upvalue(cap);
                        {
                            self.print_indent();
                            tprintln!("        upvalue: {:?} -> {:?}", cap, upv_id);
                        }

                        upvalues.push(upv_id);

                        next_ndx += 1;
                    }

                    let forced_this = forced_this.map(|reg| self.get_operand(reg));
                    let module_id = self.data.top().header().fn_id.0;
                    let fnid = bytecode::FnId(module_id, *fnid);
                    let closure = Closure::JS(JSClosure {
                        fnid,
                        upvalues,
                        forced_this,
                    });

                    let oid = self.realm.heap.new_function(closure, HashMap::new());
                    self.data.top_mut().set_result(*dest, Value::Object(oid));
                }
                // This is always handled in the code for ClosureNew
                Instr::ClosureAddCapture(_) => {
                    unreachable!(
                        "interpreter bug: ClosureAddCapture should be handled with ClosureNew. (Usual cause: the bytecode compiler has placed some other instruction between ClosureAddCapture and ClosureNew.)"
                    )
                }

                Instr::UnaryMinus { dest, arg } => {
                    let arg_val: f64 = self.get_operand(*arg).expect_num()?;
                    self.data
                        .top_mut()
                        .set_result(*dest, Value::Number(-arg_val));
                }

                Instr::ImportModule(dest, module_path) => {
                    let bytecode::FnId(import_site, _) = self.data.top().header().fn_id;
                    let module_path = self.get_operand_string(*module_path)?.to_string();

                    let root_fnid = self
                        .loader
                        .load_import(&module_path, import_site)
                        .with_context(error!("while trying to import '{}'", module_path))?;

                    if let Some(module_oid) = self.realm.module_objs.get(&root_fnid.0) {
                        self.data
                            .top_mut()
                            .set_result(*dest, Value::Object(*module_oid));
                    } else {
                        // TODO Refactor with other implementations of Call?
                        let root_fn = self.loader.get_function(root_fnid).unwrap();
                        let n_instrs = root_fn.instrs().len().try_into().unwrap();

                        let call_meta = stack::CallMeta {
                            fnid: root_fnid,
                            n_instrs,
                            n_captured_upvalues: 0,
                            n_args: 0,
                            this: Value::Undefined,
                            return_value_reg: Some(*dest),
                            return_to_iid: Some(IID(self.iid.0 + 1)),
                        };

                        self.print_indent();
                        tprintln!("-- loading module {:?}", root_fnid.0);

                        self.data.push(call_meta);
                        self.iid = IID(0u16);
                        continue;
                    }
                }

                Instr::LoadNull(dest) => {
                    self.data.top_mut().set_result(*dest, Value::Null);
                }
                Instr::LoadUndefined(dest) => {
                    self.data.top_mut().set_result(*dest, Value::Undefined);
                }
                Instr::LoadThis(dest) => {
                    let value = self.data.top().header().this;
                    self.data.top_mut().set_result(*dest, value);
                }
                Instr::ArithInc(dest, src) => {
                    let val = self
                        .get_operand(*src)
                        .expect_num()
                        .map_err(|_| error!("bytecode bug: ArithInc on non-number"))?;
                    self.data
                        .top_mut()
                        .set_result(*dest, Value::Number(val + 1.0));
                }
                Instr::ArithDec(dest, src) => {
                    let val = self
                        .get_operand(*src)
                        .expect_num()
                        .map_err(|_| error!("bytecode bug: ArithDec on non-number"))?;
                    self.data
                        .top_mut()
                        .set_result(*dest, Value::Number(val - 1.0));
                }
                Instr::IsInstanceOf(dest, obj, sup) => {
                    let result = self.is_instance_of(*obj, *sup);
                    self.data.top_mut().set_result(*dest, Value::Bool(result));
                }
                Instr::NewIterator { dest: _, obj: _ } => todo!(),
                Instr::IteratorGetCurrent { dest: _, iter: _ } => todo!(),
                Instr::IteratorAdvance { iter: _ } => todo!(),
                Instr::JmpIfIteratorFinished { iter: _, dest: _ } => todo!(),

                Instr::Throw(_) => todo!(),

                Instr::StrCreateEmpty(dest) => {
                    let oid = self.realm.heap.new_string(String::new());
                    self.data.top_mut().set_result(*dest, Value::Object(oid));
                }
                Instr::StrAppend(buf_reg, tail) => {
                    let value = self.str_append(buf_reg, tail)?;
                    self.data.top_mut().set_result(*buf_reg, value);
                }

                Instr::GetGlobal { dest, key } => {
                    let key = self.get_operand(*key);
                    let key = Self::value_to_index_or_key(&self.realm.heap, &key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;

                    let gobj = self.realm.heap.get(self.realm.global_obj).unwrap();

                    let value = gobj
                        .as_object()
                        .get_own_element_or_property(key)
                        .unwrap_or(Value::Undefined);
                    self.data.top_mut().set_result(*dest, value);
                }

                Instr::Breakpoint => {
                    // We must update self.iid now, or the Interpreter will be back here on resume,
                    // in an infinite loop
                    self.iid.0 = next_ndx;
                    return Ok(ExitInternal::Suspended);
                }
            }

            tprintln!();

            #[cfg(enable_jit)]
            if let Some(jitting) = &mut self.jitting {
                jitting.builder.interpreter_step(&InterpreterStep {
                    fnid,
                    func,
                    iid: self.iid,
                    next_iid: IID(next_ndx),
                    get_operand: &|iid| self.data.get_result(iid).clone(),
                });
            }

            // TODO Checking for breakpoints here in this hot loop is going to be *very* slow!
            let giid = bytecode::GlobalIID(fnid, self.iid);
            let out_of_fuel = match &mut self.fuel {
                Fuel::Limited(count) => {
                    assert!(*count > 0);
                    *count -= 1;
                    if *count == 0 {
                        self.fuel = Fuel::Unlimited;
                        true
                    } else {
                        false
                    }
                }
                Fuel::Unlimited => false,
            };

            // Gotta increase IID even if we're about to suspend, or we'll be back here on resume
            self.iid.0 = next_ndx;

            if out_of_fuel || self.instr_bkpts.contains_key(&giid) {
                return Ok(ExitInternal::Suspended);
            }
        }

        Ok(ExitInternal::Finished)
    }

    fn str_append(&mut self, a: &VReg, b: &VReg) -> Result<Value> {
        // TODO Make this at least *decently* efficient!
        let mut buf = self.get_operand_string(*a)?.to_owned();
        {
            let tail_ref = self.get_operand_string(*b)?;
            let tail = tail_ref.deref();
            buf.push_str(tail);
        }
        let value = literal_to_value(bytecode::Literal::String(buf), &mut self.realm.heap);
        Ok(value)
    }

    fn is_instance_of(&mut self, obj: VReg, sup: VReg) -> bool {
        let sup_oid = match self.get_operand(sup) {
            Value::Object(oid) => oid,
            _ => return false,
        };

        let obj = match self.get_operand_object(obj) {
            Ok(obj) => obj,
            Err(_) => return false,
        };
        self.realm.heap.is_instance_of(obj.as_object(), sup_oid)
    }

    fn get_operand_string(&self, vreg: bytecode::VReg) -> Result<Ref<str>> {
        self.get_operand_object(vreg)?
            .as_str()
            .map_err(|_| error!("expected string, but got another type"))
    }

    fn dump_output(self) -> FinishedData {
        let sink = try_values_to_literals(&self.sink, &self.realm.heap);
        FinishedData { sink }
    }

    fn exit_function(&mut self, callee_retval_reg: Option<VReg>) -> Option<IID> {
        let return_value = callee_retval_reg
            .map(|vreg| self.get_operand(vreg))
            .unwrap_or(Value::Undefined);
        let header = self.data.top().header();
        let caller_retval_reg = header.return_value_vreg;
        let return_to_iid = header.return_to_iid;
        self.data.pop();

        // XXX This is a bug.  Get the top frame again.  But the compiler should say something
        // first!

        if let Some(vreg) = caller_retval_reg {
            self.data.top_mut().set_result(vreg, return_value);
        }

        #[cfg(enable_jit)]
        if let Some(jitting) = &mut self.jitting {
            jitting.builder.exit_function(callee_retval_reg);
        }

        return_to_iid
    }

    fn with_numbers<F>(&mut self, dest: VReg, a: VReg, b: VReg, op: F)
    where
        F: FnOnce(f64, f64) -> f64,
    {
        let a = self.get_operand(a).expect_num().unwrap();
        let b = self.get_operand(b).expect_num().unwrap();
        self.data
            .top_mut()
            .set_result(dest, Value::Number(op(a, b)));
    }

    fn compare(&mut self, dest: VReg, a: VReg, b: VReg, test: impl Fn(ValueOrdering) -> bool) {
        let a = self.get_operand(a);
        let b = self.get_operand(b);

        let ordering = match (&a, &b) {
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b).into(),
            (Value::Number(a), Value::Number(b)) => a
                .partial_cmp(b)
                .map(|x| x.into())
                .unwrap_or(ValueOrdering::Incomparable),
            (Value::Null, Value::Null) => ValueOrdering::Equal,
            (Value::Undefined, Value::Undefined) => ValueOrdering::Equal,
            (Value::Object(a), Value::Object(b)) => {
                let a = self.realm.heap.get(*a);
                let b = self.realm.heap.get(*b);
                match (a.as_deref(), b.as_deref()) {
                    (
                        Some(heap::HeapObject::StringObject(a)),
                        Some(heap::HeapObject::StringObject(b)),
                    ) => a.string().cmp(b.string()).into(),
                    _ => ValueOrdering::Incomparable,
                }
            }
            _ => ValueOrdering::Incomparable,
        };

        self.data
            .top_mut()
            .set_result(dest, Value::Bool(test(ordering)));
    }

    fn print_indent(&self) {
        #[cfg(test)]
        for _ in 0..(self.data.len() - 1) {
            tprint!("·   ");
        }
    }

    // TODO(cleanup) inline this function? It now adds nothing
    fn get_operand(&self, vreg: bytecode::VReg) -> Value {
        let value = self.data.top().get_result(vreg);
        //  TODO(cleanup) Move to a global logger. This is just for debugging!
        #[cfg(test)]
        {
            tprintln!();
            self.print_indent();
            tprint!("        {:?} = {:?}", vreg, value);
        }
        value
    }

    fn get_operand_object(&self, vreg: bytecode::VReg) -> Result<heap::ValueObjectRef> {
        let value = self.get_operand(vreg);
        as_object_ref(value, &self.realm.heap)
            .ok_or_else(|| error!("could not use as object: {:?}", vreg))
    }

    fn get_operand_object_mut(&self, vreg: bytecode::VReg) -> Result<heap::ValueObjectMut> {
        let value = self.get_operand(vreg);
        as_object_mut(value, &self.realm.heap)
            .ok_or_else(|| error!("could not use as object: {:?}", vreg))
    }

    fn js_typeof(&mut self, value: &Value) -> Value {
        use heap::Object;
        let ty_s = match value {
            Value::Number(_) => "number",
            Value::Bool(_) => "boolean",
            Value::Object(oid) => match self.realm.heap.get(*oid).unwrap().as_object().type_of() {
                heap::Typeof::Object => "object",
                heap::Typeof::Function => "function",
                heap::Typeof::String => "string",
                heap::Typeof::Number => "number",
                heap::Typeof::Boolean => "boolean",
            },
            // TODO(cleanup) This is actually an error in our type system.  null is really a value
            // of the 'object' type
            Value::Null => "object",
            Value::Undefined => "undefined",
            Value::SelfFunction => "function",
            Value::Internal(_) => panic!("internal value has no typeof!"),
        };

        literal_to_value(
            bytecode::Literal::String(ty_s.to_string()),
            &mut self.realm.heap,
        )
    }

    fn value_to_index_or_key<'h>(
        heap: &'h heap::Heap,
        value: &Value,
    ) -> Option<heap::IndexOrKey<Ref<'h, str>>> {
        match value {
            Value::Number(n) if *n >= 0.0 => {
                let n_trunc = n.trunc();
                if *n == n_trunc {
                    let ndx = n_trunc as usize;
                    Some(heap::IndexOrKey::Index(ndx))
                } else {
                    None
                }
            }
            Value::Object(oid) => {
                let obj = heap.get(*oid)?;
                Ref::filter_map(obj, |hobj| hobj.as_str())
                    .map(|s| heap::IndexOrKey::Key(s))
                    .ok()
            }
            _ => None,
        }
    }

    /// Converts the given value to a boolean (e.g. for use by `if`)
    ///
    /// See: https://262.ecma-international.org/14.0/#sec-toboolean
    fn to_boolean(&self, value: Value) -> bool {
        match value {
            Value::Bool(bool_val) => bool_val,
            Value::Number(num) => num != 0.0,
            Value::Object(oid) => match self.realm.heap.get(oid).unwrap().deref() {
                heap::HeapObject::OrdObject(_) => true,
                heap::HeapObject::ClosureObject(_) => true,
                heap::HeapObject::StringObject(sobj) => !sobj.string().is_empty(),
            },
            Value::Null => false,
            Value::Undefined => false,
            Value::SelfFunction => true,
            Value::Internal(_) => {
                panic!("bytecode compiler bug: internal value should be unreachable")
            }
        }
    }
}

fn init_stack(loader: &mut loader::Loader, fnid: FnId) -> stack::InterpreterData {
    let mut data = stack::InterpreterData::new();
    let root_fn = loader.get_function(fnid).unwrap();
    let n_instrs = root_fn.instrs().len().try_into().unwrap();

    data.push(stack::CallMeta {
        fnid,
        n_instrs,
        n_captured_upvalues: 0,
        n_args: 0,
        this: Value::Undefined,
        return_value_reg: None,
        return_to_iid: None,
    });

    data
}

fn as_object_ref<'h>(value: Value, heap: &'h heap::Heap) -> Option<heap::ValueObjectRef<'h>> {
    match value {
        Value::Object(oid) => heap.get(oid).map(Into::into),
        Value::Number(num) => Some(heap::NumberObject(num).into()),
        Value::Bool(bool) => Some(heap::BoolObject(bool).into()),
        _ => None,
    }
}

fn as_object_mut<'h>(value: Value, heap: &'h heap::Heap) -> Option<heap::ValueObjectMut<'h>> {
    match value {
        Value::Object(oid) => heap.get_mut(oid).map(Into::into),
        Value::Number(num) => Some(heap::NumberObject(num).into()),
        Value::Bool(bool) => Some(heap::BoolObject(bool).into()),
        _ => None,
    }
}

fn try_values_to_literals(vec: &[Value], heap: &heap::Heap) -> Vec<Option<bytecode::Literal>> {
    vec.iter()
        .map(|value| try_value_to_literal(*value, &heap))
        .collect()
}

/// Create a Value based on the given Literal.
///
/// It may allocate an object in the GC-managed heap.
fn literal_to_value(lit: bytecode::Literal, heap: &mut heap::Heap) -> Value {
    match lit {
        bytecode::Literal::Number(nu) => Value::Number(nu),
        bytecode::Literal::String(st) => {
            // TODO(performance) avoid this allocation
            let oid = heap.new_string(st.clone());
            Value::Object(oid)
        }
        bytecode::Literal::Bool(bo) => Value::Bool(bo),
        bytecode::Literal::Null => Value::Null,
        bytecode::Literal::Undefined => Value::Undefined,
        bytecode::Literal::SelfFunction => todo!(),
    }
}

fn try_value_to_literal(value: Value, heap: &heap::Heap) -> Option<bytecode::Literal> {
    match value {
        Value::Number(num) => Some(bytecode::Literal::Number(num)),
        Value::Bool(b) => Some(bytecode::Literal::Bool(b)),
        Value::Object(oid) => {
            let hobj = heap.get(oid)?;
            if let heap::HeapObject::StringObject(sobj) = hobj.deref() {
                let s = sobj.string().to_owned();
                Some(bytecode::Literal::String(s.to_owned()))
            } else {
                None
            }
        }
        Value::Null => Some(bytecode::Literal::Null),
        Value::Undefined => Some(bytecode::Literal::Undefined),
        Value::SelfFunction => None,
        Value::Internal(_) => None,
    }
}

fn init_builtins(heap: &mut heap::Heap) -> heap::ObjectId {
    #![allow(non_snake_case)]

    let mut global = HashMap::new();

    {
        let Array_push = heap.new_function(Closure::Native(nf_Array_push), HashMap::new());
        let Array_pop = heap.new_function(Closure::Native(nf_Array_pop), HashMap::new());
        let mut array_proto = HashMap::new();
        array_proto.insert("push".to_string(), Value::Object(Array_push));
        array_proto.insert("pop".to_string(), Value::Object(Array_pop));
        let array_proto = heap.new_ordinary_object(array_proto);

        let Array_isArray = heap.new_function(Closure::Native(nf_Array_isArray), HashMap::new());
        let mut array_cons = HashMap::new();
        array_cons.insert("isArray".to_string(), Value::Object(Array_isArray));
        array_cons.insert("prototype".to_string(), Value::Object(array_proto));
        let array_cons = heap.new_function(Closure::Native(nf_Array), array_cons);

        global.insert("Array".to_string(), Value::Object(array_cons));
    }

    let RegExp = heap.new_function(Closure::Native(nf_RegExp), HashMap::new());
    global.insert("RegExp".to_string(), Value::Object(RegExp));

    let mut number_cons_props = HashMap::new();
    number_cons_props.insert("prototype".to_string(), Value::Object(heap.number_proto()));
    {
        let Number_prototype_toString = heap.new_function(
            Closure::Native(nf_Number_prototype_toString),
            HashMap::new(),
        );
        let oid = heap.number_proto();
        let mut number_proto = heap.get_mut(oid).unwrap();
        number_proto.as_object_mut().set_own_property(
            "toString".to_string(),
            Value::Object(Number_prototype_toString),
        )
    }
    let Number = heap.new_function(Closure::Native(nf_Number), number_cons_props);
    global.insert("Number".to_string(), Value::Object(Number));

    let String = heap.new_function(Closure::Native(nf_String), HashMap::new());
    global.insert("String".to_string(), Value::Object(String));

    let Boolean = heap.new_function(Closure::Native(nf_Boolean), HashMap::new());
    global.insert("Boolean".to_string(), Value::Object(Boolean));

    // builtins.insert("Boolean".into(), NativeFnId::BooleanNew as u32);
    // builtins.insert("Object".into(), NativeFnId::ObjectNew as u32);
    // builtins.insert("parseInt".into(), NativeFnId::ParseInt as u32);
    // builtins.insert("SyntaxError".into(), NativeFnId::SyntaxErrorNew as u32);
    // builtins.insert("TypeError".into(), NativeFnId::TypeErrorNew as u32);
    // builtins.insert("Math_floor".into(), NativeFnId::MathFloor as u32);

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    heap.new_ordinary_object(global)
}

#[allow(non_snake_case)]
fn nf_Array_isArray(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = if let Some(Value::Object(oid)) = args.get(0) {
        let obj = intrp
            .realm
            .heap
            .get(*oid)
            .ok_or_else(|| error!("no such object!"))?;
        if let heap::HeapObject::OrdObject(obj) = obj.deref() {
            obj.is_array()
        } else {
            false
        }
    } else {
        false
    };

    Ok(Value::Bool(value))
}

#[allow(non_snake_case)]
fn nf_Array_push(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let oid = this.expect_obj().unwrap();
    let mut arr = intrp.realm.heap.get_mut(oid).unwrap();
    let len = arr.as_object().len();
    let value = args.get(0).unwrap().clone();
    arr.as_object_mut().set_element(len, value);
    Ok(Value::Undefined)
}

#[allow(non_snake_case)]
fn nf_Array_pop(_intrp: &mut Interpreter, _this: &Value, _args: &[Value]) -> Result<Value> {
    todo!("nf_Array_pop")
}

#[allow(non_snake_case)]
fn nf_RegExp(intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    // TODO
    let oid = intrp.realm.heap.new_ordinary_object(HashMap::new());
    Ok(Value::Object(oid))
}

#[allow(non_snake_case)]
fn nf_Array(intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    // TODO
    let oid = intrp.realm.heap.new_array(Vec::new());
    Ok(Value::Object(oid))
}

#[allow(non_snake_case)]
fn nf_Number(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Number(0.0))
}

#[allow(non_snake_case)]
fn nf_Number_prototype_toString(
    intrp: &mut Interpreter,
    this: &Value,
    _: &[Value],
) -> Result<Value> {
    let num_value = match this {
        Value::Number(num_value) => num_value,
        _ => return Err(error!("Not a number value!")),
    };

    let num_str = num_value.to_string();
    let oid = intrp.realm.heap.new_string(num_str);
    Ok(Value::Object(oid))
}

#[allow(non_snake_case)]

fn nf_String(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    let value_str = match args.get(0) {
        Some(Value::Number(num)) => num.to_string(),
        Some(Value::Bool(true)) => "true".into(),
        Some(Value::Bool(false)) => "false".into(),
        Some(Value::Object(oid)) => match intrp.realm.heap.get(*oid).unwrap().deref() {
            heap::HeapObject::OrdObject(_) => "<object>".to_owned(),
            heap::HeapObject::StringObject(sobj) => sobj.string().clone(),
            heap::HeapObject::ClosureObject(_) => "<closure>".to_owned(),
        },
        Some(Value::Null) => "null".into(),
        Some(Value::Undefined) => "undefined".into(),
        Some(Value::SelfFunction) => "<function>".into(),
        Some(Value::Internal(_)) => unreachable!(),
        None => "".into(),
    };

    Ok(literal_to_value(
        bytecode::Literal::String(value_str),
        &mut intrp.realm.heap,
    ))
}

#[allow(non_snake_case)]
fn nf_Boolean(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Bool(false))
}

#[derive(PartialEq, Eq)]
enum ValueOrdering {
    Equal,
    Less,
    Greater,
    /// Not equal, but also not less nor greater.  Just always `!==`
    Incomparable,
}
impl From<std::cmp::Ordering> for ValueOrdering {
    fn from(cmp_ordering: std::cmp::Ordering) -> Self {
        match cmp_ordering {
            Ordering::Less => ValueOrdering::Less,
            Ordering::Equal => ValueOrdering::Equal,
            Ordering::Greater => ValueOrdering::Greater,
        }
    }
}

#[cfg(any(test, feature = "debugger"))]
pub mod debugger {
    use std::cell::Ref;
    use std::collections::HashSet;

    use crate::common::Result;
    use crate::{bytecode, InterpreterValue};
    use crate::{error, heap};

    use super::{Fuel, InstrBreakpoint, Interpreter, SourceBreakpoint};

    pub use heap::ObjectId;

    /// The only real entry point to all the debugging features present in the
    /// Interpreter.
    ///
    /// Create it it by calling Probe::attach with a reference to the Interpreter. The
    /// Interpreter will stay suspended as long as the Probe exists. (Since the
    /// interpreter only runs within a call to Interpreter::run, this implies that the
    /// Interpreter is suspended at that time. And since the Probe acquires an
    /// exclusive &mut reference to the Interpreter, you won't be able
    /// to call Interpreter::run until dropping the probe).
    ///
    /// In general, a Probe can be used to:
    ///  - query the interpreter's state (examine the call stack, local variables, etc.)
    ///  - place breakpoints
    pub struct Probe<'a, 'b> {
        interpreter: &'a mut Interpreter<'b>,
    }

    pub use crate::loader::BreakRangeID;

    #[derive(Debug)]
    pub enum BreakpointError {
        /// Breakpoint already set at the given location (can't have more than 1 at the same
        /// location).
        AlreadyThere,
        InvalidLocation,
    }

    impl BreakpointError {
        pub fn message(&self) -> &'static str {
            match self {
                BreakpointError::AlreadyThere => "A breakpoint had already been set there",
                BreakpointError::InvalidLocation => "Invalid location to place a breakpoint at",
            }
        }
    }

    impl<'a, 'b> Probe<'a, 'b> {
        pub fn attach(interpreter: &'a mut Interpreter<'b>) -> Self {
            Probe { interpreter }
        }

        pub fn giid(&self) -> bytecode::GlobalIID {
            let frame = self.interpreter.data.top();

            bytecode::GlobalIID(frame.header().fn_id, self.interpreter.iid)
        }

        pub fn sink(&self) -> &[InterpreterValue] {
            self.interpreter.sink.as_slice()
        }

        pub fn set_fuel(&mut self, fuel: Fuel) {
            self.interpreter.fuel = fuel;
        }

        /// Set a breakpoint at the specified instruction.
        ///
        /// After this operation, the interpreter will suspend itself (Interpreter::run
        /// will return Exit::Suspended), and it will be possible to examine its
        /// state (by attaching a new Probe on it).
        pub fn set_source_breakpoint(
            &mut self,
            brange_id: BreakRangeID,
        ) -> std::result::Result<(), BreakpointError> {
            if self.interpreter.source_bkpts.contains_key(&brange_id) {
                return Err(BreakpointError::AlreadyThere);
            }

            let giid = self.giid_of_break_range(brange_id)?;
            let bkpt = InstrBreakpoint {
                src_bkpt: Some(brange_id),
            };
            self.add_instr_bkpt(giid, bkpt)?;

            let prev = self
                .interpreter
                .source_bkpts
                .insert(brange_id, SourceBreakpoint);
            assert!(prev.is_none());

            Ok(())
        }

        fn giid_of_break_range(
            &mut self,
            brange_id: BreakRangeID,
        ) -> std::result::Result<bytecode::GlobalIID, BreakpointError> {
            let break_range: &bytecode::BreakRange = self
                .interpreter
                .loader
                .get_break_range(brange_id)
                .ok_or(BreakpointError::InvalidLocation)?;
            Ok(bytecode::GlobalIID(
                bytecode::FnId(brange_id.module_id(), break_range.local_fnid),
                break_range.iid_start,
            ))
        }

        ///
        /// Delete the breakpoint with the given ID.
        ///
        /// Returns true only if there was actually a breakpoint with the given ID; false
        /// if the ID did not correspond to any breakpoint.
        pub fn clear_source_breakpoint(
            &mut self,
            brange_id: BreakRangeID,
        ) -> std::result::Result<bool, BreakpointError> {
            if !self.interpreter.source_bkpts.contains_key(&brange_id) {
                return Ok(false);
            }

            let giid = self.giid_of_break_range(brange_id)?;
            let was_there = self.clear_instr_breakpoint(giid);
            assert!(was_there);

            Ok(true)
        }

        pub fn source_breakpoints(
            &self,
        ) -> impl ExactSizeIterator<Item = (BreakRangeID, &SourceBreakpoint)> {
            self.interpreter.source_bkpts.iter().map(|(k, v)| (*k, v))
        }

        pub fn set_instr_breakpoint(
            &mut self,
            giid: bytecode::GlobalIID,
        ) -> std::result::Result<(), BreakpointError> {
            let bkpt = InstrBreakpoint { src_bkpt: None };
            self.add_instr_bkpt(giid, bkpt)
        }

        fn add_instr_bkpt(
            &mut self,
            giid: bytecode::GlobalIID,
            bkpt: InstrBreakpoint,
        ) -> std::result::Result<(), BreakpointError> {
            let new_insert = self.interpreter.instr_bkpts.insert(giid, bkpt);
            match new_insert {
                None => Ok(()),
                Some(_) => Err(BreakpointError::AlreadyThere),
            }
        }

        pub fn clear_instr_breakpoint(&mut self, giid: bytecode::GlobalIID) -> bool {
            let ibkpt = self.interpreter.instr_bkpts.remove(&giid);
            if let Some(ibkpt) = &ibkpt {
                if let Some(brid) = &ibkpt.src_bkpt {
                    self.interpreter.source_bkpts.remove(brid).unwrap();
                }
            }

            ibkpt.is_some()
        }

        pub fn instr_breakpoints(&self) -> impl '_ + ExactSizeIterator<Item = bytecode::GlobalIID> {
            self.interpreter.instr_bkpts.keys().copied()
        }

        pub fn loader(&self) -> &crate::loader::Loader {
            &self.interpreter.loader
        }

        /// Returns the sequence of stack frames in the form of an iterator, ordered top
        /// to bottom.
        pub fn frames(&self) -> impl ExactSizeIterator<Item = crate::stack::Frame> {
            self.interpreter.data.frames()
        }

        /// Get the instruction pointer for the n-th frame (0 = top)
        ///
        /// Panics if `frame_ndx` is invalid.
        pub fn frame_giid(&self, frame_ndx: usize) -> bytecode::GlobalIID {
            // This function is necessary because we actually store the
            // "instruction pointer" for the i-th frame in the (i - 1)-frame,
            // (the topmost frame's is stored in the interpreter's state)

            let mut frames = self.frames();
            let (fnid, iid) = if frame_ndx == 0 {
                let iid = self.interpreter.iid;
                let fnid = frames.next().unwrap().header().fn_id;
                (fnid, iid)
            } else {
                let frame_above = frames.nth(frame_ndx - 1).unwrap();
                let frame = frames.next().unwrap();

                let fnid = frame.header().fn_id;
                let iid = frame_above.header().return_to_iid.unwrap();

                (fnid, iid)
            };

            // For the top frame: `iid` is the instruction to *resume* to.
            // For other frames: `iid` is the instruction to *return* to.
            // In either case: we actually want the instruction we suspended/called at,
            // which is the previous one.

            let iid = bytecode::IID(iid.0);
            bytecode::GlobalIID(fnid, iid)
        }

        pub fn get_object(&self, obj_id: heap::ObjectId) -> Option<Ref<heap::HeapObject>> {
            self.interpreter.realm.heap.get(obj_id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Literal;

    fn quick_run(code: &str) -> FinishedData {
        let res = std::panic::catch_unwind(|| {
            let mut prereq = prepare_vm(code);
            let vm = prereq.make_vm();
            vm.run().unwrap().expect_finished()
        });

        if let Err(err) = &res {
            println!("quick_run: error: {:?}", err);
        }
        res.unwrap()
    }

    fn prepare_vm(code: &str) -> VMPrereq {
        let filename = "<input>".to_string();
        let mut loader = loader::Loader::new(None);
        let chunk_fnid = loader
            .load_script(Some(filename), code.to_string())
            .expect("couldn't compile test script");
        VMPrereq {
            loader,
            root_fnid: chunk_fnid,
            realm: Realm::new(),
        }
    }

    struct VMPrereq {
        loader: loader::Loader,
        root_fnid: FnId,
        realm: Realm,
    }
    impl VMPrereq {
        fn make_vm(&mut self) -> Interpreter {
            Interpreter::new(&mut self.realm, &mut self.loader, self.root_fnid)
        }
    }

    #[test]
    fn test_simple_call() {
        let output = quick_run("/* Here is some simple code: */ sink(1 + 4 + 99); ");
        assert_eq!(&[Some(Literal::Number(104.0))], &output.sink[..]);
    }

    #[test]
    fn test_multiple_calls() {
        let output = quick_run("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ");
        assert_eq!(
            &[
                Some(Literal::Number(12. * 5.)),
                Some(Literal::Number(99. - 15.))
            ],
            &output.sink[..]
        );
    }

    #[test]
    fn test_if() {
        let output = quick_run(
            "
            const x = 123;
            let y = 'a';
            if (x < 200) {
                y = 'b';
            }
            sink(y);
            ",
        );

        assert_eq!(&[Some(Literal::String("b".to_owned()))], &output.sink[..]);
    }

    #[test]
    fn test_simple_fn() {
        let output = quick_run(
            "
            function foo(a, b) { return a + b; }
            sink(foo(1, 2));
            ",
        );

        assert_eq!(&[Some(Literal::Number(3.0))], &output.sink[..]);
    }

    #[test]
    fn test_fn_with_branch() {
        let output = quick_run(
            "
            function foo(mode, a, b) {
                if (mode === 'sum')
                    return a + b;
                else if (mode === 'product')
                    return a * b;
                else
                    return 'something else';
            }

            sink(foo('product', 9, 8));
            ",
        );

        assert_eq!(&[Some(Literal::Number(9.0 * 8.0))], &output.sink[..]);
    }

    #[test]
    fn test_while() {
        let output = quick_run(
            "
            function sum_range(n) {
                let i = 0;
                let ret = 0;
                while (i <= n) {
                    ret += i;
                    i++;
                }
                return ret;
            }

            sink(sum_range(4));
            ",
        );

        assert_eq!(&[Some(Literal::Number(10.0))], &output.sink[..]);
    }

    fn try_casting_bool(code: &str, expected_value: bool) {
        let output = quick_run(code);
        assert_eq!(&[Some(Literal::Bool(expected_value))], &output.sink[..]);
    }
    #[test]
    fn test_boolean_cast_from_number() {
        try_casting_bool("sink(!123.994);", false);
        try_casting_bool("sink(!-123.994);", false);
        try_casting_bool("sink(!0.0);", true);
    }
    #[test]
    fn test_boolean_cast_from_string() {
        try_casting_bool("sink(!'');", true);
        try_casting_bool("sink(!'asdlol');", false);
    }
    #[test]
    fn test_boolean_cast_from_bool() {
        try_casting_bool("sink(!false);", true);
        try_casting_bool("sink(!true);", false);
    }
    #[test]
    fn test_boolean_cast_from_object() {
        try_casting_bool("sink(!{a: 1, b: 2});", false);
        try_casting_bool("sink(!{});", false);
    }
    #[test]
    fn test_boolean_cast_from_null() {
        try_casting_bool("sink(!null);", true);
    }
    #[test]
    fn test_boolean_cast_from_undefined() {
        try_casting_bool("sink(!undefined);", true);
    }
    #[test]
    fn test_boolean_cast_from_function() {
        try_casting_bool(
            "
            function my_fun() { }
            sink(!my_fun);
            ",
            false,
        );
    }

    #[test]
    fn test_capture() {
        let output = quick_run(
            "
            let counter = 0;

            function f() {
                function g() {
                    counter++;
                }
                g();
                g();
                sink(counter);
            }

            f();
            f();
            f();
            counter -= 5;
            sink(counter);
            ",
        );

        assert_eq!(
            &[
                Some(Literal::Number(2.0)),
                Some(Literal::Number(4.0)),
                Some(Literal::Number(6.0)),
                Some(Literal::Number(1.0))
            ],
            &output.sink[..]
        );
    }

    #[test]
    fn test_object_init() {
        let output = quick_run(
            "
            const obj = {
                aString: 'asdlol123',
                aNumber: 1239423.4518923,
                anotherObject: { x: 123, y: 899 },
                aFunction: function(pt) { return 42; }
            }

            sink(obj.aString)
            sink(obj.aNumber)
            sink(obj.anotherObject.x)
            sink(obj.anotherObject.y)
            sink(obj.aFunction())
            ",
        );

        assert_eq!(5, output.sink.len());
        assert_eq!(&Some(Literal::String("asdlol123".into())), &output.sink[0]);
        assert_eq!(&Some(Literal::Number(1239423.4518923)), &output.sink[1]);
        assert_eq!(&Some(Literal::Number(123.0)), &output.sink[2]);
        assert_eq!(&Some(Literal::Number(899.0)), &output.sink[3]);
        assert_eq!(&Some(Literal::Number(42.0)), &output.sink[4]);
    }

    #[test]
    fn test_typeof() {
        let output = quick_run(
            "
            let anObj = {}

            sink(typeof undefined)
            sink(typeof anObj.aNonExistantProperty)

            sink(typeof null)
            sink(typeof {})
            sink(typeof anObj)

            sink(typeof true)
            sink(typeof false)

            sink(typeof 123.0)
            sink(typeof -99.2)
            sink(typeof (156.0/0))
            sink(typeof (-156.0/0))
            sink(typeof (0/0))

            sink(typeof '')
            sink(typeof 'a string')

            sink(typeof (function() {}))
            ",
        );

        assert_eq!(
            &output.sink[..],
            &[
                Some(Literal::String("undefined".into())),
                Some(Literal::String("undefined".into())),
                Some(Literal::String("object".into())),
                Some(Literal::String("object".into())),
                Some(Literal::String("object".into())),
                Some(Literal::String("boolean".into())),
                Some(Literal::String("boolean".into())),
                Some(Literal::String("number".into())),
                Some(Literal::String("number".into())),
                Some(Literal::String("number".into())),
                Some(Literal::String("number".into())),
                Some(Literal::String("number".into())),
                Some(Literal::String("string".into())),
                Some(Literal::String("string".into())),
                Some(Literal::String("function".into())),
                // TODO(feat) BigInt (typeof -> "bigint")
                // TODO(feat) Symbol (typeof -> "symbol")
            ]
        );
    }

    #[test]
    fn test_object_member_set() {
        let output = quick_run(
            "
            const pt = { x: 123, y: 4 }

            sink(pt.x)
            sink(pt.y)
            pt.y = 999
            sink(pt.x)
            sink(pt.y)
            ",
        );

        assert_eq!(4, output.sink.len());
        assert_eq!(&Some(Literal::Number(123.0)), &output.sink[0]);
        assert_eq!(&Some(Literal::Number(4.0)), &output.sink[1]);
        assert_eq!(&Some(Literal::Number(123.0)), &output.sink[2]);
        assert_eq!(&Some(Literal::Number(999.0)), &output.sink[3]);
    }

    #[test]
    fn test_object_prototype() {
        let output = quick_run(
            "
            const a = { count: 99, name: 'lol', pos: {x: 32, y: 99} }
            const b = { name: 'another name' }
            b.__proto__ = a
            const c = { __proto__: b, count: 0 }

            sink(c.pos.y)
            sink(c.pos.x)
            c.pos.x = 12304
            sink(b.pos.x)
            sink(c.count)
            sink(c.name)
            b.name = 'another name yet'
            sink(c.name)
            ",
        );

        assert_eq!(
            &output.sink[..],
            &[
                Some(Literal::Number(99.0)),
                Some(Literal::Number(32.0)),
                Some(Literal::Number(12304.0)),
                Some(Literal::Number(0.0)),
                Some(Literal::String("another name".into())),
                Some(Literal::String("another name yet".into())),
            ]
        );
    }

    #[test]
    fn test_for_in() {
        // TODO(small feat) This syntax is not yet implemented
        let output = quick_run(
            "
            const obj = {
                x: 12.0,
                y: 90.2,
                name: 'THE SPOT',
            };

            for (const name in obj) sink(name);
            ",
        );

        let mut sink: Vec<_> = output
            .sink
            .into_iter()
            .map(|value| match value {
                Some(Literal::String(s)) => s,
                other => panic!("not a String: {:?}", other),
            })
            .collect();
        sink.sort();
        assert_eq!(&sink[..], &["name", "x", "y"]);
    }

    #[test]
    fn test_builtin() {
        let output = quick_run(
            "
            sink(Array.isArray([1, 2, 3]));
            sink(Array.isArray('not an array'));
            ",
        );

        assert_eq!(
            &output.sink,
            &[Some(Literal::Bool(true)), Some(Literal::Bool(false))]
        );
    }

    #[test]
    fn test_this() {
        let output = quick_run(
            r#"
            function getThis() {
                return this;
            }
            function getThisViaArrowFunc() {
                const f = () => { return this; };
                return f();
            }

            const obj1 = { name: "obj1" };
            const obj2 = { name: "obj2" };
            obj1.getThis = getThis;
            obj2.getThis = getThis;
            const obj3 = { __proto__: obj1, name: "obj3" };
            const obj4 = {
              name: "obj4",
              getThis() { return this },
            };
            const obj5 = { name: "obj5" };
            obj5.getThis = obj4.getThis;

            sink(obj1.getThis().name);
            sink(obj2.getThis().name);
            sink(obj3.getThis().name);
            sink(obj5.getThis().name);
            sink(getThis());

            obj5.getThisViaArrowFunc = getThisViaArrowFunc;
            sink(obj5.getThisViaArrowFunc().name);
            "#,
        );
        assert_eq!(
            &output.sink,
            &[
                Some(Literal::String("obj1".into())),
                Some(Literal::String("obj2".into())),
                Some(Literal::String("obj3".into())),
                Some(Literal::String("obj5".into())),
                Some(Literal::Undefined),
                Some(Literal::String("obj5".into())),
            ],
        );
    }

    #[test]
    fn test_methods_on_numbers() {
        let output = quick_run(
            r#"
            const num = 123.45;

            Number.prototype.greet = function() { return "Hello, I'm " + this.toString() + "!" }

            sink(num.greet())
            "#,
        );

        assert_eq!(
            &output.sink,
            &[Some(Literal::String("Hello, I'm 123.45!".into())),],
        );
    }

    #[test]
    fn test_array_access() {
        let output = quick_run(
            r#"
            const xs = ['a', 'b', 'c'];

            sink(xs[-1])
            sink(xs[0])
            sink(xs[1])
            sink(xs[2])
            sink(xs[3])
            sink(xs.length)
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Undefined),
                Some(Literal::String("a".to_string())),
                Some(Literal::String("b".to_string())),
                Some(Literal::String("c".to_string())),
                Some(Literal::Undefined),
                Some(Literal::Number(3.0)),
            ],
        );
    }
}
