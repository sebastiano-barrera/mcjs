use std::cell::Ref;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{
    bytecode::{self, FnId, GlobalIID, Instr, VReg, IID},
    common::{Context, Result},
    error,
    heap::{self, Heap, IndexOrKey, Object},
    loader::{self, BreakRangeID},
    // jit::{self, InterpreterStep},
    stack,
    util::pop_while,
};

pub use crate::common::Error;

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
    #[cfg(feature = "debugger")]
    pub fn fnid(&self) -> FnId {
        self.fnid
    }

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

#[cfg_attr(not(enable_jit), allow(dead_code))]
#[derive(Clone, Default)]
pub struct Options {
    #[cfg(enable_jit)]
    pub jit_mode: JitMode,
}
#[cfg(enable_jit)]
#[derive(Clone, Copy, Debug)]
pub enum JitMode {
    Compile,
    UseTraces,
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

#[allow(clippy::new_without_default)]
impl Realm {
    pub fn new(loader: &mut loader::Loader) -> Realm {
        let mut heap = heap::Heap::new();
        let global_obj = init_builtins(&mut heap);
        let mut realm = Realm {
            heap,
            module_objs: HashMap::new(),
            global_obj,
        };

        let boot_script_fnid = loader.boot_script_fnid();
        Interpreter::new(&mut realm, loader, boot_script_fnid)
            .run()
            .unwrap()
            .expect_finished();

        realm
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

    current_exc: Option<Value>,
    exc_handler_stack: Vec<ExcHandler>,

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

struct ExcHandler {
    /// Size/length/height of the stack at the time when the exception handler
    /// was installed. This is used to pop the correct amount of stack frames as
    /// we unwind it during handling.
    stack_height: usize,

    /// IID to jump to.  The instruction is assumed to be in the same function where the exception
    /// handler was installed (tracked via stack_height).
    target_iid: IID,
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
        let opts = Default::default();
        Self::with_options(opts, realm, loader, fnid)
    }

    pub fn with_options(
        opts: Options,
        realm: &'a mut Realm,
        loader: &'a mut loader::Loader,
        fnid: bytecode::FnId,
    ) -> Self {
        // Initialize the stack with a single frame, corresponding to a call to fnid with no
        // parameters
        let data = init_stack(loader, realm, fnid);
        Interpreter {
            realm,
            iid: bytecode::IID(0),
            data,
            loader,
            current_exc: None,
            exc_handler_stack: Vec::new(),
            sink: Vec::new(),
            opts,
            instr_bkpts: HashMap::new(),
            source_bkpts: HashMap::new(),
            fuel: Fuel::Unlimited,
            #[cfg(enable_jit)]
            jitting: None,
        }
    }

    pub fn restart(mut self) -> Self {
        self.data = {
            let bottom_frame = self.data.frames().last().unwrap();
            let root_fnid = bottom_frame.header().fn_id;
            init_stack(self.loader, self.realm, root_fnid)
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
        while !self.data.is_empty() {
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
                self.exit_function(None);
                continue;
            }

            let instr = func.instrs()[self.iid.0 as usize];
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

                Instr::OpAdd(dest, a, b) => match self.get_operand(*a) {
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
                    self.exit_function(Some(*value));
                    continue;
                }
                Instr::Call {
                    callee,
                    this,
                    return_value,
                } => {
                    let oid = self.get_operand(*callee).expect_obj()?;
                    let ho_ref = self
                        .realm
                        .heap
                        .get(oid)
                        .ok_or_else(|| error!("invalid function (object is not callable)"))?
                        .borrow();
                    let closure: &Closure = ho_ref
                        .as_closure()
                        .ok_or_else(|| error!("can't call non-closure"))?;

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
                            let n_params = bytecode::ARGS_COUNT_MAX as usize;
                            arg_vals.truncate(n_params);
                            arg_vals.resize(n_params, Value::Undefined);
                            assert_eq!(arg_vals.len(), n_params);

                            let this = closure
                                .forced_this
                                .unwrap_or_else(|| self.get_operand(*this));
                            // "this" substitution: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Strict_mode#no_this_substitution
                            // If I understand this correctly, we don't need to box anything right
                            // now.  We just pass the value, and the callee will box it when
                            // needed.
                            let this = match (callee_func.is_strict_mode(), this) {
                                (false, Value::Null | Value::Undefined) => {
                                    Value::Object(self.realm.global_obj)
                                }
                                (_, other) => other,
                            };

                            let call_meta = stack::CallMeta {
                                fnid: closure.fnid,
                                // TODO Actually, we just need to allocate enough space for
                                // *variables*, not for instructions.  However, this is OK for now,
                                // as n_instrs is always >= n_variables.
                                n_regs: callee_func.n_regs() as u32,
                                captures: &closure.upvalues,
                                this,
                            };

                            self.data
                                .top_mut()
                                .set_return_target(return_to_iid, *return_value);

                            self.data.push(call_meta);
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
                            drop(ho_ref);

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

                    obj.set_own_element_or_property(key.to_ref(), value);
                }
                Instr::ObjGet { dest, obj, key } => {
                    let obj = self.get_operand_object(*obj)?;
                    let key = self.get_operand(*key);
                    let key = Self::value_to_index_or_key(&self.realm.heap, &key);

                    let value = match key {
                        Some(ik @ heap::IndexOrKeyOwned::Index(_)) => {
                            obj.get_own_element_or_property(ik.to_ref())
                        }
                        Some(heap::IndexOrKeyOwned::Key(key)) => {
                            self.realm.heap.get_property_chained(&obj, &key)
                        }
                        None => None,
                    }
                    .unwrap_or(Value::Undefined);

                    self.data.top_mut().set_result(*dest, value);
                }
                Instr::ObjGetKeys { dest, obj } => {
                    // TODO Something more efficient?
                    let obj = self.get_operand_object(*obj)?;
                    let keys = obj
                        .own_properties()
                        .into_iter()
                        .map(|name| Value::Object(self.realm.heap.new_string(name)))
                        .collect();

                    let keys_oid = self.realm.heap.new_array(keys);
                    self.data
                        .top_mut()
                        .set_result(*dest, Value::Object(keys_oid));
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
                        obj.delete_own_element_or_property(key.to_ref());
                    }

                    self.data.top_mut().set_result(*dest, Value::Bool(true));
                }

                Instr::ArrayPush { arr, value } => {
                    let value = self.get_operand(*value);
                    let mut arr = self
                        .get_operand_object_mut(*arr)?
                        .into_heap_cell()
                        .ok_or_else(|| error!("not an array!"))?
                        .borrow_mut();

                    let was_array = arr.array_push(value);
                    assert!(was_array);
                }
                Instr::ArrayNth { dest, arr, index } => {
                    let value = {
                        let arr = self
                            .get_operand_object(*arr)?
                            .into_heap_cell()
                            .ok_or_else(|| error!("not an array!"))?
                            .borrow();
                        let elements = arr.array_elements().unwrap();

                        let num = self.get_operand(*index).expect_num()?;
                        let num_trunc = num.trunc();
                        if num_trunc == num {
                            let ndx = num_trunc as usize;
                            elements.get(ndx).copied().unwrap_or(Value::Undefined)
                        } else {
                            Value::Undefined
                        }
                    };
                    self.data.top_mut().set_result(*dest, value);
                }
                Instr::ArraySetNth { .. } => todo!("ArraySetNth"),
                Instr::ArrayLen { dest, arr } => {
                    let len = self
                        .get_operand_object(*arr)?
                        .into_heap_cell()
                        .ok_or_else(|| error!("not an array!"))?
                        .borrow()
                        .array_elements()
                        .unwrap()
                        .len();
                    self.data
                        .top_mut()
                        .set_result(*dest, Value::Number(len as f64));
                }

                Instr::TypeOf { dest, arg: value } => {
                    let value = self.get_operand(*value);
                    let result = self.js_typeof(&value);
                    self.data.top_mut().set_result(*dest, result);
                }

                Instr::BoolOpAnd(dest, a, b) => {
                    let a = self.get_operand(*a);
                    let b = self.get_operand(*b);
                    let a_bool = self.to_boolean(a);
                    let res = if a_bool { b } else { Value::Bool(false) };
                    self.data.top_mut().set_result(*dest, res);
                }
                Instr::BoolOpOr(dest, a, b) => {
                    let a = self.get_operand(*a);
                    let b = self.get_operand(*b);
                    let a_bool = self.to_boolean(a);
                    let res = if a_bool { a } else { b };
                    self.data.top_mut().set_result(*dest, res);
                }

                Instr::ClosureNew {
                    dest,
                    fnid,
                    forced_this,
                } => {
                    let mut upvalues = Vec::new();
                    while let Some(Instr::ClosureAddCapture(cap)) =
                        func.instrs().get(next_ndx as usize)
                    {
                        let upv_id = self.data.top_mut().ensure_in_upvalue(*cap);
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

                    let oid = self.realm.heap.new_function(closure);
                    self.data.top_mut().set_result(*dest, Value::Object(oid));
                }
                // This is always handled in the code for ClosureNew
                Instr::ClosureAddCapture(_) => {
                    unreachable!(
                        "interpreter bug: ClosureAddCapture should be handled with ClosureNew. (Usual cause: the bytecode compiler has placed some other instruction between ClosureAddCapture and ClosureNew.)"
                    )
                }
                Instr::Unshare(reg) => {
                    self.data.top_mut().ensure_inline(*reg);
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
                        .load_import(&module_path, Some(import_site))
                        .with_context(error!("while trying to import '{}'", module_path))?;

                    if let Some(module_oid) = self.realm.module_objs.get(&root_fnid.0) {
                        self.data
                            .top_mut()
                            .set_result(*dest, Value::Object(*module_oid));
                    } else {
                        // TODO Refactor with other implementations of Call?
                        let root_fn = self.loader.get_function(root_fnid).unwrap();

                        let call_meta = stack::CallMeta {
                            fnid: root_fnid,
                            n_regs: root_fn.n_regs() as u32,
                            captures: &[],
                            this: Value::Undefined,
                        };
                        self.data
                            .top_mut()
                            .set_return_target(IID(self.iid.0 + 1), *dest);

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

                Instr::StrCreateEmpty(dest) => {
                    let oid = self.realm.heap.new_string(String::new());
                    self.data.top_mut().set_result(*dest, Value::Object(oid));
                }
                Instr::StrAppend(buf_reg, tail) => {
                    let value = self.str_append(buf_reg, tail)?;
                    self.data.top_mut().set_result(*buf_reg, value);
                }

                Instr::GetGlobalThis(dest) => {
                    let value = Value::Object(self.realm.global_obj);
                    self.data.top_mut().set_result(*dest, value);
                }

                Instr::Breakpoint => {
                    // We must update self.iid now, or the Interpreter will be back here on resume,
                    // in an infinite loop
                    self.iid.0 = next_ndx;
                    return Ok(ExitInternal::Suspended);
                }

                Instr::GetCurrentException(dest) => {
                    let current_exc = self.current_exc.expect("no current exception!");
                    self.data.top_mut().set_result(*dest, current_exc);
                }
                Instr::Throw(exc_value) => {
                    self.current_exc = Some(self.get_operand(*exc_value));
                    let handler = self
                        .exc_handler_stack
                        .pop()
                        .ok_or_else(|| error!("unhandled exception: {:?}", exc_value))?;

                    assert!(handler.stack_height <= self.data.len());
                    while handler.stack_height < self.data.len() {
                        // like return, but ignore the return value
                        self.data.pop();
                        let (tgt_iid, _) = self.data.top_mut().take_return_target();
                        self.iid = tgt_iid;
                    }
                    assert_eq!(handler.stack_height, self.data.len());

                    self.iid = handler.target_iid;
                    continue;
                }
                Instr::PopExcHandler => {
                    let handler = self
                        .exc_handler_stack
                        .pop()
                        .ok_or_else(|| error!("compiler bug: no exception handler to pop!"))?;
                    assert_eq!(handler.stack_height, self.data.len());
                }
                Instr::PushExcHandler(target_iid) => self.exc_handler_stack.push(ExcHandler {
                    stack_height: self.data.len(),
                    target_iid: *target_iid,
                }),
            }

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
        let b = self.get_operand(*b);

        let mut buf = self.get_operand_string(*a)?.to_owned();
        let tail = value_to_string(b, &self.realm.heap);
        buf.push_str(&tail);
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
        self.realm.heap.is_instance_of(&obj, sup_oid)
    }

    fn get_operand_string(&self, vreg: bytecode::VReg) -> Result<Ref<str>> {
        let ho = self
            .get_operand_object(vreg)?
            .into_heap_cell()
            .ok_or_else(|| error!("not a heap object"))?
            .borrow();
        Ok(Ref::filter_map(ho, |ho| ho.as_str()).unwrap())
    }

    fn dump_output(self) -> FinishedData {
        let sink = try_values_to_literals(&self.sink, &self.realm.heap);
        FinishedData { sink }
    }

    fn exit_function(&mut self, callee_retval_reg: Option<VReg>) {
        let return_value = callee_retval_reg
            .map(|vreg| self.get_operand(vreg))
            .unwrap_or(Value::Undefined);

        let height = self.data.len();
        pop_while(&mut self.exc_handler_stack, |handler| {
            handler.stack_height == height
        });

        self.data.pop();
        if !self.data.is_empty() {
            let (tgt_iid, tgt_vreg) = self.data.top_mut().take_return_target();
            self.data.top_mut().set_result(tgt_vreg, return_value);

            #[cfg(enable_jit)]
            if let Some(jitting) = &mut self.jitting {
                jitting.builder.exit_function(callee_retval_reg);
            }

            self.iid = tgt_iid;
        } else {
            // The stack is now empty, so execution can't continue
            // This instance of Interpreter must be decommissioned
        }
    }

    fn with_numbers<F>(&mut self, dest: VReg, a: VReg, b: VReg, op: F)
    where
        F: FnOnce(f64, f64) -> f64,
    {
        let a = self.get_operand(a).expect_num();
        let b = self.get_operand(b).expect_num();
        match (a, b) {
            (Ok(a), Ok(b)) => {
                self.data
                    .top_mut()
                    .set_result(dest, Value::Number(op(a, b)));
            }
            (a, b) => {
                eprintln!(
                    ">>>> WARNING: failed number op: {:?}, {:?}, {:?}",
                    dest, a, b
                );
            }
        }
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
            #[rustfmt::skip]
            (Value::Object(a_oid), Value::Object(b_oid)) => {
                let a_obj = self.realm.heap.get(*a_oid).map(|x| x.borrow());
                let b_obj = self.realm.heap.get(*b_oid).map(|x| x.borrow());

                let a_str = a_obj.as_ref().and_then(|ho| ho.as_str());
                let b_str = b_obj.as_ref().and_then(|ho| ho.as_str());

                if let (Some(a_str), Some(b_str)) = (a_str, b_str) {
                    a_str.cmp(b_str).into()
                } else if a_oid == b_oid {
                    ValueOrdering::Equal
                } else {
                    ValueOrdering::Incomparable
                }
            }
            _ => ValueOrdering::Incomparable,
        };

        self.data
            .top_mut()
            .set_result(dest, Value::Bool(test(ordering)));
    }

    // TODO(cleanup) inline this function? It now adds nothing
    fn get_operand(&self, vreg: bytecode::VReg) -> Value {
        self.data.top().get_result(vreg)
    }

    fn get_operand_object(&self, vreg: bytecode::VReg) -> Result<heap::ValueObjectRef> {
        let value = self.get_operand(vreg);
        as_object_ref(value, &self.realm.heap)
            .ok_or_else(|| error!("could not use as object: {:?}", vreg))
    }

    fn get_operand_object_mut(&self, vreg: bytecode::VReg) -> Result<heap::ValueObjectRef> {
        // TODO Remove this function, inline into all callers
        self.get_operand_object(vreg)
    }

    fn js_typeof(&mut self, value: &Value) -> Value {
        let ty_s = match value {
            Value::Number(_) => "number",
            Value::Bool(_) => "boolean",
            Value::Object(oid) => match self.realm.heap.get(*oid).unwrap().borrow().type_of() {
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

    fn value_to_index_or_key(heap: &heap::Heap, value: &Value) -> Option<heap::IndexOrKeyOwned> {
        match value {
            Value::Number(n) if *n >= 0.0 => {
                let n_trunc = n.trunc();
                if *n == n_trunc {
                    let ndx = n_trunc as usize;
                    Some(heap::IndexOrKeyOwned::Index(ndx))
                } else {
                    None
                }
            }
            Value::Object(oid) => {
                let obj = heap.get(*oid)?;
                let string = obj.borrow().as_str()?.to_owned();
                Some(heap::IndexOrKeyOwned::Key(string))
            }
            _ => None,
        }
    }

    /// Converts the given value to a boolean (e.g. for use by `if`,
    /// or operators `&&` and `||`)
    ///
    /// See: https://262.ecma-international.org/14.0/#sec-toboolean
    fn to_boolean(&self, value: Value) -> bool {
        match value {
            Value::Null => false,
            Value::Bool(bool_val) => bool_val,
            Value::Number(num) => num != 0.0,
            Value::Object(oid) => self.realm.heap.get(oid).unwrap().borrow().to_boolean(),
            Value::Undefined => false,
            Value::SelfFunction => true,
            Value::Internal(_) => {
                panic!("bytecode compiler bug: internal value should be unreachable")
            }
        }
    }
}

fn init_stack(
    loader: &mut loader::Loader,
    realm: &mut Realm,
    fnid: FnId,
) -> stack::InterpreterData {
    let mut data = stack::InterpreterData::new();
    let root_fn = loader.get_function(fnid).unwrap();

    let global_this = Value::Object(realm.global_obj);

    data.push(stack::CallMeta {
        fnid,
        n_regs: root_fn.n_regs() as u32,
        captures: &[],
        this: global_this,
    });

    data
}

fn as_object_ref(value: Value, heap: &heap::Heap) -> Option<heap::ValueObjectRef> {
    match value {
        Value::Object(oid) => heap.get(oid).map(heap::ValueObjectRef::Heap),
        Value::Number(num) => Some(heap::ValueObjectRef::Number(num, heap)),
        Value::Bool(bool) => Some(heap::ValueObjectRef::Bool(bool, heap)),
        _ => None,
    }
}

fn try_values_to_literals(vec: &[Value], heap: &heap::Heap) -> Vec<Option<bytecode::Literal>> {
    vec.iter()
        .map(|value| try_value_to_literal(*value, heap))
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
        bytecode::Literal::JsWord(jsw) => {
            // TODO(performance) avoid this allocation
            let oid = heap.new_string(jsw.to_string());
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
            hobj.borrow()
                .as_str()
                .map(|s| bytecode::Literal::String(s.to_owned()))
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
        let Array_push = heap.new_function(Closure::Native(nf_Array_push));
        let Array_pop = heap.new_function(Closure::Native(nf_Array_pop));
        let array_proto_oid = heap.array_proto();
        {
            let mut array_proto = heap.get(array_proto_oid).unwrap().borrow_mut();
            array_proto.set_own_element_or_property("push".into(), Value::Object(Array_push));
            array_proto.set_own_element_or_property("pop".into(), Value::Object(Array_pop));
        }

        let Array_isArray = heap.new_function(Closure::Native(nf_Array_isArray));
        let mut array_ctor_props = HashMap::new();
        array_ctor_props.insert("isArray".to_string(), Value::Object(Array_isArray));
        array_ctor_props.insert("prototype".to_string(), Value::Object(array_proto_oid));
        let array_ctor = heap.new_ordinary_object(array_ctor_props);
        heap.init_function(array_ctor, Closure::Native(nf_Array));

        global.insert("Array".to_string(), Value::Object(array_ctor));
    }

    let RegExp = heap.new_function(Closure::Native(nf_RegExp));
    global.insert("RegExp".to_string(), Value::Object(RegExp));

    let mut number_cons_props = HashMap::new();
    number_cons_props.insert("prototype".to_string(), Value::Object(heap.number_proto()));
    {
        let Number_prototype_toString =
            heap.new_function(Closure::Native(nf_Number_prototype_toString));
        let oid = heap.number_proto();
        let mut number_proto = heap.get(oid).unwrap().borrow_mut();
        number_proto.set_own_element_or_property(
            "toString".into(),
            Value::Object(Number_prototype_toString),
        )
    }
    let Number = heap.new_ordinary_object(number_cons_props);
    heap.init_function(Number, Closure::Native(nf_Number));
    global.insert("Number".to_string(), Value::Object(Number));

    let String = heap.new_function(Closure::Native(nf_String));
    global.insert("String".to_string(), Value::Object(String));

    let Boolean = heap.new_function(Closure::Native(nf_Boolean));
    global.insert("Boolean".to_string(), Value::Object(Boolean));

    let mut func_cons_props = HashMap::new();
    func_cons_props.insert("prototype".to_string(), Value::Object(heap.func_proto()));
    let Function = heap.new_ordinary_object(func_cons_props);
    heap.init_function(Function, Closure::Native(nf_Function));
    global.insert("Function".to_string(), Value::Object(Function));

    let cash_print = heap.new_function(Closure::Native(nf_cash_print));
    global.insert("$print".to_string(), Value::Object(cash_print));

    let func_bind = heap.new_function(Closure::Native(nf_Function_bind));
    {
        let mut func_proto = heap.get(heap.func_proto()).unwrap().borrow_mut();
        func_proto.set_own_element_or_property("bind".into(), Value::Object(func_bind));
    }

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
        obj.borrow().array_elements().is_some()
    } else {
        false
    };

    Ok(Value::Bool(value))
}

#[allow(non_snake_case)]
fn nf_Array_push(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let oid = this.expect_obj().unwrap();
    let mut arr = intrp.realm.heap.get(oid).unwrap().borrow_mut();
    let value = *args.get(0).unwrap();
    arr.array_push(value);
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
fn nf_Array(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    let this = this
        .expect_obj()
        .expect("compiler bug: new Array(...): non-object this");
    // args directly go as values
    intrp.realm.heap.init_array(this, args.to_vec());
    Ok(Value::Undefined)
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
    let value = args.get(0).copied();
    let heap = &intrp.realm.heap;

    let value_str = value
        .map(|v| value_to_string(v, heap))
        .unwrap_or_else(String::new);

    Ok(literal_to_value(
        bytecode::Literal::String(value_str),
        &mut intrp.realm.heap,
    ))
}

fn value_to_string(value: Value, heap: &Heap) -> String {
    match value {
        Value::Number(num) => num.to_string(),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::Object(oid) => heap.get(oid).unwrap().borrow().js_to_string(),
        Value::Null => "null".into(),
        Value::Undefined => "undefined".into(),
        Value::SelfFunction => "<function>".into(),
        Value::Internal(_) => unreachable!(),
    }
}

#[allow(non_snake_case)]
fn nf_Boolean(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Bool(false))
}

#[allow(non_snake_case)]
fn nf_Function(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    todo!("not yet implemented!")
}

#[allow(non_snake_case)]
fn nf_cash_print(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    for arg in args {
        if let Value::Object(obj_id) = arg {
            let obj = intrp.realm.heap.get(*obj_id).unwrap().borrow();
            if let Some(s) = obj.as_str() {
                println!("  {:?}", s);
            } else {
                let props = obj.own_properties();
                println!("{:?} [{} properties]", obj_id, props.len());

                for prop in props {
                    let value = obj.get_own_element_or_property(IndexOrKey::Key(&prop));
                    println!("  - {:?} = {:?}", prop, value);
                }
            }
        } else {
            println!("{:?}", arg);
        }
    }
    Ok(Value::Undefined)
}

#[allow(non_snake_case)]
fn nf_Function_bind(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    let js_closure = {
        let obj_id = this.expect_obj()?;
        let obj = intrp
            .realm
            .heap
            .get(obj_id)
            .ok_or_else(|| error!("no such object"))?
            .borrow();
        let closure = obj.as_closure().ok_or_else(|| error!("not a function"))?;
        match closure {
            Closure::Native(_) => return Err(error!("can't bind a native function (yet)")),
            Closure::JS(jsc) => jsc.clone(),
        }
    };

    let forced_this = Some(args.get(0).copied().unwrap_or(Value::Undefined));

    let new_closure = Closure::JS(JSClosure {
        forced_this,
        ..js_closure
    });

    let new_obj_id = intrp.realm.heap.new_function(new_closure);
    Ok(Value::Object(new_obj_id))
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

    use crate::heap;
    use crate::{bytecode, InterpreterValue};

    use super::{Fuel, InstrBreakpoint, Interpreter, SourceBreakpoint};

    pub use heap::{IndexOrKey, Object, ObjectId};

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
            self.interpreter.loader
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
            // `frame_ndx` is top-first (0 = top)
            // the stack API is bottom first (0 = bottom), so convert first
            let frame = self
                .interpreter
                .data
                .nth_frame(self.interpreter.data.len() - frame_ndx - 1);
            let fnid = frame.header().fn_id;

            let iid = if frame_ndx == 0 {
                // Top frame
                self.interpreter.iid
            } else {
                frame
                    .return_target()
                    .expect("non-top stack frame has no return target!")
                    .0
            };

            // For the top frame: `iid` is the instruction to *resume* to.
            // For other frames: `iid` is the instruction to *return* to.
            // In either case: we actually want the instruction we suspended/called at,
            // which is the previous one.

            bytecode::GlobalIID(fnid, iid)
        }

        pub fn get_object(&self, obj_id: heap::ObjectId) -> Option<Ref<heap::HeapObject>> {
            self.interpreter
                .realm
                .heap
                .get(obj_id)
                .map(|hocell| hocell.borrow())
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
        let realm = Realm::new(&mut loader);
        VMPrereq {
            loader,
            root_fnid: chunk_fnid,
            realm,
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
            // wrapping into iife makes sure that the shared variable is not a global
            (function() {
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
            })();
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
                Some(Literal::String(s)) => s.clone(),
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
    fn test_this_basic_nonstrict() {
        let output = quick_run(
            r#"
            function getThis() { return this; }
            sink(getThis());
            "#,
        );
        // `None` because it's the globalThis object due to "this substitution"
        assert_eq!(&output.sink, &[None]);
    }

    #[test]
    fn test_this_basic_strict() {
        let output = quick_run(
            r#"
            "use strict";
            function getThis() { return this; }
            sink(getThis());
            "#,
        );
        // no "this substitution" in strict mode
        assert_eq!(&output.sink, &[Some(Literal::Undefined)]);
    }

    #[test]
    fn test_this() {
        let output = quick_run(
            r#"
            "use strict"
            
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

    #[test]
    fn test_script_global() {
        let output = quick_run(
            r#"
            var x = 55
            sink(globalThis.x)
            sink(x)

            globalThis.x = 222
            sink(x)
            sink(globalThis.x)
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Number(55.0)),
                Some(Literal::Number(55.0)),
                Some(Literal::Number(222.0)),
                Some(Literal::Number(222.0)),
            ],
        );
    }

    #[test]
    fn test_constructor_prototype() {
        quick_run(
            r#"
                function Test262Error(message) {
                  this.message = message || "";
                }

                Test262Error.prototype.toString = function () {
                  return "Test262Error: " + this.message;
                };

                Test262Error.thrower = function (message) {
                  throw new Test262Error(message);
                };
            "#,
        );
    }

    #[test]
    fn test_new() {
        let output = quick_run(
            r#"
                function MyConstructor(inner) {
                    this.inner = inner
                    return 'lol'
                }

                const obj = new MyConstructor(123)
                sink(obj.inner === 123)
                sink(obj.__proto__ === MyConstructor.prototype)
                sink(obj.constructor === MyConstructor)
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Bool(true)),
                Some(Literal::Bool(true)),
                Some(Literal::Bool(true)),
            ],
        );
    }
}
