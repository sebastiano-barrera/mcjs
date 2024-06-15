use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::Rc;

use crate::error_item;
use crate::heap::JSString;
use crate::{
    bytecode::{self, FnID, Instr, VReg, IID},
    common::{self, Result},
    define_flag, error,
    heap::{self, IndexOrKey},
    loader,
};

mod builtins;
pub mod stack;

#[cfg(feature = "debugger")]
pub use stack::SlotDebug;
use stack::UpvalueID;

// Public versions of the private `RunResult` and `RunError`
pub type InterpreterResult<T> = std::result::Result<T, InterpreterError>;
pub struct InterpreterError {
    pub error: crate::common::Error,

    // When debugging support is compiled in, the state of the failed
    // interpreter is transferred into this struct, where it can be examined via
    // InterpreterError::interpreter_data()
    #[cfg(feature = "debugger")]
    intrp_state: stack::InterpreterData,
}
impl InterpreterError {
    #[cfg(feature = "debugger")]
    pub fn interpreter_state(&self) -> &stack::InterpreterData {
        &self.intrp_state
    }
}
impl std::fmt::Debug for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InterpreterError: {:?}", self.error)
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
    Object(heap::ObjectID),
    String(heap::StringID),
    Null,
    Undefined,

    Symbol(&'static str),
}

macro_rules! gen_value_expect {
    ($fn_name:ident, $variant:tt, $inner_ty:ty) => {
        impl Value {
            pub(crate) fn $fn_name(&self) -> Result<$inner_ty> {
                match self {
                    Value::$variant(inner) => Ok(*inner),
                    other => {
                        Err(error!("expected a {}, got {:?}", stringify!($variant), other).into())
                    }
                }
            }
        }
    };
}

gen_value_expect!(expect_num, Number, f64);
gen_value_expect!(expect_obj, Object, heap::ObjectID);

/// A *reference* to a closure.
///
/// It can be cloned, and the resulting value will "point" to the same closure as the
/// first one. (These semantics are also in `Value`, and `Closure` inherits them from it).
#[derive(Clone)]
pub(crate) enum Closure {
    Native(NativeClosure),
    JS(Rc<JSClosure>),
}

#[derive(Clone, Copy)]
pub(crate) struct NativeClosure(NativeFn);
type NativeFn = fn(&mut Realm, &Value, &[Value]) -> RunResult<Value>;

#[derive(Clone)]
pub struct JSClosure {
    fnid: FnID,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueID>,
    forced_this: Option<Value>,
    generator_snapshot: RefCell<Option<stack::FrameSnapshot>>,
}
impl JSClosure {
    #[cfg(feature = "debugger")]
    pub fn fnid(&self) -> FnID {
        self.fnid
    }

    pub fn upvalues(&self) -> &[UpvalueID] {
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

pub struct Realm {
    heap: heap::Heap,
    // Key is the root fnid of each module
    module_objs: HashMap<bytecode::FnID, Value>,
    global_obj: Value,
}

impl Realm {
    pub fn new(loader: &mut loader::Loader) -> Realm {
        let mut heap = heap::Heap::new();
        let global_obj = builtins::init_builtins(&mut heap);
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

    #[cfg(feature = "debugger")]
    pub fn heap(&self) -> &heap::Heap {
        &self.heap
    }
}

pub struct Interpreter<'a> {
    /// Exclusive reference to the Realm in which the code runs.
    ///
    /// It's important that this reference is exclusive: no one must access (much less
    /// manipulate) this Realm object until the Interpreter finishes executing.
    realm: &'a mut Realm,

    /// The interpreter's stack.
    ///
    /// The data stored here does not survive after the Interpreter returns.  (Well, some
    /// referenced objects may survive in the heap for a while, but the GC is supposed to
    /// collect them.)
    data: stack::InterpreterData,

    // The loader ref must never change for the whole lifecycle of the interpreter.  What would
    // happen if the same module path suddenly corresponded to a different module? Better not to
    // know
    loader: &'a mut loader::Loader,

    #[cfg(feature = "debugger")]
    dbg: Option<&'a mut debugger::DebuggingState>,
}

pub struct FinishedData {
    #[cfg(any(test, feature = "debugger"))]
    pub sink: Vec<Option<bytecode::Literal>>,
}

pub enum Exit {
    Finished(FinishedData),
    Suspended {
        #[cfg(feature = "debugger")]
        intrp_state: stack::InterpreterData,
        cause: SuspendCause,
    },
}
impl Exit {
    pub fn expect_finished(self) -> FinishedData {
        match self {
            Exit::Finished(fd) => fd,
            Exit::Suspended { .. } => {
                panic!("interpreter was interrupted, while it was expected to finish")
            }
        }
    }
}

impl<'a> Interpreter<'a> {
    pub fn new(realm: &'a mut Realm, loader: &'a mut loader::Loader, fnid: bytecode::FnID) -> Self {
        // Initialize the stack with a single frame, corresponding to a call to fnid with no
        // parameters, then put it into an Interpreter
        let mut data = init_stack(loader, realm, fnid);
        data.set_default_this(realm.global_obj);
        Interpreter {
            realm,
            data,
            loader,
            #[cfg(feature = "debugger")]
            dbg: None,
        }
    }

    #[cfg(feature = "debugger")]
    pub fn resume(
        realm: &'a mut Realm,
        loader: &'a mut loader::Loader,
        mut data: stack::InterpreterData,
    ) -> Self {
        data.set_resuming_from_breakpoint();
        Interpreter {
            realm,
            data,
            loader,
            #[cfg(feature = "debugger")]
            dbg: None,
        }
    }

    #[cfg(feature = "debugger")]
    pub fn set_debugging_state(&mut self, dbg: &'a mut debugger::DebuggingState) {
        self.dbg = Some(dbg);
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
    ///     again via the Output::Suspended variant; then it can be resumed or dropped
    ///     (destroyed).
    pub fn run(mut self) -> InterpreterResult<Exit> {
        assert!(!self.data.is_empty());

        let res = run(
            &mut self.data,
            self.realm,
            self.loader,
            #[cfg(feature = "debugger")]
            &mut self.dbg,
        );

        // We currently translate IntrpResult into Result<Exit<'a>> just to
        // avoid boiling the ocean, but long-term Result<Exit> should be
        // deleted.

        match res {
            Ok(()) => {
                #[cfg(any(test, feature = "debugger"))]
                let sink: Vec<_> = self
                    .data
                    .sink()
                    .iter()
                    .map(|value| try_value_to_literal(*value, &self.realm.heap))
                    .collect();

                Ok(Exit::Finished(FinishedData {
                    #[cfg(any(test, feature = "debugger"))]
                    sink,
                }))
            }
            Err(RunError::Exception(exc)) => {
                let head_message = {
                    let mut buf = Vec::new();
                    let params = &ShowValueParams {
                        indent: 0,
                        max_object_depth: 2,
                    };
                    show_value_ex(&mut buf, exc, self.realm, params);
                    String::from_utf8(buf).unwrap()
                };
                let mut error = error!("unhandled exception: {}", head_message);
                // the stack is maintained intact if the exception was unhandled.
                // See `throw_exc`
                for frame in self.data.frames() {
                    let hdr = frame.header();
                    let context = error_item!("while running this")
                        .with_giid(bytecode::GlobalIID(hdr.fnid, hdr.iid));
                    error.push_context(context);
                }
                Err(InterpreterError {
                    error,
                    #[cfg(feature = "debugger")]
                    intrp_state: self.data,
                })
            }
            Err(RunError::Internal(common_err)) => Err(InterpreterError {
                error: common_err,
                #[cfg(feature = "debugger")]
                intrp_state: self.data,
            }),

            #[cfg(feature = "debugger")]
            Err(RunError::Suspended(cause)) => Ok(Exit::Suspended {
                cause,
                #[cfg(feature = "debugger")]
                intrp_state: self.data,
            }),
        }
    }
}

pub fn init_stack(
    loader: &mut loader::Loader,
    realm: &mut Realm,
    fnid: FnID,
) -> stack::InterpreterData {
    let mut data = stack::InterpreterData::new();
    let root_fn = loader.get_function(fnid).unwrap();
    data.push_direct(fnid, root_fn, realm.global_obj);
    data
}

/// Just a wrapper for `run_inner` that adds some useful context about the interpreter's
/// state to errors.
///
/// See `run_inner` for details.
fn run(
    data: &mut stack::InterpreterData,
    realm: &mut Realm,
    loader: &mut loader::Loader,
    #[cfg(feature = "debugger")] dbg: &mut Option<&mut debugger::DebuggingState>,
) -> RunResult<()> {
    run_inner(
        data,
        realm,
        loader,
        #[cfg(feature = "debugger")]
        dbg,
    )
    .map_err(|err| {
        match err {
            // do nothing
            RunError::Exception(_) => err,
            #[cfg(feature = "debugger")]
            RunError::Suspended(_) => err,
            RunError::Internal(mut err) => {
                for frame in data.frames() {
                    let mut ctx = error_item!("<- interpreter was here");
                    let giid = bytecode::GlobalIID(frame.header().fnid, frame.header().iid);
                    ctx.set_giid(giid);
                    err.push_context(ctx);
                }

                RunError::Internal(err)
            }
        }
    })
}

/// Run bytecode until the stack is depleted.  This is the core of the interpreter.
///
/// Before and after each call, `data` must hold the interpreter's state in
/// enough detail that this function can resume execution even across separate
/// calls.  This function uphold this postcondition, too.
///
/// This function assumes that the stack frame has already been set up
/// beforehand, but will pop it before returning successfully. When returning
/// due to failure/suspend, the stack frame stays there, to allow for resuming.
///
/// In order to restore execution correctly, callers of this function external
/// to the function itself should always pass `0` as the `stack_level`
/// parameter. Any different value is a bug (akin to undefined behavior, that
/// is, not necessarily detected at run-time).
///
/// For external callers, this function returns when:
///  - the interpreter has finished its work (gone through the program without errors)
///  - an unhandled exception has been thrown
///  - the debugger has suspended execution (e.g. a breakpoint has been reached).
///
/// If a bug is detected (e.g. assertion failed), the function panics. The
/// interpreter's data is not unwind-safe in general, so execution can only
/// restart by initializing a new interpreter instance.
///
/// ## Implementation details
///
/// This function calls itself recursively, so that each JS stack frame maps 1:1
/// with a native `run_internal` stack frame.
///
/// For a call coming from within `run_internal` itself (recursively), it
/// returns when:
///  - the function has returned successfully (no exception)
///  - an as-of-yet-unhandled exception has been thrown (may be handled by one of the
///    parent frames or bubble up to the user unhandled)
///  - the debugger has suspended execution (e.g. a breakpoint has been reached).
///
/// Upon starting, `run_internal` calls itself again with a 1-higher
/// `stack_level` so as to restore the mapping between JS and native stacks.
/// Then it resumes JS execution.
fn run_inner(
    data: &mut stack::InterpreterData,
    realm: &mut Realm,
    loader: &mut loader::Loader,
    #[cfg(feature = "debugger")] dbg: &mut Option<&mut debugger::DebuggingState>,
) -> RunResult<()> {
    'reborrow: loop {
        if data.is_empty() {
            // We're done.
            return Ok(());
        }

        let fnid = data.top().header().fnid;
        let func = loader.get_function(fnid).unwrap();

        loop {
            let iid = data.top().header().iid;
            if iid.0 as usize == func.instrs().len() {
                // Bytecode "finished" => Implicitly return undefined
                do_return(Value::Undefined, data, realm);
                continue 'reborrow;
            }

            let instr = func.instrs()[iid.0 as usize];
            let mut next_ndx = iid.0 + 1;

            // TODO Checking for breakpoints here in this hot loop is going to be *very* slow!
            #[cfg(feature = "debugger")]
            if let Some(dbg) = dbg {
                if dbg.fuel_empty() {
                    dbg.set_fuel(debugger::Fuel::Unlimited);
                    return Err(force_suspend_for_breakpoint(data, iid));
                }

                let giid = bytecode::GlobalIID(fnid, iid);
                if let Some(bkpt) = dbg.instr_bkpt_at(&giid) {
                    suspend_for_breakpoint(data, iid)?;

                    if bkpt.delete_on_hit {
                        dbg.clear_instr_bkpt(giid);
                    }
                }
            }

            match &instr {
                Instr::LoadConst(dest, bytecode::ConstIndex(const_ndx)) => {
                    let literal = func.consts()[*const_ndx as usize].clone();
                    let value = literal_to_value(literal, &mut realm.heap);
                    data.top_mut().set_result(*dest, value);
                }

                Instr::OpAdd(dest, a, b) => match get_operand(data, *a)? {
                    Value::Number(_) => with_numbers(data, *dest, *a, *b, |x, y| x + y)?,
                    // TODO replace this with coercing
                    Value::String(_) | Value::Object(_) => {
                        let value = str_append(data, realm, *a, *b)?;
                        data.top_mut().set_result(*dest, value);
                    }
                    other => return Err(error!("unsupported operator '+' for: {:?}", other).into()),
                },
                Instr::ArithSub(dest, a, b) => with_numbers(data, *dest, *a, *b, |x, y| x - y)?,
                Instr::ArithMul(dest, a, b) => with_numbers(data, *dest, *a, *b, |x, y| x * y)?,
                Instr::ArithDiv(dest, a, b) => with_numbers(data, *dest, *a, *b, |x, y| x / y)?,

                Instr::PushToSink(operand) => {
                    #[allow(unused_variables)]
                    let value = get_operand(data, *operand)?;
                    #[cfg(test)]
                    data.push_to_sink(value);
                    #[cfg(not(test))]
                    panic!("PushToSink instruction not implemented outside of unit tests");
                }

                Instr::CmpGE(dest, a, b) => compare(data, realm, *dest, *a, *b, |ord| {
                    ord == ValueOrdering::Greater || ord == ValueOrdering::Equal
                })?,
                Instr::CmpGT(dest, a, b) => compare(data, realm, *dest, *a, *b, |ord| {
                    ord == ValueOrdering::Greater
                })?,
                Instr::CmpLT(dest, a, b) => {
                    compare(data, realm, *dest, *a, *b, |ord| ord == ValueOrdering::Less)?
                }
                Instr::CmpLE(dest, a, b) => compare(data, realm, *dest, *a, *b, |ord| {
                    ord == ValueOrdering::Less || ord == ValueOrdering::Equal
                })?,
                Instr::CmpEQ(dest, a, b) => compare(data, realm, *dest, *a, *b, |ord| {
                    ord == ValueOrdering::Equal
                })?,
                Instr::CmpNE(dest, a, b) => compare(data, realm, *dest, *a, *b, |ord| {
                    ord != ValueOrdering::Equal
                })?,

                Instr::JmpIf { cond, dest } => {
                    let cond_value = get_operand(data, *cond)?;
                    let cond_value = realm.heap.to_boolean(cond_value);
                    if cond_value {
                        next_ndx = dest.0;
                    } else {
                        // Just go to the next instruction
                    }
                }
                Instr::JmpIfNot { cond, dest } => {
                    let cond_value = get_operand(data, *cond)?;
                    let cond_value = realm.heap.to_boolean(cond_value);
                    if !cond_value {
                        next_ndx = dest.0;
                    } else {
                        // Just go to the next instruction
                    }
                }

                Instr::Copy { dst, src } => {
                    let value = get_operand(data, *src)?;
                    data.top_mut().set_result(*dst, value);
                }
                Instr::LoadCapture(dest, cap_ndx) => {
                    data.capture_to_var(*cap_ndx, *dest);
                }

                Instr::Nop => {}
                Instr::BoolNot { dest, arg } => {
                    let value = get_operand(data, *arg)?;
                    let value = Value::Bool(!realm.heap.to_boolean(value));
                    data.top_mut().set_result(*dest, value);
                }
                Instr::ToNumber { dest, arg } => {
                    let value = get_operand(data, *arg)?;
                    let value = to_number_value(value).ok_or_else(|| {
                        let message = JSString::new_from_str(&format!(
                            "cannot convert to a number: {:?}",
                            *arg
                        ));
                        let exc = make_exception(realm, "TypeError", message);
                        RunError::Exception(exc)
                    })?;
                    data.top_mut().set_result(*dest, value);
                }

                Instr::Jmp(IID(dest_ndx)) => {
                    next_ndx = *dest_ndx;
                }
                Instr::SaveFrameSnapshot(resume_iid) => {
                    data.top_mut().save_snapshot(*resume_iid);
                }
                Instr::Return(value) => {
                    let return_value = get_operand(data, *value)?;
                    do_return(return_value, data, realm);
                    continue 'reborrow;
                }
                Instr::Call {
                    callee,
                    this,
                    return_value: return_value_reg,
                } => {
                    let callee = get_operand(data, *callee)?;
                    let closure: &Closure = realm.heap.as_closure(callee).ok_or_else(|| {
                        let val_s = realm.heap.show_debug(callee);
                        error!("can't call non-closure: {}", val_s)
                    })?;

                    // NOTE The arguments have to be "read" before adding the stack frame;
                    // they will no longer be accessible
                    // afterwards
                    //
                    // TODO make the above fact false, and avoid this allocation
                    //
                    // NOTE Note that we collect into Result<Vec<_>>, and *then* try (?). This is
                    // to ensure that, if get_operand fails, we return early. otherwise we
                    // would risk that arg_vals.len() != the number of CallArg instructions
                    let res: RunResult<Vec<_>> = func
                        .instrs()
                        .iter()
                        .skip(iid.0 as usize + 1)
                        .map_while(|instr| match instr {
                            Instr::CallArg(arg_reg) => Some(*arg_reg),
                            _ => None,
                        })
                        .map(|vreg| get_operand(data, vreg))
                        .collect();
                    let mut arg_vals = res?;
                    let n_args_u16: u16 = arg_vals.len().try_into().unwrap();
                    let return_to_iid = IID(iid.0 + n_args_u16 + 1);

                    match closure {
                        Closure::JS(closure) => {
                            let n_params = bytecode::ARGS_COUNT_MAX as usize;
                            arg_vals.truncate(n_params);
                            arg_vals.resize(n_params, Value::Undefined);
                            assert_eq!(arg_vals.len(), n_params);

                            let this = get_operand(data, *this)?;

                            data.top_mut().set_return_target(*return_value_reg);
                            data.top_mut().set_resume_iid(return_to_iid);

                            data.push_call(Rc::clone(closure), this, loader);
                            for (i, arg) in arg_vals.into_iter().enumerate() {
                                data.top_mut().set_arg(bytecode::ArgIndex(i as _), arg);
                            }

                            continue 'reborrow;
                        }
                        Closure::Native(nf) => {
                            let NativeClosure(nf) = *nf;
                            let this = get_operand(data, *this)?;
                            match nf(realm, &this, &arg_vals) {
                                Ok(ret_val) => {
                                    data.top_mut().set_result(*return_value_reg, ret_val);
                                    next_ndx = return_to_iid.0;
                                }
                                Err(RunError::Exception(exc)) => {
                                    throw_exc(
                                        exc,
                                        iid,
                                        data,
                                        #[cfg(feature = "debugger")]
                                        dbg,
                                    )?;
                                    continue 'reborrow;
                                }
                                Err(other_err) => return Err(other_err),
                            };
                        }
                    }
                }
                Instr::CallArg(_) => {
                    unreachable!("interpreter bug: CallArg goes through another path!")
                }

                Instr::LoadArg(dest, arg_ndx) => {
                    // extra copy?
                    // TODO usize is a bit too wide
                    let value = data.top().get_arg(*arg_ndx).unwrap_or(Value::Undefined);
                    data.top_mut().set_result(*dest, value);
                }

                Instr::ObjCreateEmpty(dest) => {
                    let oid = realm.heap.new_ordinary_object();
                    data.top_mut().set_result(*dest, Value::Object(oid));
                }
                Instr::ObjSet { obj, key, value } => {
                    obj_set(data, realm, obj, key, value, true)?;
                }
                Instr::ObjSetN { obj, key, value } => {
                    obj_set(data, realm, obj, key, value, false)?;
                }
                Instr::ObjGet { dest, obj, key } => {
                    let obj = get_operand(data, *obj)?;
                    let key = get_operand(data, *key)?;
                    let key = value_to_index_or_key(&realm.heap, &key);

                    let property = key
                        .and_then(|key| realm.heap.get_chained(obj, key.to_ref()))
                        .unwrap_or(heap::Property::NonEnumerable(Value::Undefined));
                    let value = property_to_value(&property, &mut realm.heap)?;

                    data.top_mut().set_result(*dest, value);
                }
                Instr::ObjGetKeysOE { dest, obj } => obj_get_keys(
                    data,
                    realm,
                    obj,
                    dest,
                    OnlyEnumerable::Yes,
                    IncludeInherited::No,
                )?,
                Instr::ObjGetKeysIE { dest, obj } => obj_get_keys(
                    data,
                    realm,
                    obj,
                    dest,
                    OnlyEnumerable::Yes,
                    IncludeInherited::Yes,
                )?,
                Instr::ObjGetKeysO { dest, obj } => obj_get_keys(
                    data,
                    realm,
                    obj,
                    dest,
                    OnlyEnumerable::No,
                    IncludeInherited::No,
                )?,
                Instr::ObjDelete { obj, key } => {
                    let obj = get_operand(data, *obj)?;
                    // the boolean is necessary instead of a normal match on
                    // Option, because borrowck can't prove that the destructor
                    // won't run.
                    // TODO Change this if the heap switches to something other
                    // than RefCells (qcell?)
                    let key = get_operand(data, *key)?;
                    let key = value_to_index_or_key(&realm.heap, &key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let was_actually_obj = realm.heap.delete_own(obj, key.to_ref());
                    if !was_actually_obj {
                        let message = JSString::new_from_str(&format!("not an object: {:?}", obj));
                        let exc = make_exception(realm, "TypeError", message);
                        throw_exc(
                            exc,
                            iid,
                            data,
                            #[cfg(feature = "debugger")]
                            dbg,
                        )?;
                        // FIX No continue 'reborrow?
                    }
                }

                Instr::ArrayPush { arr, value } => {
                    let arr = get_operand(data, *arr)?;
                    let value = get_operand(data, *value)?;
                    let was_array = realm.heap.array_push(arr, value);
                    assert!(was_array);
                }
                Instr::ArrayNth { dest, arr, index } => {
                    let value = {
                        let arr = get_operand(data, *arr)?;
                        let elements = realm.heap.array_elements(arr).unwrap();

                        let num = get_operand(data, *index)?.expect_num()?;
                        let num_trunc = num.trunc();
                        if num_trunc == num {
                            let ndx = num_trunc as usize;
                            elements.get(ndx).copied().unwrap_or(Value::Undefined)
                        } else {
                            Value::Undefined
                        }
                    };
                    data.top_mut().set_result(*dest, value);
                }
                Instr::ArraySetNth { .. } => todo!("ArraySetNth"),
                Instr::ArrayLen { dest, arr } => {
                    let arr = get_operand(data, *arr)?;
                    let len = realm
                        .heap
                        .array_elements(arr)
                        .ok_or_else(|| error!("not an array!"))?
                        .len();
                    data.top_mut().set_result(*dest, Value::Number(len as f64));
                }

                Instr::TypeOf { dest, arg: value } => {
                    let value = get_operand(data, *value)?;
                    let result = js_typeof(&value, realm);
                    data.top_mut().set_result(*dest, result);
                }

                Instr::BoolOpAnd(dest, a, b) => {
                    let a = get_operand(data, *a)?;
                    let b = get_operand(data, *b)?;
                    let a_bool = realm.heap.to_boolean(a);
                    let res = if a_bool { b } else { Value::Bool(false) };
                    data.top_mut().set_result(*dest, res);
                }
                Instr::BoolOpOr(dest, a, b) => {
                    let a = get_operand(data, *a)?;
                    let b = get_operand(data, *b)?;
                    let a_bool = realm.heap.to_boolean(a);
                    let res = if a_bool { a } else { b };
                    data.top_mut().set_result(*dest, res);
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
                        let upv_id = data.top_mut().ensure_in_upvalue(*cap);
                        upvalues.push(upv_id);
                        next_ndx += 1;
                    }

                    let forced_this = forced_this.map(|reg| get_operand(data, reg)).transpose()?;
                    let closure = Closure::JS(Rc::new(JSClosure {
                        fnid: *fnid,
                        upvalues,
                        forced_this,
                        generator_snapshot: RefCell::new(None),
                    }));

                    let oid = realm.heap.new_function(closure);
                    data.top_mut().set_result(*dest, Value::Object(oid));
                }
                // This is always handled in the code for ClosureNew
                Instr::ClosureAddCapture(_) => {
                    unreachable!(
                            "interpreter bug: ClosureAddCapture should be handled with ClosureNew. (Usual cause: the bytecode compiler has placed some other instruction between ClosureAddCapture and ClosureNew.)"
                        )
                }
                Instr::Unshare(reg) => {
                    data.top_mut().ensure_inline(*reg);
                }

                Instr::UnaryMinus { dest, arg } => {
                    let arg_val: f64 = get_operand(data, *arg)?.expect_num()?;
                    data.top_mut().set_result(*dest, Value::Number(-arg_val));
                }

                Instr::ImportModule(dest, module_path) => {
                    use common::Context;

                    let module_path = get_operand_string(data, realm, *module_path)?.to_string();
                    let root_fnid = loader
                        .load_import_from_fn(&module_path, fnid)
                        .with_context(error_item!("while trying to import '{}'", module_path))?;

                    // Commit before reborrowing
                    data.top_mut().set_resume_iid(bytecode::IID(iid.0 + 1));

                    if let Some(&module_value) = realm.module_objs.get(&root_fnid) {
                        data.top_mut().set_result(*dest, module_value);
                    } else {
                        let root_fn = loader.get_function(root_fnid).unwrap();

                        data.top_mut().set_return_target(*dest);
                        data.top_mut().set_resume_iid(IID(iid.0 + 1));

                        data.push_direct(root_fnid, root_fn, Value::Undefined);
                        data.top_mut().set_is_module_root_fn();
                    }

                    // This satisfies the borrow checker
                    // (`loader.load_import_from_fn` mut-borrows the whole
                    // Loader, so the compiler can't prove that the same FnId
                    // won't correspond to a different function or even stay
                    // valid across the call)
                    continue 'reborrow;
                }

                Instr::LoadNull(dest) => {
                    data.top_mut().set_result(*dest, Value::Null);
                }
                Instr::LoadUndefined(dest) => {
                    data.top_mut().set_result(*dest, Value::Undefined);
                }
                Instr::LoadThis(dest) => {
                    let value = data.top().header().this;
                    data.top_mut().set_result(*dest, value);
                }
                Instr::ArithInc(dest, src) => {
                    let val = get_operand(data, *src)?
                        .expect_num()
                        .map_err(|_| error!("bytecode bug: ArithInc on non-number"))?;
                    data.top_mut().set_result(*dest, Value::Number(val + 1.0));
                }
                Instr::ArithDec(dest, src) => {
                    let val = get_operand(data, *src)?
                        .expect_num()
                        .map_err(|_| error!("bytecode bug: ArithDec on non-number"))?;
                    data.top_mut().set_result(*dest, Value::Number(val - 1.0));
                }
                Instr::IsInstanceOf(dest, obj, sup) => {
                    let result = is_instance_of(data, realm, *obj, *sup)?;
                    data.top_mut().set_result(*dest, Value::Bool(result));
                }
                Instr::NewIterator { dest: _, obj: _ } => todo!(),
                Instr::IteratorGetCurrent { dest: _, iter: _ } => todo!(),
                Instr::IteratorAdvance { iter: _ } => todo!(),
                Instr::JmpIfIteratorFinished { iter: _, dest: _ } => todo!(),

                Instr::StrCreateEmpty(dest) => {
                    let oid = realm.heap.new_string(JSString::empty());
                    data.top_mut().set_result(*dest, Value::String(oid));
                }
                Instr::StrAppend(buf_reg, tail) => {
                    let value = str_append(data, realm, *buf_reg, *tail)?;
                    data.top_mut().set_result(*buf_reg, value);
                }

                Instr::GetGlobalThis(dest) => {
                    data.top_mut().set_result(*dest, realm.global_obj);
                }
                Instr::GetGlobal {
                    dest,
                    name: bytecode::ConstIndex(name_cndx),
                } => {
                    let literal = func.consts()[*name_cndx as usize].clone();
                    let key_str = match &literal {
                        bytecode::Literal::String(s) => s,
                        bytecode::Literal::JsWord(jsw) => jsw.as_ref(),
                        _ => panic!(
                            "malformed bytecode: GetGlobal argument `name` not a string literal"
                        ),
                    };
                    let key = heap::IndexOrKey::Key(key_str);

                    let lookup_result = realm.heap.get_own(realm.global_obj, key);

                    let prop = match lookup_result {
                        None => None,
                        Some(prop) if prop.value() == Some(Value::Undefined) => None,
                        Some(prop) => Some(prop),
                    };

                    match prop {
                        None => {
                            let msg =
                                JSString::new_from_str(&format!("{} is not defined", key_str));
                            let msg = realm.heap.new_string(msg);

                            let exc_proto = get_builtin(realm, "ReferenceError").unwrap();
                            let exc = Value::Object(realm.heap.new_ordinary_object());
                            realm.heap.set_proto(exc, Some(exc_proto));
                            realm.heap.set_own(exc, IndexOrKey::Key("message"), {
                                let value = Value::String(msg);
                                heap::Property::Enumerable(value)
                            });

                            throw_exc(
                                exc,
                                iid,
                                data,
                                #[cfg(feature = "debugger")]
                                dbg,
                            )?;
                            continue 'reborrow;
                        }
                        Some(prop) => {
                            data.top_mut().set_result(*dest, prop.value().unwrap());
                        }
                    }
                }

                Instr::Breakpoint => {
                    suspend_for_breakpoint(data, iid)?;
                }

                Instr::GetCurrentException(dest) => {
                    let cur_exc = data
                        .get_cur_exc()
                        .expect("compiler bug: GetCurrentException but there isn't any");
                    data.top_mut().set_result(*dest, cur_exc);
                }
                Instr::Throw(exc) => {
                    let exc = get_operand(data, *exc)?;
                    throw_exc(
                        exc,
                        iid,
                        data,
                        #[cfg(feature = "debugger")]
                        dbg,
                    )?;
                    continue 'reborrow;
                }
                Instr::PopExcHandler => {
                    data.top_mut()
                        .pop_exc_handler()
                        .ok_or_else(|| error!("compiler bug: no exception handler to pop!"))?;
                }
                Instr::PushExcHandler(target_iid) => data.top_mut().push_exc_handler(*target_iid),
            }

            // TODO Inefficient, but prevents a number of bugs
            data.top_mut().set_resume_iid(bytecode::IID(next_ndx));

            #[cfg(feature = "debugger")]
            if let Some(dbg) = dbg {
                dbg.consume_1_fuel();
            }
        }
    }
}

fn make_exception(realm: &mut Realm, constructor_name: &str, message: JSString) -> Value {
    let message = realm.heap.new_string(message);

    let exc_cons = get_builtin(realm, constructor_name).unwrap();
    let exc_proto = realm
        .heap
        .get_own(Value::Object(exc_cons), IndexOrKey::Key("prototype"))
        .unwrap()
        .value()
        .unwrap()
        .expect_obj()
        .unwrap();
    let exc = Value::Object(realm.heap.new_ordinary_object());

    realm.heap.set_proto(exc, Some(exc_proto));
    realm.heap.set_own(exc, IndexOrKey::Key("message"), {
        let value = Value::String(message);
        heap::Property::Enumerable(value)
    });

    exc
}

fn get_builtin(realm: &mut Realm, builtin_name: &str) -> RunResult<heap::ObjectID> {
    realm
        .heap
        .get_own(realm.global_obj, IndexOrKey::Key(builtin_name))
        .map(|p| p.value().unwrap())
        .ok_or_else(|| error!("missing required builtin: {}", builtin_name))?
        .expect_obj()
        .map_err(|_| RunError::Internal(error!("bug: ReferenceError is not an object?!")))
}

type RunResult<T> = std::result::Result<T, RunError>;
/// The error type only used internally by `run_frame` and `run_internal`.
#[derive(Debug)]
enum RunError {
    Exception(Value),
    #[cfg(feature = "debugger")]
    Suspended(SuspendCause),
    Internal(common::Error),
}
impl From<common::Error> for RunError {
    fn from(err: common::Error) -> Self {
        RunError::Internal(err)
    }
}

#[derive(Debug)]
pub enum SuspendCause {
    Breakpoint,
    Exception(Value),
}

fn do_return(ret_val: Value, data: &mut stack::InterpreterData, realm: &mut Realm) {
    let hdr = data.top().header();
    if hdr.is_module_root_fn {
        let module_root_fnid = hdr.fnid;
        let prev = realm.module_objs.insert(module_root_fnid, ret_val);
        assert!(prev.is_none());
    }

    data.pop();
    if !data.is_empty() {
        if let Some(rv_reg) = data.top_mut().take_return_target() {
            data.top_mut().set_result(rv_reg, ret_val);
        }
    }
}

fn obj_set(
    data: &mut stack::InterpreterData,
    realm: &mut Realm,
    obj: &VReg,
    key: &VReg,
    value: &VReg,
    is_enumerable: bool,
) -> RunResult<()> {
    let obj = get_operand(data, *obj)?;
    let key = get_operand(data, *key)?;
    let key = value_to_index_or_key(&realm.heap, &key)
        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
    let value = get_operand(data, *value)?;
    realm.heap.set_own(
        obj,
        key.to_ref(),
        match is_enumerable {
            true => heap::Property::Enumerable(value),
            false => heap::Property::NonEnumerable(value),
        },
    );
    Ok(())
}

define_flag!(OnlyEnumerable);
define_flag!(IncludeInherited);

fn obj_get_keys(
    data: &mut stack::InterpreterData,
    realm: &mut Realm,
    obj: &VReg,
    dest: &VReg,
    only_enumerable: OnlyEnumerable,
    include_inherited: IncludeInherited,
) -> RunResult<()> {
    let keys = {
        let obj = get_operand(data, *obj)?;
        let mut keys = Vec::new();

        realm
            .heap
            .own_properties(obj, only_enumerable.into(), &mut keys);
        if include_inherited.into() {
            realm
                .heap
                .list_properties_prototypes(obj, only_enumerable.into(), &mut keys);
        }

        keys
    };

    let keys = keys
        .into_iter()
        .map(|name| {
            let name = JSString::new_from_str(&name);
            let new_string = realm.heap.new_string(name);
            Value::String(new_string)
        })
        .collect();
    let keys_oid = realm.heap.new_array(keys);
    data.top_mut().set_result(*dest, Value::Object(keys_oid));
    Ok(())
}

/// Throw an exception.
///
/// This function returns `Ok(())` when an exception handler has been found.
/// The interpreter's state (execution stack, exception handlers, etc.)  will
/// have been modified so as to be ready to handle the exception.  It will point
/// to the correct exception handler, and it will be sufficient to "continue"
/// the interpreter execution regularly to handle the exception correctly.
///
/// It returns `Err(RunError::Exception(...))` when the exception is unhandled
/// (i.e. no exception handlers are registered).
///
/// When feature `debugging` is enabled, this function can also return
/// `Err(RunError::Suspended(...))`. In this case, the interpreter's state is
/// prepared for a subsequent resume.
fn throw_exc(
    exc: Value,
    iid: bytecode::IID,
    data: &mut stack::InterpreterData,
    #[cfg(feature = "debugger")] dbg: &Option<&mut debugger::DebuggingState>,
) -> RunResult<()> {
    // depth of the stack frame that has an exception handler = the number of stack frames
    // to remove from the top as a result of unwinding for exception handling
    //
    // None iff there is no exception handler anywhere on the stack (exception is going to be
    // unhandled)
    let handler_depth = data
        .frames()
        .enumerate()
        .find(|(_, frame)| !frame.header().exc_handlers.is_empty())
        .map(|(ndx, _)| ndx);

    #[cfg(feature = "debugger")]
    if let Some(dbg) = dbg {
        if dbg.should_break_on_throw()
            || (dbg.should_break_on_unhandled_throw() && handler_depth.is_some())
        {
            // Save the IID so that execution can resume correctly afterwards
            data.top_mut().set_resume_iid(iid);
            return Err(RunError::Suspended(SuspendCause::Exception(exc)));
        }
    }

    match handler_depth {
        None => {
            // NOTE: We return the error but we PRESERVE the state of the stack.  The
            // interpreter's state (including the stack) is supposed to be discarded
            // before late anyway, but this gives an opportunity to use the stack for
            // adding extra context to the error value.
            return Err(RunError::Exception(exc));
        }

        Some(handler_depth) => {
            for _ in 0..handler_depth {
                data.pop();
            }

            let handler_iid = data.top_mut().pop_exc_handler().unwrap();
            data.set_cur_exc(exc);
            data.top_mut().set_resume_iid(handler_iid);
            return Ok(());
        }
    }
}

/// Save the interpreter state and return
/// `Err(RunError::Suspended(SuspendCause::Breakpoint))` *if* the interpreter is in fact
/// supposed to do so.  In case the interpreter is supposed *not* to suspend and continue
/// execution, it returns `Ok(())`.
///
/// The return value is designed so that you can just `try!` a call to it (i.e.
/// `suspend_for_breakpoint(...)?`) in the body of `run_regular` and get the proper
/// behavior.
fn suspend_for_breakpoint(data: &mut stack::InterpreterData, iid: bytecode::IID) -> RunResult<()> {
    #[cfg(not(feature = "debugger"))]
    let _ = (data, iid);

    #[cfg(feature = "debugger")]
    if !data.take_resuming_from_breakpoint() {
        return Err(force_suspend_for_breakpoint(data, iid));
    }

    // In case of cfg(not(feature = "debugger")), just always keep going
    // Don't suspend. Execute the instruction and keep going
    Ok(())
}

/// Save the interpreter state and return
/// `RunError::Suspended(SuspendCause::Breakpoint)`
#[cfg(feature = "debugger")]
fn force_suspend_for_breakpoint(data: &mut stack::InterpreterData, iid: bytecode::IID) -> RunError {
    // Important: Commit the *current* IID to the interpreter state so that:
    //  1. debugging tools can see the correct IID
    //  2. a successor interpreter will resume by *repeating* the instruction where the
    //     suspension happened. That will result in a second call to `suspend_for_breakpoint`,
    //     which will return `Ok(())` on this second run allowing execution to continue.
    data.top_mut().set_resume_iid(iid);
    RunError::Suspended(SuspendCause::Breakpoint)
}

// TODO(cleanup) inline this function? It now adds nothing
fn get_operand(data: &stack::InterpreterData, vreg: bytecode::VReg) -> RunResult<Value> {
    data.top()
        .get_result(vreg)
        .ok_or_else(|| error!("variable read before initialization").into())
}

fn get_operand_string<'r>(
    data: &stack::InterpreterData,
    realm: &'r mut Realm,
    vreg: bytecode::VReg,
) -> RunResult<JSString> {
    let obj = get_operand(data, vreg)?;
    realm
        .heap
        .as_str(obj)
        .ok_or_else(|| error!("not a string!").into())
        .cloned()
}

fn js_typeof(value: &Value, realm: &mut Realm) -> Value {
    let ty_s = match value {
        Value::Null => "object",
        Value::Undefined => "undefined",
        _ => match realm.heap.type_of(*value) {
            heap::Typeof::Object => "object",
            heap::Typeof::Function => "function",
            heap::Typeof::String => "string",
            heap::Typeof::Number => "number",
            heap::Typeof::Boolean => "boolean",
            heap::Typeof::Symbol => "symbol",
            heap::Typeof::Undefined => "undefined",
        },
    };

    let ty_s = JSString::new_from_str(ty_s);
    let ty_s = realm.heap.new_string(ty_s);
    Value::String(ty_s)
}

fn is_instance_of(
    data: &stack::InterpreterData,
    realm: &mut Realm,
    obj: VReg,
    sup: VReg,
) -> RunResult<bool> {
    let sup_oid = match get_operand(data, sup)? {
        Value::Object(oid) => oid,
        _ => return Ok(false),
    };

    let obj = get_operand(data, obj)?;
    Ok(realm.heap.is_instance_of(obj, sup_oid))
}

fn with_numbers<F>(
    data: &mut stack::InterpreterData,
    dest: VReg,
    a: VReg,
    b: VReg,
    op: F,
) -> RunResult<()>
where
    F: FnOnce(f64, f64) -> f64,
{
    let a = get_operand(data, a)?.expect_num();
    let b = get_operand(data, b)?.expect_num();
    let value = match (a, b) {
        (Ok(a), Ok(b)) => Value::Number(op(a, b)),
        (_, _) => {
            // TODO: Try to convert values to numbers. For example:
            //   { valueOf() { return 42; } } => 42
            //   "10" => 10
            Value::Number(f64::NAN)
        }
    };

    data.top_mut().set_result(dest, value);
    Ok(())
}

fn compare(
    data: &mut stack::InterpreterData,
    realm: &mut Realm,
    dest: VReg,
    a: VReg,
    b: VReg,
    test: impl Fn(ValueOrdering) -> bool,
) -> RunResult<()> {
    let a = get_operand(data, a)?;
    let b = get_operand(data, b)?;

    let ordering = match (&a, &b) {
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b).into(),
        (Value::Number(a), Value::Number(b)) => a
            .partial_cmp(b)
            .map(|x| x.into())
            .unwrap_or(ValueOrdering::Incomparable),
        // null and undefined are mutually ==
        (Value::Null | Value::Undefined, Value::Null | Value::Undefined) => ValueOrdering::Equal,
        #[rustfmt::skip]
        _ => {
            // TODO should this be implemented via coercing?
            let a_str = realm.heap.as_str(a);
            let b_str = realm.heap.as_str(b);

            // TODO Remove this special case when strings become primitives (i.e. Value::String)
            if let (Some(a_str), Some(b_str)) = (a_str, b_str) {
                a_str.view().cmp(b_str.view()).into()
            } else if let (Value::Object(a_oid), Value::Object(b_oid)) = (a, b)  {
                if a_oid == b_oid {
                    ValueOrdering::Equal
                } else {
                    ValueOrdering::Incomparable
                }
            } else {
                ValueOrdering::Incomparable
            }
        }
    };

    data.top_mut().set_result(dest, Value::Bool(test(ordering)));
    Ok(())
}

/// Converts the given value to a number, in the way that is typically invoked
/// by the unary plus operator (e.g. `+{}`, `+123.0`)
///
/// A `Some` is returned for a successful conversion. On failure (e.g. for
/// Symbol), a `None` is returned.
fn to_number_value(value: Value) -> Option<Value> {
    to_number(value).map(Value::Number)
}
fn to_number(value: Value) -> Option<f64> {
    match value {
        Value::Null => Some(0.0),
        Value::Bool(true) => Some(1.0),
        Value::Bool(false) => Some(0.0),
        Value::Number(num) => Some(num),
        Value::Object(_) => Some(f64::NAN),
        // TODO
        Value::String(_) => Some(f64::NAN),
        Value::Undefined => Some(f64::NAN),
        Value::Symbol(_) => None,
    }
}
/// Implements JavaScript's implicit conversion to string.
fn value_to_string(value: Value, heap: &heap::Heap) -> Result<JSString> {
    // TODO Sink the error mgmt that was here into js_to_string
    Ok(heap.js_to_string(value))
}

fn property_to_value(prop: &heap::Property, heap: &mut heap::Heap) -> Result<Value> {
    match prop {
        heap::Property::Enumerable(value) | heap::Property::NonEnumerable(value) => Ok(*value),
        heap::Property::Substring(jss) => Ok(Value::String(heap.new_string(jss.clone()))),
    }
}

/// Write a human-readable description of the value.
///
/// If the value is a string, it will appear in the output (possibly truncated,
/// with a note about the total length).
///
/// If the value is an object, a listing of its properties will appear in its
/// output.
fn show_value_ex<W: std::io::Write>(
    out: &mut W,
    value: Value,
    realm: &mut Realm,
    params: &ShowValueParams,
) {
    if let Some(s) = realm.heap.as_str(value) {
        writeln!(out, "{:?}", s).unwrap();
        return;
    }

    let mut props = Vec::new();
    realm.heap.own_properties(value, false, &mut props);

    write!(out, "{} ", realm.heap.show_debug(value)).unwrap();
    writeln!(out, "[{} properties]", props.len()).unwrap();

    if params.max_object_depth > 0 {
        for key in props {
            for _ in 0..params.indent {
                write!(out, "    ").unwrap();
            }

            let property = realm.heap.get_own(value, IndexOrKey::Key(&key)).unwrap();
            write!(out, "  - {:?} = ", key).unwrap();
            show_value_ex(
                out,
                // get_own is not allowed to return a Property without a value
                // (such as Property::String)
                property.value().unwrap(),
                realm,
                &ShowValueParams {
                    indent: params.indent + 1,
                    max_object_depth: params.max_object_depth - 1,
                },
            );
        }
    }
}

#[derive(Default, Clone, Copy)]
struct ShowValueParams {
    indent: u8,
    max_object_depth: u8,
}

fn str_append(
    data: &stack::InterpreterData,
    realm: &mut Realm,
    a: VReg,
    b: VReg,
) -> RunResult<Value> {
    // TODO Make this at least *decently* efficient!
    let a = get_operand_string(data, realm, a)?;
    let b = get_operand(data, b)?;

    let mut buf = a.view().to_vec();
    let tail = value_to_string(b, &realm.heap)?;
    buf.extend_from_slice(tail.view());
    let jss: JSString = JSString::new(buf);
    let sid = realm.heap.new_string(jss);
    Ok(Value::String(sid))
}

/// Create a Value based on the given Literal.
///
/// It may allocate an object in the GC-managed heap.
fn literal_to_value(lit: bytecode::Literal, heap: &mut heap::Heap) -> Value {
    match lit {
        bytecode::Literal::Number(nu) => Value::Number(nu),
        bytecode::Literal::String(st) => {
            // TODO(performance) avoid this allocation
            let jss = JSString::new_from_str(&st);
            let oid = heap.new_string(jss);
            Value::String(oid)
        }
        bytecode::Literal::Symbol(sym) => Value::Symbol(sym),
        bytecode::Literal::JsWord(jsw) => {
            // TODO(performance) avoid this allocation
            let jss = JSString::new_from_str(&jsw.to_string());
            let oid = heap.new_string(jss);
            Value::String(oid)
        }
        bytecode::Literal::Bool(bo) => Value::Bool(bo),
        bytecode::Literal::Null => Value::Null,
        bytecode::Literal::Undefined => Value::Undefined,
    }
}

#[cfg(any(test, feature = "debugger"))]
fn try_value_to_literal(value: Value, heap: &heap::Heap) -> Option<bytecode::Literal> {
    match value {
        Value::Number(num) => Some(bytecode::Literal::Number(num)),
        Value::Bool(b) => Some(bytecode::Literal::Bool(b)),
        Value::String(_) => heap
            .as_str(value)
            .map(|s| bytecode::Literal::String(s.to_string())),
        Value::Object(_) => None,
        Value::Symbol(sym) => Some(bytecode::Literal::Symbol(sym)),
        Value::Null => Some(bytecode::Literal::Null),
        Value::Undefined => Some(bytecode::Literal::Undefined),
    }
}

fn value_to_index_or_key(heap: &heap::Heap, value: &Value) -> Option<heap::IndexOrKeyOwned> {
    match value {
        Value::Number(n) if *n >= 0.0 => {
            let n_trunc = n.trunc();
            if *n == n_trunc {
                let ndx = n_trunc as u32;
                Some(heap::IndexOrKeyOwned::Index(ndx))
            } else {
                None
            }
        }
        // TODO Use string coercing?
        Value::Symbol(sym) => Some(heap::IndexOrKeyOwned::Symbol(sym)),
        _ => {
            let string = heap.as_str(*value)?.to_string();
            Some(heap::IndexOrKeyOwned::Key(string))
        }
    }
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

#[cfg(feature = "debugger")]
pub mod debugger {
    use std::collections::HashMap;

    use super::stack;
    use crate::GlobalIID;
    use crate::{bytecode, loader};

    pub use super::SuspendCause;
    pub use crate::loader::BreakRangeID;

    #[derive(Debug)]
    pub enum BreakpointError {
        /// Breakpoint already set at the given location (can't have more than 1 at the
        /// same location).
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
    impl std::fmt::Display for BreakpointError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.message())
        }
    }
    impl std::error::Error for BreakpointError {}

    pub fn giid(data: &stack::InterpreterData) -> bytecode::GlobalIID {
        let frame = data.top();
        bytecode::GlobalIID(frame.header().fnid, frame.header().iid)
    }

    fn giid_of_break_range(
        loader: &loader::Loader,
        brange_id: BreakRangeID,
    ) -> std::result::Result<bytecode::GlobalIID, BreakpointError> {
        let break_range: &bytecode::BreakRange = loader
            .get_break_range(brange_id)
            .ok_or(BreakpointError::InvalidLocation)?;
        Ok(bytecode::GlobalIID(break_range.fnid, break_range.iid_start))
    }

    /// Get the instruction pointer for the n-th frame (0 = top)
    ///
    /// Panics if `frame_ndx` is invalid.
    pub fn frame_giid(data: &stack::InterpreterData, frame_ndx: usize) -> bytecode::GlobalIID {
        // `frame_ndx` is top-first (0 = top)
        // the stack API is bottom first (0 = bottom), so convert first
        let frame = data.nth_frame(data.len() - frame_ndx - 1);
        let fnid = frame.header().fnid;
        let iid = frame.header().iid;
        bytecode::GlobalIID(fnid, iid)
    }

    /// Additional debugging-specific state added to the interpreter.
    pub struct DebuggingState {
        /// Instruction breakpoints
        instr_bkpts: HashMap<GlobalIID, InstrBreakpoint>,

        /// Source breakpoints, indexed by their ID.
        ///
        /// Each source breakpoint corresponds to exactly to one instruction breakpoint,
        /// which is added/deleted together with it.
        source_bkpts: HashMap<BreakRangeID, SourceBreakpoint>,

        /// This special flag can be used to cause the interpreter to suspend
        /// after N instructions. It's conceptually similar to a breakpoint that
        /// automatically follows call/return.
        ///
        /// See the type, `Fuel`.
        fuel: Fuel,

        /// If true, the interpreter will suspend right after an unhandled
        /// exception is thrown, but before the stack is unwound in the attempt
        /// to handle it.   This is useful to look at the interpreter's state
        /// before it gets completely undone after the handling attempt.
        break_on_unhandled_throw: bool,

        /// If true, the interpreter will suspend upon reaching any `Instr::Throw`.
        break_on_throw: bool,
    }

    #[derive(Debug, PartialEq, Eq)]
    pub enum Fuel {
        Limited(usize),
        Unlimited,
    }

    // There is nothing here for now. The mere existence of an entry in
    // Interpreter.source_bktps is enough (but some addtional parameters might have to be
    // includede here later)
    pub struct SourceBreakpoint;

    #[derive(Default)]
    pub struct InstrBreakpoint {
        src_bkpt: Option<BreakRangeID>,

        /// Delete this breakpoint as soon as it's hit.
        ///
        /// Typical use case is temporary breakpoints (which are in turn used for the
        /// debugger's function "next", which doesn't follow call/return).
        pub delete_on_hit: bool,
    }

    // This instance is important because module users must not be able to write private
    // members of InstrBreakpoint, not even for initialization.

    impl DebuggingState {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            DebuggingState {
                instr_bkpts: HashMap::new(),
                source_bkpts: HashMap::new(),
                fuel: Fuel::Unlimited,
                break_on_unhandled_throw: false,
                break_on_throw: false,
            }
        }

        pub fn set_fuel(&mut self, fuel: Fuel) {
            self.fuel = fuel;
        }
        pub fn fuel_empty(&self) -> bool {
            self.fuel == Fuel::Limited(0)
        }
        pub fn consume_1_fuel(&mut self) {
            match &mut self.fuel {
                Fuel::Limited(count) => {
                    assert!(*count > 0);
                    *count -= 1;
                }
                Fuel::Unlimited => {}
            }
        }

        //
        // Source breakpoints
        //

        /// Set a breakpoint at the specified instruction.
        ///
        /// After this operation, the interpreter will suspend itself (Interpreter::run
        /// will return Exit::Suspended), and it will be possible to examine its
        /// state (by attaching a new Probe on it).
        pub fn set_source_breakpoint(
            &mut self,
            brange_id: BreakRangeID,
            loader: &loader::Loader,
        ) -> std::result::Result<(), BreakpointError> {
            if self.source_bkpts.contains_key(&brange_id) {
                return Err(BreakpointError::AlreadyThere);
            }

            let giid = giid_of_break_range(loader, brange_id)?;
            let bkpt = InstrBreakpoint {
                src_bkpt: Some(brange_id),
                delete_on_hit: false,
            };
            self.set_instr_bkpt(giid, bkpt)?;

            let prev = self.source_bkpts.insert(brange_id, SourceBreakpoint);
            assert!(prev.is_none());

            Ok(())
        }

        /// Delete the breakpoint with the given ID.
        ///
        /// Returns true only if there was actually a breakpoint with the given ID; false
        /// if the ID did not correspond to any breakpoint.
        pub fn clear_source_breakpoint(
            &mut self,
            brange_id: BreakRangeID,
            loader: &loader::Loader,
        ) -> std::result::Result<bool, BreakpointError> {
            if !self.source_bkpts.contains_key(&brange_id) {
                return Ok(false);
            }

            let giid = giid_of_break_range(loader, brange_id)?;
            let was_there = self.clear_instr_bkpt(giid);
            assert!(was_there);

            Ok(true)
        }

        pub fn source_breakpoints(
            &self,
        ) -> impl ExactSizeIterator<Item = (BreakRangeID, &SourceBreakpoint)> {
            self.source_bkpts.iter().map(|(k, v)| (*k, v))
        }

        //
        // Instruction breakpoints
        //

        pub fn set_instr_bkpt(
            &mut self,
            giid: bytecode::GlobalIID,
            bkpt: InstrBreakpoint,
        ) -> std::result::Result<(), BreakpointError> {
            let new_insert = self.instr_bkpts.insert(giid, bkpt);
            match new_insert {
                None => Ok(()),
                Some(_) => Err(BreakpointError::AlreadyThere),
            }
        }

        pub fn clear_instr_bkpt(&mut self, giid: bytecode::GlobalIID) -> bool {
            let ibkpt = self.instr_bkpts.remove(&giid);

            if let Some(ibkpt) = &ibkpt {
                if let Some(brid) = &ibkpt.src_bkpt {
                    self.source_bkpts.remove(brid).unwrap();
                }
            }

            ibkpt.is_some()
        }

        pub fn instr_bkpts(&self) -> impl '_ + ExactSizeIterator<Item = bytecode::GlobalIID> {
            self.instr_bkpts.keys().copied()
        }

        pub fn instr_bkpt_at(&self, giid: &bytecode::GlobalIID) -> Option<&InstrBreakpoint> {
            self.instr_bkpts.get(giid)
        }

        pub fn set_break_on_throw(&mut self, value: bool) {
            self.break_on_throw = value;
        }
        pub fn should_break_on_throw(&self) -> bool {
            self.break_on_throw
        }

        pub fn set_break_on_unhandled_throw(&mut self, value: bool) {
            self.break_on_unhandled_throw = value;
        }
        pub fn should_break_on_unhandled_throw(&self) -> bool {
            self.break_on_unhandled_throw
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{bytecode::Literal, bytecode_compiler};

    fn quick_run_script(code: &str) -> FinishedData {
        let mut loader = loader::Loader::new_cwd();
        let chunk_fnid = loader
            .load_script_anon(code.to_string())
            .expect("couldn't compile test script");
        complete_run(&mut loader, chunk_fnid)
    }

    fn complete_run(loader: &mut crate::Loader, root_fnid: FnID) -> FinishedData {
        let mut realm = Realm::new(loader);
        let vm = Interpreter::new(&mut realm, loader, root_fnid);
        match vm.run() {
            Ok(exit) => exit.expect_finished(),
            Err(err_box) => {
                let error = err_box.error.with_loader(&loader);
                panic!("{:?}", error);
            }
        }
    }

    #[test]
    fn test_simple_call() {
        let output = quick_run_script("/* Here is some simple code: */ sink(1 + 4 + 99); ");
        assert_eq!(&[Some(Literal::Number(104.0))], &output.sink[..]);
    }

    #[test]
    fn test_multiple_calls() {
        let output =
            quick_run_script("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ");
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
        let output = quick_run_script(
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
        let output = quick_run_script(
            "
            function foo(a, b) { return a + b; }
            sink(foo(1, 2));
            ",
        );

        assert_eq!(&[Some(Literal::Number(3.0))], &output.sink[..]);
    }

    #[test]
    fn test_fn_with_branch() {
        let output = quick_run_script(
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
        let output = quick_run_script(
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

    #[test]
    fn test_switch() {
        let output = quick_run_script(
            "
            function trySwitch(x) {
                switch(x) {
                case 'a':
                case 'b':
                    sink(1);
                case 'c':
                    sink(2);
                    break;

                case 'd':
                    sink(3);
                }
                sink(99);
            }

            trySwitch('b');
            trySwitch('d');
            trySwitch('c');
            trySwitch('y');
            trySwitch('a');
            ",
        );
        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Number(1.0)),
                Some(Literal::Number(2.0)),
                Some(Literal::Number(99.0)),
                //
                Some(Literal::Number(3.0)),
                Some(Literal::Number(99.0)),
                //
                Some(Literal::Number(2.0)),
                Some(Literal::Number(99.0)),
                //
                Some(Literal::Number(99.0)),
                //
                Some(Literal::Number(1.0)),
                Some(Literal::Number(2.0)),
                Some(Literal::Number(99.0)),
            ]
        );
    }

    fn try_casting_bool(code: &str, expected_value: bool) {
        let output = quick_run_script(code);
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
    fn test_number_cast() {
        let output = quick_run_script(
            "
            sink(Number(123.0));
            sink(Number('-123.45e9'));
            sink(Number(true));
            sink(Number(false));
            sink(Number(null));
            sink(Number({a: 3}));
            sink(Number());
            ",
        );

        assert!(matches!(
            &output.sink[..],
            &[
                Some(Literal::Number(123.0)),
                Some(Literal::Number(-123450000000.0)),
                Some(Literal::Number(1.0)),
                Some(Literal::Number(0.0)),
                Some(Literal::Number(0.0)),
                Some(Literal::Number(a)),
                Some(Literal::Number(b)),
            ]
            if a.is_nan() && b.is_nan()
        ))
    }

    // Un-ignore when Symbol.for is implemented
    #[test]
    #[ignore]
    fn test_number_cast_symbol() {
        let output = quick_run_script(
            "
            try { Number(Symbol.for('asd')) }
            catch (err) { sink(err.name) }
            ",
        );

        assert_eq!(
            &output.sink,
            &[Some(Literal::String("TypeError".to_string())),]
        )
    }
    #[test]
    fn test_capture() {
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
    fn test_number_constructor_prototype() {
        let mut loader = loader::Loader::new_cwd();
        let realm = Realm::new(&mut loader);
        let number = realm
            .heap
            .get_chained(realm.global_obj, heap::IndexOrKey::Key("Number"))
            .unwrap()
            .value()
            .unwrap();
        let prototype = realm
            .heap
            .get_chained(number, heap::IndexOrKey::Key("prototype"))
            .unwrap()
            .value()
            .unwrap();
        assert_eq!(prototype, Value::Object(realm.heap.number_proto()));
    }

    #[test]
    fn test_methods_on_numbers() {
        let output = quick_run_script(
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
        let output = quick_run_script(
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
        let output = quick_run_script(
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
    fn test_script_global_fn_nonstrict() {
        let output = quick_run_script(
            r#"
            function x() { return 55 }
            sink(globalThis.x())
            sink(x())
            "#,
        );

        assert_eq!(
            &output.sink,
            &[Some(Literal::Number(55.0)), Some(Literal::Number(55.0)),],
        );
    }

    #[test]
    fn test_script_global_fn_strict() {
        let output = quick_run_script(
            r#"
            "use strict";
            function x() { return 55 }
            sink(globalThis.x())
            sink(x())
            "#,
        );

        assert_eq!(
            &output.sink,
            &[Some(Literal::Number(55.0)), Some(Literal::Number(55.0)),],
        );
    }

    #[test]
    fn test_constructor_prototype() {
        quick_run_script(
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
        let output = quick_run_script(
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

    #[test]
    fn test_temporal_dead_zone_none() {
        let _ = quick_run_script(
            r#"
                "use strict"
                let x = 12
                sink(foo())
                function foo() { return x }
            "#,
        );
    }

    #[test]
    #[should_panic]
    fn test_temporal_dead_zone() {
        let _ = quick_run_script(
            r#"
                "use strict"
                sink(foo())
                let x = 12
                function foo() { return x }
            "#,
        );
    }

    #[test]
    fn test_reference_error() {
        let output = quick_run_script(
            r#"
                try {
                    aVariableThatDoesNotExist;
                } catch (e) {
                    sink(e instanceof ReferenceError);
                }
            "#,
        );

        assert_eq!(&output.sink, &[Some(Literal::Bool(true))]);
    }

    #[test]
    fn test_unwinding_on_exception() {
        let output = quick_run_script(
            r#"
                try {
                    (function() {
                        throw 42;
                    })()
                } catch (e) {
                    sink(e);
                }
            "#,
        );

        assert_eq!(&output.sink, &[Some(Literal::Number(42.0))]);
    }

    #[test]
    fn test_void_operator() {
        let output = quick_run_script("sink(123); sink(void 123);");
        assert_eq!(
            &output.sink,
            &[Some(Literal::Number(123.0)), Some(Literal::Undefined)]
        );
    }

    #[test]
    fn test_eval_completion_value() {
        let output = quick_run_script("sink(eval('11'))");
        assert_eq!(&output.sink, &[Some(Literal::Number(11.0))]);
    }

    #[test]
    fn test_array_properties() {
        let output = quick_run_script(
            r#"
            const arr = ['a', 123, false]
            for (const name in arr) {
              sink(name);
              sink(arr[name]);
            }
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::String("0".to_string())),
                Some(Literal::String("a".to_string())),
                Some(Literal::String("1".to_string())),
                Some(Literal::Number(123.0)),
                Some(Literal::String("2".to_string())),
                Some(Literal::Bool(false)),
            ]
        );
    }

    #[test]
    fn test_generator_basic() {
        let output = quick_run_script(
            r#"
function* makeGenerator() {
  yield 'first';
  for (let i=0; i < 5; i++) {
    yield i * 2;
  }
  yield 123;
}

const generator = makeGenerator();
let item;
do {
  item = generator.next();
  sink(item.value);
} while (!item.done);
        "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::String("first".to_string())),
                Some(Literal::Number(0.0)),
                Some(Literal::Number(2.0)),
                Some(Literal::Number(4.0)),
                Some(Literal::Number(6.0)),
                Some(Literal::Number(8.0)),
                Some(Literal::Number(123.0)),
                Some(Literal::Undefined),
            ]
        );
    }

    #[test]
    fn test_for_of_generator() {
        let output = quick_run_script(
            r#"
                function* getSomeNumbers() {
                    for (let i=0; i < 5; i++) {
                        yield i * 3;
                    }
                }

                for (const value of getSomeNumbers()) {
                    sink(value);
                }
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Number(0.0)),
                Some(Literal::Number(3.0)),
                Some(Literal::Number(6.0)),
                Some(Literal::Number(9.0)),
                Some(Literal::Number(12.0)),
            ]
        );
    }

    #[test]
    fn test_short_circuiting_and() {
        let output = quick_run_script(
            r#"
                function getSomethingElse() { sink(2); return 456; }
                function getFalsy() { sink(1); return ""; }
                sink(getFalsy() && getSomethingElse());
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Number(1.0)),
                Some(Literal::String("".to_string())),
            ]
        );
    }

    #[test]
    fn test_short_circuiting_or() {
        let output = quick_run_script(
            r#"
                function getSomethingElse() { sink(2); return 456; }
                function getTruthy() { sink(1); return 123; }
                sink(getTruthy() || getSomethingElse());
            "#,
        );

        assert_eq!(
            &output.sink,
            &[Some(Literal::Number(1.0)), Some(Literal::Number(123.0)),]
        );
    }

    #[test]
    fn test_module_default_export() {
        let mut loader = loader::Loader::new_cwd();
        let root_fnid = loader
            .load_code_forced(
                loader::FileID::File(PathBuf::from("/virtualtest/root.mjs")),
                r#"
                    import the_thing from "./somewhere/the_thing.mjs";
                    sink(the_thing.the_content);
                "#
                .to_string(),
                bytecode_compiler::SourceType::Module,
            )
            .unwrap();

        loader
            .load_code_forced(
                loader::FileID::File(PathBuf::from("/virtualtest/somewhere/the_thing.mjs")),
                r#"
                    export default { the_content: 123 };
                "#
                .to_string(),
                bytecode_compiler::SourceType::Module,
            )
            .unwrap();

        let finished_data = complete_run(&mut loader, root_fnid);
        assert_eq!(&finished_data.sink, &[Some(Literal::Number(123.0)),]);
    }

    #[test]
    fn test_module_named_export() {
        let mut loader = loader::Loader::new_cwd();
        let root_fnid = loader
            .load_code_forced(
                loader::FileID::File(PathBuf::from("/virtualtest/root.mjs")),
                r#"
                    import * as the_thing from "./somewhere/the_thing.mjs";
                    sink(the_thing.the_content);
                    sink(the_thing.double_the_content());
                "#
                .to_string(),
                bytecode_compiler::SourceType::Module,
            )
            .unwrap();

        loader
            .load_code_forced(
                loader::FileID::File(PathBuf::from("/virtualtest/somewhere/the_thing.mjs")),
                r#"
                    export const the_content = 123;
                    export function double_the_content() {
                        return 2 * the_content;
                    };
                "#
                .to_string(),
                bytecode_compiler::SourceType::Module,
            )
            .unwrap();

        let finished_data = complete_run(&mut loader, root_fnid);
        assert_eq!(
            &finished_data.sink,
            &[Some(Literal::Number(123.0)), Some(Literal::Number(246.0))]
        );
    }

    #[test]
    fn test_delete_basic() {
        let output = quick_run_script(
            "
            const obj = {a: 123};
            sink(obj.a);
            delete obj.a;
            sink(obj.a);
        ",
        );
        assert_eq!(
            &output.sink,
            &[Some(Literal::Number(123.0)), Some(Literal::Undefined)]
        );
    }

    #[test]
    fn test_to_number() {
        let output = quick_run_script(
            "
            sink(+123.0);
            sink(+{});
            sink(+null);
            sink(+undefined);
            sink(+true);
            sink(+false);
        ",
        );

        let expected = &[123.0, f64::NAN, 0.0, f64::NAN, 1.0, 0.0];

        assert_eq!(output.sink.len(), expected.len());
        for (out, &exp) in output.sink.iter().zip(expected) {
            let out = if let Some(Literal::Number(n)) = out {
                *n
            } else {
                panic!("expected Literal::Number, got {:?}", out);
            };

            // Check for equality, including NaN
            assert_eq!(f64::to_bits(out), f64::to_bits(exp));
        }
    }

    #[test]
    fn test_string_index() {
        let output = quick_run_script(
            "const s = 'asdlol123';
            sink(s[0]);
            sink(s[4]);
            sink(s[8]);
            sink(s[9]);",
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::String("a".to_string())),
                Some(Literal::String("o".to_string())),
                Some(Literal::String("3".to_string())),
                Some(Literal::Undefined),
            ]
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_string_codePointAt() {
        let output = quick_run_script(
            "const s = 'asdlol123';
            for (let i=0; i < s.length; ++i) {
                sink(s.codePointAt(i));
            }
            sink(s.codePointAt(s.length));
            ",
        );

        let ref_string = "asdlol123";
        let ref_string_u16: Vec<_> = ref_string.encode_utf16().collect();

        assert_eq!(output.sink.len(), ref_string.len() + 1);

        for (i, &code_point) in ref_string_u16.iter().enumerate() {
            assert_eq!(output.sink[i], Some(Literal::Number(code_point as f64)));
        }
        assert_eq!(output.sink[ref_string.len()], Some(Literal::Undefined));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_string_fromCodePoint() {
        let output = quick_run_script(
            "const s = 'asdlol123';
            for (let i=0; i < s.length; ++i) {
                sink(String.fromCodePoint(s.codePointAt(i)));
            }

            try {
                String.fromCodePoint(undefined);
            } catch (err) {
                sink(err.name)
            }
            ",
        );

        let ref_string = "asdlol123";

        assert_eq!(output.sink.len(), ref_string.len() + 1);

        for (i, ch) in ref_string.chars().enumerate() {
            assert_eq!(output.sink[i], Some(Literal::String(ch.into())));
        }
        assert_eq!(
            output.sink[ref_string.len()],
            Some(Literal::String("RangeError".into()))
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_RegExp_test() {
        let output = quick_run_script(
            r#"const re = /^\d{2,5}$/;
            sink(re.test('12'));
            sink(re.test('123'));
            sink(re.test('1234'));
            sink(re.test('12345'));
            sink(re.test('x12345'));
            sink(re.test('123456'));
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Bool(true)),
                Some(Literal::Bool(true)),
                Some(Literal::Bool(true)),
                Some(Literal::Bool(true)),
                Some(Literal::Bool(false)),
                Some(Literal::Bool(false)),
            ]
        );
    }

    mod debugging {
        use super::*;
        use crate::Loader;

        #[test]
        #[cfg(feature = "debugger")]
        fn test_inline_breakpoint() {
            const SOURCE_CODE: &str = r#"
                function foo() {
                    sink(1);
                    debugger;
                    sink(2);
                }

                foo();
            "#;

            let mut loader = Loader::new_cwd();
            let main_fnid = loader.load_script_anon(SOURCE_CODE.to_string()).unwrap();

            let mut realm = Realm::new(&mut loader);

            let exit = Interpreter::new(&mut realm, &mut loader, main_fnid)
                .run()
                .expect("interpreter failed");

            let intrp_state = match exit {
                Exit::Finished(_) => panic!("finished instead of interrupting"),
                Exit::Suspended { intrp_state, .. } => intrp_state,
            };

            assert_eq!(&intrp_state.sink, &[Value::Number(1.0)]);

            let finish_data = Interpreter::resume(&mut realm, &mut loader, intrp_state)
                .run()
                .unwrap()
                .expect_finished();
            assert_eq!(
                &finish_data.sink,
                &[
                    Some(bytecode::Literal::Number(1.0)),
                    Some(bytecode::Literal::Number(2.0)),
                ]
            );
        }

        #[test]
        #[cfg(feature = "debugger")]
        fn test_pos_breakpoint() {
            let mut loader = Loader::new_cwd();

            const SOURCE_CODE: &str = r#"
                function foo() {
                    sink(1);
                    sink(2);
                }

                foo();
            "#;
            let main_fnid = loader.load_script_anon(SOURCE_CODE.to_string()).unwrap();

            // Hardcoded. Must be updated if breakme-0.js changes
            let pos = swc_common::BytePos(166);

            let mut realm = Realm::new(&mut loader);

            // Resolve into the (potentially multiple) GIIDs
            let break_range_ids: Vec<_> = loader
                .resolve_break_loc(main_fnid, pos)
                .unwrap()
                .into_iter()
                .map(|(brid, _)| brid)
                .collect();

            let mut dbg = debugger::DebuggingState::new();

            for brid in break_range_ids {
                dbg.set_source_breakpoint(brid, &loader).unwrap();

                let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);
                interpreter.set_debugging_state(&mut dbg);

                let intrp_state = match interpreter.run().expect("interpreter failed") {
                    Exit::Finished(_) => panic!("interpreter finished instead of breaking"),
                    Exit::Suspended { intrp_state, .. } => intrp_state,
                };

                let giid = debugger::giid(&intrp_state);
                eprintln!("we are at: {:?}; sink = {:?}", giid, intrp_state.sink);
                if giid.0 == bytecode::FnID(2) {
                    assert_eq!(&intrp_state.sink, &[Value::Number(1.0)]);
                } else {
                    assert_eq!(&intrp_state.sink, &[]);
                }

                let finish_data = Interpreter::resume(&mut realm, &mut loader, intrp_state)
                    .run()
                    .expect("interpreter failed")
                    .expect_finished();
                assert_eq!(
                    &finish_data.sink,
                    &[
                        Some(bytecode::Literal::Number(1.0)),
                        Some(bytecode::Literal::Number(2.0)),
                    ]
                );
            }
        }
    }
}
