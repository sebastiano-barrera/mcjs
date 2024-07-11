use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::Rc;

use crate::error_item;
use crate::heap::JSString;
use crate::{
    bytecode::{self, FnID, Instr, VReg, IID},
    common::{self, Result},
    define_flag, error, heap, loader,
};

mod builtins;
pub mod stack;
mod tests;

#[cfg(feature = "debugger")]
pub use stack::SlotDebug;

use stack::UpvalueRef;

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
impl Value {
    fn is_primitive(&self) -> bool {
        !matches!(self, Value::Object(_))
    }
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

/// A closure.
///
/// It can be cloned, and the resulting value will be independent from the
/// source. It will be backed by the same function, and use the same upvalues,
/// have the same "forced this", etc., but these can be changed independently.
#[derive(Clone)]
pub struct Closure {
    func: Func,
    is_strict: bool,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueRef>,
    forced_this: Option<Value>,
    generator_snapshot: RefCell<Option<stack::FrameSnapshot>>,
}
impl Closure {
    pub fn upvalues(&self) -> &[UpvalueRef] {
        &self.upvalues
    }

    /// Create a simple closure backed by a native function.
    ///
    /// This is a shorthand for the common case where native functions are
    /// trivially wrapped into a closure, without upvalues, generator state,
    /// etc.
    pub(crate) fn new_native(nf: NativeFn) -> Self {
        Closure {
            func: Func::Native(nf),
            upvalues: Vec::new(),
            forced_this: None,
            generator_snapshot: RefCell::new(None),
            // by convention, native functions are always strict
            is_strict: true,
        }
    }
}

#[derive(Clone, Copy)]
enum Func {
    Native(NativeFn),
    JS(JSFn),
}

pub(crate) type NativeFn =
    fn(&mut Realm, &mut loader::Loader, &Value, &[Value]) -> RunResult<Value>;
#[derive(Clone, Copy)]
pub struct JSFn(FnID);

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.func {
            Func::Native(nf) => {
                write!(f, "<native closure {:?} ", (nf as *const ()))?;
            }
            Func::JS(JSFn(fnid)) => {
                write!(f, "<closure {:?} ", fnid)?;
            }
        };

        write!(f, "| ")?;
        for upv in &self.upvalues {
            write!(f, "{:?} ", upv)?;
        }
        write!(f, ">")
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

    pub fn heap(&self) -> &heap::Heap {
        &self.heap
    }

    pub fn heap_mut(&mut self) -> &mut heap::Heap {
        &mut self.heap
    }

    pub fn global_obj(&self) -> Value {
        self.global_obj
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
    pub ret_val: Value,
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

    fn new_call(
        realm: &'a mut Realm,
        loader: &'a mut crate::Loader,
        closure: Rc<Closure>,
        this: Value,
        args: &[Value],
    ) -> Self {
        // Initialize the stack with a single frame, corresponding to a call to fnid with no
        // parameters, then put it into an Interpreter
        let mut data = stack::InterpreterData::new();
        data.set_default_this(realm.global_obj);

        data.push_call(closure, this, loader);
        for i in 0..bytecode::ARGS_COUNT_MAX {
            let arg = args.get(i as usize).unwrap_or(&Value::Undefined);
            data.top_mut().set_arg(bytecode::ArgIndex(i as _), *arg);
        }

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
                    ret_val: self.data.final_return_value(),
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
    loop {
        let res = run_inner(
            data,
            realm,
            loader,
            #[cfg(feature = "debugger")]
            dbg,
        );

        match res {
            Ok(_) => return res,
            Err(RunError::Exception(exc)) => {
                // run_inner does not handle exception; we do it here

                // `handle_exception` returns:
                //  - Err(RunError::Exception(_)) if the exception is unhandled => bubble up to
                //    caller
                //  - Ok(()) => there IS an handler and `data` is ready to resume execution from the
                //    handler, just call run_inner again
                handle_exception(
                    exc,
                    data,
                    #[cfg(feature = "debugger")]
                    dbg,
                )?;

                // just loop back, resume exec
            }
            #[cfg(feature = "debugger")]
            Err(RunError::Suspended(_)) => return res,
            Err(RunError::Internal(mut err)) => {
                // just add some context and bubble it up
                for frame in data.frames() {
                    let mut ctx = error_item!("<- interpreter was here");
                    let giid = bytecode::GlobalIID(frame.header().fnid, frame.header().iid);
                    ctx.set_giid(giid);
                    err.push_context(ctx);
                }

                return Err(RunError::Internal(err));
            }
        }
    }
}

/// Run bytecode. This is the core of the interpreter.
///
/// Keeps going, until either:
///  - the stack is depleted (execution has finished normally);
///  - an exception is thrown
///  - the interpreter suspends (e.g. for a debugger breakpoint)
///
/// Right before this function is called, `data` is presumed to hold the
/// interpreter's stack in a consistent state.  At least one stack frame must
/// be present, and the top frame represents the function and instruction
/// where execution resumes.  
///
/// In case of exception or breakpoint, `data` is left representing
/// the interpreter's state at the time of suspension. It can be inspected,
/// manipulated, and eventually used in a new call to `run_inner` to resume
/// execution.
///
/// JavaScript exceptions are NOT handled by this function.
///
/// If a bug is detected (e.g. assertion failed), the function panics. The
/// interpreter's data is not unwind-safe in general, so execution can only
/// restart by initializing a new Interpreter instance.
fn run_inner(
    data: &mut stack::InterpreterData,
    realm: &mut Realm,
    loader: &mut loader::Loader,
    #[cfg(feature = "debugger")] dbg: &mut Option<&mut debugger::DebuggingState>,
) -> RunResult<()> {
    loop {
        if data.is_empty() {
            // We're done.
            return Ok(());
        }

        let fnid = data.top().header().fnid;
        let func = loader.get_function(fnid).unwrap();

        let iid = data.top().header().iid;
        if iid.0 as usize == func.instrs().len() {
            // Bytecode "finished" => Implicitly return undefined
            do_return(Value::Undefined, data, realm);
            continue;
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

            Instr::OpAdd(dest, a, b) => {
                let a = get_operand(data, *a)?;
                let b = get_operand(data, *b)?;

                assert!(a.is_primitive());
                assert!(b.is_primitive());

                let result = match (a, b) {
                    (Value::String(_), _) | (_, Value::String(_)) => {
                        let a_strv = to_string_or_throw(a, realm, loader)?;
                        let b_strv = to_string_or_throw(b, realm, loader)?;

                        let a_str = realm.heap.as_str(a_strv).unwrap().view();
                        let b_str = realm.heap.as_str(b_strv).unwrap().view();

                        let mut concated = Vec::with_capacity(a_str.len() + b_str.len());
                        concated.extend_from_slice(a_str);
                        concated.extend_from_slice(b_str);
                        let concated = JSString::new_from_utf16(concated);
                        let concated = realm.heap.new_string(concated);

                        Value::String(concated)
                    }
                    _ => {
                        let an = to_number_or_throw(a, realm)?;
                        let bn = to_number_or_throw(b, realm)?;
                        Value::Number(an + bn)
                    }
                };

                data.top_mut().set_result(*dest, result);
            }
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
                let cond_value = to_boolean(cond_value, &realm.heap);
                if cond_value {
                    next_ndx = dest.0;
                } else {
                    // Just go to the next instruction
                }
            }
            Instr::JmpIfNot { cond, dest } => {
                let cond_value = get_operand(data, *cond)?;
                let cond_value = to_boolean(cond_value, &realm.heap);
                if !cond_value {
                    next_ndx = dest.0;
                } else {
                    // Just go to the next instruction
                }
            }
            Instr::JmpIfPrimitive { arg, dest } => {
                let arg = get_operand(data, *arg)?;
                if arg.is_primitive() {
                    next_ndx = dest.0;
                }
            }
            Instr::JmpIfNotClosure { arg, dest } => {
                let arg = get_operand(data, *arg)?;
                if realm.heap.as_closure(arg).is_none() {
                    next_ndx = dest.0;
                }
            }
            Instr::JmpIfNumberNotInteger { arg, dest } => {
                let arg = get_operand(data, *arg)?;
                let num = arg.expect_num().expect(
                    "compiler bug: argument to JmpIfNumberNotInteger is expected to be number",
                );
                if num.fract() != 0.0 {
                    next_ndx = dest.0;
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
                let value = Value::Bool(!to_boolean(value, &realm.heap));
                data.top_mut().set_result(*dest, value);
            }
            Instr::ToNumber { dest, arg } => {
                let value = get_operand(data, *arg)?;
                let value = to_number_or_throw(value, realm)?;
                data.top_mut().set_result(*dest, Value::Number(value));
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
                continue;
            }
            Instr::Call {
                callee,
                this,
                return_value: return_value_reg,
            } => {
                let callee = get_operand(data, *callee)?;
                let closure = match realm.heap.as_closure(callee) {
                    Some(c) => Rc::clone(&c),
                    None => {
                        let val_s = realm.heap.show_debug(callee);
                        let msg = &format!("can't call non-closure: {:?}", val_s);
                        let exc = make_exception(realm, "TypeError", msg);
                        return Err(RunError::Exception(exc));
                    }
                };

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

                let n_params = bytecode::ARGS_COUNT_MAX as usize;
                arg_vals.truncate(n_params);

                let this = get_operand(data, *this)?;
                // Perform "this substitution", in preparation for a function call.
                // See: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Strict_mode#no_this_substitution
                let this = match (closure.forced_this, closure.is_strict, this) {
                    (Some(value), _, _) => value,
                    (_, false, Value::Null | Value::Undefined) => realm.global_obj,
                    _ => this,
                };
                let this = if closure.is_strict {
                    this
                } else {
                    to_object_or_throw(this, realm)?
                };

                match closure.func {
                    Func::JS(_) => {
                        data.top_mut().set_return_target(*return_value_reg);
                        data.top_mut().set_resume_iid(return_to_iid);

                        data.push_call(closure, this, loader);
                        for i in 0..bytecode::ARGS_COUNT_MAX {
                            let arg = arg_vals.get(i as usize).unwrap_or(&Value::Undefined);
                            data.top_mut().set_arg(bytecode::ArgIndex(i as _), *arg);
                        }

                        // stack is prepared, just continue turning the crank to run
                        // the callee
                    }
                    Func::Native(nf) => {
                        let ret_val = nf(realm, loader, &this, &arg_vals)?;
                        data.top_mut().set_result(*return_value_reg, ret_val);
                        data.top_mut().set_resume_iid(return_to_iid);
                    }
                }
                continue;
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
                if let Value::Null | Value::Undefined = obj {
                    let message = "cannot read property of null or undefined";
                    let exc = make_exception(realm, "TypeError", message);
                    return Err(RunError::Exception(exc));
                }
                // fine; obj is an object or will be treated as such by heap's API
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
                if let Value::Null | Value::Undefined = obj {
                    let msg = "can't delete properties from null or undefined";
                    let exc = make_exception(realm, "TypeError", msg);
                    return Err(RunError::Exception(exc));
                }

                let key = get_operand(data, *key)?;
                let key = value_to_index_or_key(&realm.heap, &key)
                    .ok_or_else(|| error!("invalid object key: {:?}", key))?;

                let was_actually_obj = realm.heap.delete_own(obj, key.to_ref());
                if !was_actually_obj {
                    let message = &format!("not an object: {:?}", obj);
                    let exc = make_exception(realm, "TypeError", message);
                    return Err(RunError::Exception(exc));
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

            Instr::TypeOf { dest, arg: value } => {
                let value = get_operand(data, *value)?;
                let result = js_typeof(&value, realm);
                data.top_mut().set_result(*dest, result);
            }

            Instr::BoolOpAnd(dest, a, b) => {
                let a = get_operand(data, *a)?;
                let b = get_operand(data, *b)?;
                let a_bool = to_boolean(a, &realm.heap);
                let res = if a_bool { b } else { Value::Bool(false) };
                data.top_mut().set_result(*dest, res);
            }
            Instr::BoolOpOr(dest, a, b) => {
                let a = get_operand(data, *a)?;
                let b = get_operand(data, *b)?;
                let a_bool = to_boolean(a, &realm.heap);
                let res = if a_bool { a } else { b };
                data.top_mut().set_result(*dest, res);
            }

            Instr::ClosureNew {
                dest,
                fnid,
                forced_this,
            } => {
                let mut upvalues = Vec::new();
                while let Some(Instr::ClosureAddCapture(cap)) = func.instrs().get(next_ndx as usize)
                {
                    let upv_id = data.top_mut().ensure_in_upvalue(*cap);
                    upvalues.push(upv_id);
                    next_ndx += 1;
                }

                let is_strict = loader.get_function(*fnid).unwrap().is_strict_mode();
                let forced_this = forced_this.map(|reg| get_operand(data, reg)).transpose()?;
                let closure = Rc::new(Closure {
                    func: Func::JS(JSFn(*fnid)),
                    upvalues,
                    forced_this,
                    generator_snapshot: RefCell::new(None),
                    is_strict,
                });

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

                    debug_assert!(root_fn.is_strict_mode());
                    data.push_direct(root_fnid, root_fn, Value::Undefined);
                    data.top_mut().set_is_module_root_fn();
                }

                // This satisfies the borrow checker
                // (`loader.load_import_from_fn` mut-borrows the whole
                // Loader, so the compiler can't prove that the same FnId
                // won't correspond to a different function or even stay
                // valid across the call)
                continue;
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

            Instr::StrCreateEmpty(dest) => {
                let oid = realm.heap.new_string(JSString::empty());
                data.top_mut().set_result(*dest, Value::String(oid));
            }
            Instr::StrAppend(buf_reg, tail) => {
                let value = str_append(data, *buf_reg, *tail, realm, loader)?;
                data.top_mut().set_result(*buf_reg, value);
            }
            Instr::StrFromCodePoint { dest, arg } => {
                let arg = get_operand(data, *arg)?;
                let code_point_f64 = arg
                    .expect_num()
                    .expect("compiler bug: argument to StrFromCodePoint is expected to be number");
                assert!(
                    code_point_f64.fract() == 0.0,
                    "compiler bug: argument to StrFromCodePoint is expected to be integer number"
                );
                let code_point: u16 = (code_point_f64 as usize)
                    .try_into()
                    .expect("compiler bug: argument to StrFromCodePoint does not fit into u16");

                let jss = JSString::new_from_utf16(vec![code_point]);
                let oid = realm.heap.new_string(jss);
                data.top_mut().set_result(*dest, Value::String(oid));
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
                    _ => {
                        panic!("malformed bytecode: GetGlobal argument `name` not a string literal")
                    }
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
                        let msg = JSString::new_from_str(&format!("{} is not defined", key_str));
                        let msg = realm.heap.new_string(msg);

                        let exc_proto = get_builtin(realm, "ReferenceError").unwrap();
                        let exc = Value::Object(realm.heap.new_ordinary_object());
                        realm.heap.set_proto(exc, Some(exc_proto));
                        realm.heap.set_own(exc, heap::IndexOrKey::Key("message"), {
                            let value = Value::String(msg);
                            heap::Property::Enumerable(value)
                        });

                        return Err(RunError::Exception(exc));
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
                return Err(RunError::Exception(exc));
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

/// Perform number coercion (see `to_number`).
///
/// On error, throw a TypeError (as per JS semantics.)
fn to_number_or_throw(arg: Value, realm: &mut Realm) -> RunResult<f64> {
    to_number(arg, realm)?.ok_or_else(|| {
        let message = &format!("cannot convert to a number: {:?}", arg);
        let exc = make_exception(realm, "TypeError", message);
        RunError::Exception(exc)
    })
}

/// Perform number coercion. Converts the given value to a number, in the way
/// that is typically invoked by the unary plus operator (e.g. `+{}`, `+123.0`)
///
/// (The BigInt type is not yet implemented, so neither is BigInt coercion. As a
/// consequence, this method also implements "numeric" coercion, as far as this
/// interpreter is capable.)
///
/// A `Some` is returned for a successful conversion. On failure (e.g. for
/// Symbol), a `None` is returned.
fn to_number(value: Value, realm: &mut Realm) -> RunResult<Option<f64>> {
    match value {
        Value::Null => Ok(Some(0.0)),
        Value::Bool(true) => Ok(Some(1.0)),
        Value::Bool(false) => Ok(Some(0.0)),
        Value::Number(num) => Ok(Some(num)),
        Value::Object(_) => {
            // Converting an object to a number is handled by first converting
            // it to a primitive (via the $ToPrimitive direct from/complex
            // instruction), then passing the result to to_number
            panic!("compiler bug: to_number must be called with a primitive, not an object")
        }
        Value::Undefined => Ok(Some(f64::NAN)),
        Value::Symbol(_) => Ok(None),
        Value::String(_) => {
            let jss = realm.heap.as_str(value).unwrap();
            // hopefully not too costly...
            // TODO cache utf16 -> utf8 conversion?
            let s = jss.to_string();
            let val = match s.as_str().trim() {
                "" => 0.0,
                "+Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                _ => s.parse().unwrap_or(f64::NAN),
            };
            Ok(Some(val))
        }
    }
}

/// Implements JavaScript's implicit conversion to string.
pub(crate) fn to_string_or_throw(
    value: Value,
    realm: &mut Realm,
    loader: &mut loader::Loader,
) -> RunResult<Value> {
    // TODO we always allocate on the string heap. it should be possible to
    // - use interned strings
    // - avoid any allocation if value is Value::String(_)
    use std::borrow::Cow;

    let s: Cow<str> = match value {
        Value::Number(n) => n.to_string().into(),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::String(_) => return Ok(value),
        Value::Object(_) => {
            // TODO Move this behavior to Object.prototype.toString
            // match heap.as_closure(value) {
            //     Some(_) => "<closure>",
            //     None => "<object>",
            // }
            // .into()

            let prim = to_primitive(value, realm, loader)?;
            assert!(prim.is_primitive());
            return to_string_or_throw(prim, realm, loader);
        }
        Value::Symbol(_) => {
            let exc = make_exception(
                realm,
                "TypeError",
                "Cannot convert Symbol value to a string",
            );
            return Err(RunError::Exception(exc));
        }
        Value::Null => "null".into(),
        Value::Undefined => "undefined".into(),
    };

    let jss = JSString::new_from_str(s.as_ref());
    let sid = realm.heap.new_string(jss);
    Ok(Value::String(sid))
}

/// Converts the given value to a boolean (e.g. for use by `if`,
/// or operators `&&` and `||`)
///
/// See:
///  - https://262.ecma-international.org/14.0/#sec-toboolean
///  - https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Boolean#boolean_coercion
pub(crate) fn to_boolean(value: Value, heap: &heap::Heap) -> bool {
    match value {
        Value::Number(0.0) => false,
        Value::Number(n) if n.is_nan() => false,
        Value::Number(_) => true,
        Value::Bool(b) => b,
        Value::String(_) => {
            let string = heap.as_str(value).unwrap();
            !string.view().is_empty()
        }
        Value::Object(_) => true,
        Value::Symbol(_) => true,
        Value::Null | Value::Undefined => false,
    }
}

/// Convert to object.
///
/// Simply returns None, if the conversion fails. For a version with
/// JS-compatible error management, see `to_object_or_throw`.
///
/// For details on object coercion, see: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object#object_coercion
fn to_object(value: Value, heap: &mut heap::Heap) -> Option<Value> {
    match value {
        Value::Null | Value::Undefined => None,
        Value::Object(_) => Some(value),
        Value::Number(_) | Value::Bool(_) | Value::String(_) | Value::Symbol(_) => {
            let oid = heap.wrap_primitive(value);
            Some(Value::Object(oid))
        }
    }
}

/// Convert to object.  On failure, throws TypeError.
///
/// See `to_object`.
fn to_object_or_throw(value: Value, realm: &mut Realm) -> RunResult<Value> {
    to_object(value, &mut realm.heap).ok_or_else(|| {
        let message = &format!("cannot convert to object: {:?}", value);
        let exc = make_exception(realm, "TypeError", message);
        RunError::Exception(exc)
    })
}

/// Convert the given value to a primitive, via PRIMITIVE COERCION.
///
/// Note that this procedure may invoke the value's `valueOf` and `toString`
/// methods, even if the value is already a primitive.  This is because
/// JavaScript allows overriding those methods even for built-in primitive
/// wrappers such as Number and String.  For example:
///
/// ```js
/// (123).valueOf()    // 123
/// Number.prototype.valueOf = function() { return 'HEYOOO' }
/// (123).valueOf()    // 'HEYOOO'
/// ````
fn to_primitive(value: Value, realm: &mut Realm, loader: &mut loader::Loader) -> RunResult<Value> {
    // This whole ordeal needs to happen even if value is not an object, because some
    // funny guy could set a custom function to a wrapper's valueOf:
    #![allow(non_snake_case)]

    if value.is_primitive() {
        return Ok(value);
    }
    debug_assert!(matches!(value, Value::Object(_)));

    // call x.valueOf(); return it if not an object
    if let Some(valueOf) = realm
        .heap
        .get_chained(value, heap::IndexOrKey::Key("valueOf"))
    {
        let valueOf = valueOf.value().unwrap();
        if let Some(valueOf) = realm.heap.as_closure(valueOf) {
            let ret_val = call_closure(realm, loader, Rc::clone(&valueOf), value, &[])?;
            if ret_val.is_primitive() {
                return Ok(ret_val);
            }
        }
    }

    if let Some(toString) = realm
        .heap
        .get_chained(value, heap::IndexOrKey::Key("toString"))
    {
        let toString = toString.value().unwrap();
        if let Some(toString) = realm.heap.as_closure(toString) {
            let ret_val = call_closure(realm, loader, Rc::clone(&toString), value, &[])?;

            if ret_val.is_primitive() {
                return Ok(ret_val);
            }
        }
    }

    let message = format!("Cannot convert to primitive: {:?}", value);
    let exc = make_exception(realm, "TypeError", &message);
    return Err(RunError::Exception(exc));
}

fn call_closure(
    realm: &mut Realm,
    loader: &mut crate::Loader,
    closure: Rc<Closure>,
    this: Value,
    args: &[Value; 0],
) -> RunResult<Value> {
    match closure.func {
        Func::Native(nf) => nf(realm, loader, &this, args),
        Func::JS(_) => {
            let ret_val = Interpreter::new_call(realm, loader, closure, this, args)
                .run()?
                .expect_finished()
                .ret_val;
            Ok(ret_val)
        }
    }
}

fn make_exception(realm: &mut Realm, constructor_name: &str, message: &str) -> Value {
    let message = JSString::new_from_str(message);
    let message = realm.heap.new_string(message);

    let exc_cons = Value::Object(get_builtin(realm, constructor_name).unwrap());
    let exc_proto = realm
        .heap
        .get_own(exc_cons, heap::IndexOrKey::Key("prototype"))
        .unwrap()
        .value()
        .unwrap()
        .expect_obj()
        .unwrap();
    let exc = Value::Object(realm.heap.new_ordinary_object());

    realm.heap.set_proto(exc, Some(exc_proto));
    realm.heap.set_own(exc, heap::IndexOrKey::Key("message"), {
        let value = Value::String(message);
        heap::Property::Enumerable(value)
    });
    realm.heap.set_own(
        exc,
        heap::IndexOrKey::Key("constructor"),
        heap::Property::NonEnumerable(exc_cons),
    );

    exc
}

fn get_builtin(realm: &Realm, builtin_name: &str) -> RunResult<heap::ObjectID> {
    realm
        .heap
        .get_own(realm.global_obj, heap::IndexOrKey::Key(builtin_name))
        .map(|p| p.value().unwrap())
        .ok_or_else(|| error!("missing required builtin: {}", builtin_name))?
        .expect_obj()
        .map_err(|_| RunError::Internal(error!("bug: ReferenceError is not an object?!")))
}

pub type RunResult<T> = std::result::Result<T, RunError>;

/// The error type only used internally by `run_frame` and `run_internal`.
#[derive(Debug)]
pub enum RunError {
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
impl From<InterpreterError> for RunError {
    fn from(intrp_err: InterpreterError) -> Self {
        intrp_err.error.into()
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
    data.set_return_value(ret_val);
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
        if let Value::Null | Value::Undefined = obj {
            let message = "cannot traverse keys of null or undefined";
            let exc = make_exception(realm, "TypeError", message);
            return Err(RunError::Exception(exc));
        }

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
fn handle_exception(
    exc: Value,
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

            let property = realm
                .heap
                .get_own(value, heap::IndexOrKey::Key(&key))
                .unwrap();
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
    a: VReg,
    b: VReg,
    realm: &mut Realm,
    loader: &mut loader::Loader,
) -> RunResult<Value> {
    // TODO Make this at least *decently* efficient!
    let a = get_operand_string(data, realm, a)?;
    let b = get_operand(data, b)?;

    let mut buf = a.view().to_vec();
    let tail = to_string_or_throw(b, realm, loader)?;
    let tail = realm.heap.as_str(tail).unwrap();
    buf.extend_from_slice(tail.view());
    let jss: JSString = JSString::new_from_utf16(buf);
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
