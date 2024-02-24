use std::cell::Ref;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::{
    bytecode::{self, FnId, Instr, VReg, IID},
    common::{self, Result},
    error,
    heap::{self, IndexOrKey, Object},
    loader,
    stack::{self, UpvalueId},
};

mod builtins;

// Public versions of the private `Result` and `Error` above
pub type InterpreterResult<'a, T> = std::result::Result<T, InterpreterError<'a>>;
pub struct InterpreterError<'a> {
    pub error: crate::common::Error,

    // This struct owns the failed interpreter, but we explicitly disallow doing
    // anything else with it other than examining it (read-only, via &)
    #[cfg_attr(not(feature = "debugger"), allow(dead_code))]
    interpreter: Interpreter<'a>,
}
impl<'a> InterpreterError<'a> {
    #[cfg(feature = "debugger")]
    pub fn probe<'s>(&'s mut self) -> debugger::Probe<'s, 'a> {
        debugger::Probe::attach(&mut self.interpreter)
    }
}
impl<'a> std::fmt::Debug for InterpreterError<'a> {
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
                    other => {
                        Err(error!("expected a {}, got {:?}", stringify!($variant), other).into())
                    }
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

type NativeFunction = fn(&mut Realm, &Value, &[Value]) -> Result<Value>;

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

pub struct Realm {
    heap: heap::Heap,
    module_objs: HashMap<bytecode::ModuleId, heap::ObjectId>,
    global_obj: heap::ObjectId,
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

    /// Instruction breakpoints
    #[cfg(feature = "debugger")]
    dbg: debugger::InterpreterState,
}

pub struct FinishedData {
    pub sink: Vec<Option<bytecode::Literal>>,
}

pub enum Exit<'a> {
    Finished(FinishedData),
    Suspended {
        interpreter: Interpreter<'a>,
        cause: SuspendCause,
    },
}
impl<'a> Exit<'a> {
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
    pub fn new(realm: &'a mut Realm, loader: &'a mut loader::Loader, fnid: bytecode::FnId) -> Self {
        // Initialize the stack with a single frame, corresponding to a call to fnid with no
        // parameters
        let data = init_stack(loader, realm, fnid);
        Interpreter {
            realm,
            data,
            loader,
            #[cfg(feature = "debugger")]
            dbg: debugger::InterpreterState::new(),
        }
    }

    // From this section, it seems that `Interpreter` can only be created
    // (`Interpreter::new`) and used+destroyed (`Interpreter::run`). So why not
    // just have a function without an intermediate struct? Because gives the
    // user an opportunity to hook the `debugger::Probe` API.

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
    pub fn run(mut self) -> InterpreterResult<'a, Exit<'a>> {
        assert!(!self.data.is_empty());

        let res = run_frame(
            &mut self.data,
            self.realm,
            self.loader,
            0,
            #[cfg(feature = "debugger")]
            &mut self.dbg,
        );

        // We currently translate IntrpResult into Result<Exit<'a>> just to
        // avoid boiling the ocean, but long-term Result<Exit> should be
        // deleted.

        match res {
            Ok(_) => {
                // return value discarded

                #[cfg(test)]
                let sink: Vec<_> = self
                    .data
                    .sink()
                    .iter()
                    .map(|value| try_value_to_literal(*value, &self.realm.heap))
                    .collect();

                #[cfg(not(test))]
                let sink = Vec::new();

                Ok(Exit::Finished(FinishedData { sink }))
            }
            Err(RunError::Exception(exc)) => Err(InterpreterError {
                error: error!("unhandled exception: {:?}", exc),
                interpreter: self,
            }),
            Err(RunError::Internal(common_err)) => Err(InterpreterError {
                error: common_err,
                interpreter: self,
            }),
            Err(RunError::Suspended(cause)) => Ok(Exit::Suspended {
                interpreter: self,
                cause,
            }),
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

/// Run bytecode.  This is the core of the interpreter.
///
/// Before/after each call, `data` must hold/holds the interpreter's state in
/// enough detail that this function can resume execution even across separate
/// calls.
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
///  - the interpreter has finished its work (gone through the program without
///    errors)
///  - an unhandled exception has been thrown
///  - the debugger has suspended execution (e.g. a breakpoint has been
///    reached).
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
///  - an as-of-yet-unhandled exception has been thrown (may be handled by one
///    of the parent frames or bubble up to the user unhandled)
///  - the debugger has suspended execution (e.g. a breakpoint has been
///    reached).
///
/// Upon starting, `run_internal` calls itself again with a 1-higher
/// `stack_level` so as to restore the mapping between JS and native stacks.
/// Then it resumes JS execution.
fn run_frame<'a>(
    data: &'a mut stack::InterpreterData,
    realm: &'a mut Realm,
    loader: &'a mut loader::Loader,
    // TODO Define a StackLevel newtype
    stack_level: usize,

    #[cfg(feature = "debugger")] dbg: &'a mut debugger::InterpreterState,
) -> RunResult<Value> {
    #[allow(clippy::len_zero)]
    {
        assert!(data.len() > 0);
        assert!(stack_level < data.len());
    }

    let t = crate::tracing::section("run_frame");

    let mut cur_exc = None;

    t.log("stack_level", &format!("{}/{}", stack_level, data.len()));

    loop {
        let err = if stack_level < data.len() - 1 {
            t.log("loop", "climbing stack");

            // We're not the top of the stack, which means we're suspended waiting for a called function.
            // Keep running code until it's our turn.
            let res = run_frame(
                data,
                realm,
                loader,
                stack_level + 1,
                #[cfg(feature = "debugger")]
                dbg,
            );

            match res {
                Ok(ret_val) => {
                    t.log("returned", "after climbing the stack");
                    if let Some(rv_reg) = data.top_mut().take_return_target() {
                        data.top_mut().set_result(rv_reg, ret_val);
                    }
                    continue;
                }
                Err(err) => err,
            }
        } else {
            let loop_header = format!(
                "regular execution {:?}{:?}",
                data.top().header().fnid,
                data.top().header().iid,
            );
            t.log("loop", &loop_header);

            let res = run_regular(
                data,
                realm,
                loader,
                stack_level,
                cur_exc.unwrap_or(Value::Undefined),
                #[cfg(feature = "debugger")]
                dbg,
            );

            match res {
                Ok(ret_val) => {
                    t.log("returned", "regularly");
                    data.pop();
                    return Ok(ret_val);
                }
                Err(err) => err,
            }
        };

        match err {
            RunError::Exception(exc) => {
                t.log("returned", &format!("exception thrown: {:?}", exc));

                if let Some(handler_iid) = data.top_mut().pop_exc_handler() {
                    t.log(
                        "exception handling",
                        &format!("internally @ {:?}", handler_iid),
                    );

                    // Cut the stack short to here, because we left it up in the 'else' clause
                    data.truncate(stack_level + 1);

                    cur_exc = Some(exc);
                    data.top_mut().set_resume_iid(handler_iid);
                    // loop back; next call to run_regular will handle the
                    // exception
                } else {
                    // no handler, let the caller do something about it
                    t.log("exception handling", "unwinding");

                    data.pop();
                    return Err(err);
                }
            }

            RunError::Internal(_) => {
                t.log("returned", "due to internal error");
                // data.pop();
                return Err(err);
            }

            // Keep the stack around in this case
            RunError::Suspended(ref cause) => {
                t.log("returned", &format!("suspending due to {:?}", cause));
                return Err(err);
            }
        }
    }
}

type RunResult<T> = std::result::Result<T, RunError>;
/// The error type only used internally by `run_frame` and `run_internal`.
#[derive(Debug)]
enum RunError {
    Exception(Value),
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

fn run_regular<'a>(
    data: &'a mut stack::InterpreterData,
    realm: &'a mut Realm,
    loader: &'a mut loader::Loader,
    // TODO Define a StackLevel newtype
    stack_level: usize,
    cur_exc: Value,
    #[cfg(feature = "debugger")] dbg: &'a mut debugger::InterpreterState,
) -> RunResult<Value> {
    'reborrow: loop {
        // Reload the func and instr IID and locate the function. Keep it around
        // until we need to suspend and resume our burrow (e.g. across a call to
        // run_frame)
        let fnid = data.top().header().fnid;
        let mut iid = data.top().header().iid;
        let func = loader.get_function(fnid).unwrap();

        loop {
            // Each (native) call to `run_regular` corresponds to part of a call to a JavaScript
            // function. As such, it stays within the same JavaScript stack frame.
            assert_eq!(data.len() - 1, stack_level);
            assert_eq!(data.top().header().fnid, fnid);

            if iid.0 as usize == func.instrs().len() {
                // Bytecode "finished" => Implicitly return undefined
                return Ok(Value::Undefined);
            }
            let instr = func.instrs()[iid.0 as usize];
            let mut next_ndx = iid.0 + 1;

            match &instr {
                Instr::LoadConst(dest, bytecode::ConstIndex(const_ndx)) => {
                    let literal = func.consts()[*const_ndx as usize].clone();
                    let value = literal_to_value(literal, &mut realm.heap);
                    data.top_mut().set_result(*dest, value);
                }

                Instr::OpAdd(dest, a, b) => match get_operand(data, *a)? {
                    Value::Number(_) => with_numbers(data, *dest, *a, *b, |x, y| x + y)?,
                    Value::Object(_) => {
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
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => {
                            return Err(
                                error!(" invalid if condition (not boolean): {:?}", other).into()
                            )
                        }
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
                    let value = Value::Bool(!to_boolean(value, realm));
                    data.top_mut().set_result(*dest, value);
                }
                Instr::Jmp(IID(dest_ndx)) => {
                    next_ndx = *dest_ndx;
                }
                Instr::Return(value) => {
                    let return_value = get_operand(data, *value)?;
                    return Ok(return_value);
                }
                Instr::Call {
                    callee,
                    this,
                    return_value: return_value_reg,
                } => {
                    let oid = get_operand(data, *callee)?.expect_obj()?;
                    let ho_ref = realm
                        .heap
                        .get(oid)
                        .ok_or_else(|| error!("invalid function (object is not callable)"))?
                        .borrow();
                    let closure: &Closure = ho_ref
                        .as_closure()
                        .ok_or_else(|| error!("can't call non-closure"))?;

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
                            let callee_func = loader.get_function(closure.fnid).unwrap();
                            let n_params = bytecode::ARGS_COUNT_MAX as usize;
                            arg_vals.truncate(n_params);
                            arg_vals.resize(n_params, Value::Undefined);
                            assert_eq!(arg_vals.len(), n_params);

                            let this = closure
                                .forced_this
                                .map(Ok)
                                .unwrap_or_else(|| get_operand(data, *this))?;
                            // "this" substitution: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Strict_mode#no_this_substitution
                            // If I understand this correctly, we don't need to box anything right
                            // now.  We just pass the value, and the callee will box it when
                            // needed.
                            let this = match (callee_func.is_strict_mode(), this) {
                                (false, Value::Null | Value::Undefined) => {
                                    Value::Object(realm.global_obj)
                                }
                                (_, other) => other,
                            };

                            // Only used if we re-enter run_internal via suspend+resume
                            data.top_mut().set_return_target(*return_value_reg);

                            // Commit the IID that we want to return to before
                            // calling run_frame to properly resume, either
                            // "from the top" (run_frame returns) or "from the
                            // bottom" (stack rebuilt by run_frame)
                            data.top_mut().set_resume_iid(return_to_iid);

                            data.push(stack::CallMeta {
                                fnid: closure.fnid,
                                n_regs: callee_func.n_regs() as u32,
                                captures: &closure.upvalues,
                                this,
                            });
                            for (i, arg) in arg_vals.into_iter().enumerate() {
                                data.top_mut().set_arg(bytecode::ArgIndex(i as _), arg);
                            }

                            // Important: run ho_ref destructor early ('unref' the
                            // RefCell via ReadGuard::drop), so that we can
                            // re-borrow for run_frame
                            drop(ho_ref);
                            let ret_val = run_frame(
                                data,
                                realm,
                                loader,
                                stack_level + 1,
                                #[cfg(feature = "debugger")]
                                dbg,
                            )?;

                            assert_eq!(stack_level, data.len() - 1);
                            if let Some(rv_reg) = data.top_mut().take_return_target() {
                                data.top_mut().set_result(rv_reg, ret_val);
                            }

                            continue 'reborrow;
                        }
                        Closure::Native(nf) => {
                            let nf = *nf;
                            drop(ho_ref);

                            let this = get_operand(data, *this)?;
                            let ret_val = nf(realm, &this, &arg_vals)?;
                            data.top_mut().set_result(*return_value_reg, ret_val);
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
                    let value = data.top().get_arg(*arg_ndx).unwrap_or(Value::Undefined);
                    data.top_mut().set_result(*dest, value);
                }

                Instr::ObjCreateEmpty(dest) => {
                    let oid = realm.heap.new_ordinary_object(HashMap::new());
                    data.top_mut().set_result(*dest, Value::Object(oid));
                }
                Instr::ObjSet { obj, key, value } => {
                    let mut obj = get_operand_object(data, realm, *obj)?;
                    let key = get_operand(data, *key)?;
                    let key = value_to_index_or_key(&realm.heap, &key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = get_operand(data, *value)?;

                    obj.set_own_element_or_property(key.to_ref(), value);
                }
                Instr::ObjGet { dest, obj, key } => {
                    let obj = get_operand_object(data, realm, *obj)?;
                    let key = get_operand(data, *key)?;
                    let key = value_to_index_or_key(&realm.heap, &key);

                    let value = match key {
                        Some(ik @ heap::IndexOrKeyOwned::Index(_)) => {
                            obj.get_own_element_or_property(ik.to_ref())
                        }
                        Some(heap::IndexOrKeyOwned::Key(key)) => {
                            realm.heap.get_property_chained(&obj, &key)
                        }
                        None => None,
                    }
                    .unwrap_or(Value::Undefined);

                    data.top_mut().set_result(*dest, value);
                }
                Instr::ObjGetKeys { dest, obj } => {
                    // TODO Something more efficient?
                    let obj = get_operand_object(data, realm, *obj)?;
                    let keys = obj
                        .own_properties()
                        .into_iter()
                        .map(|name| Value::Object(realm.heap.new_string(name)))
                        .collect();

                    let keys_oid = realm.heap.new_array(keys);
                    data.top_mut().set_result(*dest, Value::Object(keys_oid));
                }
                Instr::ObjDelete { dest, obj, key } => {
                    // TODO Adjust return value: true for all cases except when the property is an
                    // own non-configurable property, in which case false is returned in non-strict
                    // mode. (Source: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/delete)
                    {
                        let mut obj = get_operand_object(data, realm, *obj)?;
                        let key = get_operand(data, *key)?;
                        let key = value_to_index_or_key(&realm.heap, &key)
                            .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                        obj.delete_own_element_or_property(key.to_ref());
                    }

                    data.top_mut().set_result(*dest, Value::Bool(true));
                }

                Instr::ArrayPush { arr, value } => {
                    let value = get_operand(data, *value)?;
                    let mut arr = get_operand_object(data, realm, *arr)?
                        .into_heap_cell()
                        .ok_or_else(|| error!("not an array!"))?
                        .borrow_mut();

                    let was_array = arr.array_push(value);
                    assert!(was_array);
                }
                Instr::ArrayNth { dest, arr, index } => {
                    let value = {
                        let arr = get_operand_object(data, realm, *arr)?
                            .into_heap_cell()
                            .ok_or_else(|| error!("not an array!"))?
                            .borrow();
                        let elements = arr.array_elements().unwrap();

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
                    let len = get_operand_object(data, realm, *arr)?
                        .into_heap_cell()
                        .ok_or_else(|| error!("not an array!"))?
                        .borrow()
                        .array_elements()
                        .unwrap()
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
                    let a_bool = to_boolean(a, realm);
                    let res = if a_bool { b } else { Value::Bool(false) };
                    data.top_mut().set_result(*dest, res);
                }
                Instr::BoolOpOr(dest, a, b) => {
                    let a = get_operand(data, *a)?;
                    let b = get_operand(data, *b)?;
                    let a_bool = to_boolean(a, realm);
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
                    let module_id = data.top().header().fnid.0;
                    let fnid = bytecode::FnId(module_id, *fnid);
                    let closure = Closure::JS(JSClosure {
                        fnid,
                        upvalues,
                        forced_this,
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

                    let bytecode::FnId(import_site, _) = fnid;
                    let module_path = get_operand_string(data, realm, *module_path)?.to_string();

                    let root_fnid = loader
                        .load_import(&module_path, Some(import_site))
                        .with_context(error!("while trying to import '{}'", module_path))?;

                    // Commit before reborrowing
                    data.top_mut().set_resume_iid(bytecode::IID(iid.0 + 1));

                    if let Some(module_oid) = realm.module_objs.get(&root_fnid.0) {
                        data.top_mut().set_result(*dest, Value::Object(*module_oid));
                    } else {
                        let root_fn = loader.get_function(root_fnid).unwrap();

                        data.push(stack::CallMeta {
                            fnid: root_fnid,
                            n_regs: root_fn.n_regs() as u32,
                            captures: &[],
                            this: Value::Undefined,
                        });

                        // We don't care about return value: don't set a return
                        // value target reg, discard it here
                        run_frame(
                            data,
                            realm,
                            loader,
                            stack_level + 1,
                            #[cfg(feature = "debugger")]
                            dbg,
                        )?;
                    }

                    // This satisfies the borrow checker (`loader.load_import`
                    // mut-borrows the whole Loader, so the compiler can't prove
                    // that the same FnId won't correspond to a different
                    // function or even stay valid across the call)
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
                    let oid = realm.heap.new_string(String::new());
                    data.top_mut().set_result(*dest, Value::Object(oid));
                }
                Instr::StrAppend(buf_reg, tail) => {
                    let value = str_append(data, realm, *buf_reg, *tail)?;
                    data.top_mut().set_result(*buf_reg, value);
                }

                Instr::GetGlobalThis(dest) => {
                    let value = Value::Object(realm.global_obj);
                    data.top_mut().set_result(*dest, value);
                }
                Instr::GetGlobal {
                    dest,
                    name: bytecode::ConstIndex(name_cndx),
                } => {
                    let literal = func.consts()[*name_cndx as usize].clone();
                    let key = {
                        let str_literal = match &literal {
                            bytecode::Literal::String(s) => s,
                            bytecode::Literal::JsWord(jsw) => jsw.as_ref(),
                            _ => panic!(
                            "malformed bytecode: GetGlobal argument `name` not a string literal"
                        ),
                        };
                        heap::IndexOrKey::Key(str_literal)
                    };

                    let global_this = realm.heap.get(realm.global_obj).unwrap().borrow();
                    let lookup_result = global_this.get_own_element_or_property(key);

                    match lookup_result {
                        None | Some(Value::Undefined) => {
                            let exc_proto = global_this
                                .get_own_element_or_property(IndexOrKey::Key("ReferenceError"))
                                .expect("missing required builtin: ReferenceError")
                                .expect_obj()
                                .expect("bug: ReferenceError is not an object?!");
                            // sadly, the borrowck needs some hand-holding here
                            drop(global_this);
                            let exc_oid = realm.heap.new_ordinary_object(HashMap::new());
                            {
                                let mut exc = realm.heap.get(exc_oid).unwrap().borrow_mut();
                                exc.set_proto(Some(exc_proto));
                            }

                            // Duplicate with the Instr::Throw implementation. Not sure how to improve.
                            let exc = Value::Object(exc_oid);
                            throw_exc(
                                exc,
                                #[cfg(feature = "debugger")]
                                dbg,
                            )?;
                        }
                        Some(value) => {
                            data.top_mut().set_result(*dest, value);
                        }
                    }
                }

                Instr::Breakpoint => {
                    // We must set the 'return-to' IID now, or the Interpreter will be back here on resume,
                    // in an infinite loop.
                    data.top_mut().set_resume_iid(bytecode::IID(next_ndx));
                    return Err(RunError::Suspended(SuspendCause::Breakpoint));
                }

                Instr::GetCurrentException(dest) => {
                    data.top_mut().set_result(*dest, cur_exc);
                }
                Instr::Throw(exc) => {
                    let exc = get_operand(data, *exc)?;

                    throw_exc(
                        exc,
                        #[cfg(feature = "debugger")]
                        dbg,
                    )?;
                }
                Instr::PopExcHandler => {
                    data.top_mut()
                        .pop_exc_handler()
                        .ok_or_else(|| error!("compiler bug: no exception handler to pop!"))?;
                }
                Instr::PushExcHandler(target_iid) => data.top_mut().push_exc_handler(*target_iid),
            }

            // Gotta increase IID even if we're about to suspend, or we'll be back here on resume
            iid.0 = next_ndx;

            // TODO Checking for breakpoints here in this hot loop is going to be *very* slow!
            #[cfg(feature = "debugger")]
            {
                let giid = bytecode::GlobalIID(fnid, iid);
                let out_of_fuel = dbg.consume_1_fuel();
                if out_of_fuel || dbg.is_breakpoint_at(&giid) {
                    // Important: Commit a pre-increased IID before suspending,
                    // so that:
                    //  1. debugging tools can see the correct IID
                    //  2. cycling suspend/resume (which is what "NEXT" is in
                    //     the debugger) won't be an infinite loop.
                    data.top_mut().set_resume_iid(iid);
                    return Err(RunError::Suspended(SuspendCause::Breakpoint));
                }
            }
        }
    }
}

fn throw_exc(
    exc: Value,
    #[cfg(feature = "debugger")] dbg: &mut debugger::InterpreterState,
) -> RunResult<()> {
    #[cfg(feature = "debugger")]
    if dbg.should_break_on_throw()
        || (dbg.should_break_on_unhandled_throw() && !data.any_exception_handler())
    {
        // Save the IID so that execution can resume correctly afterwards
        data.top_mut().set_resume_iid(iid);
        return Err(RunError::Suspended(SuspendCause::Exception(exc)));
    }

    Err(RunError::Exception(exc))
}

// TODO(cleanup) inline this function? It now adds nothing
fn get_operand(data: &stack::InterpreterData, vreg: bytecode::VReg) -> RunResult<Value> {
    data.top()
        .get_result(vreg)
        .ok_or_else(|| error!("variable read before initialization").into())
}

fn get_operand_object<'r>(
    data: &stack::InterpreterData,
    realm: &'r Realm,
    vreg: bytecode::VReg,
) -> RunResult<heap::ValueObjectRef<'r>> {
    let value = get_operand(data, vreg)?;
    as_object_ref(value, &realm.heap)
        .ok_or_else(|| error!("could not use as object: {:?}", vreg).into())
}

fn get_operand_string<'r>(
    data: &stack::InterpreterData,
    realm: &'r Realm,
    vreg: bytecode::VReg,
) -> RunResult<Ref<'r, str>> {
    let ho = get_operand_object(data, realm, vreg)?
        .into_heap_cell()
        .ok_or_else(|| error!("not a heap object"))?
        .borrow();
    Ok(Ref::filter_map(ho, |ho| ho.as_str()).unwrap())
}

fn js_typeof(value: &Value, realm: &mut Realm) -> Value {
    let ty_s = match value {
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::Object(oid) => match realm.heap.get(*oid).unwrap().borrow().type_of() {
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

    literal_to_value(bytecode::Literal::String(ty_s.to_string()), &mut realm.heap)
}

fn as_object_ref(value: Value, heap: &heap::Heap) -> Option<heap::ValueObjectRef> {
    match value {
        Value::Object(oid) => heap.get(oid).map(heap::ValueObjectRef::Heap),
        Value::Number(num) => Some(heap::ValueObjectRef::Number(num, heap)),
        Value::Bool(bool) => Some(heap::ValueObjectRef::Bool(bool, heap)),
        _ => None,
    }
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

    let obj = match get_operand_object(data, realm, obj) {
        Ok(obj) => obj,
        Err(_) => return Ok(false),
    };

    Ok(realm.heap.is_instance_of(&obj, sup_oid))
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
        (Value::Null, Value::Null) => ValueOrdering::Equal,
        (Value::Undefined, Value::Undefined) => ValueOrdering::Equal,
        #[rustfmt::skip]
        (Value::Object(a_oid), Value::Object(b_oid)) => {
            let a_obj = realm.heap.get(*a_oid).map(|x| x.borrow());
            let b_obj = realm.heap.get(*b_oid).map(|x| x.borrow());

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

    data.top_mut().set_result(dest, Value::Bool(test(ordering)));
    Ok(())
}

/// Converts the given value to a boolean (e.g. for use by `if`,
/// or operators `&&` and `||`)
///
/// See: https://262.ecma-international.org/14.0/#sec-toboolean
fn to_boolean(value: Value, realm: &Realm) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(bool_val) => bool_val,
        Value::Number(num) => num != 0.0,
        Value::Object(oid) => realm.heap.get(oid).unwrap().borrow().to_boolean(),
        Value::Undefined => false,
        Value::SelfFunction => true,
        Value::Internal(_) => {
            panic!("bytecode compiler bug: internal value should be unreachable")
        }
    }
}

fn value_to_string(value: Value, heap: &heap::Heap) -> String {
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

fn str_append(
    data: &stack::InterpreterData,
    realm: &mut Realm,
    a: VReg,
    b: VReg,
) -> RunResult<Value> {
    // TODO Make this at least *decently* efficient!
    let b = get_operand(data, b)?;

    let mut buf = get_operand_string(data, realm, a)?.to_owned();
    let tail = value_to_string(b, &realm.heap);
    buf.push_str(&tail);
    let value = literal_to_value(bytecode::Literal::String(buf), &mut realm.heap);
    Ok(value)
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

#[cfg(test)]
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
    use std::cell::Ref;
    use std::collections::HashMap;

    use crate::{bytecode, InterpreterValue};
    use crate::{heap, GlobalIID};

    use super::Interpreter;

    pub use super::SuspendCause;
    pub use crate::loader::BreakRangeID;
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

            bytecode::GlobalIID(frame.header().fnid, frame.header().iid)
        }

        pub fn sink(&self) -> &[InterpreterValue] {
            self.interpreter.data.sink()
        }

        pub fn set_fuel(&mut self, fuel: Fuel) {
            self.interpreter.dbg.fuel = fuel;
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
            if self.interpreter.dbg.source_bkpts.contains_key(&brange_id) {
                return Err(BreakpointError::AlreadyThere);
            }

            let giid = self.giid_of_break_range(brange_id)?;
            let bkpt = InstrBreakpoint {
                src_bkpt: Some(brange_id),
            };
            self.add_instr_bkpt(giid, bkpt)?;

            let prev = self
                .interpreter
                .dbg
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
            if !self.interpreter.dbg.source_bkpts.contains_key(&brange_id) {
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
            self.interpreter
                .dbg
                .source_bkpts
                .iter()
                .map(|(k, v)| (*k, v))
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
            let new_insert = self.interpreter.dbg.instr_bkpts.insert(giid, bkpt);
            match new_insert {
                None => Ok(()),
                Some(_) => Err(BreakpointError::AlreadyThere),
            }
        }

        pub fn clear_instr_breakpoint(&mut self, giid: bytecode::GlobalIID) -> bool {
            let ibkpt = self.interpreter.dbg.instr_bkpts.remove(&giid);
            if let Some(ibkpt) = &ibkpt {
                if let Some(brid) = &ibkpt.src_bkpt {
                    self.interpreter.dbg.source_bkpts.remove(brid).unwrap();
                }
            }

            ibkpt.is_some()
        }

        pub fn instr_breakpoints(&self) -> impl '_ + ExactSizeIterator<Item = bytecode::GlobalIID> {
            self.interpreter.dbg.instr_bkpts.keys().copied()
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
            let fnid = frame.header().fnid;
            let iid = frame.header().iid;
            bytecode::GlobalIID(fnid, iid)
        }

        pub fn get_object(&self, obj_id: heap::ObjectId) -> Option<Ref<heap::HeapObject>> {
            self.interpreter
                .realm
                .heap
                .get(obj_id)
                .map(|hocell| hocell.borrow())
        }

        pub fn break_on_throw(&mut self) -> bool {
            self.interpreter.dbg.break_on_throw
        }
        pub fn set_break_on_throw(&mut self, value: bool) {
            self.interpreter.dbg.break_on_throw = value;
        }

        pub fn break_on_unhandled_throw(&mut self) -> bool {
            self.interpreter.dbg.break_on_unhandled_throw
        }
        pub fn set_break_on_unhandled_throw(&mut self, value: bool) {
            self.interpreter.dbg.break_on_unhandled_throw = value;
        }
    }

    /// Extra stuff that is debugging-specific and added to the intepreter.
    pub struct InterpreterState {
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

        /// If true, the interpreter will suspend right after an unhandled
        /// exception is thrown, but before the stack is unwound in the attempt
        /// to handle it.   This is useful to look at the interpreter's state
        /// before it gets completely undone after the handling attempt.
        break_on_unhandled_throw: bool,

        /// If true, the interpreter will suspend upon reaching any `Instr::Throw`.
        break_on_throw: bool,
    }

    pub enum Fuel {
        Limited(usize),
        Unlimited,
    }

    // There is nothing here for now. The mere existence of an entry in Interpreter.source_bktps is
    // enough (but some addtional parameters might have to be includede here later)
    pub struct SourceBreakpoint;

    pub struct InstrBreakpoint {
        src_bkpt: Option<BreakRangeID>,
    }

    impl InterpreterState {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            InterpreterState {
                instr_bkpts: HashMap::new(),
                source_bkpts: HashMap::new(),
                fuel: Fuel::Unlimited,
                break_on_unhandled_throw: false,
                break_on_throw: false,
            }
        }

        /// Consume 1 unit of fuel.  Returns true iff the tank is empty.
        pub fn consume_1_fuel(&mut self) -> bool {
            match &mut self.fuel {
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
            }
        }

        pub fn is_breakpoint_at(&self, giid: &bytecode::GlobalIID) -> bool {
            self.instr_bkpts.contains_key(giid)
        }

        pub(crate) fn should_break_on_unhandled_throw(&self) -> bool {
            self.break_on_unhandled_throw
        }

        pub(crate) fn should_break_on_throw(&self) -> bool {
            self.break_on_throw
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
    fn test_script_global_fn_nonstrict() {
        let output = quick_run(
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
        let output = quick_run(
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

    #[test]
    fn test_temporal_dead_zone_none() {
        let _ = quick_run(
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
        let _ = quick_run(
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
        let output = quick_run(
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
        let output = quick_run(
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
        let output = quick_run("sink(123); sink(void 123);");
        assert_eq!(
            &output.sink,
            &[Some(Literal::Number(123.0)), Some(Literal::Undefined)]
        );
    }

    #[test]
    fn test_eval_completion_value() {
        let output = quick_run("sink(eval('11'))");
        assert_eq!(&output.sink, &[Some(Literal::Number(11.0))]);
    }
}
