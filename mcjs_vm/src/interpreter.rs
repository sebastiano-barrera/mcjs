use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::{cell::Ref, rc::Rc};

use crate::error_item;
use crate::{
    bytecode::{self, FnId, Instr, VReg, IID},
    common::{self, Result},
    define_flag, error,
    heap::{self, IndexOrKey, Object},
    loader,
};

mod builtins;
pub mod stack;

#[cfg(feature = "debugger")]
pub use stack::SlotDebug;
use stack::UpvalueId;

// Public versions of the private `Result` and `Error` above
pub type InterpreterResult<T> = std::result::Result<T, Box<InterpreterError>>;
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
    Object(heap::ObjectId),
    Null,
    Undefined,
    SelfFunction,

    Symbol(&'static str),
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

/// A *reference* to a closure.
///
/// It can be cloned, and the resulting value will "point" to the same closure as the
/// first one. (These semantics are also in `Value`, and `Closure` inherits them from it).
#[derive(Clone)]
pub enum Closure {
    Native(NativeFunction),
    JS(Rc<JSClosure>),
}

type NativeFunction = fn(&mut Realm, &Value, &[Value]) -> Result<Value>;

#[derive(Clone)]
pub struct JSClosure {
    fnid: FnId,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueId>,
    forced_this: Option<Value>,
    generator_snapshot: RefCell<Option<stack::FrameSnapshot>>,
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
    // Key is the root fnid of each module
    module_objs: HashMap<bytecode::FnId, Value>,
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
    pub fn new(realm: &'a mut Realm, loader: &'a mut loader::Loader, fnid: bytecode::FnId) -> Self {
        // Initialize the stack with a single frame, corresponding to a call to fnid with no
        // parameters, then put it into an Interpreter
        let mut data = init_stack(loader, realm, fnid);
        data.set_default_this(Value::Object(realm.global_obj));
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
                Err(Box::new(InterpreterError {
                    error,
                    #[cfg(feature = "debugger")]
                    intrp_state: self.data,
                }))
            }
            Err(RunError::Internal(common_err)) => Err(Box::new(InterpreterError {
                error: common_err,
                #[cfg(feature = "debugger")]
                intrp_state: self.data,
            })),

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
    fnid: FnId,
) -> stack::InterpreterData {
    let mut data = stack::InterpreterData::new();
    let root_fn = loader.get_function(fnid).unwrap();
    let global_this = Value::Object(realm.global_obj);
    data.push_direct(fnid, root_fn, global_this);
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
                    let cond_value = to_boolean(cond_value, realm);
                    if cond_value {
                        next_ndx = dest.0;
                    } else {
                        // Just go to the next instruction
                    }
                }
                Instr::JmpIfNot { cond, dest } => {
                    let cond_value = get_operand(data, *cond)?;
                    let cond_value = to_boolean(cond_value, realm);
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
                    let value = Value::Bool(!to_boolean(value, realm));
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
                    let obj = get_operand_object(data, realm, *obj)?;
                    let key = get_operand(data, *key)?;
                    let key = value_to_index_or_key(&realm.heap, &key);

                    let value = key
                        .and_then(|key| realm.heap.get_property_chained(&obj, key.to_ref()))
                        .map(|p| p.value)
                        .unwrap_or(Value::Undefined);

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
                Instr::ObjDelete { dest, obj, key } => {
                    // TODO Adjust return value: true for all cases except when the property is an
                    // own non-configurable property, in which case false is returned in non-strict
                    // mode. (Source: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/delete)
                    {
                        let mut obj = get_operand_object(data, realm, *obj)?;
                        let key = get_operand(data, *key)?;
                        let key = value_to_index_or_key(&realm.heap, &key)
                            .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                        obj.delete_own(key.to_ref());
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
                    let lookup_result = global_this.get_own(key);

                    match lookup_result {
                        None
                        | Some(heap::Property {
                            value: Value::Undefined,
                            ..
                        }) => {
                            let exc_proto = global_this
                                .get_own(IndexOrKey::Key("ReferenceError"))
                                .map(|p| p.value)
                                .expect("missing required builtin: ReferenceError")
                                .expect_obj()
                                .expect("bug: ReferenceError is not an object?!");
                            // sadly, the borrowck needs some hand-holding here
                            drop(global_this);
                            let exc_oid = realm.heap.new_ordinary_object();
                            {
                                let mut exc = realm.heap.get(exc_oid).unwrap().borrow_mut();
                                exc.set_proto(Some(exc_proto));
                            }

                            // Duplicate with the Instr::Throw implementation. Not sure how to
                            // improve.
                            let exc = Value::Object(exc_oid);
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
                            data.top_mut().set_result(*dest, prop.value);
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
    let mut obj = get_operand_object(data, realm, *obj)?;
    let key = get_operand(data, *key)?;
    let key = value_to_index_or_key(&realm.heap, &key)
        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
    let value = get_operand(data, *value)?;
    obj.set_own(
        key.to_ref(),
        heap::Property {
            value,
            is_enumerable,
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
    let obj = get_operand_object(data, realm, *obj)?;
    let mut keys = Vec::new();

    obj.own_properties(only_enumerable.into(), &mut keys);
    if include_inherited.into() {
        realm
            .heap
            .list_properties_prototypes(&obj, only_enumerable.into(), &mut keys);
    }

    let keys = keys
        .into_iter()
        .map(|name| Value::Object(realm.heap.new_string(name)))
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
        Value::Symbol(_) => "symbol",
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
        Value::Symbol(_) => true,
    }
}

/// Implements JavaScript's implicit conversion to string.
fn value_to_string(value: Value, heap: &heap::Heap) -> Result<String> {
    match value {
        Value::Number(num) => Ok(num.to_string()),
        Value::Bool(true) => Ok("true".into()),
        Value::Bool(false) => Ok("false".into()),
        Value::Object(oid) => Ok(heap.get(oid).unwrap().borrow().js_to_string()),
        Value::Null => Ok("null".into()),
        Value::Undefined => Ok("undefined".into()),
        Value::SelfFunction => Ok("<function>".into()),
        Value::Internal(_) => unreachable!(),
        Value::Symbol(_) => Err(error!("Cannot convert Symbol value into a String")),
    }
}

/// Write into `wrt` a human-readable description of the value.
///
/// If the value is a string, it will appear in the output.
///
/// If the value is an object, a listing of its properties will appear in its output.
fn show_value<W: std::io::Write>(out: &mut W, value: Value, realm: &mut Realm) {
    show_value_ex(out, value, realm, &ShowValueParams::default())
}

fn show_value_ex<W: std::io::Write>(
    out: &mut W,
    value: Value,
    realm: &mut Realm,
    params: &ShowValueParams,
) {
    if let Value::Object(obj_id) = value {
        let obj = realm.heap.get(obj_id).unwrap().borrow();
        if let Some(s) = obj.as_str() {
            writeln!(out, "{:?}", s).unwrap();
        } else {
            let mut props = Vec::new();
            obj.own_properties(false, &mut props);
            writeln!(out, "{:?} [{} properties]", obj_id, props.len()).unwrap();

            drop(obj);
            if params.max_object_depth > 0 {
                for key in props {
                    for _ in 0..params.indent {
                        write!(out, "    ").unwrap();
                    }

                    let property = {
                        let obj = realm.heap.get(obj_id).unwrap().borrow();
                        obj.get_own(IndexOrKey::Key(&key)).unwrap()
                    };
                    write!(out, "  - {:?} = ", key).unwrap();
                    show_value_ex(
                        out,
                        property.value,
                        realm,
                        &ShowValueParams {
                            indent: params.indent + 1,
                            max_object_depth: params.max_object_depth - 1,
                        },
                    );
                }
            }
        }
    } else {
        writeln!(out, "{:?}", value).unwrap();
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
    let b = get_operand(data, b)?;

    let mut buf = get_operand_string(data, realm, a)?.to_owned();
    let tail = value_to_string(b, &realm.heap)?;
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
        bytecode::Literal::Symbol(sym) => Value::Symbol(sym),
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

#[cfg(any(test, feature = "debugger"))]
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
        Value::Symbol(sym) => Some(bytecode::Literal::Symbol(sym)),
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
        Value::Symbol(sym) => Some(heap::IndexOrKeyOwned::Symbol(sym)),
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
    use std::collections::HashMap;

    use super::stack;
    use crate::{bytecode, loader};
    use crate::{heap, GlobalIID};

    pub use super::SuspendCause;
    pub use crate::loader::BreakRangeID;
    pub use heap::{IndexOrKey, Object, ObjectId};

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
    use super::*;
    use crate::bytecode::Literal;

    fn quick_run(code: &str) -> FinishedData {
        let res = std::panic::catch_unwind(|| {
            let mut prereq = prepare_vm(code);
            let vm = prereq.make_vm();
            match vm.run() {
                Ok(exit) => exit.expect_finished(),
                Err(err_box) => {
                    let error = err_box.error.with_loader(&prereq.loader);
                    panic!("{:?}", error);
                }
            }
        });

        if let Err(err) = &res {
            println!("quick_run: error: {:?}", err);
        }
        res.unwrap()
    }

    fn prepare_vm(code: &str) -> VMPrereq {
        let mut loader = loader::Loader::new_cwd();
        let chunk_fnid = loader
            .load_script_anon(code.to_string())
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

    #[test]
    fn test_array_properties() {
        let output = quick_run(
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
        let output = quick_run(
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
        let output = quick_run(
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
        let output = quick_run(
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
        let output = quick_run(
            r#"
                function getSomethingElse() { sink(2); return 456; }
                function getTruthy() { sink(1); return 123; }
                sink(getTruthy() || getSomethingElse());
            "#,
        );

        assert_eq!(
            &output.sink,
            &[
                Some(Literal::Number(1.0)),
                Some(Literal::Number(123.0)),
            ]
        );
    }

    mod debugging {
        use std::path::PathBuf;

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
                if giid.0 == bytecode::FnId(2) {
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
