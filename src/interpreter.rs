use lazy_static::lazy_static;
use swc_atoms::JsWord;

use std::{
    borrow::Cow,
    cell::{Ref, RefCell, RefMut},
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex},
};

use crate::{
    bytecode::{self, FnId, GlobalIID, Instr, VReg, IID},
    bytecode_compiler,
    common::Result,
    error,
    heap::{self, ObjectHeap},
    // jit::{self, InterpreterStep},
    stack,
    util::Mask,
};

pub use crate::common::Error;

/// A value that can be input, output, or processed by the program at runtime.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Number(f64),
    String(Cow<'static, str>),
    Bool(bool),
    Object(heap::ObjectId),
    Null,
    Undefined,
    SelfFunction,

    Internal(usize),
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(Cow::Owned(value))
    }
}

impl From<&'static str> for Value {
    fn from(value: &'static str) -> Self {
        Value::String(Cow::Borrowed(value))
    }
}

impl From<bytecode::Literal> for Value {
    fn from(bc_value: bytecode::Literal) -> Self {
        match bc_value {
            bytecode::Literal::Number(nu) => Value::Number(nu),
            bytecode::Literal::String(st) => Value::String(st.into()),
            bytecode::Literal::Bool(bo) => Value::Bool(bo),
            bytecode::Literal::Null => Value::Null,
            bytecode::Literal::Undefined => Value::Undefined,
            bytecode::Literal::SelfFunction => todo!(),
        }
    }
}

impl Value {
    pub(crate) fn expect_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(val) => Ok(*val),
            _ => Err(error!("expected a boolean")),
        }
    }

    pub(crate) fn expect_num(&self) -> Result<f64> {
        match self {
            Value::Number(val) => Ok(*val),
            _ => Err(error!("expected a number")),
        }
    }

    pub(crate) fn into_str(mut self) -> Result<String> {
        match self {
            Value::String(s) => Ok(s.into_owned()),
            _ => Err(error!("expected a string")),
        }
    }

    pub(crate) fn expect_str(&self) -> Result<&str> {
        match self {
            Value::String(s) => Ok(s.as_ref()),
            _ => Err(error!("expected a string")),
        }
    }
}

#[derive(Clone)]
pub enum Closure {
    Native(&'static NativeFunction),
    JS(JSClosure),
}

type NativeFunction = dyn Fn(&mut Interpreter, &[Value]) -> Result<Value>;

#[derive(Clone, PartialEq, Eq)]
pub struct JSClosure {
    fnid: FnId,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueId>,
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

pub struct Interpreter<'a> {
    iid: bytecode::IID,
    data: stack::InterpreterData,
    heap: heap::ObjectHeap,

    #[cfg(enable_jit)]
    jitting: Option<Jitting>,

    modules: HashMap<bytecode::ModuleId, heap::ObjectId>,
    global_obj: heap::ObjectId,

    codebase: &'a bytecode::Codebase,
    sink: Vec<Value>,
    // traces: HashMap<String, (jit::Trace, jit::NativeThunk)>,
    opts: Options,
}

#[cfg(enable_jit)]
struct Jitting {
    fnid: FnId,
    iid: IID,
    trace_id: String,
    builder: jit::TraceBuilder,
}

impl<'a> Interpreter<'a> {
    pub fn new(codebase: &'a bytecode::Codebase) -> Self {
        let mut heap = heap::ObjectHeap::new();
        let global_obj = init_builtins(&mut heap);
        Interpreter {
            iid: bytecode::IID(0),
            data: stack::InterpreterData::new(),
            heap,
            #[cfg(enable_jit)]
            jitting: None,
            modules: HashMap::new(),
            global_obj,
            codebase,
            sink: Vec::new(),
            opts: Default::default(),
        }
    }

    pub fn options_mut(&mut self) -> &mut Options {
        &mut self.opts
    }

    pub fn take_sink(&mut self) -> Vec<Value> {
        let mut swap_area = Vec::new();
        std::mem::swap(&mut swap_area, &mut self.sink);
        swap_area
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

    pub fn run_module(&mut self, module_id: bytecode::ModuleId) -> Result<()> {
        let fnid = self
            .codebase
            .get_module_root_fn(module_id)
            .ok_or_else(|| error!("no such module: {module_id:?}"))?;
        let root_fn = self.codebase.get_function(fnid).unwrap();
        let n_instrs = root_fn.instrs().len().try_into().unwrap();

        assert!(self.data.len() == 0);
        self.data.push(stack::CallMeta {
            fnid,
            n_instrs,
            n_captured_upvalues: 0,
            n_args: 0,
            return_value_reg: None,
            call_iid: None,
        });

        self.run_until_done()?;

        #[cfg(enable_jit)]
        if let Some(jitting) = intrp.jitting {
            if let Some(trace) = jitting.builder.build() {
                #[cfg(test)]
                {
                    eprintln!(" ---- compiled trace");
                    trace.dump();
                }

                let native_thunk = trace.compile();
                let prev = self.traces.insert(jitting.trace_id, (trace, native_thunk));
                assert!(prev.is_none());
            }
        }

        Ok(())
    }

    fn run_until_done(&mut self) -> Result<()> {
        while self.data.len() != 0 {
            let fnid = self.data.fnid();
            let func = self.get_function(fnid);

            assert!(
                self.iid.0 as usize <= func.instrs().len(),
                "can't proceed to instruction at index {} (func has {})",
                self.iid.0,
                func.instrs().len()
            );
            if self.iid.0 as usize == func.instrs().len() {
                if let Some(call_iid) = self.exit_function(None) {
                    self.iid.0 = call_iid.0 + 1;
                    continue;
                } else {
                    return Ok(());
                }
            }

            let instr = &func.instrs()[self.iid.0 as usize];

            self.print_indent();
            eprint!("{:4}: {:?}", self.iid.0, instr);
            if let Instr::LoadConst(_, const_ndx) = instr {
                let lit = &func.consts()[const_ndx.0 as usize];
                eprint!(" = ({:?})", lit);
            }
            eprintln!();

            let mut next_ndx = self.iid.0 + 1;
            drop(func);

            #[cfg(enable_jit)]
            if let Some(tanch) = func.get_trace_anchor(self.iid) {
                match self.flags.jit_mode {
                    JitMode::Compile => {
                        if self.jitting.is_none() {
                            let builder = jit::TraceBuilder::start(
                                func.instrs().len(),
                                jit::CloseMode::FunctionExit,
                            );
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

            match instr {
                Instr::LoadConst(dest, bytecode::ConstIndex(const_ndx)) => {
                    let value = func.consts()[*const_ndx as usize].clone();
                    self.data.set_result(*dest, value.into());
                }

                Instr::ArithAdd(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x + y),
                Instr::ArithSub(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x - y),
                Instr::ArithMul(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x * y),
                Instr::ArithDiv(dest, a, b) => self.with_numbers(*dest, *a, *b, |x, y| x / y),

                Instr::PushToSink(operand) => {
                    let value = self.get_operand(*operand);
                    self.sink.push(value);
                }
                Instr::CmpGE(dest, a, b) => {
                    self.compare(*dest, *a, *b, |x, y| x >= y, |x, y| x >= y, |x, y| x >= y)
                }
                Instr::CmpGT(dest, a, b) => {
                    self.compare(*dest, *a, *b, |x, y| x > y, |x, y| x > y, |x, y| x > y)
                }
                Instr::CmpLT(dest, a, b) => {
                    self.compare(*dest, *a, *b, |x, y| x < y, |x, y| x < y, |x, y| x < y)
                }
                Instr::CmpLE(dest, a, b) => {
                    self.compare(*dest, *a, *b, |x, y| x <= y, |x, y| x <= y, |x, y| x <= y)
                }
                Instr::CmpEQ(dest, a, b) => {
                    self.compare(*dest, *a, *b, |x, y| x == y, |x, y| x == y, |x, y| x == y)
                }
                Instr::CmpNE(dest, a, b) => {
                    self.compare(*dest, *a, *b, |x, y| x != y, |x, y| x != y, |x, y| x != y)
                }

                Instr::JmpIf { cond, dest } => {
                    let cond_value = self.get_operand(*cond);
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => panic!("invalid if condition (not boolean): {:?}", other),
                    }
                }

                Instr::Copy { dst, src } => {
                    self.data.set_result(*dst, self.get_operand(*src));
                }
                Instr::LoadCapture(dest, cap_ndx) => {
                    self.data.capture_to_var(*cap_ndx, *dest);
                }

                Instr::Nop => {}
                Instr::BoolNot(var) => {
                    let value = match self.get_operand(*var) {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        Value::Number(num) => Value::Bool(num == 0.0),
                        Value::String(str) => Value::Bool(str.is_empty()),
                        Value::Object(_) => Value::Bool(false),
                        Value::Null => Value::Bool(true),
                        Value::Undefined => Value::Bool(true),
                        Value::SelfFunction => Value::Bool(false),
                        Value::Internal(_) => {
                            panic!("bytecode compiler bug: internal value should be unreachable")
                        }
                    };
                    self.data.set_result(*var, value);
                }
                Instr::Jmp(IID(dest_ndx)) => {
                    next_ndx = *dest_ndx;
                }
                Instr::Return(value) => {
                    if let Some(call_iid) = self.exit_function(Some(*value)) {
                        self.iid.0 = call_iid.0 + 1;
                        continue;
                    } else {
                        return Ok(());
                    }
                }
                Instr::Call {
                    callee,
                    return_value,
                } => {
                    let oid = self
                        .get_operand_object(*callee)
                        .expect("invalid function (not an object)");
                    let closure: &Closure = self
                        .heap
                        .get_closure(oid)
                        .expect("invalid function (object is not callable)");

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
                    next_ndx += TryInto::<u16>::try_into(arg_vals.len()).unwrap();

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

                            let callee_func = self.codebase.get_function(closure.fnid).unwrap();
                            let n_args = callee_func.n_params().0;
                            arg_vals.truncate(n_args as usize);
                            let call_meta = stack::CallMeta {
                                fnid: closure.fnid,
                                n_instrs: callee_func.instrs().len().try_into().unwrap(),
                                n_captured_upvalues: closure.upvalues.len().try_into().unwrap(),
                                n_args,
                                return_value_reg: Some(*return_value),
                                call_iid: Some(self.iid),
                            };

                            self.print_indent();
                            eprintln!(
                                "-- fn #{} [{} captures]",
                                call_meta.fnid.0, call_meta.n_captured_upvalues
                            );

                            self.data.push(call_meta);
                            for (capndx, capture) in closure.upvalues.iter().enumerate() {
                                self.data.set_capture(capndx, *capture);
                            }
                            for (i, arg) in arg_vals.into_iter().enumerate() {
                                self.data.set_arg(i, arg);
                            }
                            self.iid = IID(0u16);

                            // Important: we don't execute the tail part of the instruction's
                            // execution. This makes it easier
                            // to keep a consistent value in `func` and other
                            // variables, and avoid subtle bugs
                            continue;
                        }
                        Closure::Native(nf) => {
                            let ret_val = (*nf)(self, &arg_vals)?;
                            self.data.set_result(*return_value, ret_val);
                        }
                    }
                }
                Instr::CallArg(_) => {
                    unreachable!("interpreter bug: CallArg goes through another path!")
                }

                Instr::LoadArg(dest, bytecode::ArgIndex(arg_ndx)) => {
                    // TODO extra copy?
                    // TODO usize is a bit too wide
                    let value = self.data.get_arg(*arg_ndx as usize).clone();
                    self.data.set_result(*dest, value);
                }

                Instr::ObjCreateEmpty(dest) => {
                    let oid = self.heap.new_object();
                    self.data.set_result(*dest, Value::Object(oid));
                }
                Instr::ObjSet { obj, key, value } => {
                    let oid = self
                        .get_operand_object(*obj)
                        // TODO(big feat) use TypeError exception here
                        .unwrap_or_else(|| panic!("ObjSet: cannot set properties of: {:?}", obj));
                    let key = self.get_operand(*key);
                    let key = heap::ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self.get_operand(*value);

                    self.heap.set_property(oid, key, value);
                }
                Instr::ObjGet { dest, obj, key } => {
                    let oid = self
                        .get_operand_object(*obj)
                        // TODO(big feat) use TypeError exception here
                        .unwrap_or_else(|| panic!("ObjGet: cannot read properties of: {:?}", obj));
                    let key = self.get_operand(*key);
                    let key = heap::ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self
                        .heap
                        .get_property(oid, &key)
                        .unwrap_or(Value::Undefined);
                    self.data.set_result(*dest, value.clone());
                }
                Instr::ObjGetKeys { dest, obj } => {
                    let oid = self.get_operand_object(*obj).unwrap_or_else(|| {
                        panic!("ObjGetKeys: can't get keys of non-object: {:?}", obj)
                    });
                    let keys = self.heap.get_keys_as_array(oid);
                    self.data.set_result(*dest, Value::Object(keys));
                }
                Instr::ObjDelete { dest, obj, key } => {
                    // TODO Adjust return value: true for all cases except when the property is an
                    // own non-configurable property, in which case false is returned in non-strict
                    // mode. (Source: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/delete)
                    let oid = self.get_operand_object(*obj).unwrap_or_else(|| {
                        panic!("ObjGetKeys: can't delete property of non-object: {:?}", obj)
                    });
                    let key = self.get_operand(*key);
                    let key = heap::ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;

                    self.heap.delete_property(oid, &key);
                    self.data.set_result(*dest, Value::Bool(true));
                }

                Instr::ArrayPush { arr, value } => {
                    todo!("ArrayPush does nothing for now")
                }
                Instr::ArrayNth { dest, arr, index } => {
                    let oid = self.get_operand_object(*arr).unwrap_or_else(|| {
                        panic!("ArrayNth: can't get element of non-array: {:?}", arr)
                    });

                    let num = self.get_operand(*index).expect_num()?;
                    let num_trunc = num.trunc();
                    let value = if num_trunc == num {
                        let ndx = num_trunc as usize;
                        self.heap.array_nth(oid, ndx).unwrap_or(Value::Undefined)
                    } else {
                        Value::Undefined
                    };

                    self.data.set_result(*dest, value);
                }
                Instr::ArraySetNth { .. } => todo!("ArraySetNth"),
                Instr::ArrayLen { dest, arr } => {
                    let oid = self.get_operand_object(*arr).unwrap_or_else(|| {
                        panic!("ArrayLen: can't get length of non-object/array: {:?}", arr)
                    });
                    let len: usize = self.heap.array_len(oid);
                    self.data.set_result(*dest, Value::Number(len as f64));
                }

                Instr::TypeOf { dest, value } => {
                    let value = self.get_operand(*value);
                    self.data.set_result(*dest, self.js_typeof(&value));
                }

                Instr::BoolOpAnd(dest, a, b) => {
                    let a: bool = self.get_operand(*a).expect_bool()?;
                    let b: bool = self.get_operand(*b).expect_bool()?;
                    let res = a && b;
                    self.data.set_result(*dest, Value::Bool(res));
                }
                Instr::BoolOpOr(dest, a, b) => {
                    let a: bool = self.get_operand(*a).expect_bool()?;
                    let b: bool = self.get_operand(*b).expect_bool()?;
                    let res = a || b;
                    self.data.set_result(*dest, Value::Bool(res));
                }

                Instr::ClosureNew(dest, fnid) => {
                    let mut upvalues = Vec::new();
                    while let Instr::ClosureAddCapture(cap) = func.instrs()[next_ndx as usize] {
                        let upv_id = self.data.ensure_in_upvalue(cap);
                        #[cfg(test)]
                        {
                            self.print_indent();
                            eprintln!("        upvalue: {:?} -> {:?}", cap, upv_id);
                        }

                        upvalues.push(upv_id);

                        next_ndx += 1;
                    }

                    let closure = Closure::JS(JSClosure {
                        fnid: *fnid,
                        upvalues,
                    });

                    let oid = self.heap.new_function(closure);
                    self.data.set_result(*dest, Value::Object(oid));
                }
                // This is always handled in the code for ClosureNew
                Instr::ClosureAddCapture(_) => {
                    unreachable!(
                        "interpreter bug: ClosureAddCapture should be handled with ClosureNew"
                    )
                }

                Instr::UnaryMinus(var) => {
                    let arg: f64 = self.get_operand(*var).expect_num()?;
                    self.data.set_result(*var, Value::Number(-arg));
                }

                Instr::GetModule(dest, module_id) => {
                    if let Some(module_oid) = self.modules.get(module_id) {
                        self.data.set_result(*dest, Value::Object(*module_oid));
                    } else {
                        // TODO Refactor with other implementations of Call?
                        let root_fnid = self
                            .codebase
                            .get_module_root_fn(*module_id)
                            .expect("no such module ID");
                        let root_fn = self.codebase.get_function(root_fnid).unwrap();
                        let n_instrs = root_fn.instrs().len().try_into().unwrap();

                        let call_meta = stack::CallMeta {
                            fnid: root_fnid,
                            n_instrs,
                            n_captured_upvalues: 0,
                            n_args: 0,
                            return_value_reg: Some(*dest),
                            call_iid: Some(self.iid),
                        };

                        self.print_indent();
                        eprintln!("-- loading module m#{}", module_id.0);

                        self.data.push(call_meta);
                        self.iid = IID(0u16);
                        continue;
                    }
                }

                Instr::LoadNull(dest) => {
                    self.data.set_result(*dest, Value::Null);
                }
                Instr::LoadUndefined(dest) => {
                    self.data.set_result(*dest, Value::Undefined);
                }
                Instr::ArithInc(dest, src) => {
                    let val = self
                        .get_operand(*src)
                        .expect_num()
                        .expect("bytecode bug: ArithInc on non-number");
                    self.data.set_result(*dest, Value::Number(val + 1.0));
                }
                Instr::ArithDec(dest, src) => {
                    let val = self
                        .get_operand(*src)
                        .expect_num()
                        .expect("bytecode bug: ArithDec on non-number");
                    self.data.set_result(*dest, Value::Number(val - 1.0));
                }
                Instr::IsInstanceOf(_dest, _obj, _sup) => todo!("IsInstanceOf"),
                Instr::NewIterator { dest, obj } => todo!(),
                Instr::IteratorGetCurrent { dest, iter } => todo!(),
                Instr::IteratorAdvance { iter } => todo!(),
                Instr::JmpIfIteratorFinished { iter, dest } => todo!(),

                Instr::Throw(_) => todo!(),

                Instr::StrCreateEmpty(dest) => self
                    .data
                    .set_result(*dest, Value::String(String::new().into())),
                Instr::StrAppend(buf_reg, tail) => {
                    // TODO Make this *decently* efficient!
                    let buf = self.get_operand(*buf_reg);
                    let mut buf: String = buf.into_str()?;

                    let tail = self.get_operand(*tail);
                    let tail: &str = tail.expect_str()?;

                    buf.push_str(tail);
                    self.data.set_result(*buf_reg, Value::String(buf.into()));
                }

                Instr::GetGlobal { dest, key } => {
                    let key = self.get_operand(*key);
                    let key = heap::ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;

                    let value = self
                        .heap
                        .get_property(self.global_obj, &key)
                        .unwrap_or(Value::Undefined);
                    self.data.set_result(*dest, value);
                }
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

            self.iid.0 = next_ndx;
        }

        Ok(())
    }

    fn get_function(&mut self, fnid: FnId) -> &'a bytecode::Function {
        // TODO make it so that func is "gotten" and unwrapped only when strictly necessary
        self.codebase.get_function(fnid).unwrap()
    }

    fn exit_function(&mut self, callee_retval_reg: Option<VReg>) -> Option<IID> {
        let return_value = callee_retval_reg
            .map(|vreg| self.get_operand(vreg))
            .unwrap_or(Value::Undefined);
        let caller_retval_reg: Option<VReg> = self.data.caller_retval_reg();
        let call_iid: Option<IID> = self.data.call_iid();
        self.data.pop();

        if let Some(vreg) = caller_retval_reg {
            self.data.set_result(vreg, return_value);
        }

        #[cfg(enable_jit)]
        if let Some(jitting) = &mut self.jitting {
            jitting.builder.exit_function(callee_retval_reg);
        }

        call_iid
    }

    fn with_numbers<F>(&mut self, dest: VReg, a: VReg, b: VReg, op: F)
    where
        F: FnOnce(f64, f64) -> f64,
    {
        let a = self.get_operand(a).expect_num().unwrap();
        let b = self.get_operand(b).expect_num().unwrap();
        self.data.set_result(dest, Value::Number(op(a, b)));
    }

    fn compare<FB, FN, FS>(
        &mut self,
        dest: VReg,
        a: VReg,
        b: VReg,
        op_bool: FB,
        op_num: FN,
        op_str: FS,
    ) where
        FB: FnOnce(bool, bool) -> bool,
        FN: FnOnce(f64, f64) -> bool,
        FS: FnOnce(&str, &str) -> bool,
    {
        let a = self.get_operand(a);
        let b = self.get_operand(b);

        let result = match (&a, &b) {
            (Value::Bool(a), Value::Bool(b)) => op_bool(*a, *b),
            (Value::Number(a), Value::Number(b)) => op_num(*a, *b),
            (Value::String(a), Value::String(b)) => op_str(a, b),
            _ => false,
        };

        self.data.set_result(dest, Value::Bool(result));
    }

    fn get_operand_object(&self, operand: VReg) -> Option<heap::ObjectId> {
        let obj = self.get_operand(operand);
        if let Value::Object(oid) = obj {
            Some(oid)
        } else {
            None
        }
    }

    fn get_operand_function(&self, operand: VReg) -> &Closure {
        let oid = self
            .get_operand_object(operand)
            .expect("invalid function (not an object)");
        self.heap
            .get_closure(oid)
            .expect("invalid function (object is not callable)")
    }

    fn print_indent(&self) {
        for _ in 0..(self.data.len() - 1) {
            eprint!("    ");
        }
    }

    // TODO(cleanup) inline this function? It now adds nothing
    fn get_operand(&self, vreg: bytecode::VReg) -> Value {
        let value = self.data.get_result(vreg).clone();
        //  TODO(cleanup) Move to a global logger. This is just for debugging!
        #[cfg(test)]
        {
            self.print_indent();
            eprintln!("        {:?} = {:?}", vreg, value);
        }
        value
    }

    fn js_typeof(&self, value: &Value) -> Value {
        let ty_s = match value {
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Bool(_) => "boolean",
            Value::Object(oid) => match self.heap.get_closure(*oid) {
                Some(_) => "function",
                None => "object",
            },
            // TODO(cleanup) This is actually an error in our type system.  null is really a value
            // of the 'object' type
            Value::Null => "object",
            Value::Undefined => "undefined",
            Value::SelfFunction => "function",
            Value::Internal(_) => panic!("internal value has no typeof!"),
        };

        ty_s.to_string().into()
    }
}

fn init_builtins(heap: &mut heap::ObjectHeap) -> heap::ObjectId {
    #![allow(non_snake_case)]

    let global = heap.new_object();

    {
        let array = heap.new_object();
        let Array_isArray = heap.new_function(Closure::Native(&nf_Array_isArray));
        heap.set_property(
            array,
            "isArray".to_string().into(),
            Value::Object(Array_isArray),
        );
        heap.set_property(global, "Array".to_string().into(), Value::Object(array));
    }

    let RegExp = heap.new_function(Closure::Native(&nf_RegExp));
    heap.set_property(global, "RegExp".to_string().into(), Value::Object(RegExp));

    // builtins.insert("Boolean".into(), NativeFnId::BooleanNew as u32);
    // builtins.insert("Object".into(), NativeFnId::ObjectNew as u32);
    // builtins.insert("Number".into(), NativeFnId::NumberNew as u32);
    // builtins.insert("String".into(), NativeFnId::StringNew as u32);
    // builtins.insert("Array".into(), NativeFnId::ArrayNew as u32);
    // builtins.insert("RegExp".into(), NativeFnId::RegExpNew as u32);
    // builtins.insert("parseInt".into(), NativeFnId::ParseInt as u32);
    // builtins.insert("SyntaxError".into(), NativeFnId::SyntaxErrorNew as u32);
    // builtins.insert("TypeError".into(), NativeFnId::TypeErrorNew as u32);
    // builtins.insert("Math_floor".into(), NativeFnId::MathFloor as u32);

    // TODO(big feat) pls impl all Node.js API, ok? thxbye

    global
}

fn nf_Array_isArray(arg: &mut Interpreter, args: &[Value]) -> Result<Value> {
    todo!("nf_Array_isArray")
}

fn nf_RegExp(intrp: &mut Interpreter, args: &[Value]) -> Result<Value> {
    let oid = intrp.heap.new_object();
    Ok(Value::Object(oid))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Output {
        sink: Vec<Value>,
    }

    use crate::fs::MockLoader;

    fn quick_run(code: &str) -> Result<Output> {
        let module_id = bytecode::ModuleId(123);

        let mut mock_loader = Box::new(MockLoader::new());
        mock_loader.add_module("the_script.js".to_owned(), module_id, code.to_owned());

        let codebase = {
            let bcparams = bytecode_compiler::BuilderParams {
                loader: mock_loader,
            };
            let mut builder = bcparams.to_builder();
            builder
                .compile_file("the_script.js".to_string())
                .expect("compile error");
            builder.build()
        };

        codebase.dump();

        let mut vm = Interpreter::new(&codebase);
        vm.run_module(module_id)?;
        Ok(Output {
            sink: vm.take_sink(),
        })
    }

    #[test]
    fn test_simple_call() {
        let output = quick_run("/* Here is some simple code: */ sink(1 + 4 + 99); ").unwrap();
        assert_eq!(&[Value::Number(104.0)], &output.sink[..]);
    }

    #[test]
    fn test_multiple_calls() {
        let output =
            quick_run("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ").unwrap();
        assert_eq!(
            &[Value::Number(12. * 5.), Value::Number(99. - 15.)],
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
        )
        .unwrap();

        let val: Value = "b".to_string().into();
        assert_eq!(&[val], &output.sink[..]);
    }

    #[test]
    fn test_simple_fn() {
        let output = quick_run(
            "
            function foo(a, b) { return a + b; }
            sink(foo(1, 2));
            ",
        )
        .unwrap();

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        assert_eq!(&[Value::Number(3.0)], &output.sink[..]);
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
        )
        .unwrap();

        assert_eq!(&[Value::Number(9.0 * 8.0)], &output.sink[..]);
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
        )
        .unwrap();

        assert_eq!(&[Value::Number(10.0)], &output.sink[..]);
    }

    fn try_casting_bool(code: &str, expected_value: bool) {
        let output = quick_run(code).unwrap();
        assert_eq!(&[Value::Bool(expected_value)], &output.sink[..]);
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
        )
        .unwrap();

        assert_eq!(
            &[
                Value::Number(2.0),
                Value::Number(4.0),
                Value::Number(6.0),
                Value::Number(1.0)
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
            sink(obj.aFunction)
            ",
        )
        .unwrap();

        assert_eq!(5, output.sink.len());
        assert_eq!(&Value::String("asdlol123".into()), &output.sink[0]);
        assert_eq!(&Value::Number(1239423.4518923), &output.sink[1]);
        assert_eq!(&Value::Number(123.0), &output.sink[2]);
        assert_eq!(&Value::Number(899.0), &output.sink[3]);
        assert!(matches!(&output.sink[4], Value::Object(_)));
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
        )
        .unwrap();

        assert_eq!(
            &output.sink[..],
            &[
                Value::String("undefined".into()).into(),
                Value::String("undefined".into()).into(),
                Value::String("object".into()).into(),
                Value::String("object".into()).into(),
                Value::String("object".into()).into(),
                Value::String("boolean".into()).into(),
                Value::String("boolean".into()).into(),
                Value::String("number".into()).into(),
                Value::String("number".into()).into(),
                Value::String("number".into()).into(),
                Value::String("number".into()).into(),
                Value::String("number".into()).into(),
                Value::String("string".into()).into(),
                Value::String("string".into()).into(),
                Value::String("function".into()).into(),
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
        )
        .unwrap();

        assert_eq!(4, output.sink.len());
        assert_eq!(&Value::Number(123.0), &output.sink[0]);
        assert_eq!(&Value::Number(4.0), &output.sink[1]);
        assert_eq!(&Value::Number(123.0), &output.sink[2]);
        assert_eq!(&Value::Number(999.0), &output.sink[3]);
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
        )
        .unwrap();

        assert_eq!(
            &output.sink[..],
            &[
                Value::Number(99.0),
                Value::Number(32.0),
                Value::Number(12304.0),
                Value::Number(0.0),
                Value::String("another name".into()),
                Value::String("another name yet".into()),
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
        )
        .unwrap();

        let mut sink: Vec<_> = output
            .sink
            .into_iter()
            .map(|value| match value {
                Value::String(s) => s,
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
        )
        .unwrap();

        assert_eq!(&output.sink, &[Value::Bool(true), Value::Bool(false)]);
    }
}
