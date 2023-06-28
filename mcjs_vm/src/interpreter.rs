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
    stack_access,
    util::Mask,
};

pub use crate::common::Error;

/// A value that can be input, output, or processed by the program at runtime.
///
/// Design notes: Value is `Copy` in an effort to make it as dumb as possible (easy to
/// copy/move/delete/etc.), as otherwise it becomes really hard to keep memory safety in the
/// interpreter's stack (which is shared with JIT-compiled code, which is inherently unsafe, so we
/// have to make some compromises).
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

impl Value {
    pub(crate) fn expect_bool(&self) -> Result<bool> {
        match self {
            Value::Bool(val) => Ok(*val),
            other => Err(error!("expected an boolean, got {:?}", other)),
        }
    }

    pub(crate) fn expect_num(&self) -> Result<f64> {
        match self {
            Value::Number(val) => Ok(*val),
            other => Err(error!("expected a number, got {:?}", other)),
        }
    }

    pub(crate) fn expect_obj(&self) -> Result<heap::ObjectId> {
        match self {
            Value::Object(oid) => Ok(*oid),
            other => Err(error!("expected an object, got {:?}", other)),
        }
    }
}

#[derive(Clone)]
pub enum Closure {
    Native(&'static NativeFunction),
    JS(JSClosure),
}

type NativeFunction = dyn Fn(&mut Interpreter, &Value, &[Value]) -> Result<Value>;

#[derive(Clone, PartialEq)]
pub struct JSClosure {
    fnid: FnId,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueId>,
    forced_this: Option<Value>,
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

    pub fn take_sink(&mut self) -> Vec<Option<bytecode::Literal>> {
        let mut values = Vec::new();
        std::mem::swap(&mut values, &mut self.sink);

        values
            .into_iter()
            .map(|val| self.try_value_to_literal(val))
            .collect()
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
            this: Value::Undefined,
            return_value_reg: None,
            return_to_iid: None,
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
            let fnid = self.data.header().fn_id;
            let func = self.get_function(fnid);

            assert!(
                self.iid.0 as usize <= func.instrs().len(),
                "can't proceed to instruction at index {} (func has {})",
                self.iid.0,
                func.instrs().len()
            );
            if self.iid.0 as usize == func.instrs().len() {
                if let Some(return_to_iid) = self.exit_function(None) {
                    self.iid = return_to_iid;
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
            eprint!("    ");

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
                    let literal = func.consts()[*const_ndx as usize].clone();
                    let value = self.literal_to_value(literal);
                    self.data.set_result(*dest, value);
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
                    let value = self.get_operand(*src);
                    self.data.set_result(*dst, value);
                }
                Instr::LoadCapture(dest, cap_ndx) => {
                    self.data.capture_to_var(*cap_ndx, *dest);
                }

                Instr::Nop => {}
                Instr::BoolNot { dest, arg } => {
                    let value = match self.get_operand(*arg) {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        Value::Number(num) => Value::Bool(num == 0.0),
                        Value::Object(oid) => match self.heap.get_string(oid) {
                            Some(str) => Value::Bool(str.is_empty()),
                            None => Value::Bool(false),
                        },
                        Value::Null => Value::Bool(true),
                        Value::Undefined => Value::Bool(true),
                        Value::SelfFunction => Value::Bool(false),
                        Value::Internal(_) => {
                            panic!("bytecode compiler bug: internal value should be unreachable")
                        }
                    };
                    self.data.set_result(*dest, value);
                }
                Instr::Jmp(IID(dest_ndx)) => {
                    next_ndx = *dest_ndx;
                }
                Instr::Return(value) => {
                    if let Some(return_to_iid) = self.exit_function(Some(*value)) {
                        self.iid = return_to_iid;
                        continue;
                    } else {
                        return Ok(());
                    }
                }
                Instr::Call {
                    callee,
                    this,
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
                    let n_args_u16 = TryInto::<u16>::try_into(arg_vals.len()).unwrap();
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

                            let callee_func = self.codebase.get_function(closure.fnid).unwrap();
                            let n_params = callee_func.n_params().0;
                            arg_vals.truncate(n_params as usize);
                            arg_vals.resize(n_params as usize, Value::Undefined);
                            assert_eq!(arg_vals.len(), n_params as usize);
                            eprintln!("     - call with {} params", n_params);
                            for (i, arg) in arg_vals.iter().enumerate() {
                                eprintln!("     - call arg[{}]: {:?}", i, arg);
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

                            eprintln!();
                            self.print_indent();
                            eprintln!(
                                "-- fn #{} [{} captures]",
                                call_meta.fnid.0, call_meta.n_captured_upvalues
                            );

                            self.data.push(call_meta);
                            for (capndx, capture) in closure.upvalues.iter().enumerate() {
                                let capndx = bytecode::CaptureIndex(
                                    capndx.try_into().expect("too many captures!"),
                                );
                                self.print_indent();
                                eprintln!("    capture[{}] = {:?}", capndx.0, capture);
                                self.data.set_capture(capndx, *capture);
                                assert_eq!(self.data.get_capture(capndx), *capture);
                            }
                            for (i, arg) in arg_vals.into_iter().enumerate() {
                                self.data.set_arg(bytecode::ArgIndex(i as _), arg);
                            }
                            self.iid = IID(0u16);

                            // Important: we don't execute the tail part of the instruction's
                            // execution. This makes it easier
                            // to keep a consistent value in `func` and other
                            // variables, and avoid subtle bugs
                            continue;
                        }
                        Closure::Native(nf) => {
                            let this = self.get_operand(*this);
                            let ret_val = (*nf)(self, &this, &arg_vals)?;
                            self.data.set_result(*return_value, ret_val);
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
                    let value = self.data.get_arg(*arg_ndx).unwrap_or(Value::Undefined);
                    eprint!("-> {:?}", value);
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
                    let key = self
                        .value_to_object_key(&key)
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
                    let key = self
                        .value_to_object_key(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self
                        .heap
                        .get_property(oid, &key)
                        .unwrap_or(Value::Undefined);
                    eprint!("  -> {:?}", value);
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
                    let key = self
                        .value_to_object_key(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;

                    self.heap.delete_property(oid, &key);
                    self.data.set_result(*dest, Value::Bool(true));
                }

                Instr::ArrayPush { arr, value } => {
                    let oid = self.get_operand_object(*arr).unwrap_or_else(|| {
                        panic!("ArrayPush: can't get element of non-array: {:?}", arr)
                    });
                    let value = self.get_operand(*value);
                    self.heap.array_push(oid, value);
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

                Instr::TypeOf { dest, arg: value } => {
                    let value = self.get_operand(*value);
                    let result = self.js_typeof(&value);
                    eprint!("-> {:?}", result);
                    self.data.set_result(*dest, result);
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

                Instr::ClosureNew {
                    dest,
                    fnid,
                    forced_this,
                } => {
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

                    let forced_this = forced_this.map(|reg| self.get_operand(reg));
                    let closure = Closure::JS(JSClosure {
                        fnid: *fnid,
                        upvalues,
                        forced_this,
                    });

                    let oid = self.heap.new_function(closure);
                    self.data.set_result(*dest, Value::Object(oid));
                }
                // This is always handled in the code for ClosureNew
                Instr::ClosureAddCapture(_) => {
                    unreachable!(
                        "interpreter bug: ClosureAddCapture should be handled with ClosureNew. (Usual cause: the bytecode compiler has placed some other instruction between ClosureAddCapture and ClosureNew.)"
                    )
                }

                Instr::UnaryMinus { dest, arg } => {
                    let arg_val: f64 = self.get_operand(*arg).expect_num()?;
                    self.data.set_result(*dest, Value::Number(-arg_val));
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
                            this: Value::Undefined,
                            return_value_reg: Some(*dest),
                            return_to_iid: Some(IID(self.iid.0 + 1)),
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
                Instr::LoadThis(dest) => {
                    self.data.set_result(*dest, self.data.header().this);
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
                Instr::IsInstanceOf(dest, obj, sup) => {
                    let result = if let Some(obj_oid) = self.get_operand_object(*obj) {
                        let sup_oid = self.get_operand_object(*sup).unwrap();
                        self.heap.is_instance_of(obj_oid, sup_oid)
                    } else {
                        false
                    };
                    self.data.set_result(*dest, Value::Bool(result));
                }
                Instr::NewIterator { dest: _, obj: _ } => todo!(),
                Instr::IteratorGetCurrent { dest: _, iter: _ } => todo!(),
                Instr::IteratorAdvance { iter: _ } => todo!(),
                Instr::JmpIfIteratorFinished { iter: _, dest: _ } => todo!(),

                Instr::Throw(_) => todo!(),

                Instr::StrCreateEmpty(dest) => {
                    let oid = self.heap.new_string(String::new());
                    self.data.set_result(*dest, Value::Object(oid));
                }
                Instr::StrAppend(buf_reg, tail) => {
                    // TODO Make this at least *decently* efficient!
                    let buf = self.get_operand_string(*buf_reg);
                    let mut buf: String = buf.to_owned();

                    let tail: &str = self.get_operand_string(*tail);

                    buf.push_str(tail);
                    let value = self.literal_to_value(bytecode::Literal::String(buf));
                    self.data.set_result(*buf_reg, value);
                }

                Instr::GetGlobal { dest, key } => {
                    let key = self.get_operand(*key);
                    let key = self
                        .value_to_object_key(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;

                    let value = self
                        .heap
                        .get_property(self.global_obj, &key)
                        .unwrap_or(Value::Undefined);
                    self.data.set_result(*dest, value);
                }
            }

            eprintln!();

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
        let header = self.data.header();
        let caller_retval_reg = header.return_value_vreg;
        let return_to_iid = header.return_to_iid;
        self.data.pop();

        // XXX This is a bug.  Get the top frame again.  But the compiler should say something
        // first!

        if let Some(vreg) = caller_retval_reg {
            self.data.set_result(vreg, return_value);
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
        // TODO Terrible.   We have to re-parse the whole frame header before being able to call
        // set_result!
        self.data.set_result(dest, Value::Number(op(a, b)));
    }

    fn compare<FB, FN, FS>(
        &mut self,
        dest: VReg,
        a: VReg,
        b: VReg,
        op_bool: FB,
        op_num: FN,
        _op_str: FS,
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
            (Value::Null, Value::Null) => op_bool(true, true),
            (Value::Undefined, Value::Undefined) => op_bool(true, true),
            _ => false,
        };

        // TODO Terrible.   We have to re-parse the whole frame header before being able to call
        // set_result!
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

    fn get_operand_string(&self, operand: VReg) -> &str {
        let oid = self
            .get_operand_object(operand)
            .expect("invalid string (not an object)");
        self.heap
            .get_string(oid)
            .expect("invalid string (object is not a string)")
    }

    fn print_indent(&self) {
        for _ in 0..(self.data.len() - 1) {
            eprint!("    ");
        }
    }

    // TODO(cleanup) inline this function? It now adds nothing
    fn get_operand(&self, vreg: bytecode::VReg) -> Value {
        // TODO Terrible.   We have to re-parse the whole frame header before being able to call
        // set_result!
        let value = self.data.get_result(vreg);
        //  TODO(cleanup) Move to a global logger. This is just for debugging!
        #[cfg(test)]
        {
            eprintln!();
            self.print_indent();
            eprint!("        {:?} = {:?}", vreg, value);
        }
        value
    }

    fn js_typeof(&mut self, value: &Value) -> Value {
        let ty_s = match value {
            Value::Number(_) => "number",
            Value::Bool(_) => "boolean",
            Value::Object(oid) => match self.heap.get_typeof(*oid) {
                heap::Typeof::Object => "object",
                heap::Typeof::Function => "function",
                heap::Typeof::String => "string",
            },
            // TODO(cleanup) This is actually an error in our type system.  null is really a value
            // of the 'object' type
            Value::Null => "object",
            Value::Undefined => "undefined",
            Value::SelfFunction => "function",
            Value::Internal(_) => panic!("internal value has no typeof!"),
        };

        self.literal_to_value(bytecode::Literal::String(ty_s.to_string()))
    }

    /// Create a Value based on the given Literal.\
    ///
    /// It may allocate an object in the GC-managed heap.
    fn literal_to_value(&mut self, lit: bytecode::Literal) -> Value {
        match lit {
            bytecode::Literal::Number(nu) => Value::Number(nu),
            bytecode::Literal::String(st) => {
                // TODO(performance) avoid this allocation
                let oid = self.heap.new_string(st.clone());
                Value::Object(oid)
            }
            bytecode::Literal::Bool(bo) => Value::Bool(bo),
            bytecode::Literal::Null => Value::Null,
            bytecode::Literal::Undefined => Value::Undefined,
            bytecode::Literal::SelfFunction => todo!(),
        }
    }

    fn try_value_to_literal(&self, value: Value) -> Option<bytecode::Literal> {
        match value {
            Value::Number(num) => Some(bytecode::Literal::Number(num)),
            Value::Bool(b) => Some(bytecode::Literal::Bool(b)),
            Value::Object(oid) => match self.heap.get_string(oid) {
                Some(s) => Some(bytecode::Literal::String(s.to_owned())),
                None => None,
            },
            Value::Null => Some(bytecode::Literal::Null),
            Value::Undefined => Some(bytecode::Literal::Undefined),
            Value::SelfFunction => None,
            Value::Internal(_) => None,
        }
    }

    fn value_to_object_key(&self, value: &Value) -> Option<heap::ObjectKey> {
        match value {
            Value::Number(n) if *n >= 0.0 => {
                let n_trunc = n.trunc();
                if *n == n_trunc {
                    let ndx = n_trunc as usize;
                    Some(heap::ObjectKey::ArrayIndex(ndx))
                } else {
                    None
                }
            }
            Value::Object(oid) => self.heap.get_string(*oid).map(|s| s.to_owned().into()),
            _ => None,
        }
    }
}

fn init_builtins(heap: &mut heap::ObjectHeap) -> heap::ObjectId {
    #![allow(non_snake_case)]

    let global = heap.new_object();

    {
        let array_cons = heap.new_function(Closure::Native(&nf_Array));
        heap.set_property(
            global,
            "Array".to_string().into(),
            Value::Object(array_cons),
        );
        let Array_isArray = heap.new_function(Closure::Native(&nf_Array_isArray));
        heap.set_property(
            array_cons,
            "isArray".to_string().into(),
            Value::Object(Array_isArray),
        );

        let array_proto = heap.new_object();
        heap.set_property(
            array_cons,
            "prototype".to_string().into(),
            Value::Object(array_proto),
        );

        let Array_push = heap.new_function(Closure::Native(&nf_Array_push));
        heap.set_property(
            array_proto,
            "push".to_string().into(),
            Value::Object(Array_push),
        );

        let Array_pop = heap.new_function(Closure::Native(&nf_Array_pop));
        heap.set_property(
            array_proto,
            "pop".to_string().into(),
            Value::Object(Array_pop),
        );
    }

    let RegExp = heap.new_function(Closure::Native(&nf_RegExp));
    heap.set_property(global, "RegExp".to_string().into(), Value::Object(RegExp));

    let Number = heap.new_function(Closure::Native(&nf_Number));
    heap.set_property(global, "Number".to_string().into(), Value::Object(Number));

    let String = heap.new_function(Closure::Native(&nf_String));
    heap.set_property(global, "String".to_string().into(), Value::Object(String));

    let Boolean = heap.new_function(Closure::Native(&nf_Boolean));
    heap.set_property(global, "Boolean".to_string().into(), Value::Object(Boolean));

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

#[allow(non_snake_case)]
fn nf_Array_isArray(intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    let value = if let Some(Value::Object(oid)) = args.get(0) {
        intrp.heap.is_array(*oid).unwrap_or(false)
    } else {
        false
    };

    Ok(Value::Bool(value))
}

#[allow(non_snake_case)]
fn nf_Array_push(intrp: &mut Interpreter, this: &Value, args: &[Value]) -> Result<Value> {
    // TODO Proper error handling, instead of these unwrap
    let oid = this.expect_obj().unwrap();
    intrp.heap.array_push(oid, args.get(0).unwrap().clone());
    Ok(Value::Undefined)
}

#[allow(non_snake_case)]
fn nf_Array_pop(_intrp: &mut Interpreter, _this: &Value, _args: &[Value]) -> Result<Value> {
    todo!("nf_Array_pop")
}

#[allow(non_snake_case)]
fn nf_RegExp(intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    // TODO
    let oid = intrp.heap.new_object();
    Ok(Value::Object(oid))
}

#[allow(non_snake_case)]
fn nf_Array(intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    // TODO
    let oid = intrp.heap.new_object();
    Ok(Value::Object(oid))
}

#[allow(non_snake_case)]
fn nf_Number(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Number(0.0))
}

#[allow(non_snake_case)]
fn nf_String(_intrp: &mut Interpreter, _this: &Value, args: &[Value]) -> Result<Value> {
    let value_str = match args.get(0) {
        Some(Value::Number(num)) => num.to_string(),
        Some(Value::Bool(true)) => "true".into(),
        Some(Value::Bool(false)) => "false".into(),
        Some(Value::Object(oid)) => match _intrp.heap.get_typeof(*oid) {
            heap::Typeof::Object => "<object>".to_owned(),
            heap::Typeof::Function => "<function>".to_owned(),
            heap::Typeof::String => _intrp.heap.get_string(*oid).unwrap().to_owned(),
        },
        Some(Value::Null) => "null".into(),
        Some(Value::Undefined) => "undefined".into(),
        Some(Value::SelfFunction) => "<function>".into(),
        Some(Value::Internal(_)) => unreachable!(),
        None => "".into(),
    };

    Ok(_intrp.literal_to_value(bytecode::Literal::String(value_str)))
}

#[allow(non_snake_case)]
fn nf_Boolean(_intrp: &mut Interpreter, _this: &Value, _: &[Value]) -> Result<Value> {
    Ok(Value::Bool(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Output {
        sink: Vec<Option<Literal>>,
    }

    use crate::{bytecode::Literal, fs::MockLoader};

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
        assert_eq!(&[Some(Literal::Number(104.0))], &output.sink[..]);
    }

    #[test]
    fn test_multiple_calls() {
        let output =
            quick_run("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ").unwrap();
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
        )
        .unwrap();

        assert_eq!(&[Some(Literal::String("b".to_owned()))], &output.sink[..]);
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
        )
        .unwrap();

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
        )
        .unwrap();

        assert_eq!(&[Some(Literal::Number(10.0))], &output.sink[..]);
    }

    fn try_casting_bool(code: &str, expected_value: bool) {
        let output = quick_run(code).unwrap();
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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();
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
}
