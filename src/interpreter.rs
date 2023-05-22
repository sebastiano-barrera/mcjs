use lazy_static::lazy_static;

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
    bytecode::{self, ArithOp, BoolOp, CmpOp, FnId, GlobalIID, Instr, IID},
    bytecode_compiler,
    common::Result,
    error, heap,
    jit::{self, InterpreterStep},
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
    // TODO(cleanup) Delete, Closure supersedes this
    SelfFunction,
    Closure(Closure),
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

impl From<Closure> for Value {
    fn from(closure: Closure) -> Self {
        Value::Closure(closure)
    }
}

impl From<bytecode::Value> for Value {
    fn from(bc_value: bytecode::Value) -> Self {
        match bc_value {
            bytecode::Value::Number(nu) => Value::Number(nu),
            bytecode::Value::String(st) => Value::String(st.into()),
            bytecode::Value::Bool(bo) => Value::Bool(bo),
            bytecode::Value::Null => Value::Null,
            bytecode::Value::Undefined => Value::Undefined,
            bytecode::Value::SelfFunction => todo!(),
        }
    }
}

impl Value {
    fn js_typeof(&self) -> Value {
        let ty_s = match self {
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Bool(_) => "boolean",
            Value::Object(_) => "object",
            // TODO(cleanup) This is actually an error in our type system.  null is really a value
            // of the 'object' type
            Value::Null => "object",
            Value::Undefined => "undefined",
            Value::SelfFunction => "function",
            Value::Closure(_) => "function",
        };

        ty_s.to_string().into()
    }

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
}

#[derive(Clone, PartialEq, Eq)]
pub enum Closure {
    Native(bytecode::NativeFnId),
    JS(JSClosure),
}

#[derive(Clone, PartialEq, Eq)]
pub struct JSClosure {
    fnid: FnId,
    /// TODO There oughta be a better data structure for this
    upvalues: Vec<UpvalueId>,
}

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Closure::Native(bid) => {
                write!(f, "<native closure {}>", bid)
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

// TODO Rename to a better name ("Driver"?)
pub struct VM {
    include_paths: Vec<PathBuf>,
    modules: HashMap<String, bytecode::Module>,
    opts: VMOptions,
    sink: Vec<Value>,
    traces: HashMap<String, (jit::Trace, jit::NativeThunk)>,
}

#[derive(Default)]
pub struct VMOptions {
    pub debug_dump_module: bool,
}

pub struct NotADirectoryError(PathBuf);
impl std::fmt::Debug for NotADirectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "provided path is not a directory: {}", self.0.display())
    }
}

impl VM {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        VM {
            include_paths: Vec::new(),
            modules: HashMap::new(),
            opts: Default::default(),
            sink: Vec::new(),
            traces: HashMap::new(),
        }
    }

    pub fn options_mut(&mut self) -> &mut VMOptions {
        &mut self.opts
    }

    pub fn take_sink(&mut self) -> Vec<Value> {
        let mut swap_area = Vec::new();
        std::mem::swap(&mut swap_area, &mut self.sink);
        swap_area
    }

    pub fn take_trace(&mut self) -> Option<jit::Trace> {
        todo!("(cleanup) delete this method")
    }

    pub fn get_trace(&self, trace_id: &str) -> Option<&(jit::Trace, jit::NativeThunk)> {
        self.traces.get(trace_id)
    }

    pub fn trace_ids(&self) -> impl ExactSizeIterator<Item = &String> {
        self.traces.keys()
    }

    pub fn add_include_path(
        &mut self,
        path: PathBuf,
    ) -> std::result::Result<(), NotADirectoryError> {
        if path.is_dir() {
            self.include_paths.push(path);
            Ok(())
        } else {
            Err(NotADirectoryError(path))
        }
    }

    pub fn load_module(&mut self, path: &str) -> Result<()> {
        self.load_module_ex(path, Default::default())
    }
    pub fn load_module_ex(&mut self, path: &str, flags: InterpreterFlags) -> Result<()> {
        use std::io::Read;

        let (file_path, key): (PathBuf, String) = self.find_module(path.to_string())?;

        let text = {
            let mut source_file = std::fs::File::open(file_path).map_err(Error::from)?;
            let mut buf = String::new();
            source_file.read_to_string(&mut buf).map_err(Error::from)?;
            buf
        };

        let mut bc_compiler = bytecode_compiler::Compiler::new();
        set_native_fns(&mut bc_compiler);
        let module = bc_compiler.compile_file(key.clone(), text).unwrap();

        if self.opts.debug_dump_module {
            eprintln!("=== loaded module: {}", key);
            module.dump();
        }

        self.run_module(&module, flags)
    }

    fn find_module(&self, require_path: String) -> Result<(PathBuf, String)> {
        let mut key = require_path;
        if !key.ends_with(".js") {
            key.push_str(".js");
        }
        let require_path = Path::new(&key[..]);

        for inc_path in self.include_paths.iter() {
            let potential_path = inc_path.join(require_path);
            if potential_path.is_file() {
                return Ok((potential_path.canonicalize().unwrap(), key));
            }
        }

        Err(error!("no such module: {key}"))
    }

    pub fn run_script(&mut self, script_text: String, flags: InterpreterFlags) -> Result<()> {
        let mut bc_compiler = bytecode_compiler::Compiler::new();
        set_native_fns(&mut bc_compiler);
        let module = bc_compiler
            .compile_file("<input>".to_string(), script_text)
            .unwrap();

        #[cfg(test)]
        {
            module.dump();
        }

        self.run_module(&module, flags)
    }

    fn run_module(&mut self, module: &bytecode::Module, flags: InterpreterFlags) -> Result<()> {
        let mut intrp = Interpreter::new(self, flags, module);
        intrp.run()?;
        let Interpreter { sink, jitting, .. } = intrp;

        if let Some(jitting) = jitting {
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

        self.sink = sink;
        Ok(())
    }
}

struct Interpreter<'a> {
    vm: &'a mut VM,
    flags: InterpreterFlags,
    jitting: Option<Jitting>,

    heap: heap::ObjectHeap,

    // TODO (big feat) This interpreter does not support having multiple modules running. can you
    // believe it?!
    // module_id: bytecode::IID,
    module: &'a bytecode::Module,
    data: stack::InterpreterData,
    iid: bytecode::IID,

    sink: Vec<Value>,
}

struct Jitting {
    fnid: FnId,
    iid: IID,
    trace_id: String,
    builder: jit::TraceBuilder,
}

impl<'a> Interpreter<'a> {
    fn new(parent_vm: &'a mut VM, flags: InterpreterFlags, module: &'a bytecode::Module) -> Self {
        eprintln!("Interpreter: flags: {:?}", flags);

        let mut stack = stack::InterpreterData::new();
        let main_func = module.get_function(FnId::ROOT_FN).unwrap();
        eprintln!("-- main");
        stack.push(stack::CallMeta {
            fnid: FnId::ROOT_FN,
            n_instrs: main_func.instrs().len().try_into().unwrap(),
            n_captured_upvalues: 0,
            n_args: 0,
            call_iid: None,
        });

        Interpreter {
            vm: parent_vm,
            heap: heap::ObjectHeap::new(),
            flags,
            jitting: None,
            module,
            iid: bytecode::IID(0),
            data: stack,
            sink: Vec::new(),
        }
    }

    fn run(&mut self) -> Result<()> {
        loop {
            if self.data.len() == 0 {
                return Ok(());
            }

            let fnid = self.data.fnid();
            // TODO make it so that func is "gotten" and unwrapped only when strictly necessary
            let func = self.module.get_function(fnid).unwrap();

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
            eprintln!("{:4}: {:?}", self.iid.0, instr);

            let mut next_ndx = self.iid.0 + 1;

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
                Instr::Const(value) => {
                    self.data.set_result(self.iid, value.clone().into());
                }
                Instr::Arith { op, a, b } => {
                    let a = self.get_operand(*a);
                    let b = self.get_operand(*b);

                    if let (Value::Number(a), Value::Number(b)) = (&a, &b) {
                        self.data.set_result(
                            self.iid,
                            Value::Number(match op {
                                ArithOp::Add => a + b,
                                ArithOp::Sub => a - b,
                                ArithOp::Mul => a * b,
                                ArithOp::Div => a / b,
                            }),
                        );
                    } else {
                        panic!("invalid operands for arith op: {:?}; {:?}", a, b);
                    }
                }
                Instr::PushSink(operand) => {
                    let value = self.get_operand(*operand);
                    self.sink.push(value);
                }
                Instr::Cmp { op, a, b } => {
                    let a = self.get_operand(*a);
                    let b = self.get_operand(*b);

                    match (&a, &b) {
                        (Value::Number(a), Value::Number(b)) => {
                            self.data.set_result(
                                self.iid,
                                Value::Bool(match op {
                                    CmpOp::GE => a >= b,
                                    CmpOp::GT => a > b,
                                    CmpOp::LT => a < b,
                                    CmpOp::LE => a <= b,
                                    CmpOp::EQ => a == b,
                                    CmpOp::NE => a != b,
                                }),
                            );
                        }
                        (Value::String(a), Value::String(b)) => {
                            let ordering = a.cmp(b);
                            self.data.set_result(
                                self.iid,
                                Value::Bool(matches!(
                                    (op, ordering),
                                    (CmpOp::GE, Ordering::Greater)
                                        | (CmpOp::GE, Ordering::Equal)
                                        | (CmpOp::GT, Ordering::Greater)
                                        | (CmpOp::LT, Ordering::Less)
                                        | (CmpOp::LE, Ordering::Less)
                                        | (CmpOp::LE, Ordering::Equal)
                                        | (CmpOp::EQ, Ordering::Equal)
                                        | (CmpOp::NE, Ordering::Greater)
                                        | (CmpOp::NE, Ordering::Less)
                                )),
                            );
                        }

                        _ => {
                            self.data.set_result(self.iid, Value::Bool(false));
                            // panic!("invalid operands for cmp op: {:?}; {:?}", a,
                            // b);
                        }
                    }
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

                Instr::SetVar { var, value } => {
                    self.data.set_result(*var, self.get_operand(*value));
                }
                Instr::GetCapture(cap_ndx) => {
                    self.data.capture_to_var(*cap_ndx, self.iid);
                }

                Instr::Nop => {}
                Instr::Not(value) => {
                    let value = match self.get_operand(*value) {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        Value::Number(num) => Value::Bool(num == 0.0),
                        Value::String(str) => Value::Bool(str.is_empty()),
                        Value::Object(_) => Value::Bool(false),
                        Value::Null => Value::Bool(true),
                        Value::Undefined => Value::Bool(true),
                        Value::SelfFunction => Value::Bool(false),
                        Value::Closure(_) => Value::Bool(false),
                    };
                    self.data.set_result(self.iid, value);
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
                Instr::Call { callee, args } => {
                    let closure = match self.get_operand(*callee) {
                        Value::Closure(closure) => closure,
                        other => panic!("invalid callee (not a function): {:?}", other),
                    };

                    match closure {
                        Closure::JS(closure) => {
                            // The arguments have to be "read" before adding the stack frame;
                            // they will no longer be accessible
                            // afterwards
                            let arg_vals: Vec<_> =
                                args.iter().map(|arg| self.get_operand(*arg)).collect();
                            let callee_func = self.module.get_function(closure.fnid).unwrap();
                            let n_captures = closure.upvalues.len().try_into().unwrap();
                            let n_instrs = callee_func.instrs().len().try_into().unwrap();

                            if let Some(jitting) = &mut self.jitting {
                                jitting.builder.set_args(args);
                                jitting.builder.enter_function(
                                    self.iid,
                                    *callee,
                                    n_instrs as usize,
                                );
                            }

                            self.data.push(stack::CallMeta {
                                fnid: closure.fnid,
                                n_instrs,
                                n_captured_upvalues: n_captures,
                                n_args: arg_vals.len().try_into().unwrap(),
                                call_iid: Some(self.iid),
                            });

                            self.print_indent();
                            eprintln!("-- fn #{} [{} captures]", closure.fnid.0, n_captures);

                            for (capndx, capture) in closure.upvalues.iter().enumerate() {
                                self.data.set_capture(capndx, *capture);
                            }
                            for (i, arg) in arg_vals.into_iter().enumerate() {
                                self.data.set_arg(i, arg);
                            }

                            self.iid = IID(0u32);
                            // Important: we don't execute the tail part of the instruction's
                            // execution. This makes it easier
                            // to keep a consistent value in `func` and other
                            // variables, and avoid subtle bugs
                            continue;
                        }
                        Closure::Native(nfid) => {
                            let ret_val = self.invoke_native(nfid, args)?;
                            self.data.set_result(self.iid, ret_val);
                        }
                    }
                }

                Instr::GetArg(arg_ndx) => {
                    // TODO extra copy?
                    let value = self.data.get_arg(*arg_ndx).clone();
                    self.data.set_result(self.iid, value);
                }

                Instr::ObjNew => {
                    let oid = self.heap.new_object();
                    self.data.set_result(self.iid, Value::Object(oid));
                }
                Instr::ObjSet { obj, key, value } => {
                    let obj = self.get_operand(*obj);
                    let mut oid = if let Value::Object(obj) = obj {
                        obj
                    } else {
                        panic!("ObjSet: not an object");
                    };
                    let key = self.get_operand(*key);
                    let key = heap::ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self.get_operand(*value);

                    self.heap.set_property(oid, key, value);
                }
                Instr::ObjGet { obj, key } => {
                    let obj = self.get_operand(*obj);
                    let obj = if let Value::Object(obj) = obj {
                        obj
                    } else {
                        panic!("ObjSet: not an object");
                    };
                    let key = self.get_operand(*key);
                    let key = heap::ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self
                        .heap
                        .get_property(obj, &key)
                        .unwrap_or(&Value::Undefined);
                    self.data.set_result(self.iid, value.clone());
                }
                Instr::ArrayNew => {
                    eprintln!("ArrayNew does nothing for now");
                }
                Instr::ArrayPush(_arr, _elem) => {
                    eprintln!("ArrayPush does nothing for now");
                }
                Instr::TypeOf(arg) => {
                    let value = self.get_operand(*arg);
                    self.data.set_result(self.iid, value.js_typeof());
                }
                Instr::BoolOp { op, a, b } => {
                    let a: bool = self.get_operand(*a).expect_bool()?;
                    let b: bool = self.get_operand(*b).expect_bool()?;
                    let res = match op {
                        BoolOp::And => a && b,
                        BoolOp::Or => a || b,
                    };
                    self.data.set_result(self.iid, Value::Bool(res));
                }

                Instr::ClosureNew { fnid } => {
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

                    self.data.set_result(self.iid, Value::Closure(closure));
                }
                // This is always handled in the code for ClosureNew
                Instr::ClosureAddCapture(_) => unreachable!(),
                Instr::GetNativeFn(nfid) => {
                    self.data
                        .set_result(self.iid, Closure::Native(*nfid).into());
                }

                Instr::UnaryMinus(arg) => {
                    let arg: f64 = self.get_operand(*arg).expect_num()?;
                    self.data.set_result(self.iid, Value::Number(-arg));
                }
            }

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
    }

    fn exit_function(&mut self, return_value_iid: Option<IID>) -> Option<IID> {
        let return_value = return_value_iid
            .map(|iid| self.get_operand(iid))
            .unwrap_or(Value::Undefined);
        let call_iid: Option<_> = self.data.call_iid();
        self.data.pop();

        if let Some(call_iid) = call_iid {
            self.data.set_result(call_iid, return_value);
        }
        if let Some(jitting) = &mut self.jitting {
            jitting.builder.exit_function(return_value_iid);
        }

        call_iid
    }

    fn print_indent(&self) {
        for _ in 0..(self.data.len() - 1) {
            eprint!("    ");
        }
    }

    // TODO(cleanup) inline this function. It now adds nothing
    fn get_operand(&self, iid: bytecode::IID) -> Value {
        let value = self.data.get_result(iid).clone();
        //  TODO(cleanup) Move to a global logger. This is just for debugging!
        // self.print_indent();
        // eprintln!("        {:?} = {:?}", iid, value);
        value
    }

    fn invoke_native(&mut self, nfid: u32, args_iids: &[IID]) -> Result<Value> {
        // TODO Anything more efficient is appropriate here?
        let args: Vec<_> = args_iids.iter().map(|iid| self.get_operand(*iid)).collect();
        if nfid == NativeFnId::Require as u32 {
            nf_require(self, &args)
        } else if nfid == NativeFnId::StringNew as u32 {
            nf_string_new(self, &args)
        } else {
            panic!("undefined native function with ID {nfid}")
        }
    }
}

enum NativeFnId {
    Require,
    StringNew,
    ArrayNew,
}

fn set_native_fns(bc_compiler: &mut bytecode_compiler::Compiler) {
    bc_compiler.add_builtin_fn("require".into(), NativeFnId::Require as u32);
    bc_compiler.add_builtin_fn("String".into(), NativeFnId::StringNew as u32);
    bc_compiler.add_builtin_fn("Array".into(), NativeFnId::ArrayNew as u32);
    // TODO(big feat) pls impl all Node.js API, ok? thxbye
}

fn nf_require(intp: &mut Interpreter, args: &[Value]) -> Result<Value> {
    let arg0 = args.iter().next();
    match arg0 {
        Some(Value::String(path)) => {
            intp.vm.load_module(path)?;
            Ok(Value::Undefined)
        }
        _ => Err(error!("invalid args for require()")),
    }
}

fn nf_string_new(arg: &mut Interpreter, args: &[Value]) -> Result<Value> {
    todo!("nf_string_new")
}

#[derive(Clone, Debug)]
pub struct InterpreterFlags {
    pub indent_level: u8,
    pub jit_mode: JitMode,
}

#[derive(Clone, Copy, Debug)]
pub enum JitMode {
    Compile,
    UseTraces,
}

#[derive(Clone, Debug)]
pub struct TracerFlags {
    pub start_depth: u32,
}

impl Default for InterpreterFlags {
    fn default() -> Self {
        Self {
            indent_level: 0,
            jit_mode: JitMode::UseTraces,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Output {
        sink: Vec<Value>,
    }

    fn quick_run(code: &str) -> Result<Output> {
        let mut vm = VM::new();
        vm.run_script(code.to_string(), Default::default())?;
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
        assert!(matches!(&output.sink[4], Value::Closure(_)));
    }

    #[test]
    fn test_typeof() {
        todo!("check all cases for `typeof`");
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
        todo!("create an object with a prototype, then lookup a property that requires walking the prototype chain")
    }

    #[test]
    #[ignore]
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

        assert_eq!(&output.sink[..], &["x".into(), "y".into(), "name".into(),]);
    }
}
