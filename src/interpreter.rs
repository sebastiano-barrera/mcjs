use lazy_static::lazy_static;

use std::{
    borrow::Cow,
    cell::{Ref, RefCell, RefMut},
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    rc::Rc,
    sync::Mutex,
};

pub use crate::common::Error;
use crate::{
    bytecode_compiler,
    common::Result,
    error,
    jit::{self, InterpreterStep},
    stack,
    util::Mask,
};

/// A value that can be input, output, or processed by the program at runtime.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Number(f64),
    String(Cow<'static, str>),
    Bool(bool),
    Object(Object),
    Null,
    Undefined,
    // TODO Delete, Closure supersedes this
    SelfFunction,
    // TODO Delete, Closure supersedes this
    NativeFunction(u32),
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

impl Value {
    fn js_typeof(&self) -> Value {
        let ty_s = match self {
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Bool(_) => "boolean",
            Value::Object(_) => "object",
            // TODO This is actually an error in our type system.  null is really a value of the 'object' type
            Value::Null => "object",
            Value::Undefined => "undefined",
            Value::SelfFunction => "function",
            Value::NativeFunction(_) => "function",
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

#[derive(Debug, PartialEq, Clone)]
pub struct Object(Rc<RefCell<HashMap<ObjectKey, Value>>>);

impl Object {
    fn new() -> Self {
        Object(Rc::new(RefCell::new(HashMap::new())))
    }

    fn get<'a>(&'a self, key: &ObjectKey) -> Option<Ref<'a, Value>> {
        let hashmap = self.0.borrow();
        Ref::filter_map(hashmap, |map| map.get(key)).ok()
    }

    fn set(&mut self, key: ObjectKey, value: Value) {
        let mut map = self.0.borrow_mut();
        map.insert(key, value);
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum ObjectKey {
    String(String),
}
impl ObjectKey {
    fn from_value(value: &Value) -> Option<ObjectKey> {
        if let Value::String(s) = value {
            Some(ObjectKey::String(s.to_string()))
        } else {
            None
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Closure {
    fnid: FnId,
    lexical_parent: Option<stack::FrameId>,
}

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<closure {:?} <- {:?}>", self.fnid, self.lexical_parent)
    }
}

// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u32);

/// NOTE cross-module function calls are unsupported yet
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalIID {
    pub fnid: FnId,
    pub iid: IID,
}

impl std::fmt::Debug for IID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.0)
    }
}

#[derive(Debug)]
pub enum Instr {
    Nop,
    Const(Value),
    Not(Operand),
    UnaryMinus(Operand),
    Arith {
        op: ArithOp,
        a: Operand,
        b: Operand,
    },
    Cmp {
        op: CmpOp,
        a: Operand,
        b: Operand,
    },
    BoolOp {
        op: BoolOp,
        a: Operand,
        b: Operand,
    },
    JmpIf {
        cond: Operand,
        dest: IID,
    },
    Jmp(IID),
    Set {
        var_id: StaticVarId,
        value: Operand,
    },
    PushSink(Operand),
    Return(Operand),
    GetArg(usize),
    Call {
        callee: Operand,
        // smallvec?
        args: Vec<Operand>,
    },

    ClosureNew {
        fnid: FnId,
    },

    ObjNew,
    ObjSet {
        obj: Operand,
        key: Operand,
        value: Operand,
    },
    ObjGet {
        obj: Operand,
        key: Operand,
    },

    // Temporary; should be replaced by objects, just like all other "classes"
    ArrayNew,
    ArrayPush(Operand, Operand),

    TypeOf(Operand),

    StartTrace {
        trace_id: String,
        wait_loop: bool,
    },
}

impl Instr {
    fn read_vars<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = StaticVarId>> {
        Box::new(self.read_operands().filter_map(|oper| match oper {
            Operand::Var(var_id) => Some(var_id.clone()),
            _ => None,
        }))
    }

    fn read_operands<'a>(&'a self) -> Box<dyn 'a + Iterator<Item = &Operand>> {
        match self {
            Instr::Nop => Box::new(std::iter::empty()),
            Instr::Const(_) => Box::new(std::iter::empty()),
            Instr::Not(oper) => Box::new([oper].into_iter()),
            Instr::UnaryMinus(oper) => Box::new([oper].into_iter()),
            Instr::Arith { op: _, a, b } => Box::new([a, b].into_iter()),
            Instr::Cmp { op: _, a, b } => Box::new([a, b].into_iter()),
            Instr::BoolOp { op: _, a, b } => Box::new([a, b].into_iter()),
            Instr::JmpIf { cond, .. } => Box::new(std::iter::once(cond)),
            Instr::Jmp(_) => Box::new(std::iter::empty()),
            Instr::Set { var_id: _, value } => Box::new(std::iter::once(value)),
            Instr::PushSink(arg) => Box::new(std::iter::once(arg)),
            Instr::Return(arg) => Box::new(std::iter::once(arg)),
            Instr::GetArg(_) => Box::new(std::iter::empty()),
            Instr::Call { callee: _, args } => Box::new(args.iter()),
            Instr::ObjNew => Box::new(std::iter::empty()),
            Instr::ObjSet { obj, key, value } => Box::new([obj, key, value].into_iter()),
            Instr::ObjGet { obj, key } => Box::new([obj, key].into_iter()),
            Instr::ArrayNew => Box::new(std::iter::empty()),
            Instr::ArrayPush(arr, value) => Box::new([arr, value].into_iter()),
            Instr::TypeOf(arg) => Box::new([arg].into_iter()),
            Instr::ClosureNew { .. } => Box::new(std::iter::empty()),
            Instr::StartTrace { .. } => Box::new(std::iter::empty()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarIndex(pub(crate) u16);
/// Static identifier for a variable, used in the bytecode as operand.  
///
/// Designates a variable in the bytecode, not at runtime.  If the function
/// containing the variable declaration is called multiple times (e.g. via
/// recursion, or by producing multiple closures from the same function
/// literal), this identifier is the same every time, but the *runtime*
/// identifier will change.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticVarId {
    pub(crate) fnid: FnId,
    pub(crate) var_ndx: VarIndex,
}

impl std::fmt::Debug for StaticVarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "var[{}:{}]", self.fnid.0, self.var_ndx.0)
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub enum Operand {
    // Evaluates immediately to a constant value.
    Value(Value),
    // Evaluates to the value of the indicated variable.
    Var(StaticVarId),
    // Evaluates to the result of the indicated instruction.
    IID(IID),
}

impl std::fmt::Debug for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(val) => val.fmt(f),
            Self::IID(iid) => iid.fmt(f),
            Self::Var(varid) => varid.fmt(f),
        }
    }
}
impl From<IID> for Operand {
    fn from(iid: IID) -> Self {
        Operand::IID(iid)
    }
}
impl From<Value> for Operand {
    fn from(value: Value) -> Self {
        Operand::Value(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp {
    GE,
    GT,
    LT,
    LE,
    EQ,
    NE,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoolOp {
    And,
    Or,
}

pub struct VM {
    include_paths: Vec<PathBuf>,
    modules: HashMap<String, Module>,
    opts: VMOptions,
    sink: Vec<Value>,

    traces: HashMap<String, (jit::Trace, jit::NativeThunk)>,
}

pub struct VMOptions {
    pub debug_dump_module: bool,
}
impl Default for VMOptions {
    fn default() -> Self {
        VMOptions {
            debug_dump_module: false,
        }
    }
}

pub struct NotADirectoryError(PathBuf);
impl std::fmt::Debug for NotADirectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "provided path is not a directory: {}", self.0.display())
    }
}

type NativeFunc = fn(&mut Interpreter, &[Value]) -> Result<Value>;

lazy_static! {
    static ref NATIVE_FUNCS: HashMap<u32, NativeFunc> = {
        let mut map = HashMap::new();
        map.insert(VM::NFID_REQUIRE, nf_require as NativeFunc);
        map
    };
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

impl VM {
    const NFID_REQUIRE: u32 = 1;
    const NFID_STRING_NEW: u32 = 2;

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
        todo!("delete this method")
    }

    pub fn get_trace(&self, trace_id: &str) -> Option<&(jit::Trace, jit::NativeThunk)> {
        self.traces.get(trace_id)
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
        set_builtins(&mut bc_compiler);
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
        set_builtins(&mut bc_compiler);
        let module = bc_compiler
            .compile_file("<input>".to_string(), script_text)
            .unwrap();

        #[cfg(test)]
        {
            module.dump();
        }

        self.run_module(&module, flags)
    }

    fn run_module(&mut self, module: &Module, flags: InterpreterFlags) -> Result<()> {
        let mut intrp = Interpreter::new(self, flags);
        intrp.run_module(module)?;
        let Interpreter { sink, jitting, .. } = intrp;

        if let Some(jitting) = jitting {
            if let Some(trace) = jitting.builder.build() {
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
    frame_graph: stack::FrameGraph,
    stack_depth: u32,
    sink: Vec<Value>,
}

struct Jitting {
    fnid: FnId,
    iid: IID,
    trace_id: String,
    builder: jit::TraceBuilder,
}

impl<'a> Interpreter<'a> {
    fn new(parent_vm: &'a mut VM, flags: InterpreterFlags) -> Self {
        eprintln!("Interpreter: flags: {:?}", flags);

        Interpreter {
            vm: parent_vm,
            flags,
            jitting: None,
            frame_graph: stack::FrameGraph::new(),
            stack_depth: 0,
            sink: Vec::new(),
        }
    }

    fn run_module(&mut self, module: &Module) -> Result<()> {
        let root_closure = Closure {
            fnid: FnId::ROOT_FN,
            lexical_parent: None,
        };
        self.run_module_fn(module, &root_closure, &[])?;
        Ok(())
    }

    fn print_indent(&self) {
        for _ in 0..(self.stack_depth - 1) {
            eprint!("      | ");
        }
    }

    // TODO Change this to return &Value instead
    fn get_operand(
        &self,
        operand: &Operand,
        values_buf: &Vec<Value>,
        cur_fid: stack::FrameId,
    ) -> Value {
        // TODO Refactor this mess
        resolve_operand(operand, values_buf, &self.frame_graph, cur_fid)
    }

    fn run_module_fn(
        &mut self,
        module: &Module,
        closure: &Closure,
        args: &[Value],
    ) -> Result<Value> {
        let func = module.fns.get(&closure.fnid).unwrap();

        let lexical_parent = closure.lexical_parent.clone();
        let n_local_vars = func.n_slots as usize;
        let frame = self
            .frame_graph
            .new_frame(closure.fnid, n_local_vars, lexical_parent);
        self.stack_depth += 1;

        let res = self._run_module_fn_inner(module, closure, frame, func, args);

        // TODO Garbage collect stack frames
        self.stack_depth -= 1;
        res
    }

    fn _run_module_fn_inner(
        &mut self,
        module: &Module,
        closure: &Closure,
        cur_frame_id: stack::FrameId,
        func: &Function,
        args: &[Value],
    ) -> Result<Value> {
        let instrs = &func.instrs;

        let mut values_buf = vec![Value::Undefined; instrs.len()];

        if let Some(jitting) = &mut self.jitting {
            jitting
                .builder
                .enter_function(cur_frame_id, func.n_slots as usize);
        }

        let mut ndx = 0;
        let mut return_value = None;
        let mut jit_waiting_loop = None;

        while return_value.is_none() {
            if ndx >= instrs.len() {
                break;
            }
            let instr = &instrs[ndx];
            let mut next_ndx = ndx + 1;
            let iid = IID(ndx as u32);
            let next_iid = IID(next_ndx as u32);

            self.print_indent();
            eprintln!("i{:<4} {:?}", ndx, instr);

            match instr {
                Instr::Const(value) => {
                    values_buf[ndx] = value.clone();
                }
                Instr::Arith { op, a, b } => {
                    let a = self.get_operand(a, &mut values_buf, cur_frame_id);
                    let b = self.get_operand(b, &mut values_buf, cur_frame_id);

                    if let (Value::Number(a), Value::Number(b)) = (&a, &b) {
                        values_buf[ndx] = Value::Number(match op {
                            ArithOp::Add => a + b,
                            ArithOp::Sub => a - b,
                            ArithOp::Mul => a * b,
                            ArithOp::Div => a / b,
                        });
                    } else {
                        panic!("invalid operands for arith op: {:?}; {:?}", a, b);
                    }
                }
                Instr::PushSink(operand) => {
                    let value = self.get_operand(operand, &mut values_buf, cur_frame_id);
                    self.sink.push(value);
                }
                Instr::Cmp { op, a, b } => {
                    let a = self.get_operand(a, &mut values_buf, cur_frame_id);
                    let b = self.get_operand(b, &mut values_buf, cur_frame_id);

                    match (&a, &b) {
                        (Value::Number(a), Value::Number(b)) => {
                            values_buf[ndx] = Value::Bool(match op {
                                CmpOp::GE => a >= b,
                                CmpOp::GT => a > b,
                                CmpOp::LT => a < b,
                                CmpOp::LE => a <= b,
                                CmpOp::EQ => a == b,
                                CmpOp::NE => a != b,
                            });
                        }
                        (Value::String(a), Value::String(b)) => {
                            let ordering = a.cmp(b);
                            values_buf[ndx] = Value::Bool(matches!(
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
                            ));
                        }

                        _ => {
                            values_buf[ndx] = Value::Bool(false);
                            // panic!("invalid operands for cmp op: {:?}; {:?}", a, b);
                        }
                    }
                }
                Instr::JmpIf { cond, dest } => {
                    let cond_value = self.get_operand(cond, &mut values_buf, cur_frame_id);
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0 as usize;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => panic!("invalid if condition (not boolean): {:?}", other),
                    }
                }
                Instr::Set { var_id, value } => {
                    let value = self.get_operand(value, &values_buf, cur_frame_id);

                    let target_frame_id = self
                        .frame_graph
                        .get_lexical_scope(cur_frame_id, var_id.fnid)
                        .expect("invalid var_id (no lexical scope)");
                    self.frame_graph
                        .set_var(target_frame_id, var_id.var_ndx, value);
                }
                Instr::Nop => {}
                Instr::Not(value) => {
                    let value = self.get_operand(value, &values_buf, cur_frame_id);
                    values_buf[ndx] = match value {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        Value::Number(num) => Value::Bool(num == 0.0),
                        Value::String(str) => Value::Bool(str.is_empty()),
                        Value::Object(_) => Value::Bool(false),
                        Value::Null => Value::Bool(true),
                        Value::Undefined => Value::Bool(true),
                        Value::SelfFunction => Value::Bool(false),
                        Value::NativeFunction(_) => Value::Bool(false),
                        Value::Closure(_) => Value::Bool(false),
                    };
                }
                Instr::Jmp(dest) => {
                    next_ndx = dest.0 as usize;
                }
                Instr::Return(value) => {
                    return_value = Some(self.get_operand(value, &values_buf, cur_frame_id));
                }
                Instr::Call { callee, args } => {
                    let callee = self.get_operand(callee, &values_buf, cur_frame_id);
                    let arg_values: Vec<Value> = args
                        .iter()
                        .map(|oper| self.get_operand(oper, &values_buf, cur_frame_id))
                        .collect();
                    match callee {
                        Value::Closure(closure) => {
                            self.print_indent();
                            eprintln!("      : call {closure:?}",);

                            if let Some(jitting) = &mut self.jitting {
                                jitting.builder.set_args(&args, &|fnid| {
                                    self.frame_graph
                                        .get_lexical_scope(cur_frame_id, fnid)
                                        .unwrap()
                                });
                            }
                            let ret_val = self.run_module_fn(module, &closure, &arg_values[..])?;
                            values_buf[ndx] = ret_val;
                        }
                        Value::NativeFunction(nfid) => {
                            let nf = NATIVE_FUNCS
                                .get(&nfid)
                                .ok_or(error!("no such native function: {nfid}"))?;
                            self.print_indent();
                            todo!("interpreter: tell the JIT about this native call");
                            eprintln!("      : native call");
                            let ret_val = nf(self, arg_values.as_slice())?;
                            values_buf[ndx] = ret_val;
                        }
                        _ => {
                            panic!("invalid callee (not a function): {:?}", callee);
                        }
                    }
                }

                Instr::GetArg(arg_ndx) => {
                    values_buf[ndx] = args[*arg_ndx].clone();
                }

                Instr::ObjNew => {
                    values_buf[ndx] = Value::Object(Object::new());
                }
                Instr::ObjSet { obj, key, value } => {
                    let obj = self.get_operand(obj, &values_buf, cur_frame_id);
                    let mut obj = if let Value::Object(obj) = obj {
                        obj
                    } else {
                        panic!("ObjSet: not an object");
                    };
                    let key = self.get_operand(key, &values_buf, cur_frame_id);
                    let key = ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = self.get_operand(value, &values_buf, cur_frame_id);
                    obj.set(key, value);
                }
                Instr::ObjGet { obj, key } => {
                    let obj = self.get_operand(obj, &values_buf, cur_frame_id);
                    let obj = if let Value::Object(obj) = obj {
                        obj
                    } else {
                        panic!("ObjSet: not an object");
                    };
                    let key = self.get_operand(key, &values_buf, cur_frame_id);
                    let key = ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = obj.get(&key);
                    values_buf[ndx] = value
                        .unwrap_or_else(|| {
                            panic!("ObjGet: no property for key `{:?}` in object", key)
                        })
                        .clone();
                }
                Instr::ArrayNew => {
                    eprintln!("ArrayNew does nothing for now");
                }
                Instr::ArrayPush(_arr, _elem) => {
                    eprintln!("ArrayPush does nothing for now");
                }
                Instr::TypeOf(arg) => {
                    let value = self.get_operand(arg, &values_buf, cur_frame_id);
                    values_buf[ndx] = value.js_typeof().into();
                }
                Instr::BoolOp { op, a, b } => {
                    let a: bool = self
                        .get_operand(a, &values_buf, cur_frame_id)
                        .expect_bool()?;
                    let b: bool = self
                        .get_operand(b, &values_buf, cur_frame_id)
                        .expect_bool()?;
                    let res = match op {
                        BoolOp::And => a && b,
                        BoolOp::Or => a || b,
                    };
                    values_buf[ndx] = Value::Bool(res);
                }
                Instr::ClosureNew { fnid } => {
                    let closure = Closure {
                        fnid: *fnid,
                        lexical_parent: Some(cur_frame_id.clone()),
                    };
                    self.print_indent();
                    eprintln!("      created closure: {:?}", closure);
                    values_buf[ndx] = Value::Closure(closure);
                }
                Instr::UnaryMinus(arg) => {
                    let arg: f64 = self
                        .get_operand(arg, &values_buf, cur_frame_id)
                        .expect_num()?;
                    values_buf[ndx] = Value::Number(-arg);
                }

                Instr::StartTrace {
                    trace_id,
                    wait_loop,
                } => match self.flags.jit_mode {
                    JitMode::Compile => {
                        if self.jitting.is_none() {
                            if *wait_loop {
                                jit_waiting_loop = Some(trace_id);
                            } else {
                                let builder = jit::TraceBuilder::start(
                                    cur_frame_id,
                                    func.n_slots as usize,
                                    jit::CloseMode::FunctionExit,
                                );
                                self.jitting = Some(Jitting {
                                    builder,
                                    fnid: closure.fnid,
                                    iid,
                                    trace_id: trace_id.clone(),
                                });
                            }
                        }
                    }
                    JitMode::UseTraces => {
                        let (trace, thunk) = self
                            .vm
                            .get_trace(trace_id)
                            .unwrap_or_else(|| panic!("no such trace with ID `{trace_id}`"));

                        let mut snap: Vec<_> = trace
                            .snapshot_map()
                            .iter()
                            .map(|value| self.get_operand(value, &mut values_buf, cur_frame_id))
                            .collect();

                        // TODO Use return value
                        thunk.run(&mut snap);
                    }
                },
            }

            if let Some(trace_id) = jit_waiting_loop {
                if func.is_loop_head(iid) {
                    let fnid = closure.fnid;
                    let global_iid = GlobalIID { fnid, iid };
                    let builder = jit::TraceBuilder::start(
                        cur_frame_id,
                        func.n_slots as usize,
                        jit::CloseMode::Loop(global_iid),
                    );
                    self.jitting = Some(Jitting {
                        builder,
                        fnid,
                        iid,
                        trace_id: trace_id.clone(),
                    });
                    jit_waiting_loop = None;
                }
            }

            if let Some(jitting) = &mut self.jitting {
                let frame_graph = &self.frame_graph;
                jitting.builder.interpreter_step(&InterpreterStep {
                    values_buf: &values_buf,
                    fnid: closure.fnid,
                    func: &func,
                    iid,
                    next_iid,
                    fnid_to_frameid: &|fnid| {
                        self.frame_graph
                            .get_lexical_scope(cur_frame_id, fnid)
                            .unwrap()
                    },
                    get_operand: &|operand| {
                        resolve_operand(operand, &values_buf, frame_graph, cur_frame_id)
                    },
                });
            }

            ndx = next_ndx;
        }

        if let Some(jitting) = self.jitting.as_mut() {
            jitting.builder.exit_function();
        }
        Ok(return_value.unwrap_or(Value::Undefined))
    }
}

fn resolve_operand(
    operand: &Operand,
    values_buf: &Vec<Value>,
    frame_graph: &stack::FrameGraph,
    cur_fid: stack::FrameId,
) -> Value {
    match operand {
        Operand::Value(value) => value.clone(),
        Operand::IID(IID(ndx)) => {
            let value = values_buf[*ndx as usize].clone();
            //  TODO Move to a global logger. This is just for debugging!
            //  self.print_indent();
            eprintln!("      ↑ {:?} = {:?}", operand, value);
            value
        }
        Operand::Var(var_id) => {
            let fid = frame_graph.get_lexical_scope(cur_fid, var_id.fnid).unwrap();
            let value = frame_graph.get_var(fid, var_id.var_ndx).unwrap().clone();
            //  TODO Move to a global logger. This is just for debugging!
            //  self.print_indent();
            eprintln!("      ↑ {:?} = {:?}", var_id, value);
            value
        }
    }
}

fn set_builtins(bc_compiler: &mut bytecode_compiler::Compiler) {
    bc_compiler.bind_native(
        "require".into(),
        Value::NativeFunction(VM::NFID_REQUIRE).into(),
    );
    bc_compiler.bind_native(
        "String".into(),
        Value::NativeFunction(VM::NFID_STRING_NEW).into(),
    );
    // TODO pls impl all Node.js API, ok? thxbye
    bc_compiler.bind_native("Object".into(), Value::Object(Object::new()).into());
    bc_compiler.bind_native("Array".into(), Value::Object(Object::new()).into());
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

#[derive(Clone, Copy, Hash, Debug, PartialEq, Eq)]
pub struct FnId(pub u32);
impl FnId {
    pub const ROOT_FN: FnId = FnId(0);
}

pub struct Module {
    fns: HashMap<FnId, Function>,
}

impl Module {
    pub fn new(fns: HashMap<FnId, Function>) -> Self {
        Module { fns }
    }

    pub(crate) fn functions(&self) -> &HashMap<FnId, Function> {
        &self.fns
    }

    pub(crate) fn dump(&self) {
        eprintln!("=== module");
        for (fnid, func) in self.fns.iter() {
            eprintln!("fn #{} [{} vars]:", fnid.0, func.n_slots);
            for (ndx, instr) in func.instrs.iter().enumerate() {
                let lh = func.loop_heads.get(&IID(ndx as u32));
                eprintln!(
                    "  {:4}{:4}: {:?}",
                    if lh.is_some() { ">>" } else { "" },
                    ndx,
                    instr,
                );
                if let Some(lh) = lh {
                    eprint!("            (phis: ");
                    for var in &lh.interloop_vars {
                        eprint!("{:?}, ", var);
                    }
                    eprintln!(")");
                }
            }
        }
        eprintln!("---");
    }
}

pub struct Function {
    instrs: Box<[Instr]>,
    loop_heads: HashMap<IID, LoopInfo>,
    n_slots: u16,
}
pub struct LoopInfo {
    // Variables that change in value during each cycle, in such a way that
    // each cycle sees the value in  the previous cycle.  Phi instructions are
    // added based on this set.
    interloop_vars: HashSet<StaticVarId>,
}
impl Function {
    pub(crate) fn new(instrs: Box<[Instr]>, n_slots: u16) -> Function {
        let loop_heads = find_loop_heads(&instrs[..]);
        Function {
            instrs,
            loop_heads,
            n_slots,
        }
    }

    pub(crate) fn instrs(&self) -> &[Instr] {
        self.instrs.as_ref()
    }

    pub(crate) fn is_loop_head(&self, iid: IID) -> bool {
        self.loop_heads.contains_key(&iid)
    }

    pub(crate) fn n_slots(&self) -> u16 {
        self.n_slots
    }
}

fn find_loop_heads(instrs: &[Instr]) -> HashMap<IID, LoopInfo> {
    // The set of interloop variables is the set of variables where, within a
    // loop, at least one read happens before a write.
    let mut heads = HashMap::new();

    // It ain't linear, but it does the job (plus I don't think
    // there should be so many nesting levels for loops within the
    // same function...)
    for (end_ndx, inst) in instrs.iter().enumerate() {
        match inst {
            Instr::Jmp(dest) | Instr::JmpIf { dest, .. } if dest.0 as usize <= end_ndx => {
                // Loop goes from end_ndx to dest

                let dest_ndx = dest.0 as usize;
                let mut interloop_vars = HashSet::new();
                let mut reads = HashSet::new();
                for ndx in dest_ndx..end_ndx {
                    let inst = &instrs[ndx];
                    for read_var in inst.read_vars() {
                        reads.insert(read_var);
                    }

                    if let Instr::Set { var_id, .. } = inst {
                        if reads.remove(var_id) {
                            interloop_vars.insert(var_id.clone());
                        }
                    }
                }

                heads.insert(dest.clone(), LoopInfo { interloop_vars });
            }
            _ => {}
        }
    }

    heads
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
    fn test_object_init() {
        let output = quick_run(
            "sink({
                aString: 'asdlol123',
                aNumber: 1239423.4518923,
                anotherObject: { x: 123, y: 899 },
                aFunction: function(pt) { return 42; }
            })",
        )
        .unwrap();

        assert_eq!(1, output.sink.len());
        let obj = if let Value::Object(obj) = &output.sink[0] {
            obj
        } else {
            panic!("not an object")
        };

        let val = obj.get(&ObjectKey::String("aString".to_string())).unwrap();
        assert_eq!(&*val, &"asdlol123".into());
    }

    #[ignore]
    #[test]
    fn test_object_member_set() {
        panic!("not yet implemented");
    }

    #[ignore]
    #[test]
    fn test_for_in() {
        // TODO This syntax is not yet implemented
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
