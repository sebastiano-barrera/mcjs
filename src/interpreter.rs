use lazy_static::lazy_static;

use std::{
    borrow::Cow,
    cell::{Ref, RefCell},
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
    SelfFunction,
    LocalFn(FnId),
    NativeFunction(u32),
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
            Value::LocalFn(_) => "function",
            Value::NativeFunction(_) => "function",
        };

        ty_s.to_string().into()
    }

    pub(crate) fn expect_bool(&self) -> Result<bool> {
        if let Value::Bool(val) = self {
            Ok(*val)
        } else {
            Err(error!("expected a boolean"))
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
// Instruction ID. Can identify an instruction, or its result.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub struct IID(pub u32);

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
        var_id: VarId,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StackFrameId(pub(crate) u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarIndex(pub(crate) u16);
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId {
    pub(crate) stack_fid: StackFrameId,
    pub(crate) var_ndx: VarIndex,
}

impl std::fmt::Debug for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "var[{}:{}]", self.stack_fid.0, self.var_ndx.0)
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub enum Operand {
    // Evaluates immediately to a constant value.
    Value(Value),
    // Evaluates to the value of the indicated variable.
    Var(VarId),
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
    trace_builder: Option<jit::TraceBuilder>,
    sink: Vec<Value>,
    opts: VMOptions,
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

type NativeFunc = fn(&mut VM, &[Value]) -> Result<Value>;

lazy_static! {
    static ref NATIVE_FUNCS: HashMap<u32, NativeFunc> = {
        let mut map = HashMap::new();
        map.insert(VM::NFID_REQUIRE, nf_require as NativeFunc);
        map
    };
}

fn nf_require(vm: &mut VM, args: &[Value]) -> Result<Value> {
    let arg0 = args.iter().next();
    match arg0 {
        Some(Value::String(path)) => {
            vm.load_module(path)?;
            Ok(Value::Undefined)
        }
        _ => Err(error!("invalid args for require()")),
    }
}

#[derive(Clone)]
struct StackFrameHandle(Rc<RefCell<Box<[Value]>>>);

impl StackFrameHandle {
    fn create(n_slots: usize) -> Self {
        let frame = vec![Value::Undefined; n_slots].into_boxed_slice();
        StackFrameHandle(Rc::new(RefCell::new(frame)))
    }
}

impl VM {
    const NFID_REQUIRE: u32 = 1;
    const NFID_STRING_NEW: u32 = 2;

    pub fn new() -> Self {
        VM {
            include_paths: Vec::new(),
            modules: HashMap::new(),
            trace_builder: None,
            sink: Vec::new(),
            opts: Default::default(),
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
        self.trace_builder.take().and_then(|tb| tb.build())
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

        #[cfg(test)]
        let dump_module = true;
        #[cfg(not(test))]
        let dump_module = self.opts.debug_dump_module;

        if dump_module {
            eprintln!("=== loaded module: {}", key);
            module.dump();
        }

        if flags.tracer_flags.is_some() {
            self.trace_builder = Some(jit::TraceBuilder::new());
        }
        self.run_module_fn(&module, FnId::ROOT_FN, &[], flags)?;
        Ok(())
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

        if flags.tracer_flags.is_some() {
            self.trace_builder = Some(jit::TraceBuilder::new());
        }
        self.run_module_fn(&module, FnId::ROOT_FN, &[], flags)?;
        Ok(())
    }

    fn run_module_fn(
        &mut self,
        module: &Module,
        fnid: FnId,
        args: &[Value],
        flags: InterpreterFlags,
    ) -> Result<Value> {
        let indent = "      | ".repeat(flags.indent_level as usize);

        let should_trace = flags
            .tracer_flags
            .as_ref()
            .map(|tf| tf.start_depth == 0)
            .unwrap_or(false);

        let func = module.fns.get(&fnid).unwrap();
        let instrs = &func.instrs;

        let mut values_buf = vec![Value::Undefined; instrs.len()];
        let stack_frame = StackFrameHandle::create(func.n_slots as usize);

        let get_operand = |results: &Vec<Value>, operand: &Operand| match operand {
            Operand::Value(value) => value.clone(),
            Operand::IID(IID(ndx)) => {
                let value = results[*ndx as usize].clone();
                eprintln!("{}      ↓ {:?} = {:?}", indent, operand, value);
                value
            }
            Operand::Var(var_id @ VarId { stack_fid, var_ndx }) => {
                if stack_fid.0 == 0 {
                    let frame = stack_frame.0.borrow();
                    let var_ndx = var_ndx.0 as usize;
                    let value = frame.get(var_ndx).unwrap().clone();
                    eprintln!("{}      ↓ {:?} = {:?}", indent, var_id, value);
                    value
                } else {
                    todo!("outer stack frames not passed yet!")
                }
            }
        };

        let mut ndx = 0;
        let mut return_value = None;
        while return_value.is_none() {
            if ndx >= instrs.len() {
                break;
            }
            let instr = &instrs[ndx];
            let mut next_ndx = ndx + 1;

            match instr {
                Instr::Const(value) => {
                    values_buf[ndx] = value.clone();
                }
                Instr::Arith { op, a, b } => {
                    let a = get_operand(&mut values_buf, a);
                    let b = get_operand(&mut values_buf, b);

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
                    let value = get_operand(&mut values_buf, operand);
                    self.sink.push(value);
                }
                Instr::Cmp { op, a, b } => {
                    let a = get_operand(&mut values_buf, a);
                    let b = get_operand(&mut values_buf, b);

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
                    let cond_value = get_operand(&mut values_buf, cond);
                    match cond_value {
                        Value::Bool(true) => {
                            next_ndx = dest.0 as usize;
                        }
                        Value::Bool(false) => {} // Just go to the next instruction
                        other => panic!("invalid if condition (not boolean): {:?}", other),
                    }
                }
                Instr::Set { var_id, value } => {
                    let VarId { stack_fid, var_ndx } = var_id;
                    if stack_fid.0 == 0 {
                        let mut frame = stack_frame.0.borrow_mut();
                        let var_ndx = var_ndx.0 as usize;
                        let slot = frame.get_mut(var_ndx).unwrap();
                        *slot = get_operand(&values_buf, value);
                    } else {
                        todo!("Set: outer stack frames not passed yet!")
                    }
                }
                Instr::Nop => {}
                Instr::Not(value) => {
                    let value = get_operand(&values_buf, value);
                    values_buf[ndx] = match value {
                        Value::Bool(bool_val) => Value::Bool(!bool_val),
                        _ => {
                            Value::Bool(false)

                            // panic!("invalid operand for `not` (not boolean): {:?}", other);
                        }
                    };
                }
                Instr::Jmp(dest) => {
                    next_ndx = dest.0 as usize;
                }
                Instr::Return(value) => {
                    return_value = Some(get_operand(&values_buf, value));
                }
                Instr::Call { callee, args } => {
                    let callee = get_operand(&values_buf, callee);
                    match callee {
                        Value::LocalFn(callee_fnid) => {
                            let arg_values: Vec<Value> = args
                                .iter()
                                .map(|oper| get_operand(&values_buf, oper))
                                .collect();

                            eprintln!("{indent}      : call");
                            if should_trace {
                                let tb = self.trace_builder.as_mut().unwrap();
                                tb.enter_function(&args[..], &values_buf[..]);
                            }

                            let ret_val = self.run_module_fn(
                                module,
                                callee_fnid,
                                &arg_values[..],
                                flags.for_call(),
                            )?;

                            values_buf[ndx] = ret_val;
                        }
                        Value::NativeFunction(nfid) => {
                            let arg_values: Vec<Value> = args
                                .iter()
                                .map(|oper| get_operand(&values_buf, oper))
                                .collect();
                            let nf = NATIVE_FUNCS
                                .get(&nfid)
                                .ok_or(error!("no such native function: {nfid}"))?;
                            eprintln!("{indent}      : native call");
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
                    let obj = get_operand(&values_buf, obj);
                    let mut obj = if let Value::Object(obj) = obj {
                        obj
                    } else {
                        panic!("ObjSet: not an object");
                    };
                    let key = get_operand(&values_buf, key);
                    let key = ObjectKey::from_value(&key)
                        .ok_or_else(|| error!("invalid object key: {:?}", key))?;
                    let value = get_operand(&values_buf, value);
                    obj.set(key, value);
                }
                Instr::ObjGet { obj, key } => {
                    let obj = if let Value::Object(obj) = get_operand(&values_buf, obj) {
                        obj
                    } else {
                        panic!("ObjSet: not an object");
                    };
                    let key = get_operand(&values_buf, key);
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
                    let value = get_operand(&values_buf, arg);
                    values_buf[ndx] = value.js_typeof().into();
                }
                Instr::BoolOp { op, a, b } => {
                    let a: bool = get_operand(&values_buf, a).expect_bool()?;
                    let b: bool = get_operand(&values_buf, b).expect_bool()?;
                    let res = match op {
                        BoolOp::And => a && b,
                        BoolOp::Or => a || b,
                    };
                    values_buf[ndx] = Value::Bool(res);
                }
            }

            match instr {
                Instr::Call { .. } => {}
                _ => eprintln!("{}i{:<4} {:?}", indent, ndx, instr),
            }

            if should_trace {
                let trace_builder = self.trace_builder.as_mut().unwrap();
                trace_builder.interpreter_step(&InterpreterStep {
                    values_buf: &values_buf,
                    fnid,
                    func: &func,
                    iid: IID(ndx as u32),
                    next_iid: IID(next_ndx as u32),
                });
            }

            ndx = next_ndx;
        }

        if should_trace {
            if let Some(trace_builder) = self.trace_builder.as_mut() {
                trace_builder.exit_function();
            }
        }
        Ok(return_value.unwrap_or(Value::Undefined))
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

#[derive(Clone)]
pub struct InterpreterFlags {
    pub indent_level: u8,
    pub tracer_flags: Option<TracerFlags>,
}
impl InterpreterFlags {
    fn for_call(&self) -> InterpreterFlags {
        let mut ret = self.clone();
        ret.indent_level += 1;
        if let Some(tf) = &mut ret.tracer_flags {
            if tf.start_depth > 0 {
                tf.start_depth -= 1;
            }
        }
        return ret;
    }
}

#[derive(Clone)]
pub struct TracerFlags {
    pub start_depth: u32,
}

impl Default for InterpreterFlags {
    fn default() -> Self {
        Self {
            indent_level: 0,
            tracer_flags: None,
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
                eprintln!(
                    "  {:4}{:4}: {:?}",
                    if func.loop_heads.contains(&IID(ndx as u32)) {
                        ">>"
                    } else {
                        ""
                    },
                    ndx,
                    instr
                );
            }
        }
        eprintln!("---");
    }
}

pub struct Function {
    instrs: Box<[Instr]>,
    loop_heads: HashSet<IID>,
    n_slots: u16,
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
        self.loop_heads.contains(&iid)
    }
}

fn find_loop_heads(instrs: &[Instr]) -> HashSet<IID> {
    let mut heads = HashSet::new();
    for (ndx, inst) in instrs.iter().enumerate() {
        match inst {
            Instr::Jmp(IID(dest_ndx))
            | Instr::JmpIf {
                dest: IID(dest_ndx),
                ..
            } if *dest_ndx as usize <= ndx => {
                heads.insert(IID(*dest_ndx));
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

    #[test]
    fn test_for_in() {
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
