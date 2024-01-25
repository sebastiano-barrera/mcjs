use std::collections::{HashMap, HashSet};

use swc_atoms::JsWord;

use super::js_to_past;

use crate::bytecode::{self, Instr, Literal};
use crate::common::{Context, Error, Result};
use crate::error;

pub fn compile_function<'a>(
    module_builder: &'a mut ModuleBuilder,
    globals: &'a HashSet<JsWord>,
    captures: Vec<js_to_past::DeclName>,
    func: &js_to_past::Function,
) -> Result<bytecode::LocalFnId> {
    let lfnid = module_builder.gen_id();
    let mut fnb = FnBuilder::new(lfnid, globals, module_builder);

    let mut names = HashMap::new();
    for (cap_ndx, cap_decl) in captures.into_iter().enumerate() {
        let cap_ndx = cap_ndx.try_into().unwrap();
        let reg = fnb.regs.gen();
        fnb.instrs
            .push(Instr::LoadCapture(reg, bytecode::CaptureIndex(cap_ndx)));
        names.insert(cap_decl, reg);
    }
    fnb.scopes.push(Scope {
        names,
        ..Default::default()
    });

    compile_block(&mut fnb, &func.body)?;

    fnb.scopes.pop().unwrap();
    assert_eq!(fnb.scopes.len(), 0);

    let function = fnb.build(func.span);
    module_builder.put_fn(lfnid, function);

    Ok(lfnid)
}

fn compile_stmt(fnb: &mut FnBuilder, stmt: &js_to_past::Stmt) -> Result<Option<bytecode::VReg>> {
    let iid_start = fnb.instrs.peek_iid();

    let ret = match &stmt.op {
        js_to_past::StmtOp::Block(block) => compile_block(fnb, block),
        js_to_past::StmtOp::Break(block_id) => {
            let iid = fnb.instrs.peek_iid();
            fnb.instrs.push(Instr::Breakpoint);

            let scope: &mut Scope = get_block_scope(&mut fnb.scopes, *block_id);
            scope.deferred_break.push(iid);

            Ok(None)
        }
        js_to_past::StmtOp::Continue(block_id) => {
            compile_unshare(fnb, block_id);

            let iid = fnb.instrs.peek_iid();
            fnb.instrs.push(Instr::Breakpoint);

            let scope: &mut Scope = get_block_scope(&mut fnb.scopes, *block_id);
            scope.deferred_continue.push(iid);

            Ok(None)
        }
        js_to_past::StmtOp::Unshare(block_id) => {
            compile_unshare(fnb, block_id);
            Ok(None)
        }
        js_to_past::StmtOp::If { test, cons, alt } => {
            let test = compile_stmt(fnb, test)?.expect("malformed PAST: if's test has no value");
            let jmpif = fnb.instrs.reserve();

            let alt_value = compile_stmt(fnb, alt)?;
            let out_reg = alt_value.unwrap_or_else(|| fnb.regs.gen());
            let alt_exit = fnb.instrs.reserve();

            let cons_iid = fnb.instrs.peek_iid();
            let cons_value = compile_stmt(fnb, cons)?;
            match cons_value {
                Some(reg) => fnb.instrs.push(Instr::Copy {
                    dst: out_reg,
                    src: reg,
                }),
                None => fnb.instrs.push(Instr::LoadUndefined(out_reg)),
            }

            let exit_target = fnb.instrs.peek_iid();
            *fnb.instrs.get_mut(alt_exit) = Instr::Jmp(exit_target);
            *fnb.instrs.get_mut(jmpif) = Instr::JmpIf {
                cond: test,
                dest: cons_iid,
            };

            Ok(Some(out_reg))
        }
        js_to_past::StmtOp::Undefined => {
            let reg = fnb.regs.gen();
            fnb.instrs.push(Instr::LoadUndefined(reg));
            Ok(Some(reg))
        }
        js_to_past::StmtOp::Null => {
            let reg = fnb.regs.gen();
            fnb.instrs.push(Instr::LoadNull(reg));
            Ok(Some(reg))
        }
        js_to_past::StmtOp::This => {
            let reg = fnb.regs.gen();
            fnb.instrs.push(Instr::LoadThis(reg));
            Ok(Some(reg))
        }
        js_to_past::StmtOp::Read(name) => {
            let reg = compile_read(fnb, name);
            Ok(Some(reg))
        }
        js_to_past::StmtOp::AssignParam(dest, arg_ndx) => {
            let value = fnb.regs.gen();
            fnb.instrs
                .push(Instr::LoadArg(value, bytecode::ArgIndex(*arg_ndx)));
            compile_write(fnb, dest, value);
            Ok(None)
        }
        js_to_past::StmtOp::Assign(dest, value) => {
            let value = compile_expr(fnb, value)?;
            compile_write(fnb, dest, value);
            Ok(None)
        }
        js_to_past::StmtOp::Unary(op, arg) => {
            let dest = fnb.regs.gen();
            let arg = compile_expr(fnb, arg)?;
            match op {
                swc_ecma_ast::UnaryOp::Minus => {
                    fnb.instrs.push(Instr::UnaryMinus { dest, arg });
                }
                swc_ecma_ast::UnaryOp::Plus => {}
                swc_ecma_ast::UnaryOp::Bang => {
                    fnb.instrs.push(Instr::BoolNot { dest, arg });
                }
                swc_ecma_ast::UnaryOp::TypeOf => {
                    fnb.instrs.push(Instr::TypeOf { dest, arg });
                }
                swc_ecma_ast::UnaryOp::Tilde
                | swc_ecma_ast::UnaryOp::Void
                | swc_ecma_ast::UnaryOp::Delete => panic!("unsupported unary op: {:?}", op),
            }
            Ok(Some(dest))
        }
        js_to_past::StmtOp::Binary(op, left, right) => {
            let dest = fnb.regs.gen();
            let left = compile_expr(fnb, left)?;
            let right = compile_expr(fnb, right)?;

            let instr = match op {
                swc_ecma_ast::BinaryOp::Add => Instr::OpAdd(dest, left, right),
                swc_ecma_ast::BinaryOp::Sub => Instr::ArithSub(dest, left, right),
                swc_ecma_ast::BinaryOp::Mul => Instr::ArithMul(dest, left, right),
                swc_ecma_ast::BinaryOp::Div => Instr::ArithDiv(dest, left, right),
                swc_ecma_ast::BinaryOp::Lt => Instr::CmpLT(dest, left, right),
                swc_ecma_ast::BinaryOp::LtEq => Instr::CmpLE(dest, left, right),
                swc_ecma_ast::BinaryOp::Gt => Instr::CmpGT(dest, left, right),
                swc_ecma_ast::BinaryOp::GtEq => Instr::CmpGE(dest, left, right),
                swc_ecma_ast::BinaryOp::EqEqEq => Instr::CmpEQ(dest, left, right),
                swc_ecma_ast::BinaryOp::NotEqEq => Instr::CmpNE(dest, left, right),

                // TODO TODO TODO This does not implement any of the 'wat' semantics of JavaScript
                // See https://www.destroyallsoftware.com/talks/wat
                swc_ecma_ast::BinaryOp::EqEq => Instr::CmpEQ(dest, left, right),
                swc_ecma_ast::BinaryOp::NotEq => Instr::CmpNE(dest, left, right),

                swc_ecma_ast::BinaryOp::LogicalAnd => Instr::BoolOpAnd(dest, left, right),
                swc_ecma_ast::BinaryOp::LogicalOr => Instr::BoolOpOr(dest, left, right),

                swc_ecma_ast::BinaryOp::InstanceOf => Instr::IsInstanceOf(dest, left, right),
                _ => panic!("unsupported binary op: {:?}", op),
            };

            fnb.instrs.push(instr);
            Ok(Some(dest))
        }
        js_to_past::StmtOp::StringLiteral(str_lit) => Ok(Some(compile_load_const(
            fnb,
            bytecode::Literal::String(str_lit.to_string()),
        ))),
        js_to_past::StmtOp::NumberLiteral(num_lit) => Ok(Some(compile_load_const(
            fnb,
            bytecode::Literal::Number(*num_lit),
        ))),
        js_to_past::StmtOp::BoolLiteral(bool_lit) => Ok(Some(compile_load_const(
            fnb,
            bytecode::Literal::Bool(*bool_lit),
        ))),
        js_to_past::StmtOp::ArrayCreate => {
            let array = fnb.regs.gen();

            let constructor: bytecode::VReg = compile_read_global(fnb, "Array");
            let this = fnb.regs.gen();
            fnb.instrs.push(Instr::LoadUndefined(this));
            fnb.instrs.push(Instr::Call {
                return_value: array,
                this,
                callee: constructor,
            });
            compile_new(fnb, array, constructor);

            Ok(Some(array))
        }
        js_to_past::StmtOp::ArrayPush(arr, value) => {
            let arr = compile_read(fnb, arr);
            let value = compile_expr(fnb, value)?;
            fnb.instrs.push(Instr::ArrayPush { arr, value });
            Ok(None)
        }
        js_to_past::StmtOp::ObjectCreate => {
            let obj = fnb.regs.gen();
            fnb.instrs.push(Instr::ObjCreateEmpty(obj));
            Ok(Some(obj))
        }
        js_to_past::StmtOp::ObjectGet { obj, key } => {
            let obj = compile_expr(fnb, obj.as_ref())?;
            let key = compile_expr(fnb, key.as_ref())?;
            let dest = fnb.regs.gen();
            fnb.instrs.push(Instr::ObjGet { dest, obj, key });
            Ok(Some(dest))
        }
        js_to_past::StmtOp::ObjectSet { obj, key, value } => {
            let obj = compile_expr(fnb, obj.as_ref())?;
            let key = compile_expr(fnb, key.as_ref())?;
            let value = compile_expr(fnb, value.as_ref())?;
            fnb.instrs.push(Instr::ObjSet { obj, key, value });
            Ok(None)
        }
        js_to_past::StmtOp::CreateClosure { func } => {
            let mut cap_names = Vec::new();
            let mut cap_regs = Vec::new();

            for name in func.unbound_names.iter() {
                let name = js_to_past::DeclName::Js(name.clone());
                match resolve_name(fnb, &name) {
                    Loc::Reg(reg) => {
                        cap_names.push(name.clone());
                        cap_regs.push(reg);
                    }
                    Loc::Global(_) => {
                        // skip: the inner function will *also* see the name as global and
                        // compile the appropriate access to globalThis
                    }
                }
            }

            let lfnid = compile_function(fnb.module_builder, fnb.globals, cap_names, func)?;
            let dest = fnb.regs.gen();
            fnb.instrs.push(Instr::ClosureNew {
                dest,
                fnid: lfnid,
                forced_this: None,
            });

            for reg in cap_regs {
                fnb.instrs.push(Instr::ClosureAddCapture(reg));
            }

            Ok(Some(dest))
        }
        js_to_past::StmtOp::Call {
            is_new,
            callee,
            args,
        } => {
            // Some things expressed in the `f(...)` call syntax are not actually calls to
            // anything, but have a special meaning
            match &callee.as_ref().op {
                js_to_past::StmtOp::Read(js_to_past::DeclName::Js(name)) if name == "sink" => {
                    for arg in args {
                        let var = compile_expr(fnb, &arg)?;
                        fnb.instrs.push(Instr::PushToSink(var));
                    }

                    let ret = fnb.regs.gen();
                    compile_load_const(fnb, Literal::Undefined);
                    Ok(Some(ret))
                }
                js_to_past::StmtOp::Read(js_to_past::DeclName::Js(name)) if name == "require" => {
                    if args.len() != 1 {
                        return Err(error!("`require` takes a single argument only"));
                    }
                    let import_path = compile_expr(fnb, &args[0])?;

                    let ret = fnb.regs.gen();
                    fnb.instrs.push(Instr::ImportModule(ret, import_path));
                    Ok(Some(ret))
                }
                js_to_past::StmtOp::Read(js_to_past::DeclName::Js(name)) if name == "eval" => {
                    return Err(error!("`eval` not supported"));
                }
                _ => {
                    let mut arg_regs = Vec::new();
                    for arg in args {
                        let reg = compile_expr(fnb, arg)?;
                        arg_regs.push(reg);
                    }

                    let (this, callee) = match &callee.as_ref().op {
                        js_to_past::StmtOp::ObjectGet { obj, key } => {
                            let obj = compile_expr(fnb, obj)?;
                            let key = compile_expr(fnb, key)?;
                            let callee = fnb.regs.gen();
                            fnb.instrs.push(Instr::ObjGet {
                                dest: callee,
                                obj,
                                key,
                            });
                            (obj, callee)
                        }
                        _ => {
                            let this = fnb.regs.gen();
                            let callee = compile_expr(fnb, callee)?;
                            fnb.instrs.push(Instr::LoadUndefined(this));
                            (this, callee)
                        }
                    };

                    let return_value = fnb.regs.gen();
                    fnb.instrs.push(Instr::Call {
                        return_value,
                        this,
                        callee,
                    });

                    for reg in arg_regs {
                        fnb.instrs.push(Instr::CallArg(reg));
                    }

                    if *is_new {
                        compile_new(fnb, return_value, callee);
                    }

                    Ok(Some(return_value))
                }
            }
        }
        js_to_past::StmtOp::Return(arg) => {
            let value = compile_expr(fnb, arg)?;
            fnb.instrs.push(Instr::Return(value));
            Ok(None)
        }

        js_to_past::StmtOp::Throw(arg) => {
            let arg = compile_expr(fnb, arg)?;
            fnb.instrs.push(Instr::Throw(arg));
            Ok(None)
        }
        js_to_past::StmtOp::GetCurrentException => {
            let dest = fnb.regs.gen();
            fnb.instrs.push(Instr::GetCurrentException(dest));
            Ok(Some(dest))
        }
        js_to_past::StmtOp::Try {
            main_block,
            handler_block,
            finalizer_block,
        } => {
            // TODO Allow the catch clause to access the exception object
            // We reunite the cases by *always* having a catch clause (it will run the finalizer)
            let push_handler_instr = fnb.instrs.reserve();
            compile_stmt(fnb, main_block)?;
            fnb.instrs.push(Instr::PopExcHandler);
            let jmp_after_try = fnb.instrs.reserve();

            let handler_start = fnb.instrs.peek_iid();
            compile_stmt(fnb, handler_block)?;

            let finalizer_start = fnb.instrs.peek_iid();
            compile_stmt(fnb, finalizer_block)?;
            let cur_exc = fnb.regs.gen();
            fnb.instrs.push(Instr::GetCurrentException(cur_exc));
            fnb.instrs.push(Instr::Throw(cur_exc));

            *fnb.instrs.get_mut(push_handler_instr) = Instr::PushExcHandler(handler_start);
            *fnb.instrs.get_mut(jmp_after_try) = Instr::Jmp(finalizer_start);
            Ok(None)
        }
        js_to_past::StmtOp::Debugger => {
            fnb.instrs.push(Instr::Breakpoint);
            Ok(None)
        }
    };

    if !stmt.span.is_dummy() {
        let iid_end = fnb.instrs.peek_iid();
        fnb.module_builder
            .breakable_ranges
            .push(bytecode::BreakRange {
                lo: stmt.span.lo,
                hi: stmt.span.hi,
                local_fnid: fnb.lfnid,
                iid_start,
                iid_end,
            });
    }

    ret
}

fn compile_unshare(fnb: &mut FnBuilder<'_>, block_id: &js_to_past::BlockID) {
    let mut block_found = false;

    for scope in fnb.scopes.iter().rev() {
        for reg in scope.names.values() {
            fnb.instrs.push(Instr::Unshare(*reg));
        }
        if scope.block_id == Some(*block_id) {
            block_found = true;
            break;
        }
    }

    if !block_found {
        panic!(
            "malformed PAST: Unshare({:?}): block ID not found",
            block_id
        );
    }
}

fn get_block_scope(scopes: &mut [Scope], block_id: js_to_past::BlockID) -> &mut Scope {
    scopes
        .iter_mut()
        .rev()
        .find(|scope| scope.block_id == Some(block_id))
        .unwrap()
}

fn compile_read(fnb: &mut FnBuilder<'_>, name: &js_to_past::DeclName) -> bytecode::VReg {
    match resolve_name(fnb, name) {
        Loc::Reg(reg) => reg,
        Loc::Global(name) => {
            let global_this = fnb.regs.gen();
            let dest = fnb.regs.gen();

            fnb.instrs.push(Instr::GetGlobalThis(global_this));
            let key = compile_load_const(fnb, Literal::JsWord(name));
            fnb.instrs.push(Instr::ObjGet {
                dest,
                obj: global_this,
                key,
            });

            dest
        }
    }
}

fn compile_write(fnb: &mut FnBuilder, name: &js_to_past::DeclName, value: bytecode::VReg) {
    match resolve_name(fnb, name) {
        Loc::Reg(reg) => {
            // sad, this causes a bunch of extra copies in the bytecode
            fnb.instrs.push(Instr::Copy {
                dst: reg,
                src: value,
            });
        }
        Loc::Global(name) => {
            let global_this = fnb.regs.gen();
            let value = fnb.regs.gen();

            fnb.instrs.push(Instr::GetGlobalThis(global_this));
            let key = compile_load_const(fnb, Literal::JsWord(name));
            fnb.instrs.push(Instr::ObjSet {
                obj: global_this,
                key,
                value,
            });
        }
    }
}

fn resolve_name(fnb: &mut FnBuilder, name: &js_to_past::DeclName) -> Loc {
    let local_reg = fnb
        .scopes
        .iter()
        .rev()
        .find_map(|scope| scope.names.get(name).copied().map(Loc::Reg));

    if let Some(reg) = local_reg {
        return reg;
    }

    if let js_to_past::DeclName::Js(name) = name {
        if fnb.globals.contains(name) {
            return Loc::Global(name.clone());
        }
    }
    panic!("malformed PAST: undeclared name: {:?}", name)
}

fn compile_new(fnb: &mut FnBuilder<'_>, ret: bytecode::VReg, constructor: bytecode::VReg) {
    let key = compile_load_const(fnb, Literal::String("prototype".into()));
    // reusing register `constructor` for the prototype
    fnb.instrs.push(Instr::ObjGet {
        dest: constructor,
        obj: constructor,
        key,
    });

    let key = compile_load_const(fnb, Literal::String("__proto__".into()));
    fnb.instrs.push(Instr::ObjSet {
        obj: ret,
        key,
        value: constructor,
    });

    let key = compile_load_const(fnb, Literal::String("constructor".into()));
    fnb.instrs.push(Instr::ObjSet {
        obj: ret,
        key,
        value: constructor,
    });
}

fn compile_read_global(fnb: &mut FnBuilder, name: &str) -> bytecode::VReg {
    // TODO Avoid repeating the GetGlobalThis!
    let global_this = fnb.regs.gen();
    fnb.instrs.push(Instr::GetGlobalThis(global_this));

    let key = compile_load_const(fnb, Literal::String(name.to_string()));
    let dest = fnb.regs.gen();
    fnb.instrs.push(Instr::ObjGet {
        dest,
        obj: global_this,
        key,
    });

    dest
}

fn compile_load_const(fnb: &mut FnBuilder, lit: Literal) -> bytecode::VReg {
    let dest = fnb.regs.gen();
    // Avoid string clone?
    let const_ndx = fnb.consts.push(lit);
    fnb.instrs.push(Instr::LoadConst(dest, const_ndx));

    dest
}

fn compile_expr(fnb: &mut FnBuilder, value: &js_to_past::Stmt) -> Result<bytecode::VReg> {
    let vreg = compile_stmt(fnb, value)?.expect("malformed PAST: expression has no value");
    Ok(vreg)
}

fn compile_block(fnb: &mut FnBuilder, block: &js_to_past::Block) -> Result<Option<bytecode::VReg>> {
    let names = block
        .decls
        .iter()
        .map(|decl| (decl.name.clone(), fnb.regs.gen()))
        .collect();
    fnb.scopes.push(Scope {
        block_id: Some(block.id),
        names,
        ..Default::default()
    });
    let iid_start = fnb.instrs.peek_iid();

    let mut last_reg = None;
    for stmt in &block.stmts {
        last_reg = compile_stmt(fnb, stmt)?;
    }

    let scope = fnb.scopes.pop().unwrap();

    let iid_end = fnb.instrs.peek_iid();
    for iid in scope.deferred_break {
        *fnb.instrs.get_mut(iid) = Instr::Jmp(iid_end);
    }
    for iid in scope.deferred_continue {
        *fnb.instrs.get_mut(iid) = Instr::Jmp(iid_start);
    }

    Ok(last_reg)
}

pub struct ModuleBuilder {
    fns: HashMap<bytecode::LocalFnId, bytecode::Function>,
    next_lfnid: u16,
    breakable_ranges: Vec<bytecode::BreakRange>,
}
impl ModuleBuilder {
    pub fn new(min_fnid: u16) -> Self {
        ModuleBuilder {
            fns: HashMap::new(),
            next_lfnid: min_fnid,
            breakable_ranges: Vec::new(),
        }
    }

    fn gen_id(&mut self) -> bytecode::LocalFnId {
        let lfnid = bytecode::LocalFnId(self.next_lfnid);
        self.next_lfnid += 1;
        lfnid
    }

    fn put_fn(&mut self, lfnid: bytecode::LocalFnId, function: bytecode::Function) {
        self.fns.insert(lfnid, function);
    }

    pub fn build(self, root_fnid: bytecode::LocalFnId) -> super::CompiledModule {
        super::CompiledModule {
            root_fnid,
            functions: self.fns,
            breakable_ranges: self.breakable_ranges, /* TODO */
        }
    }
}

struct FnBuilder<'a> {
    instrs: InstrBuffer,
    consts: ConstsBuffer,
    regs: RegGen,
    is_strict_mode: bool,
    scopes: Vec<Scope>,
    lfnid: bytecode::LocalFnId,
    globals: &'a HashSet<JsWord>,
    module_builder: &'a mut ModuleBuilder,
}

#[derive(Default)]
struct Scope {
    block_id: Option<js_to_past::BlockID>,
    names: HashMap<js_to_past::DeclName, bytecode::VReg>,

    /// `Break` instructions referring to this block. They need to be fixed after the block's
    /// start/end instructions are known.
    deferred_break: Vec<bytecode::IID>,

    /// `Continue` instructions referring to this block. They need to be fixed after the
    /// block's start/end instructions are known.
    deferred_continue: Vec<bytecode::IID>,
}

enum Loc {
    Reg(bytecode::VReg),
    Global(JsWord),
}

impl<'a> FnBuilder<'a> {
    fn new(
        lfnid: bytecode::LocalFnId,
        globals: &'a HashSet<JsWord>,
        module_builder: &'a mut ModuleBuilder,
    ) -> FnBuilder<'a> {
        FnBuilder {
            instrs: InstrBuffer::new(),
            consts: ConstsBuffer::new(),
            regs: RegGen::new(),
            is_strict_mode: false,
            scopes: Vec::new(),
            lfnid,
            globals,
            module_builder,
        }
    }

    fn build(self, span: swc_common::Span) -> bytecode::Function {
        // assert!(
        //     self.pending_break_instrs.is_empty(),
        //     "bytecode compiler bug: the function is over, but some break instructions were not placed yet"
        // );

        bytecode::FunctionBuilder {
            instrs: self.instrs.build(),
            consts: self.consts.build(),
            n_regs: self.regs.count(),
            // TODO TODO This is all yet to be implemented:
            ident_history: Vec::new(),
            trace_anchors: HashMap::new(),
            is_strict_mode: self.is_strict_mode,
            span,
        }
        .build()
    }
}

#[derive(Default)]
struct InstrBuffer {
    instrs: Vec<Instr>,
}
impl InstrBuffer {
    fn new() -> Self {
        Self::default()
    }
    fn push(&mut self, instr: Instr) {
        self.instrs.push(instr);
    }
    fn reserve(&mut self) -> bytecode::IID {
        let iid = self.peek_iid();
        self.instrs.push(Instr::Nop);
        iid
    }
    fn peek_iid(&mut self) -> bytecode::IID {
        bytecode::IID(self.instrs.len().try_into().unwrap())
    }
    fn get_mut(&mut self, iid: bytecode::IID) -> &mut Instr {
        self.instrs.get_mut(iid.0 as usize).unwrap()
    }

    fn build(self) -> Box<[Instr]> {
        self.instrs.into_boxed_slice()
    }
}

struct ConstsBuffer {
    consts: Vec<Literal>,
}
impl ConstsBuffer {
    fn new() -> Self {
        ConstsBuffer { consts: Vec::new() }
    }
    fn push(&mut self, lit: Literal) -> bytecode::ConstIndex {
        let index = self.consts.len();
        self.consts.push(lit);
        bytecode::ConstIndex(index.try_into().unwrap())
    }
    fn build(self) -> Box<[Literal]> {
        self.consts.into_boxed_slice()
    }
}

#[derive(Default)]
struct RegGen {
    n_regs: u8,
}
impl RegGen {
    fn new() -> Self {
        RegGen { n_regs: 0 }
    }

    fn gen(&mut self) -> bytecode::VReg {
        let reg = bytecode::VReg(self.n_regs.try_into().unwrap());
        self.n_regs += 1;
        reg
    }

    fn count(&self) -> u8 {
        self.n_regs
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, rc::Rc};

    use swc_common::SourceMap;

    use crate::{bytecode, bytecode_compiler::CompiledModule};

    #[test]
    fn test_global_var() {
        let CompiledModule {
            functions,
            root_fnid,
            ..
        } = quick_compile(
            "
                (function() {
                    (function() {
                        (function() {
                            let someLocalVar = 23;
                            console.log(someGlobalVar, someLocalVar)
                            someGlobalVar = 'lol'
                        })()
                    })()
                })()
                "
            .to_string(),
        );

        for (lfnid, func) in functions.iter() {
            println!(
                "== function {:?} {}",
                lfnid,
                if lfnid == &root_fnid { "[root]" } else { "" }
            );

            func.dump();
        }
    }

    #[test]
    fn test_let_and_var() {
        quick_compile(
            r#"(function() {
                    function f() { x = 1 }
                    let x = 5;
                    return f()
                })()
                "#
            .to_string(),
        );
    }

    #[test]
    fn test_function_before_let() {
        let compiled_module = quick_compile(
            "
                (function() {
                    function f() { x = 3 }
                    let x = 2;
                })()
                "
            .to_string(),
        );

        insta::assert_snapshot!(dump_functions(&compiled_module.functions));
    }

    #[test]
    fn test_func_decl() {
        let compiled_module = quick_compile(
            "function myFunction() {
                    return 3
                }"
            .to_string(),
        );

        insta::assert_snapshot!(dump_functions(&compiled_module.functions));
    }

    fn dump_functions(functions: &HashMap<bytecode::LocalFnId, bytecode::Function>) -> String {
        let mut ids: Vec<_> = functions.keys().copied().collect();
        ids.sort();

        let mut buf = String::new();
        for fnid in ids {
            use std::fmt::Write;

            let func = functions.get(&fnid).unwrap();
            writeln!(buf, "fn {:?}", fnid).unwrap();
            for (ndx, literal) in func.consts().iter().enumerate() {
                writeln!(buf, "  k{} {:?}", ndx, literal).unwrap();
            }
            for instr in func.instrs() {
                writeln!(buf, "  {:?}", instr).unwrap();
            }
        }

        buf
    }

    fn quick_compile(src: String) -> CompiledModule {
        let source_map = Rc::new(SourceMap::default());
        let swc_ast =
            super::super::parse_file("<input>".to_string(), src, source_map).expect("parse error");
        let past_function = super::js_to_past::compile_script(swc_ast);

        let mut module_builder = super::ModuleBuilder::new(0);
        let globals = past_function.unbound_names.iter().cloned().collect();
        let root_lfnid =
            super::compile_function(&mut module_builder, &globals, Vec::new(), &past_function)
                .expect("past->bytecode compile error");

        module_builder.build(root_lfnid)
    }
}
