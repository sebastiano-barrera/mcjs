use std::collections::{HashMap, HashSet};

use swc_atoms::JsWord;
use swc_common::{sync::Lrc, SourceMap, Span};
use swc_ecma_ast::VarDeclOrPat;
use swc_ecma_ast::{AssignOp, BinaryOp, Decl, Function, Lit, Pat, UpdateOp};

use crate::bytecode::{self, ArithOp, BoolOp, CmpOp, FnId, Instr, Literal, NativeFnId, VReg, IID};
pub use crate::common::{Context, Error, Result};
use crate::error;
use crate::util::Mask;

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
}

pub type NativeFnMap = HashMap<JsWord, NativeFnId>;

pub trait Loader {
    fn get_module_id(&mut self, filename: &str) -> bytecode::ModuleId;
    fn read_source(&self, module_id: bytecode::ModuleId) -> String;
}

pub struct BuilderParams {
    pub native_fns: NativeFnMap,
    pub loader: Box<dyn Loader>,
}

pub struct Builder {
    native_fns: NativeFnMap,
    loader: Box<dyn Loader>,
    fns: HashMap<FnId, FnBuilder>,
    fn_stack: Vec<FnBuilder>,
    next_fnid: u32,
    next_module_id: u16,

    rootfn_of_module: HashMap<bytecode::ModuleId, FnId>,
}

impl BuilderParams {
    pub fn to_builder(self) -> Builder {
        Builder {
            native_fns: self.native_fns,
            loader: self.loader,
            fns: HashMap::new(),
            fn_stack: Vec::new(),
            next_fnid: 1,
            next_module_id: 1,

            rootfn_of_module: HashMap::new(),
        }
    }
}

impl Builder {
    pub fn compile_file(&mut self, filename: String) -> Result<bytecode::ModuleId> {
        use crate::common::Context;

        // TODO(performance) Does it make sense to mmap the input file?
        let module_id = self.loader.get_module_id(&filename);
        if self.rootfn_of_module.contains_key(&module_id) {
            // The module had already been compiled
            return Ok(module_id);
        }

        let content = self.loader.read_source(module_id);

        let (source_map, ast_module) = parse_file(filename.clone(), content)
            .with_context(error!("while parsing file: {filename}"))?;

        let res = compile_module(self, &ast_module)
            .with_context(error!("while compiling module: {filename}"));
        match res {
            Ok(root_fnid) => {
                let prev = self.rootfn_of_module.insert(module_id, root_fnid);
                assert!(prev.is_none(), "duplicate module id");
            }
            Err(err) => {
                eprintln!("\nbytecode compiler error: {}\n", err.message(&source_map))
            }
        }

        Ok(module_id)
    }

    fn ensure_imported(&mut self, path: &JsWord) -> Result<bytecode::ModuleId> {
        self.compile_file(path.to_string())
    }
}

struct FnBuilder {
    fnid: FnId,
    scopes: Vec<Scope>,
    instrs: Vec<Instr>,
    consts: Vec<Literal>,
    trace_anchors: HashMap<IID, bytecode::TraceAnchor>,
    next_vreg: VReg,
    captures: Vec<JsWord>,
}

#[derive(Debug)]
struct Scope {
    // TODO Take advantage of identifier interning!
    // TODO After introducing VRegs, this could become a Vec
    vars: HashMap<String, VReg>,
}
impl Scope {
    fn new() -> Self {
        Scope {
            vars: HashMap::new(),
        }
    }
}

impl FnBuilder {
    fn new(id: FnId) -> Self {
        FnBuilder {
            fnid: id,
            scopes: vec![Scope::new()],
            instrs: Vec::new(),
            consts: Vec::new(),
            trace_anchors: HashMap::new(),
            captures: Vec::new(),
            next_vreg: 0,
        }
    }

    fn build(self) -> bytecode::Function {
        bytecode::Function::new(
            self.instrs.into_boxed_slice(),
            self.consts.into_boxed_slice(),
            self.trace_anchors,
        )
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop().unwrap();
    }

    fn inner_scope(&self) -> &Scope {
        self.scopes.last().unwrap()
    }
    fn inner_scope_mut(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap()
    }

    fn peek_iid(&self) -> IID {
        IID(self.instrs.len() as _)
    }

    fn emit(&mut self, instr: Instr) -> IID {
        let iid = self.peek_iid();
        self.instrs.push(instr);
        iid
    }

    // Assign a new upvalue index to the given variable, and emits the necessary "GetCapture"
    // instruction.
    // Subsequent calls to get_var will return  the IID for the GetCapture instruction.
    fn set_var_captured(&mut self, name: JsWord) -> VReg {
        assert!(!self.captures.contains(&name));
        self.captures.push(name.clone());

        let cap_ndx = self.captures.len().try_into().expect("too many captures!");
        self.emit(Instr::LoadCapture(cap_ndx));
        let var = self.next_vreg();
        self.define_var(name, var);

        var
    }

    fn get_var(&self, sym: &JsWord) -> Option<VReg> {
        let name = String::from_utf8_lossy(sym.as_bytes());

        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.vars.get(name.as_ref()))
            .copied()
    }

    fn define_var(&mut self, name: JsWord, vreg: VReg) {
        let name = String::from_utf8(name.as_bytes().to_owned())
            .expect("only UTF-8 identifiers are supporeted!");
        let scope = self.inner_scope_mut();
        let prev = scope.vars.insert(name.clone(), vreg);
        if prev.is_some() {
            panic!(
                "definition of var `{}` shadows previous definition (compiler limitation)",
                name
            );
        }
    }

    fn next_vreg(&mut self) -> VReg {
        let vreg = self.next_vreg;
        self.next_vreg += 1;
        vreg
    }
}

impl Builder {
    pub fn build(mut self) -> bytecode::Codebase {
        assert!(self.fn_stack.is_empty());

        let fns = self
            .fns
            .drain()
            .map(|(fn_id, fn_builder)| (fn_id, fn_builder.build()))
            .collect();

        bytecode::Codebase::new(fns, self.rootfn_of_module)
    }

    fn reserve(&mut self) -> IID {
        self.emit(Instr::Nop)
    }

    fn cur_fnb(&mut self) -> &mut FnBuilder {
        self.fn_stack.last_mut().expect("no FnBuilder!")
    }

    fn emit(&mut self, instr: Instr) -> IID {
        self.cur_fnb().emit(instr)
    }

    fn push_const(&mut self, value: bytecode::Literal) -> IID {
        match value {
            bytecode::Literal::Null => self.emit(Instr::LoadNull),
            bytecode::Literal::Undefined => self.emit(Instr::LoadUndefined),
            _ => {
                let fnb = self.cur_fnb();
                fnb.consts.push(value);
                let ndx = fnb.consts.len() - 1;
                let ndx = TryFrom::try_from(ndx).expect("too many constants!");
                self.emit(Instr::LoadConst(ndx))
            }
        }
    }

    fn peek_iid(&mut self) -> IID {
        self.cur_fnb().peek_iid()
    }

    fn get_mut(&mut self, iid: IID) -> Option<&mut Instr> {
        self.cur_fnb().instrs.get_mut(iid.0 as usize)
    }

    fn define_var(&mut self, name: JsWord, vreg: VReg) {
        self.cur_fnb().define_var(name, vreg)
    }

    fn get_var(&mut self, sym: &JsWord) -> Result<VReg> {
        // self.fn_stack.last_mut().expect("no FnBuilder!")
        let found_iid = self
            .fn_stack
            .iter_mut()
            .rev()
            .enumerate()
            .find_map(|(ndx, fnb)| fnb.get_var(sym).map(|vreg| (ndx, vreg)))
            .or_else(|| {
                let nfid = *self.native_fns.get(sym)?;
                self.cur_fnb().emit(Instr::GetNativeFn(nfid));
                let vreg = self.accu_to_vreg();
                Some((0, vreg))
            });

        match found_iid {
            // Hell yeah, found it in the current function, we have a "simple" IID
            Some((0, iid)) => Ok(iid),
            Some((_, _)) => {
                // Found it in one of the enclosing functions.  We'll have to add a capture, but we
                // know the variable is there
                let iid = self.cur_fnb().set_var_captured(sym.clone());
                Ok(iid)
            }
            None => Err(error!("unbound symbol: {}", sym)),
        }
    }

    fn accu_to_vreg(&mut self) -> VReg {
        let fnb = &mut self.cur_fnb();
        let vreg = fnb.next_vreg();
        self.emit(Instr::StoAR(vreg));
        vreg
    }

    fn start_function(&mut self, name: Option<JsWord>, params: &[swc_ecma_ast::Param]) {
        let fnid = FnId(self.next_fnid);
        self.next_fnid += 1;

        self.fn_stack.push(FnBuilder::new(fnid));

        for (param_ndx, param) in params.iter().enumerate() {
            let param_ndx = TryFrom::try_from(param_ndx).expect("too many parameters!");

            if !param.decorators.is_empty() {
                panic!("unsupported: decorators on function parameters");
            }

            match &param.pat {
                Pat::Ident(ident) => {
                    self.emit(Instr::LoadArg(param_ndx));
                    let vreg = self.accu_to_vreg();
                    self.define_var(ident.sym.clone(), vreg);
                }
                other => unsupported_node!(other),
            }
        }
    }

    fn end_function(&mut self) -> FnId {
        let fnb = self.fn_stack.pop().expect("no FnBuilder!");
        let fnid = fnb.fnid;
        self.fns.insert(fnid, fnb);
        fnid
    }

    fn place_trace_anchor(&mut self, trace_id: String) {
        let fnb = self.cur_fnb();
        let iid = fnb.peek_iid();
        fnb.trace_anchors
            .insert(iid, bytecode::TraceAnchor { trace_id });
    }
}

fn compile_module(
    builder: &mut Builder,
    ast_module: &swc_ecma_ast::Module,
) -> Result<bytecode::FnId> {
    use swc_ecma_ast::{ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    builder.start_function(None, &[]);

    builder.push_const(Literal::String("__named".to_string()));
    let lit_named = builder.accu_to_vreg();

    builder.push_const(Literal::String("__default".to_string()));
    let lit_default = builder.accu_to_vreg();

    // Exports object.  TODO Actually make it accessible via `require`!
    builder.emit(Instr::ObjNew);
    let module_obj = builder.accu_to_vreg();

    builder.emit(Instr::ObjNew);
    let mod_named_exports = builder.accu_to_vreg();
    builder.emit(Instr::ObjSet {
        obj: module_obj,
        key: lit_named,
    });

    builder.push_const(Literal::Undefined);
    let mod_default_export = builder.accu_to_vreg();
    builder.emit(Instr::ObjSet {
        obj: module_obj,
        key: lit_default,
    });

    for item in &ast_module.body {
        match item {
            ModuleItem::ModuleDecl(decl) => match decl {
                ModuleDecl::Import(decl) => {
                    if decl.type_only {
                        // TODO(small feat) make up a system for warnings
                        eprintln!(
                            "bytecode_compiler: warning: discarding type-only import statement"
                        );
                        continue;
                    }

                    let module_id = builder.compile_file(decl.src.value.to_string())?;

                    for spec in &decl.specifiers {
                        match spec {
                            // import { foo } from './a/b/c/module.js'
                            //
                            // We compile it as:
                            //    (mod) = GetModule <module id>
                            //    const foo = (mod).foo
                            swc_ecma_ast::ImportSpecifier::Named(d) => {
                                if d.is_type_only {
                                    eprintln!(
                                        "bytecode_compiler: warning: discarding type-only import specifier"
                                    );
                                } else {
                                    if d.imported.is_some() {
                                        todo!("import specifier with rename");
                                    }

                                    builder.push_const(Literal::String(d.local.sym.to_string()));
                                    let item_name = builder.accu_to_vreg();

                                    builder.emit(Instr::GetModule(module_id));
                                    builder.emit(Instr::ObjGet { key: lit_named });
                                    builder.emit(Instr::ObjGet { key: item_name });
                                    let local_var = builder.accu_to_vreg();
                                    builder.define_var(d.local.sym.clone(), local_var);
                                }
                            }
                            swc_ecma_ast::ImportSpecifier::Default(d) => {
                                builder.emit(Instr::GetModule(module_id));
                                builder.emit(Instr::ObjGet { key: lit_default });
                                let local_var = builder.accu_to_vreg();
                                builder.define_var(d.local.sym.clone(), local_var);
                            }
                            swc_ecma_ast::ImportSpecifier::Namespace(d) => {
                                builder.push_const(Literal::String("__named".to_string()));
                                let lit_named = builder.accu_to_vreg();

                                builder.emit(Instr::GetModule(module_id));
                                builder.emit(Instr::ObjGet { key: lit_named });
                                let local_var = builder.accu_to_vreg();
                                builder.define_var(d.local.sym.clone(), local_var);
                            }
                        }
                    }
                }

                ModuleDecl::ExportDecl(decl) => {
                    let mut defs = Vec::new();
                    compile_decl(builder, &decl.decl, Some(&mut defs))?;

                    for (name, vreg) in defs.into_iter() {
                        builder.push_const(Literal::String(name.to_string()));
                        let key = builder.accu_to_vreg();
                        builder.emit(Instr::LoadRA(vreg));
                        builder.emit(Instr::ObjSet {
                            obj: mod_named_exports,
                            key,
                        });
                    }
                }
                ModuleDecl::ExportNamed(_) => todo!("ExportNamed"),
                ModuleDecl::ExportDefaultDecl(decl) => match &decl.decl {
                    swc_ecma_ast::DefaultDecl::Class(_) => todo!("export default class"),
                    swc_ecma_ast::DefaultDecl::Fn(fn_expr) => {
                        let name = fn_expr.ident.as_ref().map(|ident| ident.sym.clone());
                        compile_function(builder, name, &*fn_expr.function)?;
                        builder.emit(Instr::StoAR(mod_default_export));
                    }
                    swc_ecma_ast::DefaultDecl::TsInterfaceDecl(ts_decl) => eprintln!(
                        "warning: discarded TypeScript interface export: {}",
                        ts_decl.id.to_string()
                    ),
                },
                ModuleDecl::ExportDefaultExpr(decl) => {
                    compile_expr(builder, &*decl.expr)?;
                    builder.emit(Instr::StoAR(mod_default_export));
                }
                ModuleDecl::ExportAll(_) => todo!("export all"),

                ModuleDecl::TsImportEquals(_)
                | ModuleDecl::TsExportAssignment(_)
                | ModuleDecl::TsNamespaceExport(_) => {
                    // TODO(small feat) make up a system for warnings
                    eprintln!("bytecode_compiler: warning: discarding TypeScript-only declaration");
                }
            },
            ModuleItem::Stmt(stmt) => {
                compile_stmt(builder, stmt).with_context(error!("while compiling statement"))?;
            }
            other => unsupported_node!(other),
        }
    }

    builder.emit(Instr::LoadRA(module_obj));
    builder.emit(Instr::Return);
    let root_fnid = builder.end_function();

    // The root function is the outermost scope, and therefore must capture
    // nothing.  Otherwise, we have a bug.
    assert!(builder.fns.get(&root_fnid).unwrap().captures.is_empty());

    Ok(root_fnid)
}

fn compile_stmt(builder: &mut Builder, stmt: &swc_ecma_ast::Stmt) -> Result<()> {
    use swc_ecma_ast::Stmt;

    match stmt {
        Stmt::Block(block) => {
            builder.cur_fnb().push_scope();

            for stmt in &block.stmts {
                compile_stmt(builder, stmt)
                    .with_context(error!("in block").with_span(block.span))?;
            }

            builder.cur_fnb().pop_scope();
            Ok(())
        }
        // Stmt::Empty(_) => todo!(),
        // Stmt::Debugger(_) => todo!(),
        // Stmt::With(_) => todo!(),
        Stmt::Return(stmt) => {
            if let Some(arg) = &stmt.arg {
                compile_expr(builder, arg)?;
            } else {
                builder.push_const(Literal::Undefined);
            }

            builder.emit(Instr::Return);
            Ok(())
        }
        // Stmt::Labeled(_) => todo!(),
        // Stmt::Break(_) => todo!(),
        // Stmt::Continue(_) => todo!(),
        // Stmt::Switch(_) => todo!(),
        // Stmt::Throw(_) => todo!(),
        // Stmt::Try(_) => todo!(),
        Stmt::While(while_stmt) => {
            let while_header_iid = builder.peek_iid();
            compile_expr(builder, &while_stmt.test)?;
            builder.emit(Instr::BoolNot);
            let jmpif = builder.reserve();
            compile_stmt(builder, &while_stmt.body)?;
            builder.emit(Instr::Jmp(while_header_iid));
            let while_end_iid = builder.peek_iid();

            *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                dest: while_end_iid,
            };

            Ok(())
        }

        // Stmt::DoWhile(_) => todo!(),
        // Stmt::For(_) => todo!(),
        Stmt::ForIn(forin_stmt) => {
            let item_var = match &forin_stmt.left {
                    VarDeclOrPat::VarDecl(var_decl) => var_decl.as_ref(),
                    VarDeclOrPat::Pat(_) => panic!(
                        "unsupported syntax: destructuring pattern as `<pattern>` in: `for (<pattern> in ...) {{ ... }}`"
                    ),
                };

            assert_eq!(item_var.decls.len(), 1);

            let item_var_name = match &item_var.decls[0].name {
                Pat::Ident(swc_ecma_ast::BindingIdent { id, .. }) => &id.sym,
                other => panic!(
                    "unsupported type of pattern in: `for (<pattern> in ...) {{ ... }}: {:?}",
                    other
                ),
            };

            builder.push_const(Literal::Number(0.0));
            let key_ndx = builder.accu_to_vreg();

            compile_expr(builder, &forin_stmt.right)?;
            builder.emit(Instr::ObjGetKeys);
            let keys = builder.accu_to_vreg();

            let while_begin = builder.peek_iid();
            builder.emit(Instr::LoadRA(keys));
            builder.emit(Instr::ArrayLen);
            builder.emit(Instr::Cmp(CmpOp::LT, key_ndx));
            builder.emit(Instr::BoolNot);
            let exit = builder.reserve();

            builder.cur_fnb().push_scope();
            builder.emit(Instr::LoadRA(keys));
            builder.emit(Instr::ArrayNth(key_ndx));
            let key = builder.accu_to_vreg();
            builder.define_var(item_var_name.clone(), key);

            compile_stmt(builder, &forin_stmt.body)?;

            builder.push_const(Literal::Number(1.0));
            let one = builder.accu_to_vreg();
            builder.emit(Instr::LoadRA(key_ndx));
            builder.emit(Instr::Arith(ArithOp::Add, one));
            builder.emit(Instr::StoAR(key_ndx));

            builder.cur_fnb().pop_scope();
            builder.emit(Instr::Jmp(while_begin));
            let exit_label = builder.peek_iid();

            *builder.get_mut(exit).unwrap() = Instr::JmpIf { dest: exit_label };

            Ok(())
        }

        // Stmt::ForOf(_) => todo!(),
        Stmt::Decl(decl) => compile_decl(builder, decl, None),

        Stmt::Expr(expr) => {
            compile_expr(builder, &expr.expr)?;
            Ok(())
        }

        Stmt::If(if_stmt) => {
            compile_expr(builder, &if_stmt.test)
                .with_context(error!("in if statement").with_span(if_stmt.span))?;
            builder.emit(Instr::BoolNot);
            let jmp_to_alt = builder.reserve();

            compile_stmt(builder, &if_stmt.cons)?;
            let jmp_to_end = builder.reserve();

            let alt = builder.peek_iid();
            if let Some(else_blk) = &if_stmt.alt {
                compile_stmt(builder, else_blk)?;
            }

            let end: IID = builder.peek_iid();

            *builder.get_mut(jmp_to_alt).unwrap() = Instr::JmpIf { dest: alt };
            *builder.get_mut(jmp_to_end).unwrap() = Instr::JmpIf { dest: end };
            Ok(())
        }
        other => unsupported_node!(other),
    }
}

fn compile_decl(
    builder: &mut Builder,
    decl: &Decl,
    defs: Option<&mut Vec<(JsWord, VReg)>>,
) -> Result<()> {
    match decl {
        // Decl::Class(_) => todo!(),
        Decl::Fn(fn_decl) => {
            if fn_decl.declare {
                panic!("unsupported case: fn_decl.declare");
            }

            let (name, _) = fn_decl.ident.to_id();
            let func = &fn_decl.function;
            compile_function(builder, Some(name.clone()), func)?;
            let value = builder.accu_to_vreg();
            builder.define_var(name, value);
        }

        Decl::Var(var_decl) => {
            compile_var_decl(builder, var_decl, defs)
                .with_context(error!("in variable declaration").with_span(var_decl.span))?;
        }

        Decl::TsInterface(_) | Decl::TsTypeAlias(_) | Decl::TsEnum(_) | Decl::TsModule(_) => {
            panic!("TypeScript syntax not supported (for now!)")
        }

        other => unsupported_node!(other),
    }

    Ok(())
}

fn compile_function(builder: &mut Builder, name: Option<JsWord>, func: &Function) -> Result<()> {
    if !func.decorators.is_empty() {
        panic!("unsupported: function decorators");
    }
    if func.is_async {
        panic!("unsupported: async functions");
    }
    if func.is_generator {
        panic!("unsupported: generator functions");
    }
    if func.return_type.is_some() {
        panic!("unsupported: TypeScript syntax (return type)");
    }
    if func.type_params.is_some() {
        panic!("unsupported: TypeScript syntax (return type)");
    }

    builder.start_function(name, func.params.as_slice());

    let stmts = &func.body.as_ref().expect("function without body?!").stmts;
    for stmt in stmts {
        compile_stmt(builder, stmt)
            .with_context(error!("in function declaration").with_span(func.span))?;
    }

    let inner_fnb = builder.fn_stack.pop().expect("no FnBuilder!");

    // all the calls to `get_var` must come before builder.emit(ClosureNew), to guarantee that
    // the ClosureNew and any associated ClosureAddCapture are adjacent
    // TODO(performance) avoid this allocation?
    for var_name in inner_fnb.captures.iter() {
        let vreg = builder.get_var(var_name)?;
        builder.emit(Instr::ClosureAddCapture(vreg));
    }
    builder.emit(Instr::ClosureNew {
        fnid: inner_fnb.fnid,
    });

    builder.fns.insert(inner_fnb.fnid, inner_fnb);
    Ok(())
}

fn compile_var_decl(
    builder: &mut Builder,
    var_decl: &swc_ecma_ast::VarDecl,
    mut defs: Option<&mut Vec<(JsWord, VReg)>>,
) -> Result<()> {
    use swc_ecma_ast::VarDeclKind;

    let _is_const = match var_decl.kind {
        VarDeclKind::Var => panic!("limitation: `var` bindings not supported"),
        VarDeclKind::Let => false,
        VarDeclKind::Const => true,
    };

    for decl in &var_decl.decls {
        let ident = decl
            .name
            .as_ident()
            .unwrap_or_else(|| unsupported_node!(decl));

        if let Some(expr) = &decl.init {
            compile_expr(builder, expr)?;
        } else {
            builder.push_const(Literal::Undefined);
        }
        let operand = builder.accu_to_vreg();

        let name: JsWord = ident.id.to_id().0;
        if let Some(defs) = &mut defs {
            defs.push((name.clone(), operand));
        }
        builder.define_var(name, operand);
    }

    Ok(())
}

/// Compile the given expression.
///
/// The resulting code implicitly leaves the result in the accumulator register.
fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Result<()> {
    use swc_ecma_ast::{CallExpr, Expr};

    match expr {
        Expr::Call(call_expr @ CallExpr { callee, args, .. }) => {
            let callee = callee.as_expr().ok_or_else(|| {
                error!("only calls to simple identifiers are supported for now")
                    .with_span(call_expr.span)
            })?;

            if let Some(callee) = callee.as_ident() {
                let sym = callee.sym.as_ref();
                if sym == "sink" {
                    for arg in args {
                        compile_expr(builder, &arg.expr)?;
                        builder.emit(Instr::PushToSink);
                    }

                    builder.push_const(Literal::Null);
                    return Ok(());
                } else if sym == "__start_trace" {
                    let trace_id = match args[0].expr.as_ref() {
                        Expr::Lit(Lit::Str(trace_id)) => trace_id.value.to_string(),
                        _ => {
                            panic!("__start_trace must be called with a trace ID: __start_trace('the-name-of-the-trace')")
                        }
                    };

                    builder.place_trace_anchor(trace_id);
                    builder.push_const(Literal::Undefined);
                    return Ok(());
                }
            }

            for arg in args {
                if arg.spread.is_some() {
                    panic!("unsupported: spread function parameter: function(a, b, ...)");
                }
                compile_expr(builder, &arg.expr)?;
                builder.emit(Instr::CallArg);
            }

            compile_expr(builder, callee)?;
            builder.emit(Instr::Call);
        }

        Expr::Bin(bin_expr) => {
            compile_expr(builder, &bin_expr.left)?;
            let a = builder.accu_to_vreg();
            compile_expr(builder, &bin_expr.right)?;

            let instr = match bin_expr.op {
                BinaryOp::Add => Instr::Arith(ArithOp::Add, a),
                BinaryOp::Sub => Instr::Arith(ArithOp::Sub, a),
                BinaryOp::Mul => Instr::Arith(ArithOp::Mul, a),
                BinaryOp::Div => Instr::Arith(ArithOp::Div, a),
                BinaryOp::Lt => Instr::Cmp(CmpOp::LT, a),
                BinaryOp::LtEq => Instr::Cmp(CmpOp::LE, a),
                BinaryOp::Gt => Instr::Cmp(CmpOp::GT, a),
                BinaryOp::GtEq => Instr::Cmp(CmpOp::GE, a),
                BinaryOp::EqEqEq => Instr::Cmp(CmpOp::EQ, a),
                BinaryOp::NotEqEq => Instr::Cmp(CmpOp::NE, a),

                // TODO TODO TODO This does not implement any of the 'wat' semantics of JavaScript
                // See https://www.destroyallsoftware.com/talks/wat
                BinaryOp::EqEq => Instr::Cmp(CmpOp::EQ, a),
                BinaryOp::NotEq => Instr::Cmp(CmpOp::NE, a),

                BinaryOp::LogicalAnd => Instr::BoolOp(BoolOp::And, a),
                BinaryOp::LogicalOr => Instr::BoolOp(BoolOp::Or, a),

                _ => panic!("unsupported binary op: {:?}", bin_expr.op),
            };

            builder.emit(instr);
        }

        Expr::Lit(lit) => match lit {
            Lit::Num(number) => {
                let lit = Literal::Number(number.value);
                builder.push_const(lit);
            }
            Lit::Str(s) => {
                let value = s.value.to_string();
                let lit = Literal::String(value);
                builder.push_const(lit);
            }
            Lit::Bool(bv) => {
                let lit = Literal::Bool(bv.value);
                builder.push_const(lit);
            }
            Lit::Null(_) => {
                builder.push_const(Literal::Null);
            }
            Lit::Regex(re_lit) => {
                let constructor_fid = *builder
                    .native_fns
                    .get(&JsWord::from("RegExp"))
                    .expect("undefined native constructor: RegExp");

                builder.push_const(Literal::String("__proto__".into()));
                let proto_lit = builder.accu_to_vreg();

                // TODO More efficient bytecode here?
                let re_str = re_lit.exp.to_string().into();
                builder.push_const(re_str);
                builder.emit(Instr::CallArg);
                let constructor = builder.emit(Instr::GetNativeFn(constructor_fid));
                builder.emit(Instr::Call);
                let obj = builder.accu_to_vreg();

                builder.push_const(Literal::String("prototype".into()));
                let key = builder.accu_to_vreg();
                builder.emit(Instr::GetNativeFn(constructor_fid));
                builder.emit(Instr::ObjGet { key });
                builder.emit(Instr::ObjSet {
                    obj,
                    key: proto_lit,
                });
            }
            // Lit::BigInt(_) => todo!(),
            // Lit::JSXText(_) => todo!(),
            other => unsupported_node!(other),
        },

        Expr::Ident(ident) => {
            if &ident.sym == "undefined" {
                builder.push_const(Literal::Undefined);
            } else {
                let vreg = builder.get_var(&ident.sym)?;
                builder.emit(Instr::LoadRA(vreg));
            }
        }

        Expr::Assign(asmt) => {
            compile_assignment(asmt, builder)
                .with_context(error!("in assignment").with_span(asmt.span))?;
        }

        Expr::Object(obj_expr) => {
            builder.emit(Instr::ObjNew);
            let obj = builder.accu_to_vreg();

            for prop_or_spread in obj_expr.props.iter() {
                match prop_or_spread {
                    swc_ecma_ast::PropOrSpread::Spread(spread_elm) => {
                        return Err(error!("spread syntax (...) is currently unsupported")
                            .with_span(spread_elm.dot3_token))
                    }
                    swc_ecma_ast::PropOrSpread::Prop(prop) => match prop.as_ref() {
                        swc_ecma_ast::Prop::KeyValue(kv_expr) => {
                            let key = compile_prop_name(&kv_expr.key)?;
                            builder.push_const(key);
                            let key = builder.accu_to_vreg();

                            compile_expr(builder, &kv_expr.value)?;
                            builder.emit(Instr::ObjSet { obj, key });
                        }
                        swc_ecma_ast::Prop::Shorthand(expr) => {
                            return Err(error!("unsupported node: {:?}", expr).with_span(expr.span))
                        }
                        swc_ecma_ast::Prop::Assign(expr) => {
                            return Err(
                                error!("unsupported node: {:?}", expr).with_span(expr.key.span)
                            );
                        }
                        swc_ecma_ast::Prop::Getter(expr) => {
                            return Err(error!("unsupported node: {:?}", expr).with_span(expr.span));
                        }
                        swc_ecma_ast::Prop::Setter(expr) => {
                            return Err(error!("unsupported node: {:?}", expr).with_span(expr.span));
                        }
                        swc_ecma_ast::Prop::Method(method_prop) => {
                            let key = compile_prop_name(&method_prop.key)?;
                            builder.push_const(key);
                            let key = builder.accu_to_vreg();

                            compile_function(builder, None, &*method_prop.function)?;

                            builder.emit(Instr::ObjSet { obj, key });
                        }
                    },
                }
            }

            builder.emit(Instr::LoadRA(obj));
        }

        // Expr::This(_) => todo!(),
        Expr::Array(arr_expr) => {
            use swc_ecma_ast::ExprOrSpread;

            builder.emit(Instr::ObjNew);
            let array = builder.accu_to_vreg();

            for elem in arr_expr.elems.iter() {
                match elem {
                    Some(ExprOrSpread { spread, expr }) => {
                        if spread.is_some() {
                            return Err(error!("spread syntax is currently unsupported")
                                .with_span(arr_expr.span));
                        }
                        compile_expr(builder, expr)?;
                        builder.emit(Instr::ArrayPush(array));
                    }
                    None => {
                        eprintln!(
                            "warning: an element in Expr::Array is `None` ({:?})",
                            arr_expr.span
                        );
                    }
                }
            }
        }

        Expr::Fn(fn_expr) => {
            // TODO Refactor this with Decl::Fn
            let name = fn_expr.ident.as_ref().map(|ident| ident.to_id().0);
            let func = &fn_expr.function;
            compile_function(builder, name.clone(), func)?;

            if let Some(name) = name {
                let value = builder.accu_to_vreg();
                builder.define_var(name, value);
            }
        }

        Expr::Unary(unary_expr) => {
            compile_expr(builder, &unary_expr.arg)?;
            let instr = match unary_expr.op {
                swc_ecma_ast::UnaryOp::Bang => Instr::BoolNot,
                swc_ecma_ast::UnaryOp::TypeOf => Instr::TypeOf,
                swc_ecma_ast::UnaryOp::Minus => Instr::UnaryMinus,
                other => unsupported_node!(other),
                // swc_ecma_ast::UnaryOp::Plus => todo!(),
                // swc_ecma_ast::UnaryOp::Tilde => todo!(),
                // swc_ecma_ast::UnaryOp::Void => todo!(),
                // swc_ecma_ast::UnaryOp::Delete => todo!(),
            };
            builder.emit(instr);
        }
        Expr::Update(update_expr) => {
            if let Expr::Ident(ident) = &*update_expr.arg {
                // NOTE: update_expr.prefix does not matter in this case, but
                // it will matter when this code is extended to other types of args
                let var = builder.get_var(&ident.sym)?;

                let op = match update_expr.op {
                    UpdateOp::PlusPlus => ArithOp::Add,
                    UpdateOp::MinusMinus => ArithOp::Sub,
                };

                // TODO Use integers here, when they get implemented
                builder.push_const(Literal::Number(1.0));
                let value: IID = builder.emit(Instr::Arith(op, var));
                builder.emit(Instr::StoAR(var));
            } else {
                todo!("unsupported: UpdateExpr on anything other than an identifier")
            }
        }
        Expr::Member(member_expr) => {
            compile_member_access(builder, member_expr)?;
        }

        Expr::Paren(inner) => {
            compile_expr(builder, &inner.expr)?;
        }

        // Expr::SuperProp(_) => todo!(),
        // Expr::Cond(_) => todo!(),
        // Expr::New(_) => todo!(),
        // Expr::Seq(_) => todo!(),
        // Expr::Tpl(_) => todo!(),
        // Expr::TaggedTpl(_) => todo!(),
        // Expr::Arrow(_) => todo!(),
        // Expr::Class(_) => todo!(),
        // Expr::Yield(_) => todo!(),
        // Expr::MetaProp(_) => todo!(),
        // Expr::Await(_) => todo!(),
        // Expr::JSXMember(_) => todo!(),
        // Expr::JSXNamespacedName(_) => todo!(),
        // Expr::JSXEmpty(_) => todo!(),
        // Expr::JSXElement(_) => todo!(),
        // Expr::JSXFragment(_) => todo!(),
        // Expr::TsTypeAssertion(_) => todo!(),
        // Expr::TsConstAssertion(_) => todo!(),
        // Expr::TsNonNull(_) => todo!(),
        // Expr::TsAs(_) => todo!(),
        // Expr::TsInstantiation(_) => todo!(),
        // Expr::TsSatisfies(_) => todo!(),
        // Expr::PrivateName(_) => todo!(),
        // Expr::OptChain(_) => todo!(),
        // Expr::Invalid(_) => todo!(),
        other => unsupported_node!(other),
    }

    Ok(())
}

fn compile_prop_name(prop_name: &swc_ecma_ast::PropName) -> Result<Literal> {
    match &prop_name {
        swc_ecma_ast::PropName::Ident(ident) => Ok(Literal::from(ident.sym.to_string())),
        swc_ecma_ast::PropName::Str(s) => Ok(Literal::from(s.value.to_string())),
        swc_ecma_ast::PropName::Num(num) => Ok(Literal::Number(num.value)),
        swc_ecma_ast::PropName::Computed(x) => {
            Err(error!("unsupported node: {:?}", x).with_span(x.span))
        }
        swc_ecma_ast::PropName::BigInt(x) => {
            Err(error!("unsupported node: {:?}", x).with_span(x.span))
        }
    }
}

fn compile_assignment(asmt: &swc_ecma_ast::AssignExpr, builder: &mut Builder) -> Result<()> {
    use swc_ecma_ast::{Expr, MemberExpr, MemberProp, PatOrExpr};

    if let Some(ident) = asmt.left.as_ident() {
        compile_expr(builder, &asmt.right)?;
        let rhs = builder.accu_to_vreg();

        let var = builder.get_var(&ident.sym)?;
        builder.emit(Instr::LoadRA(var));
        compile_assignment_rhs(builder, asmt.op, rhs)?;
        builder.emit(Instr::StoAR(var));
    } else if let Some(target_expr) = asmt.left.as_expr() {
        match target_expr {
            Expr::Member(member_expr) => {
                compile_expr(builder, &asmt.right)?;
                let rhs = builder.accu_to_vreg();

                let MemberAccess { obj, key } = compile_member_access(builder, member_expr)?;
                compile_assignment_rhs(builder, asmt.op, rhs)?;
                builder.emit(Instr::ObjSet { obj, key });
            }
            // We should have already handled this case in the `if let ... = asm.left.as_ident()`
            // case
            Expr::Ident(_) => unreachable!(),
            _ => {
                let error =
                    error!("assignment to an expression is unsupported").with_span(asmt.span);
                return Err(error);
            }
        }
    } else {
        panic!("unsupported pattern as assignment target: {:?}", asmt.left)
    }

    Ok(())
}

struct MemberAccess {
    obj: VReg,
    key: VReg,
}
fn compile_member_access(
    builder: &mut Builder,
    member_expr: &swc_ecma_ast::MemberExpr,
) -> Result<MemberAccess> {
    use swc_ecma_ast::MemberProp;

    match &member_expr.prop {
        MemberProp::Ident(prop_ident) => {
            let prop_ident = prop_ident.sym.to_string().into();
            builder.push_const(prop_ident);
        }
        _ => {
            let error = error!("accessing an object with non-identifer key is unsupported")
                .with_span(member_expr.span);
            return Err(error);
        }
    };
    let key = builder.accu_to_vreg();

    // obj
    compile_expr(builder, member_expr.obj.as_ref())?;
    let obj = builder.accu_to_vreg();
    builder.emit(Instr::ObjGet { key });
    Ok(MemberAccess { obj, key })
}

/// Read the operand and compute its new value, as requested by the given assignment
/// expression. This is part of the compilation procedure of assignment expressions.
fn compile_assignment_rhs(
    builder: &mut Builder,
    assign_op: swc_ecma_ast::AssignOp,
    rhs: VReg,
) -> Result<IID> {
    let value = match assign_op {
        AssignOp::Assign => builder.emit(Instr::LoadRA(rhs)),
        AssignOp::AddAssign => builder.emit(Instr::Arith(ArithOp::Add, rhs)),
        AssignOp::SubAssign => builder.emit(Instr::Arith(ArithOp::Sub, rhs)),
        AssignOp::MulAssign => builder.emit(Instr::Arith(ArithOp::Mul, rhs)),
        AssignOp::DivAssign => builder.emit(Instr::Arith(ArithOp::Div, rhs)),
        // AssignOp::ModAssign => todo!(),
        // AssignOp::LShiftAssign => todo!(),
        // AssignOp::RShiftAssign => todo!(),
        // AssignOp::ZeroFillRShiftAssign => todo!(),
        // AssignOp::BitOrAssign => todo!(),
        // AssignOp::BitXorAssign => todo!(),
        // AssignOp::BitAndAssign => todo!(),
        // AssignOp::ExpAssign => todo!(),
        // AssignOp::AndAssign => todo!(),
        // AssignOp::OrAssign => todo!(),
        // AssignOp::NullishAssign => todo!(),
        other => unsupported_node!(other),
    };
    Ok(value)
}

fn parse_file(filename: String, content: String) -> Result<(Lrc<SourceMap>, swc_ecma_ast::Module)> {
    use swc_common::{
        errors::{emitter::EmitterWriter, Handler},
        sync::Lrc,
        FileName, SourceMap,
    };
    use swc_ecma_ast::EsVersion;
    use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

    let source_map: Lrc<SourceMap> = Default::default();
    let err_handler = Handler::with_emitter(
        true, // can_emit_warnings
        true, // treat_err_as_bug
        Box::new(EmitterWriter::new(
            Box::new(std::io::stderr()),
            Some(source_map.clone()),
            false, // short_message
            true,  // teach
        )),
    );

    let source_file = source_map.new_source_file(FileName::Custom(filename), content);

    let input = StringInput::from(&*source_file);
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2015,
        input,
        None,
    );
    let mut parser = Parser::new_from(lexer);

    for e in parser.take_errors() {
        e.into_diagnostic(&err_handler).emit();
    }

    match parser.parse_module() {
        Ok(ast_module) => Ok((source_map, ast_module)),
        Err(e) => {
            e.into_diagnostic(&err_handler).emit();
            Err(error!("parse error"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NullLoader(String);
    impl NullLoader {
        const THE_MODULE_ID: bytecode::ModuleId = bytecode::ModuleId(1);
    }
    impl Loader for NullLoader {
        fn get_module_id(&self, filename: &str) -> bytecode::ModuleId {
            Self::THE_MODULE_ID
        }

        fn read_source(&self, module_id: bytecode::ModuleId) -> String {
            assert_eq!(module_id, Self::THE_MODULE_ID);
            self.0.clone()
        }
    }

    fn quick_compile(code: &str) -> bytecode::Codebase {
        let mut builder = BuilderParams {
            native_fns: HashMap::new(),
            loader: Box::new(NullLoader(code.to_string())),
        }
        .to_builder();

        builder.compile_file("<input>".to_string()).unwrap();

        builder.build()
    }

    #[test]
    fn test_bytecode_object_init() {
        let codebase = quick_compile(
            "sink({
                aString: 'asdlol123',
                aNumber: 1239423.4518923,
                anotherObject: { x: 123, y: 899 },
                aFunction: function(pt) { return 42; }
            })",
        );

        let root_fnid = codebase
            .get_module_root_fn(NullLoader::THE_MODULE_ID)
            .unwrap();
        let function = &codebase.get_function(root_fnid).unwrap();

        eprintln!("consts:");
        for (ndx, value) in function.consts().iter().enumerate() {
            eprintln!(" [{:2}] {:?}", ndx, value);
        }

        eprintln!("code:");
        for instr in function.instrs() {
            eprintln!("  {:?}", instr);
        }

        eprintln!("size of VReg: ............ {}", std::mem::size_of::<VReg>());
        eprintln!(
            "size of Instr: ........... {}",
            std::mem::size_of::<Instr>()
        );
        let size_code = function.instrs().len() * std::mem::size_of::<Instr>();
        eprintln!("size of code: ............ {}", size_code);
        eprintln!(
            "size of bytecode::Literal: {}",
            std::mem::size_of::<bytecode::Literal>()
        );
        let size_data = function.consts().len() * std::mem::size_of::<bytecode::Literal>();
        eprintln!("size of const data: ...... {}", size_data);
        eprintln!("total size: .............. {}", size_code + size_data);

        todo!()
    }

    const CODE_UPVALUES: &'static str = "
        let counter = 10;
        function foo() {
            counter++; counter++; counter++;
        }
        counter--; counter--;
        foo();
        counter--; counter--; counter--;
        foo();
        counter--;
    ";

    #[test]
    #[cfg(x)]
    fn test_upvalues_caller() {
        let module = quick_compile(CODE_UPVALUES);
        module.dump();

        let root_fn = module.get_function(FnId(0)).unwrap();
        let captures: Vec<_> = root_fn
            .instrs()
            .iter()
            .filter_map(|inst| match inst {
                Instr::ClosureAddCapture(cap_iid) => Some(cap_iid),
                _ => None,
            })
            .collect();

        assert_eq!(captures.len(), 1);
        let cap = captures[0];
        let cap_const = match &root_fn.instrs()[cap.0 as usize] {
            Instr::LoadConst(Literal::Number(val)) => val,
            _ => panic!(),
        };
        assert_eq!(*cap_const, 10.0);
    }

    #[test]
    #[cfg(x)]
    fn test_upvalues_callee() {
        let module = quick_compile(CODE_UPVALUES);
        module.dump();

        let callee = module.get_function(FnId(1)).unwrap();
        assert!(std::matches!(callee.instrs()[0], Instr::LoadCapture(0)));
    }

    #[test]
    #[cfg(x)]
    fn test_upvalues_compact_addcapinstr() {
        let module = quick_compile(
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
        );
        module.dump();

        let mut fnid = FnId(0);
        while let Some(func) = module.get_function(fnid) {
            let instrs = func.instrs();
            for (prev_instr, instr) in instrs.iter().zip(&instrs[1..]) {
                if let Instr::ClosureAddCapture(_) = instr {
                    assert!(
                        match prev_instr {
                            Instr::ClosureAddCapture(_) => true,
                            Instr::ClosureNew { .. } => true,
                            _ => false,
                        },
                        "ClosureAddCapture preceded by {:?} instead of a ClosureNew or another ClosureAddCapture", 
                        prev_instr
                    );
                }
            }

            fnid.0 += 1;
        }
    }

    #[test]
    #[cfg(x)]
    fn test_bindings_limited_scopes() {
        let res = Compiler::new().compile_file(
            "<input>".to_string(),
            "
            function f() {
                function g() {
                    // All these forms introduce a scope

                    if (false) {
                        function z() { return 'lol'; }
                    }

                    while (1) {
                        function z() { return 'lol'; }
                    }

                    {
                        function z() { return 'lol'; }
                    }

                    if (true) {
                        // Still unbound here
                        z();
                    }
                }
            }
            "
            .to_string(),
        );
        assert!(res.is_err());
    }

    #[test]
    #[cfg(x)]
    fn test_bindings_cross_scope() {
        let res = Compiler::new().compile_file(
            "<input>".to_string(),
            "
            function f() {
                function g() {
                    function z() { return 'lol'; }

                    if (true) {
                        z();
                    }
                }
            }
            "
            .to_string(),
        );

        assert!(res.is_ok());
    }

    #[test]
    #[cfg(x)]
    fn test_bindings_unbound_symbol() {
        let res = Compiler::new().compile_file(
            "<input>".to_string(),
            "
            function f() {
                function g() {
                    z();
                }
            }
            "
            .to_string(),
        );

        assert!(res.is_err());
    }
}
