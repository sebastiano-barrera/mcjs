use std::collections::{HashMap, HashSet};

use swc_atoms::JsWord;
use swc_common::{sync::Lrc, SourceMap, Span};
use swc_ecma_ast::VarDeclOrPat;
use swc_ecma_ast::{AssignOp, BinaryOp, Decl, Function, Lit, Pat, UpdateOp};

use crate::bytecode::{self, ArithOp, BoolOp, CmpOp, FnId, Instr, NativeFnId, Value, IID};
pub use crate::common::{Context, Error, Result};
use crate::error;
use crate::interpreter;
use crate::util::Mask;

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
}

type NativeFnMap = HashMap<JsWord, NativeFnId>;

pub struct Compiler {
    builtin_fns: NativeFnMap,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            builtin_fns: HashMap::new(),
        }
    }

    pub fn add_builtin_fn(&mut self, name: JsWord, bid: NativeFnId) -> Option<NativeFnId> {
        self.builtin_fns.insert(name, bid)
    }

    pub fn compile_file(&mut self, filename: String, content: String) -> Result<bytecode::Module> {
        use crate::common::Context;
        let (source_map, ast_module) = parse_file(filename.clone(), content)
            .with_context(error!("while parsing file: {filename}"))?;

        let builder = Builder::new(&self.builtin_fns);
        let res = compile_module(builder, &ast_module)
            .with_context(error!("while compiling module: {filename}"));
        if let Err(err) = &res {
            eprintln!("\nbytecode compiler error: {}\n", err.message(&source_map));
        }

        res
    }
}

struct Builder<'a> {
    native_fns: &'a NativeFnMap,
    fns: HashMap<FnId, FnBuilder>,
    fn_stack: Vec<FnBuilder>,
    next_fnid: u32,

    /// Values exported by the module as named exports, e.g.:
    ///
    /// ```js
    /// export function foobar() { ... }
    /// ```
    ///
    /// The IIDs associated to each identifier refers to the module's root function, i.e.
    /// `self.fns[FnId::ROOT_FN]`
    module_exports_named: HashMap<JsWord, IID>,
    module_export_default: Option<IID>,

    module_imports: Vec<JsWord>,
}

impl<'a> Builder<'a> {
    fn set_default_export(&mut self, value: IID) {
        let prev = self.module_export_default.replace(value);
        if prev.is_some() {
            // TODO Is this really a warning? Or an error...?
            eprintln!("Warning: default export redefined");
        }
    }

    fn ensure_imported(&mut self, path: &JsWord) -> usize {
        for (ndx, p) in self.module_imports.iter().enumerate() {
            if p == path {
                return ndx;
            }
        }
        self.module_imports.push(path.to_owned());
        self.module_imports.len() - 1
    }
}

struct FnBuilder {
    fnid: FnId,
    scopes: Vec<Scope>,
    instrs: Vec<Instr>,
    trace_anchors: HashMap<IID, bytecode::TraceAnchor>,
    last_local_ndx: u32,
    captures: Vec<JsWord>,
}

#[derive(Debug)]
struct Scope {
    // TODO Take advantage of identifier interning!
    vars: HashMap<String, IID>,
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
            trace_anchors: HashMap::new(),
            captures: Vec::new(),
            last_local_ndx: 0,
        }
    }

    fn build(self) -> bytecode::Function {
        bytecode::Function::new(self.instrs.into_boxed_slice(), self.trace_anchors)
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
        IID(self.instrs.len() as u32)
    }

    fn emit(&mut self, instr: Instr) -> IID {
        let iid = self.peek_iid();
        self.instrs.push(instr);
        iid
    }

    fn define_local(&mut self, name: JsWord, var_iid: IID) {
        self.define_var(name, var_iid);
    }

    // Assign a new upvalue index to the given variable, and emits the necessary "GetCapture"
    // instruction.
    // Subsequent calls to get_var will return  the IID for the GetCapture instruction.
    fn set_var_captured(&mut self, name: JsWord) -> IID {
        assert!(!self.captures.contains(&name));
        let cap_ndx = self.captures.len().try_into().expect("too many captures!");
        let iid = self.emit(Instr::GetCapture(cap_ndx));
        self.captures.push(name.clone());
        self.define_var(name, iid);
        iid
    }

    fn get_var(&self, sym: &JsWord) -> Option<IID> {
        let name = String::from_utf8_lossy(sym.as_bytes());

        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.vars.get(name.as_ref()))
            .copied()
    }

    fn define_var(&mut self, name: JsWord, var: IID) {
        let name = String::from_utf8(name.as_bytes().to_owned())
            .expect("only UTF-8 identifiers are supporeted!");
        let scope = self.inner_scope_mut();
        let prev = scope.vars.insert(name.clone(), var);
        if prev.is_some() {
            panic!(
                "definition of var `{}` shadows previous definition (compiler limitation)",
                name
            );
        }
    }
}

impl<'a> Builder<'a> {
    fn new(native_fns: &'a NativeFnMap) -> Self {
        let next_fnid = 1;
        assert_ne!(FnId(next_fnid), FnId::ROOT_FN);

        Builder {
            native_fns,
            fns: HashMap::new(),
            fn_stack: vec![FnBuilder::new(FnId::ROOT_FN)],
            next_fnid,
            module_exports_named: HashMap::new(),
            module_export_default: None,
            module_imports: Vec::new(),
        }
    }

    fn build(mut self) -> bytecode::Module {
        // End the root function
        let root_fnid = self.end_function();

        // The root function is the outermost scope, and therefore must capture
        // nothing.  Otherwise, we have a bug.
        assert!(self.fns.get(&root_fnid).unwrap().captures.is_empty());
        assert!(self.fn_stack.is_empty());

        let fns = self
            .fns
            .drain()
            .map(|(fn_id, fn_builder)| (fn_id, fn_builder.build()))
            .collect();
        
        bytecode::Module {
            fns,
            module_exports_named: self.module_exports_named,
            module_export_default: self.module_export_default,
            module_imports: self.module_imports,
        }
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

    fn peek_iid(&mut self) -> IID {
        self.cur_fnb().peek_iid()
    }

    fn get_mut(&mut self, iid: IID) -> Option<&mut Instr> {
        self.cur_fnb().instrs.get_mut(iid.0 as usize)
    }

    fn define_var(&mut self, name: JsWord, value_iid: IID) {
        self.cur_fnb().define_local(name, value_iid)
    }

    fn get_var(&mut self, sym: &JsWord) -> Result<IID> {
        // self.fn_stack.last_mut().expect("no FnBuilder!")
        let found_iid = self
            .fn_stack
            .iter_mut()
            .rev()
            .enumerate()
            .find_map(|(ndx, fnb)| fnb.get_var(sym).map(|iid| (ndx, iid)))
            .or_else(|| {
                let nfid = self.native_fns.get(sym)?;
                let iid = self.cur_fnb().emit(Instr::GetNativeFn(*nfid));
                Some((0, iid))
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

    fn start_function(&mut self, name: Option<JsWord>, params: &[swc_ecma_ast::Param]) {
        let fnid = FnId(self.next_fnid);
        self.next_fnid += 1;

        self.fn_stack.push(FnBuilder::new(fnid));

        for (param_ndx, param) in params.iter().enumerate() {
            if !param.decorators.is_empty() {
                panic!("unsupported: decorators on function parameters");
            }

            match &param.pat {
                Pat::Ident(ident) => {
                    let iid = self.emit(Instr::GetArg(param_ndx));
                    self.define_var(ident.sym.clone(), iid);
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
    mut builder: Builder,
    ast_module: &swc_ecma_ast::Module,
) -> Result<bytecode::Module> {
    use swc_ecma_ast::{ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    // Exports object.  TODO Actually make it accessible via `require`!
    {
        let obj = builder.emit(Instr::ObjNew);
        builder.define_var("module".into(), obj);
    }

    let mut defs = Vec::new();

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

                    let module_ndx = builder.ensure_imported(&decl.src.value);

                    for spec in &decl.specifiers {
                        match spec {
                            swc_ecma_ast::ImportSpecifier::Named(d) => {
                                if d.is_type_only {
                                    eprintln!(
                                        "bytecode_compiler: warning: discarding type-only import specifier"
                                    );
                                } else {
                                    if d.imported.is_some() {
                                        todo!("import specifier with rename");
                                    }
                                    let value = builder.emit(Instr::NamedImport {
                                        module_ndx,
                                        identifier: d.local.clone(),
                                    });
                                    builder.define_var(d.local.sym.clone(), value);
                                }
                            }
                            swc_ecma_ast::ImportSpecifier::Default(d) => {
                                let value = builder.emit(Instr::DefaultImport { module_ndx });
                                builder.define_var(d.local.sym.clone(), value);
                            }
                            swc_ecma_ast::ImportSpecifier::Namespace(d) => {
                                let value = builder.emit(Instr::AllNamedImports { module_ndx });
                                builder.define_var(d.local.sym.clone(), value);
                            }
                        }
                    }
                }

                ModuleDecl::ExportDecl(decl) => {
                    defs.clear();
                    compile_decl(&mut builder, &decl.decl, Some(&mut defs))?;

                    assert_eq!(builder.cur_fnb().fnid, FnId::ROOT_FN);
                    for (name, iid) in &defs {
                        let prev = builder.module_exports_named.insert(name.clone(), *iid);
                        if prev.is_none() {
                            // TODO Is this really a warning? Or an error...?
                            eprintln!("Warning: exported symbol redefined: {}", name);
                        }
                    }
                }
                ModuleDecl::ExportNamed(_) => todo!("ExportNamed"),
                ModuleDecl::ExportDefaultDecl(decl) => match &decl.decl {
                    swc_ecma_ast::DefaultDecl::Class(_) => todo!("export default class"),
                    swc_ecma_ast::DefaultDecl::Fn(fn_expr) => {
                        let name = fn_expr.ident.as_ref().map(|ident| ident.sym.clone());
                        let value = compile_function(&mut builder, name, &*fn_expr.function)?;
                        builder.set_default_export(value);
                    }
                    swc_ecma_ast::DefaultDecl::TsInterfaceDecl(ts_decl) => eprintln!(
                        "warning: discarded TypeScript interface export: {}",
                        ts_decl.id.to_string()
                    ),
                },
                ModuleDecl::ExportDefaultExpr(decl) => {
                    let value = compile_expr(&mut builder, &*decl.expr)?;
                    builder.set_default_export(value);
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
                compile_stmt(&mut builder, stmt)
                    .with_context(error!("while compiling statement"))?;
            }
            other => unsupported_node!(other),
        }
    }

    Ok(builder.build())
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
            let value = if let Some(arg) = &stmt.arg {
                compile_expr(builder, arg)?
            } else {
                builder.emit(Instr::Const(Value::Undefined)).into()
            };

            builder.emit(Instr::Return(value));
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
            let condition = compile_expr(builder, &while_stmt.test)?;
            let neg_condition = builder.emit(Instr::Not(condition));
            let jmpif = builder.reserve();
            compile_stmt(builder, &while_stmt.body)?;
            builder.emit(Instr::Jmp(while_header_iid));
            let while_end_iid = builder.peek_iid();

            *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                cond: neg_condition.into(),
                dest: while_end_iid,
            };

            Ok(())
        }

        // Stmt::DoWhile(_) => todo!(),
        // Stmt::For(_) => todo!(),
        Stmt::ForIn(forin_stmt) => {
            let iteree = compile_expr(builder, &forin_stmt.right)?;

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

            let keys = builder.emit(Instr::ObjGetKeys(iteree));
            let keys_count = builder.emit(Instr::ArrayLen(keys));
            let key_ndx = builder.emit(Instr::Const(Value::Number(0.0)));

            builder.cur_fnb().push_scope();

            let has_next = builder.emit(Instr::Cmp {
                op: CmpOp::GE,
                a: key_ndx,
                b: keys_count,
            });
            let exit = builder.reserve();
            let key = builder.emit(Instr::ArrayNth(keys, key_ndx));
            builder.define_var(item_var_name.clone(), key);

            compile_stmt(builder, &forin_stmt.body)?;

            let one = builder.emit(Instr::Const(Value::Number(1.0)));
            let next_key_ndx = builder.emit(Instr::Arith {
                op: ArithOp::Add,
                a: key_ndx,
                b: one,
            });
            builder.emit(Instr::SetVar {
                var: key_ndx,
                value: next_key_ndx,
            });

            builder.cur_fnb().pop_scope();
            builder.emit(Instr::Jmp(has_next));

            let exit_label = builder.peek_iid();

            *builder.get_mut(exit).unwrap() = Instr::JmpIf {
                cond: has_next,
                dest: exit_label,
            };

            Ok(())
        }

        // Stmt::ForOf(_) => todo!(),
        Stmt::Decl(decl) => compile_decl(builder, decl, None),

        Stmt::Expr(expr) => {
            compile_expr(builder, &expr.expr)?;
            Ok(())
        }

        Stmt::If(if_stmt) => {
            let condition = compile_expr(builder, &if_stmt.test)
                .with_context(error!("in if statement").with_span(if_stmt.span))?;

            if let Some(else_blk) = &if_stmt.alt {
                // with an else block:
                //      jmpif (cond) -> CONS
                //          < if false >
                //      jmp END
                // CONS     < if true >
                // END      < rest... >

                let jmpif = builder.reserve();
                compile_stmt(builder, else_blk)?;
                let jmp_end = builder.reserve();
                let if_true_start: IID = builder.peek_iid();
                compile_stmt(builder, &if_stmt.cons)?;
                let end: IID = builder.peek_iid();

                *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                    cond: condition,
                    dest: if_true_start,
                };
                *builder.get_mut(jmp_end).unwrap() = Instr::Jmp(end);

                // TODO builder should fail if any reserved instruction is left untouched
            } else {
                // without an else block:
                //      jmpif (not cond) -> END
                //          < if true >
                // END      < rest... >

                let neg_condition = builder.emit(Instr::Not(condition));
                let jmpif = builder.reserve();
                compile_stmt(builder, &if_stmt.cons)?;
                let end: IID = builder.peek_iid();

                *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                    cond: neg_condition.into(),
                    dest: end,
                };
            }

            Ok(())
        }
        other => unsupported_node!(other),
    }
}

fn compile_decl(
    builder: &mut Builder,
    decl: &Decl,
    defs: Option<&mut Vec<(JsWord, IID)>>,
) -> Result<()> {
    match decl {
        // Decl::Class(_) => todo!(),
        Decl::Fn(fn_decl) => {
            if fn_decl.declare {
                panic!("unsupported case: fn_decl.declare");
            }

            let (name, _) = fn_decl.ident.to_id();
            let func = &fn_decl.function;
            let value = compile_function(builder, Some(name.clone()), func)?;
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

fn compile_function(builder: &mut Builder, name: Option<JsWord>, func: &Function) -> Result<IID> {
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
    let cap_vars = {
        let mut cap_vars = Vec::new();
        for var_name in inner_fnb.captures.iter() {
            let iid = builder.get_var(var_name)?;
            cap_vars.push(iid);
        }
        cap_vars
    };

    let closure_iid = builder.emit(Instr::ClosureNew {
        fnid: inner_fnb.fnid,
    });
    for var in cap_vars {
        builder.emit(Instr::ClosureAddCapture(var));
    }

    builder.fns.insert(inner_fnb.fnid, inner_fnb);
    Ok(closure_iid.into())
}

fn compile_var_decl(
    builder: &mut Builder,
    var_decl: &swc_ecma_ast::VarDecl,
    mut defs: Option<&mut Vec<(JsWord, IID)>>,
) -> Result<()> {
    use swc_ecma_ast::VarDeclKind;

    let _is_const = match var_decl.kind {
        VarDeclKind::Var => panic!("limitation: `var` bindings not supported"),
        VarDeclKind::Let => false,
        VarDeclKind::Const => true,
    };

    for decl in &var_decl.decls {
        let operand = match &decl.init {
            Some(expr) => compile_expr(builder, expr)?,
            None => builder.emit(Instr::Const(Value::Undefined)).into(),
        };

        let ident = decl
            .name
            .as_ident()
            .unwrap_or_else(|| unsupported_node!(decl));

        let name: JsWord = ident.id.to_id().0;
        if let Some(defs) = &mut defs {
            defs.push((name.clone(), operand));
        }
        builder.define_var(name, operand);
    }

    Ok(())
}

fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Result<IID> {
    use swc_ecma_ast::{CallExpr, Expr};

    match expr {
        Expr::Call(call_expr @ CallExpr { callee, args, .. }) => {
            if let Some(callee) = callee.as_expr() {
                if let Some(callee) = callee.as_ident() {
                    let sym = callee.sym.as_ref();
                    if sym == "sink" {
                        for arg in args {
                            let value = compile_expr(builder, &arg.expr)?;
                            builder.emit(Instr::PushSink(value));
                        }

                        return Ok(builder.emit(Instr::Const(Value::Null)).into());
                    } else if sym.starts_with("__start_trace") {
                        let trace_id = match args[0].expr.as_ref() {
                            Expr::Lit(Lit::Str(trace_id)) => trace_id.value.to_string(),
                            _ => {
                                panic!("__start_trace must be called with a trace ID: __start_trace('the-name-of-the-trace')")
                            }
                        };

                        if sym != "__start_trace" {
                            panic!("no such JIT builtin function: {sym}")
                        }

                        builder.place_trace_anchor(trace_id);
                        return Ok(builder.emit(Instr::Const(Value::Undefined)).into());
                    }
                }

                let fn_iid = compile_expr(builder, callee)?;
                let mut args_iids = vec![];
                for arg in args {
                    if arg.spread.is_some() {
                        panic!("unsupported: spread function parameter: function(a, b, ...)");
                    }
                    let iid = compile_expr(builder, &arg.expr)?;
                    args_iids.push(iid);
                }
                return Ok(builder
                    .emit(Instr::Call {
                        callee: fn_iid,
                        args: args_iids,
                    })
                    .into());
            }

            Err(
                error!("only calls to simple identifiers are supported for now")
                    .with_span(call_expr.span),
            )
        }

        Expr::Bin(bin_expr) => {
            let a = compile_expr(builder, &bin_expr.left)?;
            let b = compile_expr(builder, &bin_expr.right)?;

            let arith = |op, a, b| Instr::Arith { a, b, op };
            let cmp = |op, a, b| Instr::Cmp { a, b, op };
            let bool_op = |op, a, b| Instr::BoolOp { a, b, op };
            let instr = match bin_expr.op {
                BinaryOp::Add => arith(ArithOp::Add, a, b),
                BinaryOp::Sub => arith(ArithOp::Sub, a, b),
                BinaryOp::Mul => arith(ArithOp::Mul, a, b),
                BinaryOp::Div => arith(ArithOp::Div, a, b),
                BinaryOp::Lt => cmp(CmpOp::LT, a, b),
                BinaryOp::LtEq => cmp(CmpOp::LE, a, b),
                BinaryOp::Gt => cmp(CmpOp::GT, a, b),
                BinaryOp::GtEq => cmp(CmpOp::GE, a, b),
                BinaryOp::EqEqEq => cmp(CmpOp::EQ, a, b),
                BinaryOp::NotEqEq => cmp(CmpOp::NE, a, b),

                // TODO TODO TODO This does not implement any of the 'wat' semantics of JavaScript
                // See https://www.destroyallsoftware.com/talks/wat
                BinaryOp::EqEq => cmp(CmpOp::EQ, a, b),
                BinaryOp::NotEq => cmp(CmpOp::NE, a, b),

                BinaryOp::LogicalAnd => bool_op(BoolOp::And, a, b),
                BinaryOp::LogicalOr => bool_op(BoolOp::Or, a, b),

                _ => panic!("unsupported binary op: {:?}", bin_expr.op),
            };

            Ok(builder.emit(instr).into())
        }

        Expr::Lit(lit) => {
            let value = match lit {
                Lit::Num(number) => builder.emit(Instr::Const(Value::Number(number.value))),
                Lit::Str(s) => {
                    let value = s.value.to_string();
                    builder.emit(Instr::Const(Value::String(value)))
                }
                Lit::Bool(bv) => builder.emit(Instr::Const(Value::Bool(bv.value))),
                Lit::Null(_) => builder.emit(Instr::Const(Value::Null)),
                Lit::Regex(re_lit) => {
                    let constructor_fid = *builder
                        .native_fns
                        .get(&JsWord::from("RegExp"))
                        .expect("undefined native constructor: RegExp");

                    // TODO More efficient bytecode here?
                    let constructor = builder.emit(Instr::GetNativeFn(constructor_fid));
                    let re_str = re_lit.exp.to_string().into();
                    let re_str = builder.emit(Instr::Const(re_str));
                    let obj = builder.emit(Instr::Call {
                        callee: constructor,
                        args: vec![re_str],
                    });

                    let prototype_literal =
                        builder.emit(Instr::Const(Value::String("prototype".into())));
                    let proto = builder.emit(Instr::ObjGet {
                        obj: constructor,
                        key: prototype_literal,
                    });

                    let key = builder.emit(Instr::Const(Value::String("__proto__".into())));
                    builder.emit(Instr::ObjSet {
                        obj,
                        key,
                        value: proto,
                    });

                    obj
                }
                // Lit::BigInt(_) => todo!(),
                // Lit::JSXText(_) => todo!(),
                other => unsupported_node!(other),
            };

            Ok(value)
        }

        Expr::Ident(ident) => {
            let value = if &ident.sym == "undefined" {
                builder.emit(Instr::Const(Value::Undefined))
            } else {
                builder.get_var(&ident.sym)?
            };
            Ok(value)
        }

        Expr::Assign(asmt) => compile_assignment(asmt, builder)
            .with_context(error!("in assignment").with_span(asmt.span)),

        Expr::Object(obj_expr) => {
            let obj = builder.emit(Instr::ObjNew);

            for prop_or_spread in obj_expr.props.iter() {
                match prop_or_spread {
                    swc_ecma_ast::PropOrSpread::Spread(spread_elm) => {
                        return Err(error!("spread syntax (...) is currently unsupported")
                            .with_span(spread_elm.dot3_token))
                    }
                    swc_ecma_ast::PropOrSpread::Prop(prop) => match prop.as_ref() {
                        swc_ecma_ast::Prop::KeyValue(kv_expr) => {
                            let value = compile_expr(builder, &kv_expr.value)?;
                            let key = compile_prop_name(&kv_expr.key)?;
                            let key = builder.emit(Instr::Const(key)).into();

                            builder.emit(Instr::ObjSet {
                                obj: obj.clone(),
                                key,
                                value,
                            });
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
                            let value = compile_function(builder, None, &*method_prop.function)?;
                            let key = compile_prop_name(&method_prop.key)?;
                            let key = builder.emit(Instr::Const(key)).into();

                            builder.emit(Instr::ObjSet {
                                obj: obj.clone(),
                                key,
                                value,
                            });
                        }
                    },
                }
            }

            Ok(obj)
        }

        // Expr::This(_) => todo!(),
        Expr::Array(arr_expr) => {
            use swc_ecma_ast::ExprOrSpread;
            let array = builder.emit(Instr::ObjNew);
            for elem in arr_expr.elems.iter() {
                match elem {
                    Some(ExprOrSpread { spread, expr }) => {
                        if spread.is_some() {
                            return Err(error!("spread syntax is currently unsupported")
                                .with_span(arr_expr.span));
                        }
                        let elem = compile_expr(builder, expr)?;
                        builder.emit(Instr::ArrayPush(array.clone(), elem));
                    }
                    None => {
                        eprintln!(
                            "warning: an element in Expr::Array is `None` ({:?})",
                            arr_expr.span
                        );
                    }
                }
            }

            Ok(array)
        }

        Expr::Fn(fn_expr) => {
            // TODO Refactor this with Decl::Fn
            let name = fn_expr.ident.as_ref().map(|ident| ident.to_id().0);
            let func = &fn_expr.function;

            let value = compile_function(builder, name.clone(), func)?;

            if let Some(name) = name {
                builder.define_var(name, value);
            }
            Ok(value)
        }

        Expr::Unary(unary_expr) => {
            match unary_expr.op {
                swc_ecma_ast::UnaryOp::Bang => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    Ok(builder.emit(Instr::Not(arg)).into())
                }
                swc_ecma_ast::UnaryOp::TypeOf => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    Ok(builder.emit(Instr::TypeOf(arg)).into())
                }
                swc_ecma_ast::UnaryOp::Minus => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    Ok(builder.emit(Instr::UnaryMinus(arg)).into())
                }
                other => unsupported_node!(other),
                // swc_ecma_ast::UnaryOp::Plus => todo!(),
                // swc_ecma_ast::UnaryOp::Tilde => todo!(),
                // swc_ecma_ast::UnaryOp::Void => todo!(),
                // swc_ecma_ast::UnaryOp::Delete => todo!(),
            }
        }
        Expr::Update(update_expr) => {
            if let Expr::Ident(ident) = &*update_expr.arg {
                // NOTE: update_expr.prefix does not matter in this case, but
                // it will matter when this code is extended to other types of args
                let var = builder.get_var(&ident.sym)?;
                // TODO Use integers here, when they get implemented
                let one = builder.emit(Instr::Const(Value::Number(1.0)));
                let op = match update_expr.op {
                    UpdateOp::PlusPlus => ArithOp::Add,
                    UpdateOp::MinusMinus => ArithOp::Sub,
                };
                let value: IID = builder.emit(Instr::Arith { op, a: var, b: one });
                builder.emit(Instr::SetVar {
                    var,
                    value: value.clone(),
                });
                Ok(value)
            } else {
                todo!("unsupported: UpdateExpr on anything other than an identifier")
            }
        }
        Expr::Member(member_expr) => {
            let (_, _, value) = compile_member_access(builder, member_expr)?;
            Ok(value)
        }

        Expr::Paren(inner) => compile_expr(builder, &inner.expr),

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
}

fn compile_prop_name(prop_name: &swc_ecma_ast::PropName) -> Result<Value> {
    match &prop_name {
        swc_ecma_ast::PropName::Ident(ident) => Ok(Value::from(ident.sym.to_string())),
        swc_ecma_ast::PropName::Str(s) => Ok(Value::from(s.value.to_string())),
        swc_ecma_ast::PropName::Num(num) => Ok(Value::Number(num.value)),
        swc_ecma_ast::PropName::Computed(x) => {
            Err(error!("unsupported node: {:?}", x).with_span(x.span))
        }
        swc_ecma_ast::PropName::BigInt(x) => {
            Err(error!("unsupported node: {:?}", x).with_span(x.span))
        }
    }
}

fn compile_assignment(asmt: &swc_ecma_ast::AssignExpr, builder: &mut Builder) -> Result<IID> {
    use swc_ecma_ast::{Expr, MemberExpr, MemberProp, PatOrExpr};

    if let Some(ident) = asmt.left.as_ident() {
        let var = builder.get_var(&ident.sym)?;
        let new_value = compile_assignment_rhs(builder, asmt, var)?;
        builder.emit(Instr::SetVar {
            var,
            value: new_value,
        });
        Ok(var)
    } else if let Some(target_expr) = asmt.left.as_expr() {
        match target_expr {
            Expr::Member(member_expr) => {
                let (obj, key, old_value) = compile_member_access(builder, member_expr)?;
                let new_value = compile_assignment_rhs(builder, asmt, old_value)?;
                builder.emit(Instr::ObjSet {
                    obj,
                    key,
                    value: new_value.clone(),
                });
                Ok(new_value)
            }
            // We should have already handled this case in the `if let ... = asm.left.as_ident()`
            // case
            Expr::Ident(_) => unreachable!(),
            _ => Err(error!("assignment to an expression is unsupported").with_span(asmt.span)),
        }
    } else {
        panic!("unsupported pattern as assignment target: {:?}", asmt.left)
    }
}

fn compile_member_access(
    builder: &mut Builder,
    member_expr: &swc_ecma_ast::MemberExpr,
) -> Result<(IID, IID, IID)> {
    use swc_ecma_ast::MemberProp;

    let obj = compile_expr(builder, member_expr.obj.as_ref())?;
    let key: IID = match &member_expr.prop {
        MemberProp::Ident(prop_ident) => {
            let prop_ident = prop_ident.sym.to_string().into();
            builder.emit(Instr::Const(prop_ident)).into()
        }
        _ => {
            return Err(
                error!("accessing an object with non-identifer key is unsupported")
                    .with_span(member_expr.span),
            )
        }
    };

    let value = builder
        .emit(Instr::ObjGet {
            obj: obj.clone(),
            key: key.clone(),
        })
        .into();
    Ok((obj, key, value))
}

/// Read the operand and compute its new value, as requested by the given assignment
/// expression. This is part of the compilation procedure of assignment expressions.
fn compile_assignment_rhs(
    builder: &mut Builder,
    asmt: &swc_ecma_ast::AssignExpr,
    target_operand: IID,
) -> Result<IID> {
    let right = compile_expr(builder, &asmt.right)?;
    let value = match asmt.op {
        AssignOp::Assign => right,
        AssignOp::AddAssign => builder
            .emit(Instr::Arith {
                op: ArithOp::Add,
                a: target_operand,
                b: right,
            })
            .into(),
        AssignOp::SubAssign => builder
            .emit(Instr::Arith {
                op: ArithOp::Sub,
                a: target_operand,
                b: right,
            })
            .into(),
        AssignOp::MulAssign => builder
            .emit(Instr::Arith {
                op: ArithOp::Mul,
                a: target_operand,
                b: right,
            })
            .into(),
        AssignOp::DivAssign => builder
            .emit(Instr::Arith {
                op: ArithOp::Div,
                a: target_operand,
                b: right,
            })
            .into(),
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

    fn quick_compile(code: &str) -> bytecode::Module {
        Compiler::new()
            .compile_file("<input>".to_string(), code.to_string())
            .unwrap()
    }

    #[test]
    fn test_bytecode_object_init() {
        let module = quick_compile(
            "sink({
                aString: 'asdlol123',
                aNumber: 1239423.4518923,
                anotherObject: { x: 123, y: 899 },
                aFunction: function(pt) { return 42; }
            })",
        );

        let root_fn = module.get_function(FnId::ROOT_FN).unwrap();
        let instrs = root_fn.instrs();
        let the_obj_id = instrs
            .iter()
            .find_map(|instr| match instr {
                Instr::PushSink(iid) => Some(*iid),
                _ => None,
            })
            .unwrap();
        let keys_iids: HashSet<_> = instrs
            .iter()
            .filter_map(|instr| match instr {
                Instr::ObjSet { obj, key, .. } if *obj == the_obj_id => Some(key),
                _ => None,
            })
            .collect();
        let keys: Vec<bytecode::Value> = instrs
            .iter()
            .enumerate()
            .filter(|(ndx, _)| keys_iids.contains(&IID(*ndx as _)))
            .filter_map(|(_, inst)| match inst {
                Instr::Const(value) => Some(value.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            keys.as_slice(),
            &[
                bytecode::Value::String("aString".into()),
                bytecode::Value::String("aNumber".into()),
                bytecode::Value::String("anotherObject".into()),
                bytecode::Value::String("aFunction".into()),
            ]
        );
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
            Instr::Const(Value::Number(val)) => val,
            _ => panic!(),
        };
        assert_eq!(*cap_const, 10.0);
    }

    #[test]
    fn test_upvalues_callee() {
        let module = quick_compile(CODE_UPVALUES);
        module.dump();

        let callee = module.get_function(FnId(1)).unwrap();
        assert!(std::matches!(callee.instrs()[0], Instr::GetCapture(0)));
    }

    #[test]
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
