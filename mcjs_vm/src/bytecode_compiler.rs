use std::collections::{HashMap, HashSet};

use swc_atoms::JsWord;
use swc_common::{sync::Lrc, SourceMap, Span};
use swc_ecma_ast::{
    ArrowExpr, AssignOp, BinaryOp, Decl, ExportDecl, ForHead, Function, Lit, Pat, UpdateOp, VarDecl,
};

use crate::bytecode::{self, FnId, Instr, Literal, NativeFnId, VReg, IID};
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
    fn get_module_id(&mut self, filename: &str) -> Option<bytecode::ModuleId>;
    fn read_source(&self, module_id: bytecode::ModuleId) -> String;
}

pub struct BuilderParams {
    pub loader: Box<dyn Loader>,
}

pub struct Builder {
    loader: Box<dyn Loader>,
    fns: HashMap<FnId, FnBuilder>,
    next_fnid: u32,
    next_module_id: u16,

    rootfn_of_module: HashMap<bytecode::ModuleId, FnId>,

    // --- below this line, things are valid during the compilation of a single module.  at the
    // end, these are reset
    fn_stack: Vec<FnBuilder>,
}

impl BuilderParams {
    pub fn to_builder(self) -> Builder {
        Builder {
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
        let module_id = self
            .loader
            .get_module_id(&filename)
            .ok_or_else(|| error!("no such file: {}", filename))?;
        if self.rootfn_of_module.contains_key(&module_id) {
            // The module had already been compiled
            return Ok(module_id);
        }

        let content = self.loader.read_source(module_id);

        let (source_map, ast_module) = parse_file(filename.clone(), content)
            .with_context(error!("while parsing file: {filename}"))?;

        let old_fn_stack_len = self.fn_stack.len();

        let res = compile_module(self, &ast_module);
        // In case of an error, fn_stack needs to be reset to allow for another compilation in
        // the future, or to resume the previous compilation (of the importing modulue).
        assert!(self.fn_stack.len() >= old_fn_stack_len);
        self.fn_stack.truncate(old_fn_stack_len);

        let root_fnid = res.with_context(
            error!("while compiling module: {filename}").with_source_map(source_map),
        )?;

        let prev = self.rootfn_of_module.insert(module_id, root_fnid);
        assert!(prev.is_none(), "duplicate module id");

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
    next_vreg: u8,
    n_params: bytecode::ArgIndex,
    // Places where a break instruction must be placed when the break target is finally "written"
    // by the compiler.
    pending_break_instrs: Vec<IID>,
    pending_continue_instrs: Vec<IID>,
    captures: Vec<String>,
}

/// Represents the location where a certain variable's value is stored.  This location
/// determines the correct instruction to use to read/write the variable.
#[derive(Clone, Copy, Debug)]
enum Loc {
    VReg(VReg),
    Arg(bytecode::ArgIndex),
    Capture(bytecode::CaptureIndex),
}

#[derive(Debug)]
struct Scope {
    // TODO Take advantage of identifier interning!
    // TODO After introducing VRegs, this could become a Vec
    vars: HashMap<JsWord, Loc>,
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
            n_params: bytecode::ArgIndex(0),
            pending_break_instrs: Vec::new(),
            pending_continue_instrs: Vec::new(),
        }
    }

    fn build(self) -> bytecode::Function {
        assert!(
            self.pending_break_instrs.is_empty(),
            "bytecode compiler bug: the function is over, but some break instructions were not placed yet"
        );
        bytecode::Function::new(
            self.instrs.into_boxed_slice(),
            self.consts.into_boxed_slice(),
            self.n_params,
            self.trace_anchors,
        )
    }

    fn get_mut(&mut self, iid: IID) -> Option<&mut Instr> {
        self.instrs.get_mut(iid.0 as usize)
    }
    fn reserve(&mut self) -> IID {
        self.emit(Instr::Nop)
    }
    fn reserve_break(&mut self) {
        let iid = self.peek_iid();
        self.emit(Instr::Nop);
        self.pending_break_instrs.push(iid);
    }
    fn resolve_break_to(&mut self, break_target: IID) {
        for iid in self.pending_break_instrs.drain(0..) {
            *self.instrs.get_mut(iid.0 as usize).unwrap() = Instr::Jmp(break_target);
        }
    }
    fn reserve_continue(&mut self) {
        let iid = self.peek_iid();
        self.emit(Instr::Nop);
        self.pending_continue_instrs.push(iid);
    }
    fn resolve_continue_to(&mut self, continue_target: IID) {
        for iid in self.pending_continue_instrs.drain(0..) {
            *self.instrs.get_mut(iid.0 as usize).unwrap() = Instr::Jmp(continue_target);
        }
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
    fn set_var_captured(&mut self, name: JsWord) -> Loc {
        let name_str = name.to_string();
        assert!(!self.captures.iter().any(|cap| cap == &name_str));

        self.captures.push(name_str);
        let cap_ndx: u16 = (self.captures.len() - 1)
            .try_into()
            .expect("too many captures!");
        let cap_ndx = bytecode::CaptureIndex(cap_ndx);
        debug_assert_eq!(
            self.captures[cap_ndx.0 as usize].as_bytes(),
            name.as_bytes()
        );

        // The variable now has to be defined in the outermost scope of this function.  Otherwise
        // it's likely that subsequent lookups will fail
        let loc = Loc::Capture(cap_ndx);
        let prev = self
            .scopes
            .first_mut()
            .unwrap()
            .vars
            .insert(name.clone(), loc);
        if prev.is_some() {
            // definition of var $name shadows previous definition
        }

        loc
    }

    fn get_var(&self, name: &JsWord) -> Option<Loc> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.vars.get(name))
            .copied()
    }

    fn define_var(&mut self, name: JsWord, loc: Loc) {
        if let Loc::Arg(arg_ndx) = loc {
            if self.n_params.0 < arg_ndx.0 + 1 {
                self.n_params.0 = arg_ndx.0 + 1;
            }
        }

        let scope = self.inner_scope_mut();
        let prev = scope.vars.insert(name, loc);
        if prev.is_some() {
            // definition of var $name shadows previous definition
        }
    }

    fn next_vreg(&mut self) -> VReg {
        let vreg_ndx = self.next_vreg;
        self.next_vreg += 1;
        VReg(vreg_ndx)
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
        self.cur_fnb().reserve()
    }

    fn reserve_break(&mut self) {
        self.cur_fnb().reserve_break();
    }
    fn resolve_break_to(&mut self, break_target: IID) {
        self.cur_fnb().resolve_break_to(break_target)
    }
    fn reserve_continue(&mut self) {
        self.cur_fnb().reserve_continue();
    }
    fn resolve_continue_to(&mut self, continue_target: IID) {
        self.cur_fnb().resolve_continue_to(continue_target)
    }
    fn get_mut(&mut self, iid: IID) -> Option<&mut Instr> {
        self.cur_fnb().get_mut(iid)
    }

    fn cur_fnb(&mut self) -> &mut FnBuilder {
        self.fn_stack.last_mut().expect("no FnBuilder!")
    }

    fn emit(&mut self, instr: Instr) -> IID {
        self.cur_fnb().emit(instr)
    }

    fn set_const(&mut self, vreg: VReg, value: bytecode::Literal) {
        match value {
            bytecode::Literal::Null => self.emit(Instr::LoadNull(vreg)),
            bytecode::Literal::Undefined => self.emit(Instr::LoadUndefined(vreg)),
            _ => {
                let fnb = self.cur_fnb();
                fnb.consts.push(value);
                let ndx = fnb.consts.len() - 1;
                let ndx = TryFrom::try_from(ndx).expect("too many constants!");
                self.emit(Instr::LoadConst(vreg, bytecode::ConstIndex(ndx)))
            }
        };
    }

    fn peek_iid(&mut self) -> IID {
        self.cur_fnb().peek_iid()
    }

    fn define_var(&mut self, name: JsWord, loc: Loc) {
        self.cur_fnb().define_var(name, loc)
    }

    fn find_vreg(&mut self, sym: &JsWord) -> Option<(usize, Loc)> {
        self.fn_stack
            .iter()
            .rev()
            .enumerate()
            .find_map(|(ndx, fnb)| fnb.get_var(sym).map(|vreg| (ndx, vreg)))
            .or_else(|| {
                let vreg = self.new_vreg();
                self.set_const(vreg, Literal::String(sym.to_string()));
                self.emit(Instr::GetGlobal {
                    dest: vreg,
                    key: vreg,
                });
                Some((0, Loc::VReg(vreg)))
            })
    }

    fn get_var(&mut self, sym: &JsWord) -> Result<Loc> {
        match self.find_vreg(sym) {
            // Hell yeah, found it in the current function, we have a "simple" IID
            Some((0, vreg)) => Ok(vreg),
            Some((_, _)) => {
                // Found it in one of the enclosing functions.  We'll have to add a capture, but we
                // know the variable is there
                let vreg = self.cur_fnb().set_var_captured(sym.clone());

                {
                    let (depth, _) = self.find_vreg(sym).unwrap();
                    assert_eq!(depth, 0);
                }

                Ok(vreg)
            }
            None => Err(error!("unbound symbol: {}", sym)),
        }
    }

    fn read_var(&mut self, sym: &JsWord) -> Result<VReg> {
        let loc = self.get_var(sym)?;
        let vreg = match loc {
            Loc::VReg(vreg) => vreg,
            Loc::Arg(arg_ndx) => {
                let vreg = self.new_vreg();
                self.emit(Instr::LoadArg(vreg, arg_ndx));
                vreg
            }
            Loc::Capture(cap_ndx) => {
                let vreg = self.new_vreg();
                self.emit(Instr::LoadCapture(vreg, cap_ndx));
                vreg
            }
        };
        Ok(vreg)
    }

    fn new_vreg(&mut self) -> VReg {
        self.cur_fnb().next_vreg()
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
                    self.define_var(ident.sym.clone(), Loc::Arg(bytecode::ArgIndex(param_ndx)));
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
    use swc_ecma_ast::{ExportDecl, ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    builder.start_function(None, &[]);

    let lit_named = builder.new_vreg();
    builder.set_const(lit_named, Literal::String("__named".to_string()));

    let lit_default = builder.new_vreg();
    builder.set_const(lit_default, Literal::String("__default".to_string()));

    // Exports object.
    let module_obj = builder.new_vreg();
    builder.emit(Instr::ObjCreateEmpty(module_obj));

    let mod_named_exports = builder.new_vreg();
    builder.emit(Instr::ObjCreateEmpty(mod_named_exports));

    let mut mod_default_export = None;

    // We have to handle hoisting here like we do for blocks.  See [#hoisting].

    for item in &ast_module.body {
        match item {
            ModuleItem::Stmt(Stmt::Decl(Decl::Var(var_decl)))
            | ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(ExportDecl {
                decl: Decl::Var(var_decl),
                ..
            })) => compile_var_decl_namedef(builder, var_decl),

            ModuleItem::Stmt(Stmt::Decl(Decl::Fn(fn_decl)))
            | ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(ExportDecl {
                decl: Decl::Fn(fn_decl),
                ..
            })) => compile_fn_decl_namedef(builder, fn_decl),

            _ => {}
        }
    }

    for item in &ast_module.body {
        match item {
            ModuleItem::Stmt(Stmt::Decl(Decl::Fn(fn_decl)))
            | ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(ExportDecl {
                decl: Decl::Fn(fn_decl),
                ..
            })) => compile_fn_decl_assignment(builder, fn_decl)?,

            _ => {}
        }
    }

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

                                    let item_name = builder.new_vreg();
                                    builder.set_const(
                                        item_name,
                                        Literal::String(d.local.sym.to_string()),
                                    );

                                    let local_var = builder.new_vreg();
                                    builder.emit(Instr::GetModule(local_var, module_id));
                                    builder.emit(Instr::ObjGet {
                                        dest: local_var,
                                        obj: local_var,
                                        key: lit_named,
                                    });
                                    builder.emit(Instr::ObjGet {
                                        dest: local_var,
                                        obj: local_var,
                                        key: item_name,
                                    });
                                    builder.define_var(d.local.sym.clone(), Loc::VReg(local_var));
                                }
                            }
                            swc_ecma_ast::ImportSpecifier::Default(d) => {
                                let local_var = builder.new_vreg();
                                builder.emit(Instr::GetModule(local_var, module_id));
                                builder.emit(Instr::ObjGet {
                                    dest: local_var,
                                    obj: local_var,
                                    key: lit_default,
                                });
                                builder.define_var(d.local.sym.clone(), Loc::VReg(local_var));
                            }
                            swc_ecma_ast::ImportSpecifier::Namespace(d) => {
                                let lit_named = builder.new_vreg();
                                builder
                                    .set_const(lit_named, Literal::String("__named".to_string()));

                                let local_var = builder.new_vreg();
                                builder.emit(Instr::GetModule(local_var, module_id));
                                builder.emit(Instr::ObjGet {
                                    dest: local_var,
                                    obj: local_var,
                                    key: lit_named,
                                });
                                builder.define_var(d.local.sym.clone(), Loc::VReg(local_var));
                            }
                        }
                    }
                }

                ModuleDecl::ExportDecl(export_decl) => match &export_decl.decl {
                    // Skip: already processed in previous phase
                    Decl::Fn(_) => {}
                    // Only do assignment part
                    Decl::Var(var_decl) => {
                        compile_var_decl_assignment(builder, &*var_decl)?;
                    }
                    Decl::Class(_)
                    | Decl::TsInterface(_)
                    | Decl::TsTypeAlias(_)
                    | Decl::TsEnum(_)
                    | Decl::TsModule(_)
                    | Decl::Using(_) => unsupported_node!(export_decl.decl),
                },
                ModuleDecl::ExportNamed(_) => todo!("ExportNamed"),
                ModuleDecl::ExportDefaultDecl(decl) => match &decl.decl {
                    swc_ecma_ast::DefaultDecl::Class(_) => todo!("export default class"),
                    swc_ecma_ast::DefaultDecl::Fn(fn_expr) => {
                        let name = fn_expr.ident.as_ref().map(|ident| ident.sym.clone());
                        let vreg = compile_function(builder, name, &*fn_expr.function)?;
                        mod_default_export = Some(vreg);
                    }
                    swc_ecma_ast::DefaultDecl::TsInterfaceDecl(ts_decl) => eprintln!(
                        "warning: discarded TypeScript interface export: {}",
                        ts_decl.id.to_string()
                    ),
                },
                ModuleDecl::ExportDefaultExpr(decl) => {
                    let vreg = compile_expr(builder, &*decl.expr)?;
                    mod_default_export = Some(vreg);
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
        }
    }

    for item in &ast_module.body {
        if let ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(ExportDecl {
            decl: Decl::Var(var_decl),
            ..
        })) = item
        {
            for decl in &var_decl.decls {
                let name = get_var_decl_name(decl);
                let value = builder
                    .read_var(&name)
                    .with_context(error!("in decl").with_span(decl.span))?;

                let key = builder.new_vreg();
                builder.set_const(key, Literal::String(name.to_string()));

                builder.emit(Instr::ObjSet {
                    obj: mod_named_exports,
                    key,
                    value,
                });
            }
        }
    }

    builder.emit(Instr::ObjSet {
        obj: module_obj,
        key: lit_named,
        value: mod_named_exports,
    });

    let mod_default_export = mod_default_export.unwrap_or_else(|| builder.new_vreg());
    builder.emit(Instr::ObjSet {
        obj: module_obj,
        key: lit_default,
        value: mod_default_export,
    });

    builder.emit(Instr::Return(module_obj));
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

            let stmts = &block.stmts;

            compile_block(builder, stmts)?;

            builder.cur_fnb().pop_scope();
            Ok(())
        }
        // Stmt::Empty(_) => todo!(),
        // Stmt::Debugger(_) => todo!(),
        // Stmt::With(_) => todo!(),
        Stmt::Return(stmt) => {
            let reg = if let Some(arg) = &stmt.arg {
                compile_expr(builder, arg)?
            } else {
                let reg = builder.new_vreg();
                builder.set_const(reg, Literal::Undefined);
                reg
            };

            builder.emit(Instr::Return(reg));
            Ok(())
        }
        // Stmt::Labeled(_) => todo!(),
        Stmt::Break(_) => {
            builder.reserve_break();
            Ok(())
        }

        Stmt::Continue(continue_stmt) => {
            assert!(
                continue_stmt.label.is_none(),
                "unsupported: continue to label"
            );
            builder.reserve_continue();
            Ok(())
        }
        Stmt::Switch(switch_stmt) => {
            let discriminant = compile_expr(builder, &switch_stmt.discriminant)?;

            // TODO(performance): any better allocation strategy?
            // NOTE: This Vec skips the `default:` label!
            let cmp_result = builder.new_vreg();
            let mut case_jumps = Vec::with_capacity(switch_stmt.cases.len());

            for case in &switch_stmt.cases {
                if let Some(test) = &case.test {
                    let value = compile_expr(builder, test)?;
                    builder.emit(Instr::CmpEQ(cmp_result, discriminant, value));
                    case_jumps.push(builder.reserve()); // Jump to label for this case
                }
            }

            let default_jump_iid = builder.reserve();
            let mut default_written = false;
            let mut case_jumps = case_jumps.into_iter();

            for case in switch_stmt.cases.iter() {
                let label = builder.peek_iid();
                if case.test.is_some() {
                    let jump_iid = case_jumps.next().unwrap();
                    *builder.get_mut(jump_iid).unwrap() = Instr::JmpIf {
                        cond: cmp_result,
                        dest: label,
                    };
                } else {
                    *builder.get_mut(default_jump_iid).unwrap() = Instr::Jmp(label);
                    default_written = true;
                }

                for stmt in &case.cons {
                    compile_stmt(builder, stmt)?;
                }
            }

            let break_target = builder.peek_iid();
            if !default_written {
                *builder.get_mut(default_jump_iid).unwrap() = Instr::Jmp(break_target);
            }
            builder.resolve_break_to(break_target);

            Ok(())
        }

        Stmt::Throw(throw_stmt) => {
            let exception = compile_expr(builder, &*throw_stmt.arg)?;
            builder.emit(Instr::Throw(exception));
            Ok(())
        }

        // Stmt::Try(_) => todo!(),
        Stmt::While(while_stmt) => {
            let while_header_iid = builder.peek_iid();

            let cond = compile_expr(builder, &while_stmt.test)?;
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let jmpif = builder.reserve();

            compile_stmt(builder, &while_stmt.body)?;
            builder.emit(Instr::Jmp(while_header_iid));
            let while_end_iid = builder.peek_iid();

            *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                cond,
                dest: while_end_iid,
            };

            Ok(())
        }

        Stmt::DoWhile(stmt) => {
            let loop_start = builder.peek_iid();
            compile_stmt(builder, &*stmt.body)?;
            let cond = compile_expr(builder, &*stmt.test)?;
            builder.emit(Instr::JmpIf {
                cond,
                dest: loop_start,
            });
            Ok(())
        }

        Stmt::For(stmt) => {
            if let Some(init) = &stmt.init {
                use swc_ecma_ast::VarDeclOrExpr;
                match init {
                    VarDeclOrExpr::VarDecl(var_decl) => {
                        // This one we don't need to hoist really.  The for statement has its own
                        // scope.
                        compile_var_decl_namedef(builder, &var_decl);
                        compile_var_decl_assignment(builder, &var_decl)?;
                    }
                    VarDeclOrExpr::Expr(expr) => {
                        compile_expr(builder, &expr)?;
                    }
                }
            }

            let loop_start = builder.peek_iid();
            let cond = match &stmt.test {
                Some(test) => Some(compile_expr(builder, &*test)?),
                None => None,
            };
            let jmpif = builder.reserve();

            compile_stmt(builder, &stmt.body)?;

            let continue_target = builder.peek_iid();
            if let Some(update) = &stmt.update {
                compile_expr(builder, &*update)?;
            }

            builder.emit(Instr::Jmp(loop_start));
            let loop_end = builder.peek_iid();

            *builder.get_mut(jmpif).unwrap() = if let Some(cond) = cond {
                Instr::JmpIf {
                    cond,
                    dest: loop_end,
                }
            } else {
                Instr::Jmp(loop_end)
            };
            builder.resolve_break_to(loop_end);
            builder.resolve_continue_to(continue_target);

            Ok(())
        }

        Stmt::ForIn(forin_stmt) => {
            let item_var = match &forin_stmt.left {
                swc_ecma_ast::ForHead::VarDecl(var_decl) => var_decl.as_ref(),
                swc_ecma_ast::ForHead::UsingDecl(_) => todo!(),
                swc_ecma_ast::ForHead::Pat(_) => panic!(
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

            let key_ndx = builder.new_vreg();
            builder.set_const(key_ndx, Literal::Number(0.0));

            let iteree = compile_expr(builder, &forin_stmt.right)?;
            let keys = builder.new_vreg();
            builder.emit(Instr::ObjGetKeys {
                dest: keys,
                obj: iteree,
            });

            let while_begin = builder.peek_iid();
            let cond = builder.new_vreg();
            builder.emit(Instr::ArrayLen {
                dest: cond,
                arr: keys,
            });
            builder.emit(Instr::CmpLT(cond, key_ndx, cond));
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let exit = builder.reserve();

            builder.cur_fnb().push_scope();
            let key = builder.new_vreg();
            builder.emit(Instr::ArrayNth {
                dest: key,
                arr: keys,
                index: key_ndx,
            });
            builder.define_var(item_var_name.clone(), Loc::VReg(key));

            compile_stmt(builder, &forin_stmt.body)?;

            builder.emit(Instr::ArithInc(key_ndx, key_ndx));

            builder.cur_fnb().pop_scope();
            builder.emit(Instr::Jmp(while_begin));
            let exit_label = builder.peek_iid();

            *builder.get_mut(exit).unwrap() = Instr::JmpIf {
                cond,
                dest: exit_label,
            };

            Ok(())
        }

        Stmt::ForOf(forof_stmt) => {
            let item_var = match &forof_stmt.left {
                ForHead::VarDecl(var_decl) => var_decl.as_ref(),
                ForHead::Pat(_) => panic!(
                    "unsupported syntax: destructuring pattern as `<pattern>` in: `for (<pattern> of ...) {{ ... }}`"
                ),
                ForHead::UsingDecl(_) => panic!(
                    "unsupported syntax: `using` declaration as for head"
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

            let iteree = compile_expr(builder, &forof_stmt.right)?;
            let iterator = builder.new_vreg();
            builder.emit(Instr::NewIterator {
                dest: iterator,
                obj: iteree,
            });

            let loop_start = builder.peek_iid();
            let exit = builder.reserve();

            builder.cur_fnb().push_scope();

            let item = builder.new_vreg();
            builder.emit(Instr::IteratorGetCurrent {
                dest: item,
                iter: iterator,
            });
            builder.define_var(item_var_name.clone(), Loc::VReg(item));
            compile_stmt(builder, &forof_stmt.body)?;

            builder.cur_fnb().pop_scope();

            builder.emit(Instr::IteratorAdvance { iter: iterator });
            builder.emit(Instr::Jmp(loop_start));
            let exit_label = builder.peek_iid();

            *builder.get_mut(exit).unwrap() = Instr::JmpIfIteratorFinished {
                iter: iterator,
                dest: exit_label,
            };

            Ok(())
        }

        // Function declarations are be processed in a dedicated phase, not here.
        // The assignment part of variable declarations is evaluated here instead.
        // See [#hoisting]
        Stmt::Decl(Decl::Var(var_decl)) => compile_var_decl_assignment(builder, &var_decl),
        Stmt::Decl(Decl::Fn(_)) => Ok(()),

        Stmt::Expr(expr) => {
            compile_expr(builder, &expr.expr)?;
            Ok(())
        }

        Stmt::If(if_stmt) => {
            let cond = compile_expr(builder, &if_stmt.test)
                .with_context(error!("in if statement").with_span(if_stmt.span))?;
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let jmp_to_alt = builder.reserve();

            compile_stmt(builder, &if_stmt.cons)?;
            let jmp_to_end = builder.reserve();

            let alt = builder.peek_iid();
            if let Some(else_blk) = &if_stmt.alt {
                compile_stmt(builder, else_blk)?;
            }

            let end: IID = builder.peek_iid();

            *builder.get_mut(jmp_to_alt).unwrap() = Instr::JmpIf { cond, dest: alt };
            *builder.get_mut(jmp_to_end).unwrap() = Instr::JmpIf { cond, dest: end };
            Ok(())
        }
        other => unsupported_node!(other),
    }
}

fn compile_block(builder: &mut Builder, stmts: &Vec<swc_ecma_ast::Stmt>) -> Result<()> {
    use swc_ecma_ast::Stmt;

    for stmt in stmts {
        match stmt {
            Stmt::Decl(Decl::Var(var_decl)) => {
                compile_var_decl_namedef(builder, var_decl);
            }
            Stmt::Decl(Decl::Fn(fn_decl)) => {
                compile_fn_decl_namedef(builder, fn_decl);
            }
            _ => (),
        }
    }

    for stmt in stmts {
        if let Stmt::Decl(Decl::Fn(fn_decl)) = stmt {
            compile_fn_decl_assignment(builder, fn_decl)?;
        }
    }

    for stmt in stmts {
        compile_stmt(builder, stmt)?;
    }

    Ok(())
}

fn compile_var_decl_namedef(builder: &mut Builder, var_decl: &swc_ecma_ast::VarDecl) {
    for decl in &var_decl.decls {
        let name = get_var_decl_name(decl);
        // Yet-unassigned vregs are implicitly initialized to Undefined, so no
        // explicit instruction is required here.
        let vreg = builder.new_vreg();
        builder.define_var(name, Loc::VReg(vreg));
    }
}

fn compile_var_decl_assignment(
    builder: &mut Builder,
    var_decl: &swc_ecma_ast::VarDecl,
) -> Result<()> {
    use swc_ecma_ast::VarDeclKind;

    let _is_const = match var_decl.kind {
        VarDeclKind::Var => false, // panic!("limitation: `var` bindings not supported"),
        VarDeclKind::Let => false,
        VarDeclKind::Const => true,
    };

    for decl in &var_decl.decls {
        let ident = decl
            .name
            .as_ident()
            .unwrap_or_else(|| unsupported_node!(decl));
        let name: JsWord = ident.id.to_id().0;
        let vreg = builder
            .read_var(&name)
            .with_context(error!("resolving identifier").with_span(ident.span))?;

        if let Some(expr) = &decl.init {
            let value = compile_expr(builder, expr)?;
            // TODO Remove this instruction.
            builder.emit(Instr::Copy {
                dst: vreg,
                src: value,
            });
        } else {
            builder.set_const(vreg, Literal::Undefined);
        }
    }

    Ok(())
}

fn get_var_decl_name(decl: &swc_ecma_ast::VarDeclarator) -> JsWord {
    let ident = decl
        .name
        .as_ident()
        .unwrap_or_else(|| unsupported_node!(decl));
    ident.id.to_id().0
}

fn compile_fn_decl_namedef(builder: &mut Builder, fn_decl: &swc_ecma_ast::FnDecl) {
    let name = get_fn_decl_name(fn_decl);
    let vreg = builder.new_vreg();
    builder.define_var(name, Loc::VReg(vreg));
}

fn compile_fn_decl_assignment(builder: &mut Builder, fn_decl: &swc_ecma_ast::FnDecl) -> Result<()> {
    if fn_decl.declare {
        panic!("unsupported case: fn_decl.declare");
    }
    let name = get_fn_decl_name(fn_decl);
    let vreg = builder
        .read_var(&name)
        .with_context(error!("resolving identifier").with_span(fn_decl.ident.span))?;
    let value = compile_function(builder, Some(name.clone()), &fn_decl.function)?;
    builder.emit(Instr::Copy {
        dst: vreg,
        src: value,
    });
    Ok(())
}

fn get_fn_decl_name(fn_decl: &swc_ecma_ast::FnDecl) -> JsWord {
    let (name, _) = fn_decl.ident.to_id();
    name
}

fn compile_function(builder: &mut Builder, name: Option<JsWord>, func: &Function) -> Result<VReg> {
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
    compile_block(builder, stmts)?;

    let inner_fnb = builder.fn_stack.pop().expect("no FnBuilder!");

    // all the calls to `get_var` must come before builder.emit(ClosureNew), to guarantee that
    // the ClosureNew and any associated ClosureAddCapture are adjacent
    let mut captures = Vec::new();
    for var_name in inner_fnb.captures.iter() {
        let var_name = JsWord::from(var_name.as_str());
        let vreg = builder.read_var(&var_name).with_context(
            error!("resolving identifier for function's capture").with_span(func.span),
        )?;
        captures.push(vreg);
    }

    let dest = builder.new_vreg();
    builder.emit(Instr::ClosureNew {
        dest,
        fnid: inner_fnb.fnid,
        forced_this: None,
    });
    for vreg in captures {
        builder.emit(Instr::ClosureAddCapture(vreg));
    }

    builder.fns.insert(inner_fnb.fnid, inner_fnb);
    Ok(dest)
}

fn compile_arrow_function(builder: &mut Builder, arrow: &ArrowExpr) -> Result<VReg> {
    if arrow.is_async {
        panic!("unsupported: async functions");
    }
    if arrow.is_generator {
        panic!("unsupported: generator functions");
    }
    if arrow.return_type.is_some() {
        panic!("unsupported: TypeScript syntax (return type)");
    }
    if arrow.type_params.is_some() {
        panic!("unsupported: TypeScript syntax (return type)");
    }

    let mut param_idents = Vec::new();
    for param_pat in arrow.params.iter() {
        param_idents.push(swc_ecma_ast::Param {
            span: arrow.span,
            decorators: Vec::new(),
            pat: param_pat.clone(),
        });
    }

    builder.start_function(None, param_idents.as_slice());

    match arrow.body.as_ref() {
        swc_ecma_ast::BlockStmtOrExpr::BlockStmt(block) => {
            compile_block(builder, &block.stmts)?;
        }
        swc_ecma_ast::BlockStmtOrExpr::Expr(expr) => {
            compile_expr(builder, &*expr)?;
        }
    }

    let inner_fnb = builder.fn_stack.pop().expect("no FnBuilder!");

    let forced_this = builder.new_vreg();
    builder.emit(Instr::LoadThis(forced_this));

    // all the calls to `get_var` must come before builder.emit(ClosureNew), to guarantee that
    // the ClosureNew and any associated ClosureAddCapture are adjacent
    // TODO(performance) avoid this allocation?
    let mut args = Vec::new();
    for var_name in inner_fnb.captures.iter() {
        let var_name = JsWord::from(var_name.as_str());
        let vreg = builder.read_var(&var_name).with_context(
            error!("resolving identifier for function's capture").with_span(arrow.span),
        )?;
        args.push(vreg);
    }

    let dest = builder.new_vreg();
    let fnid = inner_fnb.fnid;
    builder.emit(Instr::ClosureNew {
        dest,
        fnid,
        forced_this: Some(forced_this),
    });
    // all the calls to `get_var` must come before builder.emit(ClosureNew), to guarantee that
    // the ClosureNew and any associated ClosureAddCapture are adjacent
    // TODO(performance) avoid this allocation?
    for vreg in args.into_iter() {
        builder.emit(Instr::ClosureAddCapture(vreg));
    }

    builder.fns.insert(inner_fnb.fnid, inner_fnb);

    Ok(dest)
}

/// Compile the given expression.
///
/// The resulting code implicitly leaves the result in the accumulator register.
fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Result<VReg> {
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
                        let var = compile_expr(builder, &arg.expr)?;
                        builder.emit(Instr::PushToSink(var));
                    }

                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Undefined);
                    return Ok(ret);
                } else if sym == "__start_trace" {
                    let trace_id = match args[0].expr.as_ref() {
                        Expr::Lit(Lit::Str(trace_id)) => trace_id.value.to_string(),
                        _ => {
                            panic!("__start_trace must be called with a trace ID: __start_trace('the-name-of-the-trace')")
                        }
                    };

                    builder.place_trace_anchor(trace_id);
                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Undefined);
                    return Ok(ret);
                }
            }

            let mut arg_regs = Vec::new();
            for arg in args {
                if arg.spread.is_some() {
                    panic!("unsupported: spread function parameter: function(a, b, ...)");
                }
                let reg = compile_expr(builder, &arg.expr)?;
                arg_regs.push(reg);
            }

            let (this, callee) = match callee.as_ref() {
                Expr::Member(member_expr) => {
                    let MemberAccess { obj, key: _, value } =
                        compile_member_access(builder, &member_expr)?;
                    (obj, value)
                }
                other_expr => {
                    let this = builder.new_vreg();
                    builder.set_const(this, bytecode::Literal::Undefined);

                    let callee_func = compile_expr(builder, &other_expr)?;
                    (this, callee_func)
                }
            };
            let return_value = builder.new_vreg();
            builder.emit(Instr::Call {
                return_value,
                this,
                callee,
            });

            for reg in arg_regs {
                builder.emit(Instr::CallArg(reg));
            }

            Ok(return_value)
        }

        Expr::Bin(bin_expr) => {
            let a = compile_expr(builder, &bin_expr.left)?;
            let b = compile_expr(builder, &bin_expr.right)?;

            // TODO There must be a better way.  2-operand instructions?  accumulator register?
            let ret = builder.new_vreg();
            let instr = match bin_expr.op {
                BinaryOp::Add => Instr::ArithAdd(ret, a, b),
                BinaryOp::Sub => Instr::ArithSub(ret, a, b),
                BinaryOp::Mul => Instr::ArithMul(ret, a, b),
                BinaryOp::Div => Instr::ArithDiv(ret, a, b),
                BinaryOp::Lt => Instr::CmpLT(ret, a, b),
                BinaryOp::LtEq => Instr::CmpLE(ret, a, b),
                BinaryOp::Gt => Instr::CmpGT(ret, a, b),
                BinaryOp::GtEq => Instr::CmpGE(ret, a, b),
                BinaryOp::EqEqEq => Instr::CmpEQ(ret, a, b),
                BinaryOp::NotEqEq => Instr::CmpNE(ret, a, b),

                // TODO TODO TODO This does not implement any of the 'wat' semantics of JavaScript
                // See https://www.destroyallsoftware.com/talks/wat
                BinaryOp::EqEq => Instr::CmpEQ(ret, a, b),
                BinaryOp::NotEq => Instr::CmpNE(ret, a, b),

                BinaryOp::LogicalAnd => Instr::BoolOpAnd(ret, a, b),
                BinaryOp::LogicalOr => Instr::BoolOpOr(ret, a, b),

                BinaryOp::InstanceOf => Instr::IsInstanceOf(ret, a, b),
                _ => panic!("unsupported binary op: {:?}", bin_expr.op),
            };

            builder.emit(instr);

            Ok(ret)
        }

        Expr::Lit(lit) => {
            let ret = builder.new_vreg();
            match lit {
                Lit::Num(number) => {
                    builder.set_const(ret, Literal::Number(number.value));
                }
                Lit::Str(s) => {
                    let value = s.value.to_string();
                    builder.set_const(ret, Literal::String(value));
                }
                Lit::Bool(bv) => {
                    builder.set_const(ret, Literal::Bool(bv.value));
                }
                Lit::Null(_) => {
                    builder.set_const(ret, Literal::Null);
                }
                Lit::Regex(re_lit) => {
                    let constructor = builder
                        .read_var(&JsWord::from("RegExp"))
                        .expect("undefined native constructor: RegExp");
                    let re_str = re_lit.exp.to_string().into();

                    let arg = builder.new_vreg();
                    builder.set_const(arg, re_str);
                    let args = vec![arg];

                    compile_new(builder, constructor, ret, &args);
                }
                // Lit::BigInt(_) => todo!(),
                // Lit::JSXText(_) => todo!(),
                other => unsupported_node!(other),
            };
            Ok(ret)
        }

        Expr::Ident(ident) => {
            let ret = match ident.sym.as_bytes() {
                b"undefined" => {
                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Undefined);
                    ret
                }
                b"Infinity" => {
                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Number(f64::INFINITY));
                    ret
                }
                b"NaN" => {
                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Number(f64::NAN));
                    ret
                }
                _ => builder
                    .read_var(&ident.sym)
                    .with_context(error!("resolving identifier").with_span(ident.span))?,
            };
            Ok(ret)
        }

        Expr::Assign(asmt) => compile_assignment(asmt, builder)
            .with_context(error!("in assignment").with_span(asmt.span)),

        Expr::Object(obj_expr) => {
            let obj = builder.new_vreg();
            builder.emit(Instr::ObjCreateEmpty(obj));

            for prop_or_spread in obj_expr.props.iter() {
                match prop_or_spread {
                    swc_ecma_ast::PropOrSpread::Spread(spread_elm) => {
                        return Err(error!("spread syntax (...) is currently unsupported")
                            .with_span(spread_elm.dot3_token))
                    }
                    swc_ecma_ast::PropOrSpread::Prop(prop) => match prop.as_ref() {
                        swc_ecma_ast::Prop::KeyValue(kv_expr) => {
                            let key = builder.new_vreg();
                            let name = compile_prop_name(&kv_expr.key)?;
                            builder.set_const(key, name);

                            let value = compile_expr(builder, &kv_expr.value)?;

                            builder.emit(Instr::ObjSet { obj, key, value });
                        }

                        swc_ecma_ast::Prop::Method(method_prop) => {
                            let key = builder.new_vreg();
                            let name = compile_prop_name(&method_prop.key)?;
                            builder.set_const(key, name);

                            let value = compile_function(builder, None, &*method_prop.function)?;

                            builder.emit(Instr::ObjSet { obj, key, value });
                        }

                        swc_ecma_ast::Prop::Shorthand(sh) => {
                            let key = builder.new_vreg();
                            builder.set_const(key, Literal::String(sh.sym.to_string()));

                            let var = builder.read_var(&sh.sym).with_context(
                                error!("in short-hand object short-hand initializer")
                                    .with_span(sh.span),
                            )?;
                            builder.emit(Instr::ObjSet {
                                obj,
                                key,
                                value: var,
                            });
                        }

                        swc_ecma_ast::Prop::Assign(_)
                        | swc_ecma_ast::Prop::Getter(_)
                        | swc_ecma_ast::Prop::Setter(_) => {
                            return Err(error!("unsupported node: {:?}", prop_or_spread)
                                .with_span(obj_expr.span))
                        }
                    },
                }
            }

            Ok(obj)
        }

        Expr::This(_) => {
            let vreg = builder.new_vreg();
            builder.emit(Instr::LoadThis(vreg));
            Ok(vreg)
        }

        Expr::Array(arr_expr) => {
            use swc_ecma_ast::ExprOrSpread;

            let array = builder.new_vreg();

            let constructor = builder
                .read_var(&JsWord::from("Array"))
                .expect("undefined native constructor: Array");
            compile_new(builder, constructor, array, &[]);

            for elem in arr_expr.elems.iter() {
                match elem {
                    Some(ExprOrSpread { spread, expr }) => {
                        if spread.is_some() {
                            return Err(error!("spread syntax is currently unsupported")
                                .with_span(arr_expr.span));
                        }
                        let value = compile_expr(builder, expr)?;
                        builder.emit(Instr::ArrayPush { arr: array, value });
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

        Expr::Arrow(arrow_expr) => {
            // TODO Refactor this with Decl::Fn
            compile_arrow_function(builder, arrow_expr)
        }
        Expr::Fn(fn_expr) => {
            // TODO Refactor this with Decl::Fn
            let name = fn_expr.ident.as_ref().map(|ident| ident.to_id().0);
            let func = &fn_expr.function;
            let func = compile_function(builder, name.clone(), func)?;

            if let Some(name) = name {
                builder.define_var(name, Loc::VReg(func));
            }

            Ok(func)
        }

        Expr::Unary(unary_expr) => {
            let dest = builder.new_vreg();
            match unary_expr.op {
                swc_ecma_ast::UnaryOp::Bang => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    builder.emit(Instr::BoolNot { dest, arg });
                }
                swc_ecma_ast::UnaryOp::TypeOf => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    builder.emit(Instr::TypeOf { dest, arg });
                }
                swc_ecma_ast::UnaryOp::Minus => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    builder.emit(Instr::UnaryMinus { dest, arg });
                }
                swc_ecma_ast::UnaryOp::Plus => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    builder.emit(Instr::Copy {
                        dst: dest,
                        src: arg,
                    });
                }
                swc_ecma_ast::UnaryOp::Delete => {
                    let member_expr = match &*unary_expr.arg {
                        Expr::Member(member_expr) => member_expr,
                        _ => {
                            return Err(error!("`delete` operator can only be applied to an object member (e.g. `delete obj[key]`, `delete obj.property`)").with_span(unary_expr.span))
                        },
                    };

                    let obj = compile_expr(builder, &member_expr.obj)?;
                    let key = compile_obj_member_prop(builder, &member_expr.prop)?;
                    builder.emit(Instr::ObjDelete { dest, obj, key });
                }
                other => unsupported_node!(other),
                // swc_ecma_ast::UnaryOp::Tilde => todo!(),
                // swc_ecma_ast::UnaryOp::Void => todo!(),
            }
            Ok(dest)
        }
        Expr::Update(update_expr) => {
            match &*update_expr.arg {
                Expr::Ident(ident) => {
                    // NOTE: update_expr.prefix does not matter in this case, but
                    // it will matter when this code is extended to other types of args
                    let var = builder
                        .read_var(&ident.sym)
                        .with_context(error!("resolving identifier").with_span(ident.span))?;

                    compile_value_update(builder, update_expr.op, var);
                    Ok(var)
                }
                Expr::Member(member_expr) => {
                    let MemberAccess { obj, key, value } =
                        compile_member_access(builder, member_expr)?;
                    compile_value_update(builder, update_expr.op, value);
                    builder.emit(Instr::ObjSet { obj, key, value });
                    Ok(value)
                }
                other => todo!(
                    "unsupported: UpdateExpr on anything other than an identifier: {:?}",
                    other
                ),
            }
        }
        Expr::Member(member_expr) => Ok(compile_member_access(builder, member_expr)?.value),

        Expr::Paren(inner) => compile_expr(builder, &inner.expr),

        Expr::Cond(cond_expr) => {
            let ret = builder.new_vreg();

            let cond = compile_expr(builder, &cond_expr.test)?;
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let jmpif = builder.reserve();

            let value = compile_expr(builder, &cond_expr.cons)?;
            builder.emit(Instr::Copy {
                dst: ret,
                src: value,
            });
            let jmp_to_end = builder.reserve();

            let alt = builder.peek_iid();
            *builder.get_mut(jmpif).unwrap() = Instr::JmpIf { cond, dest: alt };
            let value = compile_expr(builder, &cond_expr.alt)?;
            builder.emit(Instr::Copy {
                dst: ret,
                src: value,
            });

            let end = builder.peek_iid();
            *builder.get_mut(jmp_to_end).unwrap() = Instr::Jmp(end);

            Ok(ret)
        }

        Expr::New(new_expr) => {
            let mut arg_regs = Vec::new();
            if let Some(args) = &new_expr.args {
                for arg_or_spread in args {
                    assert!(
                        arg_or_spread.spread.is_none(),
                        "unsupported: spread (...) in call args"
                    );
                    let arg = compile_expr(builder, &arg_or_spread.expr)?;
                    arg_regs.push(arg);
                }
            }

            let ret = builder.new_vreg();
            let callee = compile_expr(builder, &*new_expr.callee)?;
            compile_new(builder, callee, ret, &arg_regs);
            Ok(ret)
        }

        Expr::Tpl(tpl) => {
            assert_eq!(tpl.quasis.len(), tpl.exprs.len() + 1);

            let buf_reg = builder.new_vreg();
            builder.emit(Instr::StrCreateEmpty(buf_reg));
            let tmp_reg = builder.new_vreg();
            for (quasi, expr) in tpl.quasis.iter().zip(tpl.exprs.iter()) {
                builder.set_const(tmp_reg, Literal::String(quasi.raw.to_string()));
                builder.emit(Instr::StrAppend(buf_reg, tmp_reg));

                let value = compile_expr(builder, expr)?;
                builder.emit(Instr::StrAppend(buf_reg, value));
            }

            let last_str = tpl.quasis.last().unwrap().raw.to_string();
            builder.set_const(tmp_reg, Literal::String(last_str));
            builder.emit(Instr::StrAppend(buf_reg, tmp_reg));

            Ok(buf_reg)
        }

        // Expr::SuperProp(_) => todo!(),
        // Expr::Seq(_) => todo!(),
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

fn compile_new(builder: &mut Builder, constructor: VReg, ret: VReg, args: &[VReg]) {
    let key = builder.new_vreg();
    builder.set_const(key, bytecode::Literal::Undefined);
    builder.emit(Instr::Call {
        return_value: ret,
        this: key,
        callee: constructor,
    });
    for arg in args {
        builder.emit(Instr::CallArg(*arg));
    }

    builder.set_const(key, Literal::String("prototype".into()));
    // reusing register `constructor` for the prototype
    builder.emit(Instr::ObjGet {
        dest: constructor,
        obj: constructor,
        key,
    });

    builder.set_const(key, Literal::String("__proto__".into()));
    builder.emit(Instr::ObjSet {
        obj: ret,
        key,
        value: constructor,
    });
}

fn compile_value_update(builder: &mut Builder, op: swc_ecma_ast::UpdateOp, arg: VReg) {
    builder.emit(match op {
        UpdateOp::PlusPlus => Instr::ArithInc(arg, arg),
        UpdateOp::MinusMinus => Instr::ArithDec(arg, arg),
    });
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

fn compile_assignment(asmt: &swc_ecma_ast::AssignExpr, builder: &mut Builder) -> Result<VReg> {
    use swc_ecma_ast::{Expr, MemberExpr, MemberProp, PatOrExpr};

    if let Some(ident) = asmt.left.as_ident() {
        let rhs = compile_expr(builder, &asmt.right)?;

        let var = builder
            .read_var(&ident.sym)
            .with_context(error!("resolving identifier").with_span(ident.span))?;
        compile_assignment_rhs(builder, asmt.op, var, rhs)?;
        Ok(var)
    } else if let Some(target_expr) = asmt.left.as_expr() {
        match target_expr {
            Expr::Member(member_expr) => {
                let rhs = compile_expr(builder, &asmt.right)?;

                let MemberAccess { obj, key, value } = compile_member_access(builder, member_expr)?;
                compile_assignment_rhs(builder, asmt.op, value, rhs)?;
                builder.emit(Instr::ObjSet { obj, key, value });
                Ok(value)
            }
            // We should have already handled this case in the `if let ... = asm.left.as_ident()`
            // case
            Expr::Ident(_) => unreachable!(),
            _ => {
                let error =
                    error!("assignment to an expression is unsupported").with_span(asmt.span);
                Err(error)
            }
        }
    } else {
        panic!("unsupported pattern as assignment target: {:?}", asmt.left)
    }
}

struct MemberAccess {
    obj: VReg,
    key: VReg,
    value: VReg,
}
fn compile_member_access(
    builder: &mut Builder,
    member_expr: &swc_ecma_ast::MemberExpr,
) -> Result<MemberAccess> {
    let member_prop = &member_expr.prop;
    let key = compile_obj_member_prop(builder, member_prop)?;
    let obj = compile_expr(builder, member_expr.obj.as_ref())?;
    let value = builder.new_vreg();
    builder.emit(Instr::ObjGet {
        dest: value,
        obj,
        key,
    });

    Ok(MemberAccess { obj, key, value })
}

fn compile_obj_member_prop(
    builder: &mut Builder,
    member_prop: &swc_ecma_ast::MemberProp,
) -> Result<VReg> {
    use swc_ecma_ast::{ComputedPropName, MemberProp};

    let key = match member_prop {
        MemberProp::Ident(prop_ident) => {
            let key = builder.new_vreg();
            let prop_ident = prop_ident.sym.to_string().into();
            builder.set_const(key, prop_ident);
            key
        }
        MemberProp::Computed(ComputedPropName { expr, .. }) => compile_expr(builder, expr)?,
        other @ MemberProp::PrivateName(_) => unsupported_node!(other),
    };
    Ok(key)
}

/// Read the operand and compute its new value, as requested by the given assignment
/// expression. This is part of the compilation procedure of assignment expressions.
fn compile_assignment_rhs(
    builder: &mut Builder,
    assign_op: swc_ecma_ast::AssignOp,
    lhs: VReg,
    rhs: VReg,
) -> Result<()> {
    match assign_op {
        AssignOp::Assign => builder.emit(Instr::Copy { dst: lhs, src: rhs }),
        AssignOp::AddAssign => builder.emit(Instr::ArithAdd(lhs, lhs, rhs)),
        AssignOp::SubAssign => builder.emit(Instr::ArithSub(lhs, lhs, rhs)),
        AssignOp::MulAssign => builder.emit(Instr::ArithMul(lhs, lhs, rhs)),
        AssignOp::DivAssign => builder.emit(Instr::ArithDiv(lhs, lhs, rhs)),
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
    Ok(())
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
        fn get_module_id(&mut self, filename: &str) -> Option<bytecode::ModuleId> {
            Some(Self::THE_MODULE_ID)
        }

        fn read_source(&self, module_id: bytecode::ModuleId) -> String {
            assert_eq!(module_id, Self::THE_MODULE_ID);
            self.0.clone()
        }
    }

    fn quick_compile(code: &str) -> bytecode::Codebase {
        let mut builder = BuilderParams {
            loader: Box::new(NullLoader(code.to_string())),
        }
        .to_builder();

        builder.compile_file("<input>".to_string()).unwrap();

        builder.build()
    }

    #[test]
    #[ignore]
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
