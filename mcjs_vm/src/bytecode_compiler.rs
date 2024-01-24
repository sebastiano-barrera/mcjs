use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use swc_atoms::JsWord;
use swc_common::BytePos;
use swc_common::{sync::Lrc, Span, Spanned};
use swc_ecma_ast::{
    ArrowExpr, AssignOp, BinaryOp, BlockStmt, Decl, ExportDecl, Expr, FnDecl, ForHead, Function,
    Lit, ModuleItem, Pat, ReturnStmt, Stmt, UpdateOp, VarDecl, VarDeclKind, VarDeclarator,
};

use crate::bytecode::{self, IdentAsmt, Instr, Literal, LocalFnId, NativeFnId, VReg, IID};
use crate::common::{Context, Error, Result};
use crate::error;
use crate::util::Mask;

pub use swc_common::SourceMap;

use std::rc::Rc;

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
}

/// The state we need to track to compile a single module.  Among other things,
/// this includes things like tracking lexical scopes (so that variables are
/// resolved to simple, scope-independent IDs), FnId generation.
struct Builder {
    fns: HashMap<LocalFnId, FnBuilder>,
    fn_stack: Vec<FnBuilder>,
    next_fnid: u16,
    breakable_ranges: Vec<bytecode::BreakRange>,
    brange_stack: Vec<bytecode::BreakRange>,
    flags: CompileFlags,
    source_map: Rc<SourceMap>,
}

impl Builder {
    /// Create a new builder.
    ///
    /// At the end of the compilation process, the builder will have generated a
    /// number of functions, each with a new function ID. These function IDs are
    /// always >= than the `min_fnid` parameter. This is to make it easier to
    /// generate function IDs that don't collide with other sets of IDs
    /// generated somewhere else (for example, from a previous compilation).
    ///
    /// `source_map` is used to generate the spans where breakpoints can be set
    /// (`breakable_ranges`).
    fn new(flags: CompileFlags, source_map: Rc<SourceMap>) -> Self {
        let next_fnid = flags.min_fnid;
        assert!(next_fnid >= 1);
        Builder {
            fns: HashMap::new(),
            fn_stack: Vec::new(),
            next_fnid,
            breakable_ranges: Vec::new(),
            brange_stack: Vec::new(),
            flags,
            source_map,
        }
    }

    fn with_span(&mut self, span: &swc_common::Span) -> BuilderWithBreakRange {
        let iid_start = self.peek_iid();
        let iid_end = IID(iid_start.0 + 1);
        let local_fnid = self.cur_fnb().fnid;

        self.brange_stack.push(bytecode::BreakRange {
            lo: span.lo,
            hi: span.hi,
            local_fnid,
            iid_start,
            iid_end,
        });
        BuilderWithBreakRange(self)
    }
}

struct BuilderWithBreakRange<'a>(&'a mut Builder);

impl<'a> std::ops::Deref for BuilderWithBreakRange<'a> {
    type Target = Builder;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl<'a> std::ops::DerefMut for BuilderWithBreakRange<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<'a> Drop for BuilderWithBreakRange<'a> {
    fn drop(&mut self) {
        let mut brange = self.0.brange_stack.pop().unwrap();
        brange.iid_end = self.0.peek_iid();
        self.0.breakable_ranges.push(brange);
    }
}

pub struct CompiledChunk {
    pub root_fnid: LocalFnId,
    pub functions: HashMap<LocalFnId, bytecode::Function>,
    pub source_map: Rc<SourceMap>,
    pub breakable_ranges: Vec<bytecode::BreakRange>,
}

pub struct CompileFlags {
    pub min_fnid: u16,
    pub source_type: SourceType,
}

/// Whether we're compiling an actual module or a 'classic script'.
///
/// Influences the way that declarations work in the toplevel.  The distinction
/// is made largely at compile time, so we have to know here.
#[derive(PartialEq, Eq)]
pub enum SourceType {
    Script,
    Module,
}

/// Compile the given chunk of source code into executable bytecode.
///
/// The given `filename` is used *exclusively* for composing error messages and
/// to initialize the returned source map with a significant identifier for the
/// compiled file.  It does not have to reflect an existing entity in any file
/// system.
///
/// See `CompiledChunk` for details on the executable bytecode's representation.
pub fn compile_file(
    filename: String,
    content: String,
    source_map: Lrc<SourceMap>,
    flags: CompileFlags,
) -> Result<CompiledChunk> {
    use crate::common::Context;

    let ast_module = parse_file(filename.clone(), content, Lrc::clone(&source_map))
        .with_context(error!("while parsing file: {filename}"))?;

    let src_type_name = match flags.source_type {
        SourceType::Script => "script",
        SourceType::Module => "module",
    };
    let compiled_mod = match flags.source_type {
        SourceType::Module => compile_module(&ast_module, Lrc::clone(&source_map), flags),
        SourceType::Script => compile_script(ast_module, Lrc::clone(&source_map), flags),
    }
    .with_context(
        error!("while compiling {}: {}", src_type_name, filename)
            .with_source_map(Lrc::clone(&source_map)),
    )?;

    Ok(CompiledChunk {
        root_fnid: compiled_mod.root_fnid,
        functions: compiled_mod.functions,
        breakable_ranges: compiled_mod.breakable_ranges,
        source_map,
    })
}

struct FnBuilder {
    fnid: LocalFnId,
    global_this_vreg: Option<VReg>,
    scopes: Vec<Scope>,
    instrs: Vec<Instr>,
    consts: Vec<Literal>,
    span: Option<swc_common::Span>,
    trace_anchors: HashMap<IID, bytecode::TraceAnchor>,
    next_vreg: u8,
    // Places where a break instruction must be placed when the break target is finally "written"
    // by the compiler.
    pending_break_instrs: Vec<IID>,
    pending_continue_instrs: Vec<IID>,
    captures: Vec<String>,
    ident_history: Vec<IdentAsmt>,
    is_strict_mode: bool,

    // This is the set of registers the program has to 'unshare' in order to
    // loop back to the beginning of a scope.
    //
    // ## Why?
    //
    // JavaScript has this weird rule where multiple runs of the same block
    // (typically in a loop) generate distinct variables. The weird part (to me)
    // is observable in a test like the following:
    //
    // ```
    //   const a = []
    //   for (let i=0; a.push(function() { return i }), i < 5; i++) {
    //     ...
    //   }
    // ```
    //
    // On each iteration of the for loop, the closure added to `a` captures a
    // *new* variable, even though it's still called `i`. The 5 closures that result
    // from this loop all 'see' distinct locations on the heap, with values 0, 1, 2, 3,
    // and 4.
    //
    // 'Unsharing' means to copy the upvalue back into an inline stack slot (so
    // that it will be shared again with a new upvalue on the next cycle).
    //
    // ## Representation
    //
    // Each time a new variable is predicted to be shared (e.g. used as
    // parameter in ClosureAddCapture) it is pushed onto `unshare_on_loopback`
    // and a counter `shared_count` is increased on the current Scope.
    // Upon exiting the scope, the count is added to the parent scope's
    // `shared_count`.  This way, at any point in time, the current Scope tracks
    // how many variables have been shared at any point within it; they can be
    // found in the last N slots of `unshare_on_loopback`.
    unshare_on_loopback: Vec<VReg>,
}

enum Var {
    /// Local variable.  Its value is written by writing to this register.
    ///
    /// Note that captures are part of this category: based on the semantics of the bytecode,
    /// the interpreter's stack will contain the upvalue ID in it, so that writing to the
    /// register will result to an indirect write to the upvalue.
    Reg(VReg),

    /// Global variable, i.e. a key stored in `globalThis`.
    ///
    /// All reads and writes go through an object access to `globalThis`.
    Global(JsWord),
}

#[derive(Debug)]
struct Scope {
    vars: HashMap<JsWord, VReg>,
    shared_count: usize,
}
impl Scope {
    fn new() -> Self {
        Scope {
            vars: HashMap::new(),
            shared_count: 0,
        }
    }
}

impl<'a> FnBuilder {
    fn new(id: LocalFnId) -> Self {
        FnBuilder {
            fnid: id,
            global_this_vreg: None,
            scopes: vec![Scope::new()],
            instrs: Vec::new(),
            consts: Vec::new(),
            trace_anchors: HashMap::new(),
            captures: Vec::new(),
            next_vreg: 0,
            pending_break_instrs: Vec::new(),
            pending_continue_instrs: Vec::new(),
            span: None,
            ident_history: Vec::new(),
            is_strict_mode: false,
            unshare_on_loopback: Vec::new(),
        }
    }

    fn enable_strict_mode(&mut self) {
        self.is_strict_mode = true;
    }

    fn build(self) -> bytecode::Function {
        assert!(
            self.pending_break_instrs.is_empty(),
            "bytecode compiler bug: the function is over, but some break instructions were not placed yet"
        );

        bytecode::FunctionBuilder {
            instrs: self.instrs.into_boxed_slice(),
            consts: self.consts.into_boxed_slice(),
            n_regs: self.next_vreg,
            ident_history: self.ident_history,
            trace_anchors: self.trace_anchors,
            is_strict_mode: self.is_strict_mode,
            span: self.span.unwrap(),
        }
        .build()
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
        let closed_scope = self.scopes.pop().unwrap();
        if let Some(new_inner_scope) = self.scopes.last_mut() {
            new_inner_scope.shared_count += closed_scope.shared_count;
        }
    }
    fn unshare_on_loopback(&mut self, vreg: VReg) {
        self.inner_scope_mut().shared_count += 1;
        self.unshare_on_loopback.push(vreg);
    }

    fn emit_unshare(&mut self) {
        let count = self.inner_scope().shared_count;
        let to_unshare = &self.unshare_on_loopback[self.unshare_on_loopback.len() - count..];
        for reg in to_unshare {
            self.instrs.push(Instr::Unshare(*reg));
        }
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

    // Assign a new upvalue index to the given variable.
    // Subsequent calls to get_var will return  the IID for the GetCapture instruction.
    fn set_var_captured(&mut self, name: &JsWord) -> VReg {
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
        let reg = self.next_vreg();
        self.emit(Instr::LoadCapture(reg, cap_ndx));

        todo!("the decls structure needs to be modified in order for this to work");

        self.ident_history.push(IdentAsmt {
            iid: self.peek_iid(),
            reg,
            ident: name.clone(),
        });

        reg
    }

    fn get_global_this(&mut self) -> VReg {
        let reg = self.next_vreg();
        self.emit(Instr::GetGlobalThis(reg));
        reg
    }

    fn get_vreg(&self, name: &JsWord) -> Option<VReg> {
        self.scopes
            .iter()
            .rev() // innermost -> outwards
            .find_map(|scope| scope.vars.get(name))
            .cloned()
    }

    // TODO delete!
    fn define_var(&mut self, name: JsWord, reg: VReg) {
        let scope = self.inner_scope_mut();
        scope.vars.insert(name.clone(), reg);

        self.ident_history.push(IdentAsmt {
            iid: self.peek_iid(),
            reg,
            ident: name,
        });
    }

    fn next_vreg(&mut self) -> VReg {
        let vreg_ndx = self.next_vreg;
        self.next_vreg += 1;
        VReg(vreg_ndx)
    }
}

impl<'a> Builder {
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

    fn define_var(&mut self, name: JsWord, reg: VReg) {
        self.cur_fnb().define_var(name, reg)
    }

    fn get_vreg(&mut self, sym: &JsWord) -> Option<VReg> {
        let found = self
            .fn_stack
            .iter()
            .rev()
            .enumerate()
            .find_map(|(fn_depth, fnb)| fnb.get_vreg(sym).map(|vreg| (fn_depth, vreg)));

        match found {
            // Hell yeah, found it in the current function, we have a "simple" vreg
            Some((0, vreg)) => Some(vreg),

            // Found it in one of the enclosing functions.  We'll have to add a capture, but we
            // know the variable is there
            Some((_, _)) => Some(self.cur_fnb().set_var_captured(sym)),

            None => None,
        }
    }

    /// Determines the location where a variable is stored.  The return value can be subsequently
    /// used via `read_var` and `write_var` to access the variable properly.
    fn get_var(&mut self, sym: &JsWord) -> Var {
        match self.get_vreg(sym) {
            Some(vreg) => Var::Reg(vreg),
            None => Var::Global(sym.clone()),
        }
    }

    /// Get the value of the given variable into a register.
    ///
    /// If necessary, generates some extra instructions. For example, if the given variable is
    /// global (i.e. stored in the `globalThis` object), the necessary ObjGet is generated.
    ///
    /// Note that writing to the returned register may or may not write the variable. To make sure
    /// that the variable is properly written, finish off the write with `ensure_written`.
    fn read_var(&mut self, var: &Var) -> VReg {
        match var {
            Var::Reg(vreg) => *vreg,
            Var::Global(sym) => self.read_global(sym),
        }
    }

    fn write_var(&mut self, var: &Var, value_reg: VReg) {
        match var {
            Var::Reg(vreg) => {
                // We can save a copy from compile_assignment_rhs
                if *vreg != value_reg {
                    self.emit(Instr::Copy {
                        dst: *vreg,
                        src: value_reg,
                    });
                }
            }
            Var::Global(sym) => self.write_global(sym, value_reg),
        }
    }

    fn read_global(&mut self, sym: &JsWord) -> VReg {
        let key = self.new_vreg();
        let dest = self.new_vreg();

        let global_this = self.cur_fnb().get_global_this();
        self.set_const(key, Literal::String(sym.to_string()));
        self.emit(Instr::ObjGet {
            dest,
            obj: global_this,
            key,
        });
        dest
    }

    fn write_global(&mut self, sym: &JsWord, value: VReg) {
        let key = self.new_vreg();

        let global_this = self.cur_fnb().get_global_this();
        self.set_const(key, Literal::String(sym.to_string()));
        self.emit(Instr::ObjSet {
            obj: global_this,
            key,
            value,
        });
    }

    fn new_vreg(&mut self) -> VReg {
        self.cur_fnb().next_vreg()
    }

    #[allow(unused_variables)]
    fn start_function(&mut self) {
        let fnid = LocalFnId(self.next_fnid);
        self.next_fnid += 1;

        let mut fnb = FnBuilder::new(fnid);
        fnb.is_strict_mode = self
            .fn_stack
            .last()
            .map(|fnb| fnb.is_strict_mode)
            .unwrap_or(false);
        self.fn_stack.push(fnb);
    }

    fn end_function(&mut self, span: Span) -> LocalFnId {
        let mut fnb = self.fn_stack.pop().expect("no FnBuilder!");
        fnb.span = Some(span);
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

struct CompiledModule {
    root_fnid: LocalFnId,
    functions: HashMap<LocalFnId, bytecode::Function>,
    breakable_ranges: Vec<bytecode::BreakRange>,
}

fn compile_module(
    ast_module: &swc_ecma_ast::Module,
    source_map: Rc<SourceMap>,
    flags: CompileFlags,
) -> Result<CompiledModule> {
    use swc_ecma_ast::{ExportDecl, Expr, ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    let decls = {
        let mut decls = Vec::new();
        let flags = declset2::FindDeclFlags {
            hoist_vars: true,
            hoist_functions: false, // modules are always strict
            only_hoisted: false,
        };
        for item in &ast_module.body {
            if let ModuleItem::Stmt(stmt) = item {
                declset2::find_declarations(&mut decls, flags, stmt);
            }
        }
        declset2::check_redeclarations(decls)?
    };

    let mut builder = Builder::new(flags, source_map);
    builder.start_function();
    builder.cur_fnb().enable_strict_mode();

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

    for decl in &decls {
        let reg = builder.new_vreg();
        builder.define_var(decl.name.clone(), reg);
    }
    for decl in &decls {
        if let Some(fn_decl) = decl.kind.as_function() {
            let name = Some(fn_decl.ident.sym.clone());
            compile_function(&mut builder, name, &fn_decl.function)?;
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

                    let module_path = builder.new_vreg();
                    builder.set_const(module_path, Literal::String(decl.src.value.to_string()));

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
                                    builder.emit(Instr::ImportModule(local_var, module_path));
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
                                    builder.define_var(d.local.sym.clone(), local_var);
                                }
                            }
                            swc_ecma_ast::ImportSpecifier::Default(d) => {
                                let local_var = builder.new_vreg();
                                builder.emit(Instr::ImportModule(local_var, module_path));
                                builder.emit(Instr::ObjGet {
                                    dest: local_var,
                                    obj: local_var,
                                    key: lit_default,
                                });
                                builder.define_var(d.local.sym.clone(), local_var);
                            }
                            swc_ecma_ast::ImportSpecifier::Namespace(d) => {
                                let lit_named = builder.new_vreg();
                                builder
                                    .set_const(lit_named, Literal::String("__named".to_string()));

                                let local_var = builder.new_vreg();
                                builder.emit(Instr::ImportModule(local_var, module_path));
                                builder.emit(Instr::ObjGet {
                                    dest: local_var,
                                    obj: local_var,
                                    key: lit_named,
                                });
                                builder.define_var(d.local.sym.clone(), local_var);
                            }
                        }
                    }
                }

                ModuleDecl::ExportDecl(export_decl) => match &export_decl.decl {
                    // Skip: already processed (namedef at function level, asmt at block level)
                    Decl::Fn(_) => {}
                    // Only do assignment part
                    Decl::Var(var_decl) => {
                        compile_var_decl_assignment(&mut builder, var_decl)?;
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
                        let vreg = compile_function(&mut builder, name, &fn_expr.function)?;
                        mod_default_export = Some(vreg);
                    }
                    swc_ecma_ast::DefaultDecl::TsInterfaceDecl(ts_decl) => eprintln!(
                        "warning: discarded TypeScript interface export: {}",
                        ts_decl.id
                    ),
                },
                ModuleDecl::ExportDefaultExpr(decl) => {
                    let vreg = compile_expr(&mut builder, &decl.expr)?;
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
                compile_stmt(&mut builder, stmt)
                    .with_context(error!("while compiling statement"))?;
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
                    .get_vreg(&name)
                    .expect("compiler bug: module export has no value");

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
    let root_fnid = builder.end_function(ast_module.span());
    Ok(finish_module(builder, root_fnid))
}

fn compile_script(
    ast_module: swc_ecma_ast::Module,
    source_map: Rc<SourceMap>,
    flags: CompileFlags,
) -> Result<CompiledModule> {
    use swc_ecma_ast::{ExportDecl, Expr, ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    let function = past::compile_script(ast_module);
    assert!(function.n_parameters == 0);

    // function.unbound_names might not be empty.  In this case, it will contain the list of
    // variables that should be accessed via `globalThis`.

    let mut mod_builder = past_compiler::ModuleBuilder::new(flags.min_fnid);
    let globals = function.unbound_names.iter().cloned().collect();
    let root_fnid =
        past_compiler::compile_function(&mut mod_builder, &globals, Vec::new(), &function)?;

    // finish_module(builder, root_fnid)
    let functions = mod_builder.build();

    Ok(CompiledModule {
        root_fnid,
        functions,
        breakable_ranges: Vec::new(), /* TODO! */
    })
}

fn is_use_strict_directive(stmt: &Stmt) -> bool {
    if let Stmt::Expr(expr_stmt) = stmt {
        if let Some(Lit::Str(ref lit_str)) = expr_stmt.expr.as_lit() {
            if lit_str.value.as_bytes() == b"use strict" {
                return true;
            }
        }
    }

    false
}

fn finish_module(builder: Builder, root_fnid: LocalFnId) -> CompiledModule {
    // The root function is the outermost scope, and therefore must capture
    // nothing.  Otherwise, we have a bug.
    assert!(builder.fns.get(&root_fnid).unwrap().captures.is_empty());
    for lfnid in builder.fns.keys() {
        assert!(lfnid.0 >= builder.flags.min_fnid);
    }

    let functions = builder
        .fns
        .into_iter()
        .map(|(fnid, fnb)| (fnid, fnb.build()))
        .collect();

    CompiledModule {
        root_fnid,
        functions,
        breakable_ranges: builder.breakable_ranges,
    }
}

fn compile_stmt<'a>(builder: &mut Builder, stmt: &'a swc_ecma_ast::Stmt) -> Result<()> {
    use swc_ecma_ast::Stmt;

    let mut builder = builder.with_span(&stmt.span());

    match stmt {
        Stmt::Block(block) => compile_block(&mut builder, &block.stmts),
        Stmt::Empty(_) => {
            // no code needs to be emitted
            Ok(())
        }
        Stmt::Debugger(_) => {
            builder.emit(Instr::Breakpoint);
            Ok(())
        }
        // Stmt::With(_) => todo!(),
        Stmt::Return(stmt) => {
            let reg = if let Some(arg) = &stmt.arg {
                compile_expr(&mut builder, arg)?
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
            let discriminant = compile_expr(&mut builder, &switch_stmt.discriminant)?;

            // TODO(performance): any better allocation strategy?
            // NOTE: This Vec skips the `default:` label!
            let cmp_result = builder.new_vreg();
            let mut case_jumps = Vec::with_capacity(switch_stmt.cases.len());

            for case in &switch_stmt.cases {
                if let Some(test) = &case.test {
                    let value = compile_expr(&mut builder, test)?;
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
                    compile_stmt(&mut builder, stmt)?;
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
            let exception = compile_expr(&mut builder, &throw_stmt.arg)?;
            builder.emit(Instr::Throw(exception));
            Ok(())
        }

        Stmt::Try(try_stmt) => {
            // TODO Allow the catch clause to access the exception object
            // We reunite the cases by *always* having a catch clause (it will run the finalizer)
            let push_handler_instr = builder.reserve();
            compile_block(&mut builder, &try_stmt.block.stmts)?;
            builder.emit(Instr::PopExcHandler);
            let jmp_after_try = builder.reserve();

            let handler_start = builder.peek_iid();
            if let Some(handler) = &try_stmt.handler {
                builder.cur_fnb().push_scope();

                if let Some(param) = &handler.param {
                    // TODO I really need some sort of `compile_pattern`
                    match param {
                        Pat::Ident(ident) => {
                            let ident = ident.to_id().0;
                            let reg = builder.new_vreg();
                            builder.emit(Instr::GetCurrentException(reg));
                            builder.define_var(ident, reg);
                        }
                        _ => {
                            unsupported_node!(param);
                        }
                    }
                }

                compile_block(&mut builder, &handler.body.stmts)?;
                builder.cur_fnb().pop_scope();
            }
            let jmp_after_catch = builder.reserve();

            let finalizer_start = builder.peek_iid();
            if let Some(finalizer) = &try_stmt.finalizer {
                compile_block(&mut builder, &finalizer.stmts)?;
            }

            *builder.get_mut(push_handler_instr).unwrap() = Instr::PushExcHandler(handler_start);
            *builder.get_mut(jmp_after_try).unwrap() = Instr::Jmp(finalizer_start);
            *builder.get_mut(jmp_after_catch).unwrap() = Instr::Jmp(finalizer_start);
            Ok(())
        }

        Stmt::While(while_stmt) => {
            let while_header_iid = builder.peek_iid();

            let cond = compile_expr(&mut builder, &while_stmt.test)?;
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let jmpif = builder.reserve();

            compile_block_single_stmt(&mut builder, &while_stmt.body)?;
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
            compile_block_single_stmt(&mut builder, &stmt.body)?;
            let cond = compile_expr(&mut builder, &stmt.test)?;
            builder.emit(Instr::JmpIf {
                cond,
                dest: loop_start,
            });
            Ok(())
        }

        Stmt::For(stmt) => {
            use swc_ecma_ast::VarDeclOrExpr;

            // Two scopes:
            //   - outer (includes any variable initialized in the 'initializer' expr)
            //   - inner (corresponds to the `for` stmt body)

            // outer scope
            builder.cur_fnb().push_scope();

            {
                let mut decls = Vec::new();
                let decl_flags = declset2::FindDeclFlags {
                    hoist_vars: false,
                    hoist_functions: false,
                    only_hoisted: false,
                };
                if let Some(VarDeclOrExpr::VarDecl(var_decl)) = &stmt.init {
                    declset2::find_var_decls(&mut decls, decl_flags, var_decl);
                }
                block_start_decls(&mut builder, decls)?;
            }

            if let Some(init) = &stmt.init {
                match init {
                    VarDeclOrExpr::VarDecl(var_decl) => {
                        compile_var_decl_assignment(&mut builder, var_decl)?;
                    }
                    VarDeclOrExpr::Expr(expr) => {
                        compile_expr(&mut builder, expr)?;
                    }
                }
            }

            let loop_start = builder.peek_iid();
            let not_cond = match &stmt.test {
                Some(test) => {
                    let cond = compile_expr(&mut builder, test)?;
                    let not_cond = builder.new_vreg();
                    builder.emit(Instr::BoolNot {
                        dest: not_cond,
                        arg: cond,
                    });
                    Some(not_cond)
                }
                None => None,
            };
            let jmpif = builder.reserve();

            compile_block_single_stmt(&mut builder, &stmt.body)?;

            let continue_target = builder.peek_iid();
            if let Some(update) = &stmt.update {
                compile_expr(&mut builder, update)?;
            }

            builder.cur_fnb().emit_unshare();
            builder.emit(Instr::Jmp(loop_start));
            let loop_end = builder.peek_iid();

            *builder.get_mut(jmpif).unwrap() = if let Some(not_cond) = not_cond {
                Instr::JmpIf {
                    cond: not_cond,
                    dest: loop_end,
                }
            } else {
                Instr::Nop
            };
            builder.resolve_break_to(loop_end);
            builder.resolve_continue_to(continue_target);

            builder.cur_fnb().pop_scope();
            Ok(())
        }

        Stmt::ForIn(forin_stmt) => {
            use swc_ecma_ast::{ForHead, Pat};

            let is_strict_mode = builder.cur_fnb().is_strict_mode;

            let item_var: Cow<VarDecl> = match &forin_stmt.left {
                ForHead::VarDecl(var_decl) => Cow::Borrowed(var_decl.as_ref()),
                ForHead::UsingDecl(_) => todo!(),
                ForHead::Pat(pat) => match pat.as_ref() {
                    Pat::Ident(ident) if is_strict_mode && is_identifier_keyword(ident) => {
                        return Err(
                            error!("can't use keyword `{}` as identifier (in strict mode)", ident.sym)
                            .with_span(ident.span)
                        )
                    },
                    Pat::Ident(ident) => {
                        let decl = VarDeclarator {
                            span: ident.span,
                            name: pat.as_ref().clone(),
                            init: None,
                            definite: false,
                        };
                        Cow::Owned(VarDecl {
                            declare: false,
                            kind: VarDeclKind::Var,
                            decls: vec![decl],
                            span: ident.span,
                        })
                    },
                    _ => panic!(
                        "unsupported syntax: destructuring pattern as `<pattern>` in: `for (<pattern> in ...) {{ ... }}`"
                    ),
                },
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

            let iteree = compile_expr(&mut builder, &forin_stmt.right)?;
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
            builder.define_var(item_var_name.clone(), key);

            compile_block_single_stmt(&mut builder, &forin_stmt.body)?;

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

            let iteree = compile_expr(&mut builder, &forof_stmt.right)?;
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
            builder.define_var(item_var_name.clone(), item);
            compile_block_single_stmt(&mut builder, &forof_stmt.body)?;

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
        Stmt::Decl(Decl::Var(var_decl)) => compile_var_decl_assignment(&mut builder, var_decl),
        Stmt::Decl(Decl::Fn(_)) => Ok(()),

        Stmt::Expr(expr_stmt) => {
            compile_expr(&mut builder, &expr_stmt.expr)?;
            Ok(())
        }

        Stmt::If(if_stmt) => {
            let cond = compile_expr(&mut builder, &if_stmt.test)
                .with_context(error!("in if statement").with_span(if_stmt.span))?;
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let jmp_to_alt = builder.reserve();

            compile_block_single_stmt(&mut builder, &if_stmt.cons)?;
            let jmp_to_end = builder.reserve();

            let alt = builder.peek_iid();
            if let Some(else_blk) = &if_stmt.alt {
                compile_block_single_stmt(&mut builder, else_blk)?;
            }

            let end: IID = builder.peek_iid();

            *builder.get_mut(jmp_to_alt).unwrap() = Instr::JmpIf { cond, dest: alt };
            *builder.get_mut(jmp_to_end).unwrap() = Instr::Jmp(end);
            Ok(())
        }
        other => unsupported_node!(other),
    }
}

fn compile_block(builder: &mut Builder, stmts: &[Stmt]) -> Result<()> {
    builder.cur_fnb().push_scope();

    let mut decls = Vec::new();
    let decl_flags = declset2::FindDeclFlags {
        hoist_vars: false,
        hoist_functions: false,
        only_hoisted: false,
    };
    for stmt in stmts {
        declset2::find_declarations(&mut decls, decl_flags, stmt);
    }
    block_start_decls(builder, decls)?;

    for stmt in stmts {
        compile_stmt(builder, stmt)?;
    }

    builder.cur_fnb().pop_scope();
    Ok(())
}

fn compile_block_single_stmt(builder: &mut Builder, stmt: &Stmt) -> Result<()> {
    builder.cur_fnb().push_scope();

    let mut decls = Vec::new();
    let decl_flags = declset2::FindDeclFlags {
        hoist_vars: false,
        hoist_functions: false,
        only_hoisted: false,
    };
    declset2::find_declarations(&mut decls, decl_flags, stmt);
    block_start_decls(builder, decls)?;

    compile_stmt(builder, stmt)?;

    builder.cur_fnb().pop_scope();
    Ok(())
}

fn block_start_decls(builder: &mut Builder, decls: Vec<declset2::Decl>) -> Result<()> {
    use declset2::Kind;

    let decls = declset2::check_redeclarations(decls)?;

    let is_strict_mode = builder.cur_fnb().is_strict_mode;

    for decl in &decls {
        let defined_at_block_scope = match decl.kind {
            Kind::Var { fn_decl } => is_strict_mode && fn_decl.is_some(),
            Kind::Lexical { .. } => true,
        };
        if defined_at_block_scope {
            let reg = builder.new_vreg();
            builder.define_var(decl.name.clone(), reg);
        }
    }

    if is_strict_mode {
        for decl in decls {
            if let Some(fn_decl) = decl.kind.as_function() {
                let name = Some(decl.name.clone());
                compile_function(builder, name, &fn_decl.function)?;
            }
        }
    } else {
        // Functions already compiled at the function toplevel (see `compile_function`)
    }

    Ok(())
}

fn compile_var_decl_assignment<'a>(
    builder: &mut Builder,
    var_decl: &'a swc_ecma_ast::VarDecl,
) -> Result<()> {
    use swc_ecma_ast::VarDeclKind;

    let _is_const = match var_decl.kind {
        VarDeclKind::Var => false,
        VarDeclKind::Let => false,
        VarDeclKind::Const => true,
    };

    for decl in &var_decl.decls {
        if let Some(expr) = &decl.init {
            let ident = decl
                .name
                .as_ident()
                .unwrap_or_else(|| unsupported_node!(decl));
            let name: JsWord = ident.id.to_id().0;
            let var = builder.get_var(&name);
            let value = compile_expr(builder, expr)?;
            builder.write_var(&var, value);
        } else {
            // We do nothing.
            // The register had already been reserved for this variable in the namedef phase, and
            // it is implicitly initialized to undefined by the interpreter.
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

// TODO Dedup with the one in declset
fn get_fn_decl_name(fn_decl: &swc_ecma_ast::FnDecl) -> JsWord {
    let (name, _) = fn_decl.ident.to_id();
    name
}

fn compile_function<'a>(
    builder: &mut Builder,
    name: Option<JsWord>,
    func: &'a Function,
) -> Result<VReg> {
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

    let stmts = &func.body.as_ref().expect("function without body?!").stmts;
    let is_strict_mode = stmts.first().map(is_use_strict_directive).unwrap_or(false);

    let mut decls = Vec::new();
    let flags = declset2::FindDeclFlags {
        hoist_vars: true,
        hoist_functions: !is_strict_mode,
        only_hoisted: false,
    };
    for stmt in stmts {
        declset2::find_declarations(&mut decls, flags, stmt);
    }
    // params need to be 'declare'd here so that redeclarations can be checked.
    // regardless, param names are bound to registers later independently anyway.
    declset2::declare_params(&mut decls, &func.params);
    let decls = declset2::check_redeclarations(decls)?;

    builder.start_function();
    if is_strict_mode {
        builder.cur_fnb().enable_strict_mode();
    }

    // We need to guarantee, here, that the first ARGS_COUNT_MAX registers are
    // assigned to as many functcion arguments
    for param_ndx in 0..bytecode::ARGS_COUNT_MAX {
        // always pull a new vreg even if we're short on actual func params, so
        // that we're sure that we allocate all the first ARGS_COUNT_MAX regs
        let reg = builder.new_vreg();

        if let Some(param) = func.params.get(param_ndx as usize) {
            let pat = &param.pat;
            let name = pat
                .as_ident()
                .unwrap_or_else(|| unsupported_node!(pat))
                .sym
                .clone();
            builder.cur_fnb().define_var(name, reg);
        }
    }

    for decl in &decls {
        let reg = builder.new_vreg();
        builder.define_var(decl.name.clone(), reg);
    }
    for decl in &decls {
        if let Some(fn_decl) = decl.kind.as_function() {
            let name = Some(decl.name.clone());
            compile_function(builder, name, &fn_decl.function)?;
        }
    }

    for stmt in stmts {
        compile_stmt(builder, stmt)?;
    }

    let mut inner_fnb = builder.fn_stack.pop().expect("no FnBuilder!");
    inner_fnb.span = Some(func.span);

    // all the calls to `get_var` must come before builder.emit(ClosureNew), to guarantee that
    // the ClosureNew and any associated ClosureAddCapture are adjacent
    let mut captures = Vec::new();
    for var_name in inner_fnb.captures.iter() {
        let var_name: JsWord = var_name.as_str().into();
        let vreg = builder
            .get_vreg(&var_name)
            .expect("not yet implemented: capturing a variable that is global");
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
        builder.cur_fnb().unshare_on_loopback(vreg);
    }

    let proto = builder.new_vreg();
    let key = builder.new_vreg();
    builder.set_const(key, Literal::String("prototype".to_string()));
    builder.emit(Instr::ObjCreateEmpty(proto));
    builder.emit(Instr::ObjSet {
        obj: dest,
        key,
        value: proto,
    });
    let ctor = builder.get_var(&"Function".into());
    let ctor = builder.read_var(&ctor);
    builder.set_const(key, Literal::String("constructor".to_string()));
    builder.emit(Instr::ObjSet {
        obj: dest,
        key,
        value: ctor,
    });

    if let Some(name) = name {
        let var = builder.get_var(&name);
        builder.write_var(&var, dest);
    }

    builder.fns.insert(inner_fnb.fnid, inner_fnb);
    Ok(dest)
}

mod declset2 {
    use swc_atoms::JsWord;
    use swc_common::{Span, Spanned};
    use swc_ecma_ast::{FnDecl, Stmt};

    use crate::common::Result;
    use crate::{bytecode, error};

    #[derive(Debug, Clone, Copy)]
    pub enum Kind<'a> {
        /// `var`, `function`
        Var { fn_decl: Option<&'a FnDecl> },
        /// `let`, `const`
        Lexical { is_const: bool },
    }

    impl<'a> Kind<'a> {
        pub fn as_function(&self) -> Option<&'a FnDecl> {
            match self {
                Kind::Var { fn_decl } => *fn_decl,
                Kind::Lexical { .. } => None,
            }
        }
    }

    /// Flags for `find_declarations`.
    #[derive(Clone, Copy)]
    pub struct FindDeclFlags {
        /// If true, `var` declarations from sub-scopes are included in the returned set
        pub hoist_vars: bool,

        /// If true, `function` declarations from sub-scopes are included in the returned set
        pub hoist_functions: bool,

        /// If true, the returned set will *only* include var/function
        /// declarations enabled by the `hoist_vars` and `hoist_functions` flags         
        pub only_hoisted: bool,
    }

    impl FindDeclFlags {
        fn for_subscope_hoisting(&self) -> Self {
            FindDeclFlags {
                only_hoisted: self.only_hoisted || self.hoist_functions || self.hoist_vars,
                ..*self
            }
        }
    }

    pub struct Decl<'a> {
        pub kind: Kind<'a>,
        pub name: JsWord,
        /// Location where the original declaration appeared in the source code
        pub span: Span,
    }

    pub fn find_declarations<'a>(decls: &mut Vec<Decl<'a>>, flags: FindDeclFlags, stmt: &'a Stmt) {
        match stmt {
            Stmt::Decl(decl) => match decl {
                swc_ecma_ast::Decl::Fn(fn_decl) => {
                    if !flags.only_hoisted || flags.hoist_functions {
                        let kind = Kind::Var {
                            fn_decl: Some(fn_decl),
                        };
                        let name = fn_decl.ident.sym.clone();
                        let span = fn_decl.span();
                        decls.push(Decl { kind, name, span });
                    }
                }
                swc_ecma_ast::Decl::Var(var_decl) => {
                    find_var_decls(decls, flags, var_decl);
                }
                _ => {}
            },

            Stmt::Block(block_stmt) => {
                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();
                    for stmt in &block_stmt.stmts {
                        find_declarations(decls, flags, stmt);
                    }
                }
            }
            Stmt::If(if_stmt) => {
                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();
                    find_declarations(decls, flags, &if_stmt.cons);
                    if let Some(alt) = &if_stmt.alt {
                        find_declarations(decls, flags, alt);
                    }
                }
            }
            Stmt::While(while_stmt) => {
                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();
                    find_declarations(decls, flags, &while_stmt.body);
                }
            }
            Stmt::For(for_stmt) => {
                use swc_ecma_ast::VarDeclOrExpr;

                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();
                    if let Some(VarDeclOrExpr::VarDecl(var_decl)) = &for_stmt.init {
                        find_var_decls(decls, flags, var_decl);
                    }
                    find_declarations(decls, flags, &for_stmt.body);
                }
            }
            Stmt::Try(try_stmt) => {
                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();

                    for stmt in &try_stmt.block.stmts {
                        find_declarations(decls, flags, stmt);
                    }

                    if let Some(catch_block) = &try_stmt.handler {
                        if flags.hoist_vars {
                            if let Some(pat) = &catch_block.param {
                                let name = pat
                                    .as_ident()
                                    .unwrap_or_else(|| unsupported_node!(pat))
                                    .sym
                                    .clone();
                                let kind = Kind::Var { fn_decl: None };
                                let span = pat.span();
                                decls.push(Decl { kind, name, span });
                            }
                        }

                        for stmt in &catch_block.body.stmts {
                            find_declarations(decls, flags, stmt)
                        }
                    }

                    if let Some(finally_block) = &try_stmt.finalizer {
                        for stmt in &finally_block.stmts {
                            find_declarations(decls, flags, stmt);
                        }
                    }
                }
            }

            Stmt::ForIn(for_in_stmt) => {
                use swc_ecma_ast::ForHead;

                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();
                    if let ForHead::VarDecl(var_decl) = &for_in_stmt.left {
                        find_var_decls(decls, flags, var_decl);
                    }
                    find_declarations(decls, flags, &for_in_stmt.body);
                }
            }

            Stmt::ForOf(for_or_stmt) => {
                use swc_ecma_ast::ForHead;

                if flags.hoist_functions || flags.hoist_vars {
                    let flags = flags.for_subscope_hoisting();
                    if let ForHead::VarDecl(var_decl) = &for_or_stmt.left {
                        find_var_decls(decls, flags, var_decl);
                    }
                    find_declarations(decls, flags, &for_or_stmt.body);
                }
            }

            _ => {}
        }
    }

    pub fn find_var_decls(
        decls: &mut Vec<Decl<'_>>,
        flags: FindDeclFlags,
        var_decl: &swc_ecma_ast::VarDecl,
    ) {
        use swc_ecma_ast::VarDeclKind;

        let is_lexical = match var_decl.kind {
            VarDeclKind::Var => false,
            VarDeclKind::Let => true,
            VarDeclKind::Const => true,
        };

        let is_included = if flags.only_hoisted {
            flags.hoist_vars && !is_lexical
        } else {
            true
        };

        if is_included {
            for vd in &var_decl.decls {
                let kind = match var_decl.kind {
                    VarDeclKind::Var => Kind::Var { fn_decl: None },
                    VarDeclKind::Let => Kind::Lexical { is_const: false },
                    VarDeclKind::Const => Kind::Lexical { is_const: true },
                };

                let name = vd
                    .name
                    .as_ident()
                    .unwrap_or_else(|| unsupported_node!(vd))
                    .sym
                    .clone();
                let span = vd.span;
                decls.push(Decl { kind, name, span });
            }
        }
    }

    pub fn check_redeclarations(mut decls: Vec<Decl>) -> Result<Vec<Decl>> {
        // Sort by name so that redeclarations become adjacent.
        decls.sort_by_key(|decl| decl.name.unsafe_data());

        let mut decls = decls.into_iter().peekable();
        let mut decls_deduped = Vec::with_capacity(decls.len());

        // This scan relies on the above ordering (which acts as a grouping mechanism)
        while let Some(mut decl) = decls.next() {
            while let Some(next) = decls.peek() {
                if next.name == decl.name {
                    // next is a redeclaration of decl.name

                    match (&mut decl.kind, &next.kind) {
                        // var/function over var/function => OK
                        (Kind::Var { fn_decl: decl_fd }, Kind::Var { fn_decl: next_fd }) => {
                            // function 'override' vars in the decl scan phase
                            // (but NOTE: later, in the 'assignment' phase, the
                            // var *assignment* will override the function).

                            // when multiple function decls "collide", the last
                            // one in source code order wins
                            *decl_fd = match (*decl_fd, next_fd) {
                                (None, None) => None,
                                (None, Some(fd)) => Some(fd),
                                (Some(fd), None) => Some(fd),
                                (Some(_), Some(fd)) => Some(fd),
                            };
                        }
                        (_, _) => {
                            let err =
                                error!("Identifier '{}' has already been declared", next.name)
                                    .with_span(next.span);
                            return Err(err);
                        }
                    };

                    decls.next();
                } else {
                    break;
                }
            }

            // note: redeclarations are *not* pushed in the new vec
            decls_deduped.push(decl);
        }

        Ok(decls_deduped)
    }

    pub fn declare_params(decls: &mut Vec<Decl>, params: &[swc_ecma_ast::Param]) {
        for (param_ndx, param) in params.iter().enumerate() {
            let param_ndx: u8 = param_ndx.try_into().expect("too many parameters!");

            if param_ndx >= bytecode::ARGS_COUNT_MAX {
                panic!(
                    "unsupported: function with more than {} parameters",
                    bytecode::ARGS_COUNT_MAX
                );
            }

            if !param.decorators.is_empty() {
                panic!("unsupported: decorators on function parameters");
            }

            let name = match &param.pat {
                swc_ecma_ast::Pat::Ident(ident) => ident.sym.clone(),
                other => unsupported_node!(other),
            };

            decls.push(Decl {
                kind: Kind::Var { fn_decl: None },
                name,
                span: param.span(),
            });
        }
    }
}

fn compile_arrow_function<'a>(builder: &mut Builder, arrow: &'a ArrowExpr) -> Result<VReg> {
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

    // TODO Not 100% sure that arrow functions are always strict
    let mut decls = Vec::new();
    let flags = declset2::FindDeclFlags {
        hoist_vars: true,
        hoist_functions: false,
        only_hoisted: false,
    };
    if let Some(block_stmt) = arrow.body.as_block_stmt() {
        for stmt in &block_stmt.stmts {
            declset2::find_declarations(&mut decls, flags, stmt);
        }
    }
    // params need to be 'declare'd here so that redeclarations can be checked.
    // regardless, param names are bound to registers later independently anyway.
    for pat in &arrow.params {
        let name = pat
            .as_ident()
            .unwrap_or_else(|| unsupported_node!(pat))
            .sym
            .clone();
        decls.push(declset2::Decl {
            kind: declset2::Kind::Var { fn_decl: None },
            name,
            span: pat.span(),
        });
    }
    let decls = declset2::check_redeclarations(decls)?;

    builder.start_function();

    // We need to guarantee, here, that the first ARGS_COUNT_MAX registers are
    // assigned to as many functcion arguments
    for param_ndx in 0..bytecode::ARGS_COUNT_MAX {
        // always pull a new vreg even if we're short on actual func params, so
        // that we're sure that we allocate all the first ARGS_COUNT_MAX regs
        let reg = builder.new_vreg();

        if let Some(pat) = arrow.params.get(param_ndx as usize) {
            let name = pat
                .as_ident()
                .unwrap_or_else(|| unsupported_node!(pat))
                .sym
                .clone();
            builder.cur_fnb().define_var(name, reg);
        }
    }

    for decl in &decls {
        if let Some(fn_decl) = decl.kind.as_function() {
            let name = Some(decl.name.clone());
            compile_function(builder, name, &fn_decl.function)?;
        }
    }

    match arrow.body.as_ref() {
        swc_ecma_ast::BlockStmtOrExpr::BlockStmt(block) => {
            compile_block(builder, &block.stmts)?;
        }
        swc_ecma_ast::BlockStmtOrExpr::Expr(expr) => {
            // TODO actually make the compiled function return the expression's value
            compile_expr(builder, expr)?;
        }
    }

    let mut inner_fnb = builder.fn_stack.pop().expect("no FnBuilder!");
    inner_fnb.span = Some(arrow.span);

    let forced_this = builder.new_vreg();
    builder.emit(Instr::LoadThis(forced_this));

    // all the calls to `get_var` must come before builder.emit(ClosureNew), to guarantee that
    // the ClosureNew and any associated ClosureAddCapture are adjacent
    // TODO(performance) avoid this allocation?
    let mut args = Vec::new();
    for var_name in inner_fnb.captures.iter() {
        let var_name = JsWord::from(var_name.as_str());
        let var = builder.get_var(&var_name);
        let vreg = builder.read_var(&var);
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
fn compile_expr<'a>(builder: &mut Builder, expr: &'a swc_ecma_ast::Expr) -> Result<VReg> {
    use swc_ecma_ast::{CallExpr, Expr};

    let mut builder = builder.with_span(&expr.span());

    match expr {
        Expr::Call(call_expr @ CallExpr { callee, args, .. }) => {
            let callee = callee.as_expr().ok_or_else(|| {
                error!("only calls to simple identifiers are supported for now")
                    .with_span(call_expr.span)
            })?;

            // Some things expressed in the `f(...)` call syntax are not actually calls to
            // anything, but have a special meaning
            match callee.as_ref() {
                Expr::Ident(i) if &i.sym == "sink" => {
                    for arg in args {
                        let var = compile_expr(&mut builder, &arg.expr)?;
                        builder.emit(Instr::PushToSink(var));
                    }

                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Undefined);
                    Ok(ret)
                }
                Expr::Ident(i) if &i.sym == "__start_trace" => {
                    let trace_id = match args[0].expr.as_ref() {
                        Expr::Lit(Lit::Str(trace_id)) => trace_id.value.to_string(),
                        _ => {
                            panic!("__start_trace must be called with a trace ID: __start_trace('the-name-of-the-trace')")
                        }
                    };

                    builder.place_trace_anchor(trace_id);
                    let ret = builder.new_vreg();
                    builder.set_const(ret, Literal::Undefined);
                    Ok(ret)
                }
                Expr::Ident(i) if &i.sym == "require" => {
                    if args.len() != 1 {
                        return Err(error!("`require` takes a single argument only"));
                    }

                    let import_path = compile_expr(&mut builder, &args[0].expr)?;
                    let ret = builder.new_vreg();
                    builder.emit(Instr::ImportModule(ret, import_path));
                    Ok(ret)
                }
                Expr::Ident(i) if &i.sym == "eval" => {
                    return Err(error!("`eval` not supported"));
                }
                _ => {
                    let mut arg_regs = Vec::new();
                    for arg in args {
                        if arg.spread.is_some() {
                            panic!("unsupported: spread function parameter: function(a, b, ...)");
                        }
                        let reg = compile_expr(&mut builder, &arg.expr)?;
                        arg_regs.push(reg);
                    }

                    let (this, callee) = match callee.as_ref() {
                        Expr::Member(member_expr) => {
                            let MemberAccess { obj, key: _, value } =
                                compile_member_access(&mut builder, member_expr)?;
                            (obj, value)
                        }
                        other_expr => {
                            let this = builder.new_vreg();
                            builder.set_const(this, bytecode::Literal::Undefined);

                            let callee_func = compile_expr(&mut builder, other_expr)?;
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
            }
        }

        Expr::Bin(bin_expr) => {
            let a = compile_expr(&mut builder, &bin_expr.left)?;
            let b = compile_expr(&mut builder, &bin_expr.right)?;

            // TODO There must be a better way.  2-operand instructions?  accumulator register?
            let ret = builder.new_vreg();
            let instr = match bin_expr.op {
                BinaryOp::Add => Instr::OpAdd(ret, a, b),
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
                    let constructor = builder.read_global(&"RegExp".into());
                    let re_str = re_lit.exp.to_string().into();

                    let arg = builder.new_vreg();
                    builder.set_const(arg, re_str);
                    let args = vec![arg];

                    compile_new(&mut builder, constructor, ret, &args);
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
                _ => {
                    let var = builder.get_var(&ident.sym);
                    builder.read_var(&var)
                }
            };
            Ok(ret)
        }

        Expr::Assign(asmt) => compile_assignment(asmt, &mut builder)
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

                            let value = compile_expr(&mut builder, &kv_expr.value)?;

                            builder.emit(Instr::ObjSet { obj, key, value });
                        }

                        swc_ecma_ast::Prop::Method(method_prop) => {
                            let key = builder.new_vreg();
                            let name = compile_prop_name(&method_prop.key)?;
                            builder.set_const(key, name);

                            let value =
                                compile_function(&mut builder, None, &method_prop.function)?;

                            builder.emit(Instr::ObjSet { obj, key, value });
                        }

                        swc_ecma_ast::Prop::Shorthand(sh) => {
                            let key = builder.new_vreg();
                            builder.set_const(key, Literal::String(sh.sym.to_string()));

                            let var = builder.get_var(&sh.sym);
                            let value = builder.read_var(&var);
                            builder.emit(Instr::ObjSet { obj, key, value });
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

            let constructor = builder.read_global(&"Array".into());
            compile_new(&mut builder, constructor, array, &[]);

            for elem in arr_expr.elems.iter() {
                match elem {
                    Some(ExprOrSpread { spread, expr }) => {
                        if spread.is_some() {
                            return Err(error!("spread syntax is currently unsupported")
                                .with_span(arr_expr.span));
                        }
                        let value = compile_expr(&mut builder, expr)?;
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
            compile_arrow_function(&mut builder, arrow_expr)
        }
        Expr::Fn(fn_expr) => {
            // TODO Refactor this with Decl::Fn
            let name = fn_expr.ident.as_ref().map(|ident| ident.to_id().0);
            let func = &fn_expr.function;
            compile_function(&mut builder, name.clone(), func)
        }

        Expr::Unary(unary_expr) => {
            let dest = builder.new_vreg();
            match unary_expr.op {
                swc_ecma_ast::UnaryOp::Bang => {
                    let arg = compile_expr(&mut builder, &unary_expr.arg)?;
                    builder.emit(Instr::BoolNot { dest, arg });
                }
                swc_ecma_ast::UnaryOp::TypeOf => {
                    let arg = compile_expr(&mut builder, &unary_expr.arg)?;
                    builder.emit(Instr::TypeOf { dest, arg });
                }
                swc_ecma_ast::UnaryOp::Minus => {
                    let arg = compile_expr(&mut builder, &unary_expr.arg)?;
                    builder.emit(Instr::UnaryMinus { dest, arg });
                }
                swc_ecma_ast::UnaryOp::Plus => {
                    let arg = compile_expr(&mut builder, &unary_expr.arg)?;
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

                    let obj = compile_expr(&mut builder, &member_expr.obj)?;
                    let key = compile_obj_member_prop(&mut builder, &member_expr.prop)?;
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
                    let var = builder.get_var(&ident.sym);
                    let reg = builder.read_var(&var);
                    compile_value_update(&mut builder, update_expr.op, reg);
                    builder.write_var(&var, reg);
                    Ok(reg)
                }
                Expr::Member(member_expr) => {
                    let MemberAccess { obj, key, value } =
                        compile_member_access(&mut builder, member_expr)?;
                    compile_value_update(&mut builder, update_expr.op, value);
                    builder.emit(Instr::ObjSet { obj, key, value });
                    Ok(value)
                }
                other => todo!(
                    "unsupported: UpdateExpr on anything other than an identifier: {:?}",
                    other
                ),
            }
        }
        Expr::Member(member_expr) => Ok(compile_member_access(&mut builder, member_expr)?.value),

        Expr::Paren(inner) => compile_expr(&mut builder, &inner.expr),

        Expr::Cond(cond_expr) => {
            let ret = builder.new_vreg();

            let cond = compile_expr(&mut builder, &cond_expr.test)?;
            builder.emit(Instr::BoolNot {
                dest: cond,
                arg: cond,
            });
            let jmpif = builder.reserve();

            let value = compile_expr(&mut builder, &cond_expr.cons)?;
            builder.emit(Instr::Copy {
                dst: ret,
                src: value,
            });
            let jmp_to_end = builder.reserve();

            let alt = builder.peek_iid();
            *builder.get_mut(jmpif).unwrap() = Instr::JmpIf { cond, dest: alt };
            let value = compile_expr(&mut builder, &cond_expr.alt)?;
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
                    let arg = compile_expr(&mut builder, &arg_or_spread.expr)?;
                    arg_regs.push(arg);
                }
            }

            let ret = builder.new_vreg();
            let callee = compile_expr(&mut builder, &new_expr.callee)?;
            compile_new(&mut builder, callee, ret, &arg_regs);
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

                let value = compile_expr(&mut builder, expr)?;
                builder.emit(Instr::StrAppend(buf_reg, value));
            }

            let last_str = tpl.quasis.last().unwrap().raw.to_string();
            builder.set_const(tmp_reg, Literal::String(last_str));
            builder.emit(Instr::StrAppend(buf_reg, tmp_reg));

            Ok(buf_reg)
        }

        Expr::Seq(seq_expr) => {
            let (head, tail) = seq_expr.exprs.split_first().unwrap();
            let mut result = compile_expr(&mut builder, head)?;
            for expr in tail {
                result = compile_expr(&mut builder, expr)?;
            }

            Ok(result)
        }

        // Expr::SuperProp(_) => todo!(),
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

    builder.set_const(key, Literal::String("constructor".into()));
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

fn compile_assignment<'a>(
    asmt: &'a swc_ecma_ast::AssignExpr,
    builder: &mut Builder,
) -> Result<VReg> {
    use swc_ecma_ast::{Expr, MemberExpr, MemberProp, PatOrExpr};

    if let Some(ident) = asmt.left.as_ident() {
        let rhs = compile_expr(builder, &asmt.right)?;
        let var = builder.get_var(&ident.sym);
        compile_assignment_rhs(builder, asmt.op, &var, rhs);
        Ok(builder.read_var(&var))
    } else if let Some(target_expr) = asmt.left.as_expr() {
        match target_expr {
            Expr::Member(member_expr) => {
                let rhs = compile_expr(builder, &asmt.right)?;

                let MemberAccess { obj, key, value } = compile_member_access(builder, member_expr)?;
                compile_assignment_rhs(builder, asmt.op, &Var::Reg(value), rhs);
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
fn compile_member_access<'a>(
    builder: &mut Builder,
    member_expr: &'a swc_ecma_ast::MemberExpr,
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

fn compile_obj_member_prop<'a>(
    builder: &mut Builder,
    member_prop: &'a swc_ecma_ast::MemberProp,
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
    lhs: &Var,
    rhs: VReg,
) {
    let lhs_value = builder.read_var(lhs);

    type InstrCons = fn(VReg, VReg, VReg) -> Instr;
    let mut xform_value = |lhs_value: VReg, rhs: VReg, cons: InstrCons| {
        builder.emit(cons(lhs_value, lhs_value, rhs));
        lhs_value
    };

    let lhs_value = match assign_op {
        AssignOp::Assign => rhs,
        AssignOp::AddAssign => xform_value(lhs_value, rhs, Instr::OpAdd),
        AssignOp::SubAssign => xform_value(lhs_value, rhs, Instr::ArithSub),
        AssignOp::MulAssign => xform_value(lhs_value, rhs, Instr::ArithMul),
        AssignOp::DivAssign => xform_value(lhs_value, rhs, Instr::ArithDiv),
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

    builder.write_var(lhs, lhs_value);
}

fn parse_file(
    filename: String,
    content: String,
    source_map: Lrc<SourceMap>,
) -> Result<swc_ecma_ast::Module> {
    let err_handler = mk_error_handler(&source_map);

    let path = std::path::PathBuf::from(filename);
    let source_file = source_map.new_source_file(swc_common::FileName::Real(path), content);

    let mut parser = make_parser(&*source_file, &err_handler);
    parser.parse_module().map_err(|e| {
        e.into_diagnostic(&err_handler).emit();
        error!("parse error")
    })
}

fn make_parser<'a>(
    source_file: &'a swc_common::SourceFile,
    err_handler: &swc_common::errors::Handler,
) -> swc_ecma_parser::Parser<swc_ecma_parser::lexer::Lexer<'a>> {
    use swc_common::SourceMap;
    use swc_ecma_ast::EsVersion;
    use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

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
    parser
}

fn mk_error_handler(source_map: &Rc<SourceMap>) -> swc_common::errors::Handler {
    use swc_common::errors::{emitter::EmitterWriter, Handler};
    Handler::with_emitter(
        true,  // can_emit_warnings
        false, // treat_err_as_bug
        Box::new(EmitterWriter::new(
            Box::new(std::io::stderr()),
            Some(source_map.clone()),
            false, // short_message
            true,  // teach
        )),
    )
}

fn is_identifier_keyword(ident: &swc_ecma_ast::Ident) -> bool {
    &ident.sym == "let" || &ident.sym == "const"
}

/// The Processed AST.
///
/// An intermediate representation that results from an initial processing of
/// swc_ecma_ast's AST, designed expressly to facilitate compilation to bytecode
/// and compile-time checks.
mod past {
    use std::collections::HashSet;

    use swc_atoms::JsWord;

    #[derive(Debug)]
    pub struct Function {
        pub n_parameters: u8,
        pub unbound_names: Vec<JsWord>,
        pub body: Block,
    }

    #[derive(Debug)]
    pub struct Block {
        pub id: BlockID,
        pub decls: Vec<Decl>,
        pub stmts: Vec<Stmt>,
    }
    impl Block {
        fn empty(block_id: BlockID) -> Self {
            Block {
                id: block_id,
                decls: Vec::new(),
                stmts: Vec::new(),
            }
        }
    }

    pub struct Decl {
        pub name: DeclName,
        pub is_lexical: bool,
        is_global: bool,
    }
    impl std::fmt::Debug for Decl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Decl: {} {:?}",
                if self.is_lexical { "let" } else { "var" },
                self.name
            )
        }
    }
    #[derive(Clone, PartialEq, Eq, Hash)]
    pub enum DeclName {
        Js(JsWord),
        Tmp(TmpID),
    }

    impl std::fmt::Debug for DeclName {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let s = match self {
                DeclName::Js(word) => word.to_string(),
                DeclName::Tmp(tmpid) => format!("tmp{}", tmpid.numeric()),
            };
            write!(f, "{}", s)
        }
    }

    #[derive(Debug)]
    pub enum Stmt {
        Block(Block),
        Break(BlockID),
        Continue(BlockID),
        Unshare(BlockID),
        If {
            test: Box<Stmt>,
            cons: Box<Stmt>,
            alt: Box<Stmt>,
        },

        Undefined,
        Null,
        This,
        Read(DeclName),

        AssignParam(DeclName, u8),

        Assign(DeclName, Box<Stmt>),
        Unary(swc_ecma_ast::UnaryOp, Box<Stmt>),
        Binary(swc_ecma_ast::BinaryOp, Box<Stmt>, Box<Stmt>),

        StringLiteral(JsWord),
        NumberLiteral(f64),
        BoolLiteral(bool),

        ArrayCreate,
        ArrayPush(DeclName, Box<Stmt>),

        ObjectCreate,
        ObjectGet {
            obj: Box<Stmt>,
            key: Box<Stmt>,
        },
        ObjectSet {
            obj: Box<Stmt>,
            key: Box<Stmt>,
            value: Box<Stmt>,
        },

        CreateClosure {
            func: Box<Function>,
        },

        Call {
            is_new: bool,
            callee: Box<Stmt>,
            args: Vec<Stmt>,
        },

        Throw(Box<Stmt>),
        GetCurrentException,
        Try {
            main_block: Box<Stmt>,
            handler_block: Box<Stmt>,
            finalizer_block: Box<Stmt>,
        },

        Debugger,
    }

    mod builder {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct BlockID(u32);

        impl std::fmt::Debug for BlockID {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "block{}", self.0)
            }
        }

        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct LabelID(u32);

        impl std::fmt::Debug for LabelID {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "label{}", self.0)
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct TmpID(u32);
        impl TmpID {
            pub fn numeric(&self) -> u32 {
                self.0
            }
        }

        pub struct Builder {
            breakable_blocks: Vec<BlockID>,
            next_id: u32,
        }

        impl Builder {
            pub fn new() -> Self {
                Builder {
                    breakable_blocks: Vec::new(),
                    next_id: 0,
                }
            }

            pub fn break_target(&self) -> BlockID {
                self.breakable_blocks.last().copied().unwrap()
            }

            pub(crate) fn gen_block_id(&mut self) -> BlockID {
                let blkid = BlockID(self.next_id);
                self.next_id += 1;
                blkid
            }

            pub(crate) fn gen_label_id(&mut self) -> LabelID {
                let lblid = LabelID(self.next_id);
                self.next_id += 1;
                lblid
            }

            pub(crate) fn gen_tmp(&mut self) -> TmpID {
                let tmpid = TmpID(self.next_id);
                self.next_id += 1;
                tmpid
            }
        }
    }

    use builder::Builder;
    pub use builder::{BlockID, LabelID, TmpID};

    use crate::error;

    pub fn compile_script(ast_module: swc_ecma_ast::Module) -> Function {
        use swc_ecma_ast::ModuleItem;

        let stmts: Vec<_> = ast_module
            .body
            .into_iter()
            .map(|item| match item {
                ModuleItem::ModuleDecl(_) => panic!("invalid input: module decl in script"),
                ModuleItem::Stmt(stmt) => stmt,
            })
            .collect();

        let mut builder = Builder::new();
        let block = compile_block(&mut builder, &stmts);
        compile_function_from_parts(&mut builder, &[], Stmt::Block(block))
    }

    fn compile_block(builder: &mut Builder, swc_stmts: &[swc_ecma_ast::Stmt]) -> Block {
        let block_id = builder.gen_block_id();
        let mut block = Block {
            id: block_id,
            decls: Vec::new(),
            stmts: Vec::new(),
        };

        for stmt in swc_stmts {
            compile_stmt(builder, &mut block, stmt);
        }
        block
    }

    fn compile_block_single_stmt(builder: &mut Builder, swc_stmt: &swc_ecma_ast::Stmt) -> Block {
        let block_id = builder.gen_block_id();
        let mut block = Block {
            id: block_id,
            decls: Vec::new(),
            stmts: Vec::new(),
        };

        compile_stmt(builder, &mut block, swc_stmt);
        block
    }

    fn compile_stmt(builder: &mut Builder, block: &mut Block, stmt: &swc_ecma_ast::Stmt) {
        match stmt {
            swc_ecma_ast::Stmt::Block(block_stmt) => {
                let mut inner_block = compile_block(builder, &block_stmt.stmts);
                hoist_declarations(block, &mut inner_block.decls);
                block.stmts.push(Stmt::Block(inner_block));
            }
            swc_ecma_ast::Stmt::Empty(_) => {}
            swc_ecma_ast::Stmt::Debugger(_) => {
                block.stmts.push(Stmt::Debugger);
            }

            swc_ecma_ast::Stmt::Return(return_stmt) => {
                let expr = return_stmt
                    .arg
                    .as_ref()
                    .map(|expr| compile_expr(builder, &expr))
                    .unwrap_or(Stmt::Undefined);
                block.stmts.push(expr);
            }

            swc_ecma_ast::Stmt::Break(break_stmt) => {
                if break_stmt.label.is_some() {
                    panic!("unsupported: labeled break statement");
                }
                let break_target = builder.break_target();
                block.stmts.push(Stmt::Break(break_target));
            }
            swc_ecma_ast::Stmt::Continue(continue_stmt) => {
                if continue_stmt.label.is_some() {
                    panic!("unsupported: labeled continue statement");
                }
                let break_target = builder.break_target();
                block.stmts.push(Stmt::Continue(break_target));
            }

            swc_ecma_ast::Stmt::If(if_stmt) => {
                let test = compile_expr(builder, &if_stmt.test);

                // TODO Avoid these ephemeral Vec's

                let mut block_cons = compile_block_single_stmt(builder, &if_stmt.cons);
                hoist_declarations(&mut block_cons, &mut block.decls);

                let block_alt = if let Some(alt) = &if_stmt.alt {
                    let mut block_alt = compile_block_single_stmt(builder, alt);
                    hoist_declarations(&mut block_alt, &mut block.decls);
                    block_alt
                } else {
                    Block::empty(builder.gen_block_id())
                };

                block.stmts.push(Stmt::If {
                    test: Box::new(test),
                    cons: Box::new(Stmt::Block(block_cons)),
                    alt: Box::new(Stmt::Block(block_alt)),
                });
            }

            swc_ecma_ast::Stmt::Switch(switch_stmt) => {
                // In order to simulate the consequences of the body of the
                // switch statement being a single flat scope (with
                // declarations 'shared' between cases, and fallthrough
                // between 'case:' labels), we:
                //  - compile each case block as a separate block;
                //  - merge each block into a single superblock with an
                // if/else chain
                //  - hoist *all* declarations (regardless of what
                //  ECMAScript says)
                todo!("switch")
            }

            swc_ecma_ast::Stmt::Throw(throw_stmt) => {
                let value = compile_expr(builder, &throw_stmt.arg);
                block.stmts.push(Stmt::Throw(Box::new(value)));
            }
            swc_ecma_ast::Stmt::Try(try_stmt) => {
                let main_block = Stmt::Block(compile_block(builder, &try_stmt.block.stmts));
                let finalizer_block = try_stmt
                    .finalizer
                    .as_ref()
                    .map(|block_stmt| Stmt::Block(compile_block(builder, &block_stmt.stmts)))
                    .unwrap_or(Stmt::Undefined);

                let handler_block = {
                    let handler_block_id = builder.gen_block_id();
                    if let Some(handler) = &try_stmt.handler {
                        let mut decls = Vec::new();
                        let mut stmts = Vec::new();

                        if let Some(handler_param) = &handler.param {
                            let name = compile_name_pat(handler_param);
                            decls.push(Decl {
                                name: name.clone(),
                                is_lexical: true,
                                is_global: false,
                            });
                            stmts.push(Stmt::Assign(name, Box::new(Stmt::GetCurrentException)));
                        }

                        let handler_block = compile_block(builder, &handler.body.stmts);
                        stmts.push(Stmt::Block(handler_block));

                        Stmt::Block(Block {
                            id: handler_block_id,
                            decls,
                            stmts,
                        })
                    } else {
                        Stmt::Undefined
                    }
                };

                block.stmts.push(Stmt::Try {
                    main_block: Box::new(main_block),
                    handler_block: Box::new(handler_block),
                    finalizer_block: Box::new(finalizer_block),
                })
            }
            swc_ecma_ast::Stmt::While(while_stmt) => {
                let mut inner_block = Block::empty(builder.gen_block_id());

                let test_expr = compile_expr(builder, &while_stmt.test);
                let test_expr = Stmt::Unary(swc_ecma_ast::UnaryOp::Bang, Box::new(test_expr));
                inner_block.stmts.push(Stmt::If {
                    test: Box::new(test_expr),
                    cons: Box::new(Stmt::Break(inner_block.id)),
                    alt: Box::new(Stmt::Undefined),
                });

                compile_stmt(builder, &mut inner_block, &while_stmt.body);

                inner_block.stmts.push(Stmt::Continue(inner_block.id));

                block.stmts.push(Stmt::Block(inner_block));
            }
            swc_ecma_ast::Stmt::DoWhile(dowhile_stmt) => {
                let mut inner_block = Block::empty(builder.gen_block_id());

                compile_stmt(builder, &mut inner_block, &dowhile_stmt.body);

                let test_expr = compile_expr(builder, &dowhile_stmt.test);
                let test_expr = Stmt::Unary(swc_ecma_ast::UnaryOp::Bang, Box::new(test_expr));
                inner_block.stmts.push(Stmt::If {
                    test: Box::new(test_expr),
                    cons: Box::new(Stmt::Continue(inner_block.id)),
                    alt: Box::new(Stmt::Undefined),
                });

                block.stmts.push(Stmt::Block(inner_block));
            }
            swc_ecma_ast::Stmt::For(for_stmt) => {
                let mut outer_block = Block::empty(builder.gen_block_id());

                match &for_stmt.init {
                    Some(swc_ecma_ast::VarDeclOrExpr::VarDecl(var_decl)) => {
                        compile_var_decl(builder, &mut outer_block, var_decl)
                    }
                    Some(swc_ecma_ast::VarDeclOrExpr::Expr(expr)) => {
                        outer_block.stmts.push(compile_expr(builder, expr));
                    }
                    None => todo!(),
                }

                let mut inner_block = Block::empty(builder.gen_block_id());

                inner_block.stmts.push(Stmt::If {
                    test: Box::new(
                        for_stmt
                            .test
                            .as_ref()
                            .map(|expr| compile_expr(builder, &expr))
                            .unwrap_or(Stmt::BoolLiteral(true)),
                    ),
                    cons: Box::new(Stmt::Undefined),
                    alt: Box::new(Stmt::Break(outer_block.id)),
                });

                compile_stmt(builder, &mut inner_block, &for_stmt.body);
                if let Some(update) = &for_stmt.update {
                    inner_block.stmts.push(compile_expr(builder, update));
                }
                inner_block.stmts.push(Stmt::Unshare(outer_block.id));
                inner_block.stmts.push(Stmt::Continue(inner_block.id));

                outer_block.stmts.push(Stmt::Block(inner_block));
                block.stmts.push(Stmt::Block(outer_block));
            }

            swc_ecma_ast::Stmt::ForIn(_) => todo!(),
            swc_ecma_ast::Stmt::ForOf(_) => todo!(),

            swc_ecma_ast::Stmt::Decl(decl) => match decl {
                swc_ecma_ast::Decl::Fn(fn_decl) => {
                    let name = DeclName::Js(fn_decl.ident.sym.clone());
                    block.decls.push(Decl {
                        name,
                        is_lexical: false,
                        is_global: false,
                    });
                    let func = compile_function(builder, &fn_decl.function);
                    block.stmts.push(Stmt::CreateClosure {
                        func: Box::new(func),
                    });
                }
                swc_ecma_ast::Decl::Var(var_decl) => compile_var_decl(builder, block, var_decl),
                _ => {
                    unsupported_node!(decl)
                }
            },
            swc_ecma_ast::Stmt::Expr(expr_stmt) => {
                block.stmts.push(compile_expr(builder, &expr_stmt.expr));
            }

            swc_ecma_ast::Stmt::Labeled(_) | swc_ecma_ast::Stmt::With(_) => {
                unsupported_node!(stmt)
            }
        }
    }

    fn compile_var_decl(
        builder: &mut Builder,
        block: &mut Block,
        var_decl: &swc_ecma_ast::VarDecl,
    ) {
        let is_lexical = match var_decl.kind {
            swc_ecma_ast::VarDeclKind::Var => false,
            swc_ecma_ast::VarDeclKind::Let => true,
            swc_ecma_ast::VarDeclKind::Const => true,
        };

        for declarator in &var_decl.decls {
            let name = compile_name_pat(&declarator.name);
            block.decls.push(Decl {
                name: name.clone(),
                is_lexical,
                is_global: false,
            });

            if let Some(init) = &declarator.init {
                let value = compile_expr(builder, init);
                block.stmts.push(Stmt::Assign(name, Box::new(value)));
            }
        }
    }

    fn hoist_declarations(block: &mut Block, decls: &mut Vec<Decl>) {
        // TODO Surely a lot of allocations are happening here...
        let inner_decls = std::mem::replace(&mut block.decls, Vec::new());
        for decl in inner_decls {
            if decl.is_lexical {
                block.decls.push(decl);
            } else {
                decls.push(decl);
            }
        }

        // TODO Check redeclarations
    }

    fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Stmt {
        match expr {
            swc_ecma_ast::Expr::This(_) => Stmt::This,
            swc_ecma_ast::Expr::Array(array_expr) => {
                let block_id = builder.gen_block_id();
                let mut decls = Vec::new();
                let mut stmts = Vec::new();

                let tmp = DeclName::Tmp(builder.gen_tmp());
                decls.push(Decl {
                    name: tmp.clone(),
                    is_lexical: true,
                    is_global: false,
                });

                stmts.push(Stmt::Assign(tmp.clone(), Box::new(Stmt::ArrayCreate)));
                for value in &array_expr.elems {
                    // TODO What does `None` mean here?
                    if let Some(expr_or_spread) = value {
                        if expr_or_spread.spread.is_some() {
                            panic!("unsupported: spread syntax in array literal");
                        }

                        let value = compile_expr(builder, &expr_or_spread.expr);
                        stmts.push(Stmt::ArrayPush(tmp.clone(), Box::new(value)));
                    }
                }
                stmts.push(Stmt::Read(tmp));

                Stmt::Block(Block {
                    id: block_id,
                    decls,
                    stmts,
                })
            }
            swc_ecma_ast::Expr::Object(object_expr) => {
                let block_id = builder.gen_block_id();
                let mut decls = Vec::new();
                let mut stmts = Vec::new();

                let tmp = DeclName::Tmp(builder.gen_tmp());
                decls.push(Decl {
                    name: tmp.clone(),
                    is_lexical: true,
                    is_global: false,
                });

                stmts.push(Stmt::Assign(tmp.clone(), Box::new(Stmt::ObjectCreate)));

                for prop in &object_expr.props {
                    match prop {
                        swc_ecma_ast::PropOrSpread::Spread(_) => {
                            panic!("unsupported: spread syntax in object literal");
                        }
                        swc_ecma_ast::PropOrSpread::Prop(prop) => {
                            let (key, value) = match prop.as_ref() {
                                swc_ecma_ast::Prop::Shorthand(name) => {
                                    let key = Stmt::StringLiteral(name.sym.clone());
                                    let value = Stmt::Read(DeclName::Js(name.sym.clone()));
                                    (key, value)
                                }
                                swc_ecma_ast::Prop::KeyValue(kv) => {
                                    let key = match &kv.key {
                                        swc_ecma_ast::PropName::Ident(ident) => {
                                            Stmt::Read(DeclName::Js(ident.sym.clone()))
                                        }
                                        swc_ecma_ast::PropName::Str(str) => {
                                            Stmt::StringLiteral(JsWord::from(str.value.to_string()))
                                        }
                                        swc_ecma_ast::PropName::Num(num) => {
                                            Stmt::NumberLiteral(num.value)
                                        }
                                        _ => {
                                            unsupported_node!(kv.key)
                                        }
                                    };
                                    let value = compile_expr(builder, &kv.value);

                                    (key, value)
                                }
                                swc_ecma_ast::Prop::Assign(_)
                                | swc_ecma_ast::Prop::Getter(_)
                                | swc_ecma_ast::Prop::Setter(_)
                                | swc_ecma_ast::Prop::Method(_) => todo!(),
                            };

                            stmts.push(Stmt::ObjectSet {
                                obj: Box::new(Stmt::Read(tmp.clone())),
                                key: Box::new(key),
                                value: Box::new(value),
                            });
                        }
                    }
                }

                stmts.push(Stmt::Read(tmp));

                Stmt::Block(Block {
                    id: block_id,
                    decls,
                    stmts,
                })
            }

            swc_ecma_ast::Expr::Fn(fn_expr) => {
                let func = compile_function(builder, &fn_expr.function);
                Stmt::CreateClosure {
                    func: Box::new(func),
                }
            }
            swc_ecma_ast::Expr::Arrow(arrow_expr) => {
                let params: Vec<_> = arrow_expr.params.iter().cloned().map(From::from).collect();
                let body = match &*arrow_expr.body {
                    swc_ecma_ast::BlockStmtOrExpr::BlockStmt(block_stmts) => {
                        Stmt::Block(compile_block(builder, &block_stmts.stmts))
                    }
                    swc_ecma_ast::BlockStmtOrExpr::Expr(expr) => compile_expr(builder, expr),
                };
                let func = compile_function_from_parts(builder, &params, body);
                Stmt::CreateClosure {
                    func: Box::new(func),
                }
            }
            swc_ecma_ast::Expr::Unary(unary_expr) => {
                let arg = compile_expr(builder, &unary_expr.arg);
                Stmt::Unary(unary_expr.op, Box::new(arg))
            }
            swc_ecma_ast::Expr::Update(update_expr) => {
                let loc = compile_name(swc_ecma_ast::PatOrExpr::Expr(update_expr.arg.clone()));
                let new_value = {
                    let value = Box::new(compile_expr(builder, &update_expr.arg));
                    let one = Box::new(Stmt::NumberLiteral(1.0));
                    match update_expr.op {
                        swc_ecma_ast::UpdateOp::PlusPlus => {
                            Stmt::Binary(swc_ecma_ast::BinaryOp::Add, value, one)
                        }
                        swc_ecma_ast::UpdateOp::MinusMinus => {
                            Stmt::Binary(swc_ecma_ast::BinaryOp::Sub, value, one)
                        }
                    }
                };
                Stmt::Assign(loc, Box::new(new_value))
            }
            swc_ecma_ast::Expr::Bin(bin_expr) => {
                let left = compile_expr(builder, &bin_expr.left);
                let right = compile_expr(builder, &bin_expr.right);
                Stmt::Binary(bin_expr.op, Box::new(left), Box::new(right))
            }
            swc_ecma_ast::Expr::Assign(assign_expr) => {
                if let Some(ident) = assign_expr.left.as_ident() {
                    let loc = DeclName::Js(ident.sym.clone());
                    let value = compile_expr(builder, &assign_expr.right);
                    Stmt::Assign(loc, Box::new(value))
                } else if let Some(target_expr) = assign_expr.left.as_expr() {
                    match target_expr {
                        swc_ecma_ast::Expr::Member(member_expr) => {
                            compile_member_assignment(builder, assign_expr, member_expr)
                        }
                        // We should have already handled this case in the `if let ... = asm.left.as_ident()`
                        // case
                        swc_ecma_ast::Expr::Ident(_) => unreachable!(),
                        _ => {
                            panic!("assignment to an expression is unsupported")
                        }
                    }
                } else {
                    panic!(
                        "unsupported pattern as assignment target: {:?}",
                        assign_expr.left
                    )
                }
            }
            swc_ecma_ast::Expr::Member(member_expr) => {
                let obj = compile_expr(builder, &member_expr.obj);
                let key = compile_object_key(builder, &member_expr.prop);
                Stmt::ObjectGet {
                    obj: Box::new(obj),
                    key: Box::new(key),
                }
            }
            swc_ecma_ast::Expr::Cond(cond_expr) => {
                let test = Box::new(compile_expr(builder, &cond_expr.test));
                let cons = Box::new(compile_expr(builder, &cond_expr.cons));
                let alt = Box::new(compile_expr(builder, &cond_expr.alt));
                Stmt::If { test, cons, alt }
            }
            swc_ecma_ast::Expr::Call(call_expr) => {
                let callee = match &call_expr.callee {
                    swc_ecma_ast::Callee::Super(_) | swc_ecma_ast::Callee::Import(_) => {
                        unsupported_node!(call_expr.callee)
                    }
                    swc_ecma_ast::Callee::Expr(expr) => expr,
                };
                compile_call(builder, callee, &call_expr.args, false)
            }
            swc_ecma_ast::Expr::New(new_expr) => {
                let args = new_expr
                    .args
                    .as_ref()
                    .map(|args| args.as_slice())
                    .unwrap_or(&[]);
                compile_call(builder, &new_expr.callee, args, true)
            }
            swc_ecma_ast::Expr::Seq(seq_expr) => {
                let block_id = builder.gen_block_id();
                let stmts = seq_expr
                    .exprs
                    .iter()
                    .map(|expr| compile_expr(builder, expr))
                    .collect();
                Stmt::Block(Block {
                    id: block_id,
                    stmts,
                    decls: Vec::new(),
                })
            }
            swc_ecma_ast::Expr::Ident(ident) => Stmt::Read(DeclName::Js(ident.sym.clone())),
            swc_ecma_ast::Expr::Lit(lit) => match lit {
                swc_ecma_ast::Lit::Str(str) => Stmt::StringLiteral(str.value.to_string().into()),
                swc_ecma_ast::Lit::Bool(b) => Stmt::BoolLiteral(b.value),
                swc_ecma_ast::Lit::Null(_) => Stmt::Null,
                swc_ecma_ast::Lit::Num(num) => Stmt::NumberLiteral(num.value),
                swc_ecma_ast::Lit::BigInt(_)
                | swc_ecma_ast::Lit::Regex(_)
                | swc_ecma_ast::Lit::JSXText(_) => unsupported_node!(lit),
            },

            swc_ecma_ast::Expr::Tpl(_) => todo!(),
            swc_ecma_ast::Expr::Paren(paren_expr) => compile_expr(builder, &paren_expr.expr),

            swc_ecma_ast::Expr::SuperProp(_)
            | swc_ecma_ast::Expr::TaggedTpl(_)
            | swc_ecma_ast::Expr::Class(_)
            | swc_ecma_ast::Expr::Yield(_)
            | swc_ecma_ast::Expr::MetaProp(_)
            | swc_ecma_ast::Expr::Await(_)
            | swc_ecma_ast::Expr::JSXMember(_)
            | swc_ecma_ast::Expr::JSXNamespacedName(_)
            | swc_ecma_ast::Expr::JSXEmpty(_)
            | swc_ecma_ast::Expr::JSXElement(_)
            | swc_ecma_ast::Expr::JSXFragment(_)
            | swc_ecma_ast::Expr::TsTypeAssertion(_)
            | swc_ecma_ast::Expr::TsConstAssertion(_)
            | swc_ecma_ast::Expr::TsNonNull(_)
            | swc_ecma_ast::Expr::TsAs(_)
            | swc_ecma_ast::Expr::TsInstantiation(_)
            | swc_ecma_ast::Expr::TsSatisfies(_)
            | swc_ecma_ast::Expr::PrivateName(_)
            | swc_ecma_ast::Expr::OptChain(_)
            | swc_ecma_ast::Expr::Invalid(_) => {
                unsupported_node!(expr);
            }
        }
    }

    fn compile_member_assignment(
        builder: &mut Builder,
        assign_expr: &swc_ecma_ast::AssignExpr,
        member_expr: &swc_ecma_ast::MemberExpr,
    ) -> Stmt {
        let mut block = Block::empty(builder.gen_block_id());

        let key = compile_object_key(builder, &member_expr.prop);
        let obj = compile_expr(builder, &member_expr.obj);

        let tmp_key = create_tmp(builder, &mut block, key);
        let tmp_obj = create_tmp(builder, &mut block, obj);

        let init_value = Stmt::ObjectGet {
            obj: Box::new(Stmt::Read(tmp_obj.clone())),
            key: Box::new(Stmt::Read(tmp_key.clone())),
        };

        let rhs = compile_expr(builder, &assign_expr.right);
        let value = match assign_expr.op.to_update() {
            // regular assignment
            None => rhs,
            // updating assignmenet
            Some(binop) => Stmt::Binary(binop, Box::new(init_value), Box::new(rhs)),
        };
        block.stmts.push(Stmt::ObjectSet {
            obj: Box::new(Stmt::Read(tmp_obj)),
            key: Box::new(Stmt::Read(tmp_key)),
            value: Box::new(value),
        });

        Stmt::Block(block)
    }

    fn create_tmp(builder: &mut Builder, block: &mut Block, value: Stmt) -> DeclName {
        let tmp = DeclName::Tmp(builder.gen_tmp());
        block.decls.push(Decl {
            is_lexical: true,
            name: tmp.clone(),
            is_global: false,
        });
        block.stmts.push(Stmt::Assign(tmp.clone(), Box::new(value)));
        tmp
    }

    fn compile_object_key(builder: &mut Builder, prop: &swc_ecma_ast::MemberProp) -> Stmt {
        match prop {
            swc_ecma_ast::MemberProp::Ident(ident) => Stmt::StringLiteral(ident.sym.clone()),
            swc_ecma_ast::MemberProp::Computed(computed) => compile_expr(builder, &computed.expr),
            swc_ecma_ast::MemberProp::PrivateName(_) => {
                unsupported_node!(prop)
            }
        }
    }

    fn compile_name(pat_or_expr: swc_ecma_ast::PatOrExpr) -> DeclName {
        let pat = match pat_or_expr {
            swc_ecma_ast::PatOrExpr::Expr(expr) => match *expr {
                swc_ecma_ast::Expr::Ident(ident) => swc_ecma_ast::Pat::Ident(ident.into()),
                other => panic!("invalid target for assignment: {:?}", other),
            },
            swc_ecma_ast::PatOrExpr::Pat(pat) => *pat,
        };

        compile_name_pat(&pat)
    }

    fn compile_name_pat(pat: &swc_ecma_ast::Pat) -> DeclName {
        match pat {
            swc_ecma_ast::Pat::Ident(ident) => DeclName::Js(ident.sym.clone()),
            _ => unsupported_node!(pat),
        }
    }

    fn compile_call(
        builder: &mut Builder,
        callee: &swc_ecma_ast::Expr,
        args: &[swc_ecma_ast::ExprOrSpread],
        is_new: bool,
    ) -> Stmt {
        let callee = Box::new(compile_expr(builder, callee));
        let args = args
            .iter()
            .map(|expr_or_spread| {
                if expr_or_spread.spread.is_some() {
                    panic!("unsupported: spread syntax in call");
                }
                compile_expr(builder, &expr_or_spread.expr)
            })
            .collect();

        Stmt::Call {
            is_new,
            callee,
            args,
        }
    }

    fn compile_function(builder: &mut Builder, swc_func: &swc_ecma_ast::Function) -> Function {
        if !swc_func.decorators.is_empty() {
            panic!("unsupported: function decorators");
        }
        if swc_func.is_async {
            panic!("unsupported: async functions");
        }
        if swc_func.is_generator {
            panic!("unsupported: generator functions");
        }
        if swc_func.return_type.is_some() {
            panic!("unsupported: TypeScript syntax (return type)");
        }
        if swc_func.type_params.is_some() {
            panic!("unsupported: TypeScript syntax (return type)");
        }

        let body = &swc_func.body.as_ref().unwrap().stmts;
        let body = Stmt::Block(compile_block(builder, body));
        compile_function_from_parts(builder, &swc_func.params, body)
    }

    fn compile_function_from_parts(
        builder: &mut Builder,
        params: &[swc_ecma_ast::Param],
        body: Stmt,
    ) -> Function {
        // Two nested blocks:
        //  - outer: declares variables corresponding to function parameters
        //  - inner: the actual function body
        // var declared names are hoisted to the outer block
        let mut decls = Vec::new();
        let mut stmts = Vec::new();
        let outer_block_id = builder.gen_block_id();

        let n_parameters = params.len().try_into().expect("too many parameters!");

        for (ndx, param) in params.into_iter().enumerate() {
            let ndx = ndx.try_into().unwrap();
            let name = compile_name_pat(&param.pat);

            decls.push(Decl {
                name: name.clone(),
                is_lexical: false,
                is_global: false,
            });
            stmts.push(Stmt::AssignParam(name, ndx));
        }

        stmts.push(body);

        let root = Block {
            id: outer_block_id,
            decls,
            stmts,
        };

        let unbound_names = find_unbound_references(&root);
        Function {
            n_parameters,
            unbound_names,
            body: root,
        }
    }

    fn visit_stmts(root: &Stmt, mut handler: impl FnMut(&Stmt)) {
        let mut queue = vec![root];

        while let Some(stmt) = queue.pop() {
            handler(stmt);
            match stmt {
                Stmt::Block(block) => {
                    queue.extend(block.stmts.iter());
                }
                Stmt::Break(_) => {}
                Stmt::Continue(_) => {}
                Stmt::Unshare(_) => {}
                Stmt::If { test, cons, alt } => {
                    queue.push(test);
                    queue.push(cons);
                    queue.push(alt);
                }
                Stmt::Undefined => {}
                Stmt::Null => {}
                Stmt::This => {}
                Stmt::Read(_) => {}
                Stmt::AssignParam(_, _) => {}
                Stmt::Assign(_, value_stmt) => {
                    queue.push(value_stmt);
                }
                Stmt::Unary(_, arg) => {
                    queue.push(arg);
                }
                Stmt::Binary(_, left, right) => {
                    queue.push(left);
                    queue.push(right);
                }
                Stmt::StringLiteral(_) => {}
                Stmt::NumberLiteral(_) => {}
                Stmt::BoolLiteral(_) => {}
                Stmt::ArrayCreate => {}
                Stmt::ArrayPush(_, value_stmt) => {
                    queue.push(value_stmt);
                }
                Stmt::ObjectCreate => {}
                Stmt::ObjectGet { obj, key } => {
                    queue.push(obj);
                    queue.push(key);
                }
                Stmt::ObjectSet { obj, key, value } => {
                    queue.push(obj);
                    queue.push(key);
                    queue.push(value);
                }
                Stmt::CreateClosure { .. } => {}
                Stmt::Call { callee, args, .. } => {
                    queue.push(callee.as_ref());
                    queue.extend(args.iter());
                }
                Stmt::Throw(arg) => {
                    queue.push(arg);
                }
                Stmt::GetCurrentException => {}
                Stmt::Try {
                    main_block,
                    handler_block,
                    finalizer_block,
                } => {
                    queue.push(main_block.as_ref());
                    queue.push(handler_block.as_ref());
                    queue.push(finalizer_block.as_ref());
                }
                Stmt::Debugger => {}
            }
        }
    }

    fn visit_stmts_mut(root: &mut Stmt, mut handler: impl FnMut(&mut Stmt)) {
        let mut queue = vec![root];

        while let Some(stmt) = queue.pop() {
            handler(stmt);
            match stmt {
                Stmt::Block(block) => {
                    queue.extend(block.stmts.iter_mut());
                }
                Stmt::Break(_) => {}
                Stmt::Continue(_) => {}
                Stmt::Unshare(_) => {}
                Stmt::If { test, cons, alt } => {
                    queue.push(test);
                    queue.push(cons);
                    queue.push(alt);
                }
                Stmt::Undefined => {}
                Stmt::Null => {}
                Stmt::This => {}
                Stmt::Read(_) => {}
                Stmt::AssignParam(_, _) => {}
                Stmt::Assign(_, value_stmt) => {
                    queue.push(value_stmt);
                }
                Stmt::Unary(_, arg) => {
                    queue.push(arg);
                }
                Stmt::Binary(_, left, right) => {
                    queue.push(left);
                    queue.push(right);
                }
                Stmt::StringLiteral(_) => {}
                Stmt::NumberLiteral(_) => {}
                Stmt::BoolLiteral(_) => {}
                Stmt::ArrayCreate => {}
                Stmt::ArrayPush(_, value_stmt) => {
                    queue.push(value_stmt);
                }
                Stmt::ObjectCreate => {}
                Stmt::ObjectGet { obj, key } => {
                    queue.push(obj);
                    queue.push(key);
                }
                Stmt::ObjectSet { obj, key, value } => {
                    queue.push(obj);
                    queue.push(key);
                    queue.push(value);
                }
                Stmt::CreateClosure { .. } => {}
                Stmt::Call { callee, args, .. } => {
                    queue.push(callee.as_mut());
                    queue.extend(args.iter_mut());
                }
                Stmt::Throw(arg) => {
                    queue.push(arg);
                }
                Stmt::GetCurrentException => {}
                Stmt::Try {
                    main_block,
                    handler_block,
                    finalizer_block,
                } => {
                    queue.push(main_block.as_mut());
                    queue.push(handler_block.as_mut());
                    queue.push(finalizer_block.as_mut());
                }
                Stmt::Debugger => {}
            }
        }
    }

    fn find_unbound_references(root: &Block) -> Vec<JsWord> {
        let mut declared = HashSet::new();
        let mut referenced = HashSet::new();

        let mut add_block_decls = |block: &Block| {
            for decl in &block.decls {
                if let DeclName::Js(js_name) = &decl.name {
                    declared.insert(js_name.clone());
                }
            }
        };

        add_block_decls(root);
        for block_stmt in &root.stmts {
            visit_stmts(block_stmt, |stmt| match stmt {
                Stmt::Block(block) => {
                    add_block_decls(block);
                }
                Stmt::Read(DeclName::Js(js_name)) => {
                    referenced.insert(js_name.clone());
                }
                Stmt::CreateClosure { func } => {
                    for name in &func.unbound_names {
                        referenced.insert(name.clone());
                    }
                }
                _ => (),
            });
        }

        #[allow(unused_mut)]
        let mut unbound: Vec<_> = referenced.difference(&declared).cloned().collect();

        // Helps with insta tests
        #[cfg(test)]
        unbound.sort();

        unbound
    }

    #[cfg(test)]
    mod tests {
        use std::rc::Rc;

        use swc_common::SourceMap;

        #[test]
        fn test_let_and_var() {
            let function = quick_compile(
                r#"(function() {
                    let x = 5;
                    var x = 99;
                })()
                "#
                .to_string(),
            );
            println!("{:#?}", function);
        }

        #[test]
        fn test_simple_for() {
            let function = quick_compile(
                r#"
                let closures = []
                for (let i=0; closures.push(() => i), i < 5; ++i) {}
                "#
                .to_string(),
            );
            println!("{:#?}", function);
        }

        #[test]
        fn test_global_var() {
            let function = quick_compile(
                "
                (function() {
                (function() {
                (function() {
                    let someLocalVar = 23;
                    console.log(someGlobalVar, someLocalVar)
                })()
                })()
                })()
            "
                .to_string(),
            );
            insta::assert_debug_snapshot!(function);
        }

        fn quick_compile(src: String) -> super::Function {
            let source_map = Rc::new(SourceMap::default());
            let swc_ast = super::super::parse_file("<input>".to_string(), src, source_map)
                .expect("parse error");
            let function = super::compile_script(swc_ast);
            function
        }
    }
}

mod past_compiler {
    use std::collections::{HashMap, HashSet};

    use swc_atoms::JsWord;

    use super::past;

    use crate::bytecode::{self, Instr, Literal};
    use crate::common::{Context, Error, Result};
    use crate::error;

    pub fn compile_function<'a>(
        module_builder: &'a mut ModuleBuilder,
        globals: &'a HashSet<JsWord>,
        captures: Vec<past::DeclName>,
        func: &past::Function,
    ) -> Result<bytecode::LocalFnId> {
        let mut fnb = FnBuilder::new(module_builder, globals);

        let mut names = HashMap::new();
        for (cap_ndx, cap_decl) in captures.into_iter().enumerate() {
            let cap_ndx = cap_ndx.try_into().unwrap();
            let reg = fnb.regs.gen();
            fnb.instrs
                .push(Instr::LoadCapture(reg, bytecode::CaptureIndex(cap_ndx)));
            names.insert(cap_decl, reg);
        }
        fnb.scopes.push(Scope {
            block_id: None,
            names,
        });

        compile_block(&mut fnb, &func.body)?;

        fnb.scopes.pop();
        assert_eq!(fnb.scopes.len(), 0);

        let function = fnb.build();
        let lfnid = module_builder.put_fn(function);

        Ok(lfnid)
    }

    fn compile_stmt(fnb: &mut FnBuilder, stmt: &past::Stmt) -> Result<Option<bytecode::VReg>> {
        match stmt {
            past::Stmt::Block(block) => compile_block(fnb, block),
            past::Stmt::Break(_) => todo!(),
            past::Stmt::Continue(_) => todo!(),
            past::Stmt::Unshare(block_id) => {
                let mut block_found = false;

                for scope in fnb.scopes.iter().rev() {
                    for reg in scope.names.values() {
                        fnb.instrs.push(Instr::Unshare(*reg));
                    }
                    if scope.block_id == Some(*block_id) {
                        block_found = true;
                    }
                }

                if !block_found {
                    panic!(
                        "malformed PAST: Unshare({:?}): block ID not found",
                        block_id
                    );
                }
                Ok(None)
            }
            past::Stmt::If { test, cons, alt } => {
                let test =
                    compile_stmt(fnb, test)?.expect("malformed PAST: if's test has no value");
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
            past::Stmt::Undefined => {
                let reg = fnb.regs.gen();
                fnb.instrs.push(Instr::LoadUndefined(reg));
                Ok(Some(reg))
            }
            past::Stmt::Null => {
                let reg = fnb.regs.gen();
                fnb.instrs.push(Instr::LoadNull(reg));
                Ok(Some(reg))
            }
            past::Stmt::This => {
                let reg = fnb.regs.gen();
                fnb.instrs.push(Instr::LoadThis(reg));
                Ok(Some(reg))
            }
            past::Stmt::Read(name) => {
                let reg = compile_read(fnb, name);
                Ok(Some(reg))
            }
            past::Stmt::AssignParam(dest, arg_ndx) => {
                let value = fnb.regs.gen();
                fnb.instrs
                    .push(Instr::LoadArg(value, bytecode::ArgIndex(*arg_ndx)));
                compile_write(fnb, dest, value);
                Ok(None)
            }
            past::Stmt::Assign(dest, value) => {
                let value = compile_expr(fnb, value)?;
                compile_write(fnb, dest, value);
                Ok(None)
            }
            past::Stmt::Unary(op, arg) => {
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
                    swc_ecma_ast::UnaryOp::Tilde
                    | swc_ecma_ast::UnaryOp::TypeOf
                    | swc_ecma_ast::UnaryOp::Void
                    | swc_ecma_ast::UnaryOp::Delete => panic!("unsupported unary op: {:?}", op),
                }
                Ok(Some(dest))
            }
            past::Stmt::Binary(op, left, right) => {
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
            past::Stmt::StringLiteral(str_lit) => Ok(Some(compile_load_const(
                fnb,
                bytecode::Literal::String(str_lit.to_string()),
            ))),
            past::Stmt::NumberLiteral(num_lit) => Ok(Some(compile_load_const(
                fnb,
                bytecode::Literal::Number(*num_lit),
            ))),
            past::Stmt::BoolLiteral(bool_lit) => Ok(Some(compile_load_const(
                fnb,
                bytecode::Literal::Bool(*bool_lit),
            ))),
            past::Stmt::ArrayCreate => {
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
            past::Stmt::ArrayPush(arr, value) => {
                let arr = compile_read(fnb, arr);
                let value = compile_expr(fnb, value)?;
                fnb.instrs.push(Instr::ArrayPush { arr, value });
                Ok(None)
            }
            past::Stmt::ObjectCreate => {
                let obj = fnb.regs.gen();
                fnb.instrs.push(Instr::ObjCreateEmpty(obj));
                Ok(Some(obj))
            }
            past::Stmt::ObjectGet { obj, key } => {
                let obj = compile_expr(fnb, obj.as_ref())?;
                let key = compile_expr(fnb, key.as_ref())?;
                let dest = fnb.regs.gen();
                fnb.instrs.push(Instr::ObjGet { dest, obj, key });
                Ok(Some(dest))
            }
            past::Stmt::ObjectSet { obj, key, value } => {
                let obj = compile_expr(fnb, obj.as_ref())?;
                let key = compile_expr(fnb, key.as_ref())?;
                let value = compile_expr(fnb, value.as_ref())?;
                fnb.instrs.push(Instr::ObjSet { obj, key, value });
                Ok(None)
            }
            past::Stmt::CreateClosure { func } => {
                let mut cap_names = Vec::new();
                let mut cap_regs = Vec::new();

                for name in func.unbound_names.iter() {
                    let name = past::DeclName::Js(name.clone());
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
            past::Stmt::Call {
                is_new,
                callee,
                args,
            } => {
                // Some things expressed in the `f(...)` call syntax are not actually calls to
                // anything, but have a special meaning
                match callee.as_ref() {
                    past::Stmt::Read(past::DeclName::Js(name)) if name == "sink" => {
                        for arg in args {
                            let var = compile_expr(fnb, &arg)?;
                            fnb.instrs.push(Instr::PushToSink(var));
                        }

                        let ret = fnb.regs.gen();
                        compile_load_const(fnb, Literal::Undefined);
                        Ok(Some(ret))
                    }
                    past::Stmt::Read(past::DeclName::Js(name)) if name == "require" => {
                        if args.len() != 1 {
                            return Err(error!("`require` takes a single argument only"));
                        }
                        let import_path = compile_expr(fnb, &args[0])?;

                        let ret = fnb.regs.gen();
                        fnb.instrs.push(Instr::ImportModule(ret, import_path));
                        Ok(Some(ret))
                    }
                    past::Stmt::Read(past::DeclName::Js(name)) if name == "eval" => {
                        return Err(error!("`eval` not supported"));
                    }
                    _ => {
                        let mut arg_regs = Vec::new();
                        for arg in args {
                            let reg = compile_expr(fnb, arg)?;
                            arg_regs.push(reg);
                        }

                        let (this, callee) = match callee.as_ref() {
                            past::Stmt::ObjectGet { obj, key } => {
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
            past::Stmt::Throw(arg) => {
                let arg = compile_expr(fnb, arg)?;
                fnb.instrs.push(Instr::Throw(arg));
                Ok(None)
            }
            past::Stmt::GetCurrentException => {
                let dest = fnb.regs.gen();
                fnb.instrs.push(Instr::GetCurrentException(dest));
                Ok(Some(dest))
            }
            past::Stmt::Try {
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
            past::Stmt::Debugger => {
                fnb.instrs.push(Instr::Breakpoint);
                Ok(None)
            }
        }
    }

    fn compile_read(fnb: &mut FnBuilder<'_>, name: &past::DeclName) -> bytecode::VReg {
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

    fn compile_write(fnb: &mut FnBuilder, name: &past::DeclName, value: bytecode::VReg) {
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

    fn resolve_name(fnb: &mut FnBuilder, name: &past::DeclName) -> Loc {
        let local_reg = fnb
            .scopes
            .iter()
            .rev()
            .find_map(|scope| scope.names.get(name).copied().map(Loc::Reg));

        if let Some(reg) = local_reg {
            return reg;
        }

        if let past::DeclName::Js(name) = name {
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

    fn compile_expr(fnb: &mut FnBuilder, value: &past::Stmt) -> Result<bytecode::VReg> {
        let vreg = compile_stmt(fnb, value)?.expect("malformed PAST: expression has no value");
        Ok(vreg)
    }

    fn compile_block(
        fnb: &mut FnBuilder,
        block: &past::Block,
    ) -> std::prelude::v1::Result<Option<bytecode::VReg>, Error> {
        let names = block
            .decls
            .iter()
            .map(|decl| (decl.name.clone(), fnb.regs.gen()))
            .collect();
        fnb.scopes.push(Scope {
            block_id: Some(block.id),
            names,
        });

        let mut last_reg = None;
        for stmt in &block.stmts {
            last_reg = compile_stmt(fnb, stmt)?;
        }

        fnb.scopes.pop().unwrap();
        Ok(last_reg)
    }

    pub struct ModuleBuilder {
        fns: HashMap<bytecode::LocalFnId, bytecode::Function>,
        next_lfnid: u16,
    }
    impl ModuleBuilder {
        pub fn new(min_fnid: u16) -> Self {
            ModuleBuilder {
                fns: HashMap::new(),
                next_lfnid: min_fnid,
            }
        }

        fn put_fn(&mut self, function: bytecode::Function) -> bytecode::LocalFnId {
            let lfnid = bytecode::LocalFnId(self.next_lfnid);
            self.fns.insert(lfnid, function);
            self.next_lfnid += 1;
            lfnid
        }

        pub fn build(self) -> HashMap<bytecode::LocalFnId, bytecode::Function> {
            self.fns
        }
    }

    struct FnBuilder<'a> {
        instrs: InstrBuffer,
        consts: ConstsBuffer,
        regs: RegGen,
        is_strict_mode: bool,
        scopes: Vec<Scope>,
        module_builder: &'a mut ModuleBuilder,
        globals: &'a HashSet<JsWord>,
    }

    struct Scope {
        block_id: Option<past::BlockID>,
        names: HashMap<past::DeclName, bytecode::VReg>,
    }

    enum Loc {
        Reg(bytecode::VReg),
        Global(JsWord),
    }

    impl<'a> FnBuilder<'a> {
        fn new(
            module_builder: &'a mut ModuleBuilder,
            globals: &'a HashSet<JsWord>,
        ) -> FnBuilder<'a> {
            FnBuilder {
                instrs: InstrBuffer::new(),
                consts: ConstsBuffer::new(),
                regs: RegGen::new(),
                is_strict_mode: false,
                scopes: Vec::new(),
                module_builder,
                globals,
            }
        }

        fn build(self) -> bytecode::Function {
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
                span: swc_common::Span::default(),
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

        use crate::bytecode;

        #[test]
        fn test_global_var() {
            let (functions, root_lfnid) = quick_compile(
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
                    if lfnid == &root_lfnid { "[root]" } else { "" }
                );

                func.dump();
            }
        }

        fn quick_compile(
            src: String,
        ) -> (
            HashMap<bytecode::LocalFnId, bytecode::Function>,
            bytecode::LocalFnId,
        ) {
            let source_map = Rc::new(SourceMap::default());
            let swc_ast = super::super::parse_file("<input>".to_string(), src, source_map)
                .expect("parse error");
            let past_function = super::past::compile_script(swc_ast);

            let mut module_builder = super::ModuleBuilder::new(0);
            let globals = past_function.unbound_names.iter().cloned().collect();
            let root_lfnid =
                super::compile_function(&mut module_builder, &globals, Vec::new(), &past_function)
                    .expect("past->bytecode compile error");

            let funcs = module_builder.build();
            (funcs, root_lfnid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_bytecode_object_init() {
        let compiled = compile_file(
            "<input>".to_string(),
            "sink({
                aString: 'asdlol123',
                aNumber: 1239423.4518923,
                anotherObject: { x: 123, y: 899 },
                aFunction: function(pt) { return 42; }
            })"
            .to_string(),
            Rc::new(Default::default()),
            CompileFlags {
                min_fnid: 1,
                source_type: SourceType::Script,
            },
        )
        .unwrap();

        let root_fnid = compiled.root_fnid;
        let function = &compiled.functions.get(&root_fnid).unwrap();

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
        let compiled = compile_file("<input>".to_string(), CODE_UPVALUES.to_string()).unwrap();

        let root_fn = compiled.functions.get(&compiled.root_fnid).unwrap();
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
            Instr::LoadConst(_, Literal::Number(val)) => val,
            _ => panic!(),
        };
        assert_eq!(*cap_const, 10.0);
    }

    #[test]
    #[cfg(x)]
    fn test_upvalues_callee() {
        let compiled = compile_file("<input>".to_string(), CODE_UPVALUES).unwrap();
        let callee = compiled.functions.get(&LocalFnId(1)).unwrap();
        assert!(std::matches!(callee.instrs()[0], Instr::LoadCapture(0)));
    }

    #[test]
    #[cfg(x)]
    fn test_upvalues_compact_addcapinstr() {
        let compiled = compile_file(
            "<input>".to_string(),
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

        let mut fnid = LocalFnId(0);
        while let Some(func) = compiled.functions.get(&fnid) {
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
        let res = compile_file(
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
        let res = compile_file(
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
        let res = compile_file(
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
