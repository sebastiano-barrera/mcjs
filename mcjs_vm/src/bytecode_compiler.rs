use std::collections::{HashMap, HashSet};

use swc_atoms::JsWord;
use swc_common::{sync::Lrc, Span, Spanned};
use swc_ecma_ast::{
    ArrowExpr, AssignOp, BinaryOp, Decl, ExportDecl, FnDecl, ForHead, Function, Lit, Pat, Stmt,
    UpdateOp, VarDecl, VarDeclKind,
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
    fn new(flags: CompileFlags) -> Self {
        let next_fnid = flags.min_fnid;
        assert!(next_fnid >= 1);
        Builder {
            fns: HashMap::new(),
            fn_stack: Vec::new(),
            next_fnid,
            breakable_ranges: Vec::new(),
            brange_stack: Vec::new(),
            flags,
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

    let compiled_mod = compile_module(&ast_module, flags).with_context(
        error!("while compiling module: {filename}").with_source_map(Lrc::clone(&source_map)),
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

impl FnBuilder {
    fn new(id: LocalFnId) -> Self {
        FnBuilder {
            fnid: id,
            scopes: vec![Scope::new()],
            instrs: Vec::new(),
            consts: Vec::new(),
            trace_anchors: HashMap::new(),
            captures: Vec::new(),
            // See doc comment for bytecode::ARGS_COUNT_MAX
            next_vreg: bytecode::ARGS_COUNT_MAX,
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

        let outermost_scope = self.scopes.first_mut().unwrap();
        let prev = outermost_scope.vars.insert(name.clone(), reg);
        assert!(prev.is_none());

        self.ident_history.push(IdentAsmt {
            iid: self.peek_iid(),
            reg,
            ident: name.clone(),
        });

        reg
    }

    fn get_vreg(&self, name: &JsWord) -> Option<VReg> {
        self.scopes
            .iter()
            .rev() // innermost -> outwards
            .find_map(|scope| scope.vars.get(name))
            .cloned()
    }

    fn define_var(&mut self, name: JsWord, reg: VReg) {
        let scope = self.inner_scope_mut();
        let prev = scope.vars.insert(name.clone(), reg);
        if prev.is_some() {
            // definition of var $name shadows previous definition
        }

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

impl Builder {
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
        if sym == "globalThis" {
            let reg = self.new_vreg();
            self.emit(Instr::GetGlobalThis(reg));
            self.cur_fnb().define_var("globalThis".into(), reg);
            return Some(reg);
        }

        let found = self
            .fn_stack
            .iter()
            .rev()
            .enumerate()
            .find_map(|(ndx, fnb)| fnb.get_vreg(sym).map(|vreg| (ndx, vreg)));

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

        let global_this = self.get_vreg(&"globalThis".into()).unwrap();
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

        let global_this = self.get_vreg(&"globalThis".into()).unwrap();
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
    fn start_function(&mut self, name: Option<JsWord>, params: &[swc_ecma_ast::Param]) {
        let fnid = LocalFnId(self.next_fnid);
        self.next_fnid += 1;

        let mut fnb = FnBuilder::new(fnid);
        fnb.is_strict_mode = self
            .fn_stack
            .last()
            .map(|fnb| fnb.is_strict_mode)
            .unwrap_or(false);
        self.fn_stack.push(fnb);

        for (param_ndx, param) in params.iter().enumerate() {
            let param_ndx = TryFrom::try_from(param_ndx).expect("too many parameters!");

            if !param.decorators.is_empty() {
                panic!("unsupported: decorators on function parameters");
            }

            match &param.pat {
                Pat::Ident(ident) => {
                    let reg = self.new_vreg();
                    self.emit(Instr::LoadArg(reg, bytecode::ArgIndex(param_ndx)));
                    self.define_var(ident.sym.clone(), reg);
                }
                other => unsupported_node!(other),
            }
        }
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
    flags: CompileFlags,
) -> Result<CompiledModule> {
    use swc_ecma_ast::{ExportDecl, Expr, ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    let mut builder = Builder::new(flags);
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

    if let Some(Stmt::Expr(expr_stmt)) = ast_module.body.first().and_then(|i| i.as_stmt()) {
        if let Some(Lit::Str(ref lit_str)) = expr_stmt.expr.as_lit() {
            if lit_str.value.as_bytes() == b"use strict" {
                builder.cur_fnb().enable_strict_mode();
            }
        }
    }

    {
        // We have to handle hoisting here like we do for function body blocks.  See [#hoisting].
        // TODO any smarter way to allocate this memory? Static array somewhere?
        let mut decls = Vec::new();
        for item in &ast_module.body {
            match item {
                ModuleItem::Stmt(stmt) => {
                    find_hdecls_recursive(stmt, &mut decls);
                }
                ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(ExportDecl { decl, .. })) => {
                    match decl {
                        Decl::Fn(d) => decls.push(HoistedDeclRef::Fn(d)),
                        Decl::Var(d) => decls.push(HoistedDeclRef::Var(d)),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        compile_func_scope_namedefs(&mut builder, &decls)?;
    }

    for item in &ast_module.body {
        match item {
            ModuleItem::Stmt(Stmt::Decl(decl)) => {
                compile_decl_block_scope_part(&mut builder, decl)?;
            }
            ModuleItem::ModuleDecl(ModuleDecl::ExportDecl(export_decl)) => {
                compile_decl_block_scope_part(&mut builder, &export_decl.decl)?;
            }
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

    Ok(CompiledModule {
        root_fnid,
        functions,
        breakable_ranges: builder.breakable_ranges,
    })
}

fn compile_stmt(builder: &mut Builder, stmt: &swc_ecma_ast::Stmt) -> Result<()> {
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

            compile_stmt(&mut builder, &while_stmt.body)?;
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
            compile_stmt(&mut builder, &stmt.body)?;
            let cond = compile_expr(&mut builder, &stmt.test)?;
            builder.emit(Instr::JmpIf {
                cond,
                dest: loop_start,
            });
            Ok(())
        }

        Stmt::For(stmt) => {
            builder.cur_fnb().push_scope();

            if let Some(init) = &stmt.init {
                use swc_ecma_ast::VarDeclOrExpr;
                match init {
                    VarDeclOrExpr::VarDecl(var_decl) => {
                        compile_var_decl_block_scope_part(&mut builder, var_decl);
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

            compile_stmt(&mut builder, &stmt.body)?;

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

            compile_stmt(&mut builder, &forin_stmt.body)?;

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
            compile_stmt(&mut builder, &forof_stmt.body)?;

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

            compile_stmt(&mut builder, &if_stmt.cons)?;
            let jmp_to_end = builder.reserve();

            let alt = builder.peek_iid();
            if let Some(else_blk) = &if_stmt.alt {
                compile_stmt(&mut builder, else_blk)?;
            }

            let end: IID = builder.peek_iid();

            *builder.get_mut(jmp_to_alt).unwrap() = Instr::JmpIf { cond, dest: alt };
            *builder.get_mut(jmp_to_end).unwrap() = Instr::JmpIf { cond, dest: end };
            Ok(())
        }
        other => unsupported_node!(other),
    }
}

fn compile_block(builder: &mut Builder, stmts: &[Stmt]) -> Result<()> {
    builder.cur_fnb().push_scope();

    for stmt in stmts {
        if let Stmt::Decl(decl) = stmt {
            compile_decl_block_scope_part(builder, decl)?;
        }
    }

    for stmt in stmts {
        compile_stmt(builder, stmt)?;
    }

    builder.cur_fnb().pop_scope();
    Ok(())
}

fn compile_decl_block_scope_part(builder: &mut Builder, decl: &Decl) -> Result<()> {
    match decl {
        Decl::Var(vd) => compile_var_decl_block_scope_part(builder, vd),
        Decl::Fn(fd) => {
            compile_fn_decl_assignment(builder, fd)?;
        }
        _ => {}
    }

    Ok(())
}

fn compile_var_decl_block_scope_part(builder: &mut Builder, vd: &VarDecl) {
    match vd.kind {
        VarDeclKind::Var => {}
        VarDeclKind::Let | VarDeclKind::Const => {
            for decl in &vd.decls {
                let name = get_var_decl_name(decl);
                compile_namedef(builder, name);
            }
        }
    }
}

// Helper type.  Helps unify the cases where the decl comes from a `Decl` or a `VarDeclOrExpr`
enum HoistedDeclRef<'a> {
    Var(&'a VarDecl),
    Fn(&'a FnDecl),
}

fn find_hdecls_recursive<'a>(stmt: &'a Stmt, out: &mut Vec<HoistedDeclRef<'a>>) {
    match stmt {
        Stmt::Decl(Decl::Fn(fn_decl)) => {
            out.push(HoistedDeclRef::Fn(fn_decl));
        }
        Stmt::Decl(Decl::Var(var_decl)) => {
            out.push(HoistedDeclRef::Var(var_decl));
        }
        Stmt::Block(block_stmt) => {
            for stmt in &block_stmt.stmts {
                find_hdecls_recursive(stmt, out);
            }
        }
        Stmt::If(if_stmt) => {
            find_hdecls_recursive(&if_stmt.cons, out);
            if let Some(alt) = &if_stmt.alt {
                find_hdecls_recursive(&*alt, out);
            }
        }
        Stmt::While(while_stmt) => {
            find_hdecls_recursive(&while_stmt.body, out);
        }
        Stmt::For(for_stmt) => {
            use swc_ecma_ast::VarDeclOrExpr;
            if let Some(VarDeclOrExpr::VarDecl(var_decl)) = &for_stmt.init {
                out.push(HoistedDeclRef::Var(&*var_decl));
            }
            find_hdecls_recursive(&for_stmt.body, out);
        }
        Stmt::Try(try_stmt) => {
            for stmt in &try_stmt.block.stmts {
                find_hdecls_recursive(stmt, out);
            }
            if let Some(catch_block) = &try_stmt.handler {
                for stmt in &catch_block.body.stmts {
                    find_hdecls_recursive(stmt, out);
                }
            }
            if let Some(finally_block) = &try_stmt.finalizer {
                for stmt in &finally_block.stmts {
                    find_hdecls_recursive(stmt, out);
                }
            }
        }
        Stmt::ForIn(for_in_stmt) => {
            if let swc_ecma_ast::ForHead::VarDecl(var_decl) = &for_in_stmt.left {
                out.push(HoistedDeclRef::Var(&*var_decl));
            }
        }
        Stmt::ForOf(for_or_stmt) => {
            if let swc_ecma_ast::ForHead::VarDecl(var_decl) = &for_or_stmt.left {
                out.push(HoistedDeclRef::Var(&*var_decl));
            }
        }
        _ => {}
    }
}

/// Do the 'namedef' phase of function-scoped declarations (`var` and `function`).
///
/// In compliance with the ES spec, `var` declarations are processed first (they will *not* shadow
/// previous bindings, such function parameters) and `function` declarations are processed later
/// (they *will* shadow previous declarations). This will happen regardless of the order in
/// `decls`.
fn compile_func_scope_namedefs(builder: &mut Builder, decls: &[HoistedDeclRef]) -> Result<()> {
    // `var` variables declarations can be thought of as a name binding + an
    // assignment.
    //   - the name binding behaves as if it was moved at the beginning of the block.  the
    //   initial value is `undefined`
    //   - the assignment is left at the original position. the program will have to reach that
    //   spot before it takes effect.
    //
    //   So, these 2 block behave identically:
    //
    //   ```
    //   {
    //      $print(x);   // prints 'undefined'
    //      var x = 22;
    //      $print(x);   // prints '22'
    //   }
    //
    //   {
    //      var x;
    //      $print(x);   // prints 'undefined'
    //      x = 22;
    //      $print(x);   // prints '22'
    //   }
    //   ```
    //
    for decl in decls {
        if let HoistedDeclRef::Var(var_decl) = decl {
            match var_decl.kind {
                VarDeclKind::Var => {
                    for decl in &var_decl.decls {
                        let name = get_var_decl_name(decl);
                        if let Var::Reg(_) = builder.get_var(&name) {
                            // For `var` only:
                            // We must *not* re-bind the same name to a different register.  The name will remain
                            // bound to the same location as before. (It's almost surely a function argument.)
                        } else {
                            compile_namedef(builder, name);
                        }
                    }
                }
                VarDeclKind::Let | VarDeclKind::Const => {
                    // This name will be defined later, at the top of the
                    // innermost *block* where the definition appears, not
                    // necessarily at the top of the function level block.
                }
            }
        }
    }

    // Multiple `function` declarations can mask each other, and mask var declarations.  The last
    // one in the source code wins.
    for decl in decls {
        if let HoistedDeclRef::Fn(fn_decl) = decl {
            let name = get_fn_decl_name(fn_decl);
            compile_namedef(builder, name);
        }
    }

    Ok(())
}

fn compile_var_decl_namedef(builder: &mut Builder, var_decl: &swc_ecma_ast::VarDecl) {
    use swc_ecma_ast::VarDeclKind;

    for decl in &var_decl.decls {
        let name = get_var_decl_name(decl);
        if let Var::Reg(_) = builder.get_var(&name) {
            // We must *not* re-bind the same name to a different register.  The name will remain
            // bound to the same location as before. (It's almost surely a function argument.)
        } else {
            compile_namedef(builder, name);
        }
    }
}

fn compile_namedef(builder: &mut Builder, name: JsWord) {
    let is_script_global = builder.fn_stack.len() == 1
        && builder.cur_fnb().scopes.len() == 1
        && builder.flags.source_type == SourceType::Script;
    if is_script_global {
        // Script global => don't assign any register. Subsequent calls to `get_var` will
        // return Var::Global(_). `read_var` and `write_var` will generate object member
        // reads and writes.
    } else {
        // Not a script global => 'lock' this name to a specific register. Subsequent calls
        // to `get_var` will return Var::Reg(_), and direct writes will be generated by
        // `write_var`.
        //
        // Yet-unassigned vregs are implicitly initialized to Undefined, so no
        // explicit instruction is required here
        let reg = builder.new_vreg();
        builder.define_var(name, reg);
    }
}

fn compile_var_decl_assignment(
    builder: &mut Builder,
    var_decl: &swc_ecma_ast::VarDecl,
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

fn compile_fn_decl_assignment(builder: &mut Builder, fn_decl: &swc_ecma_ast::FnDecl) -> Result<()> {
    if fn_decl.declare {
        panic!("unsupported case: fn_decl.declare");
    }
    let name = get_fn_decl_name(fn_decl);
    let var = builder.get_var(&name);
    let value = compile_function(builder, Some(name.clone()), &fn_decl.function)?;
    builder.write_var(&var, value);
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

    for stmt in stmts {
        // only look at the topmost scope
        if let swc_ecma_ast::Stmt::Expr(expr_stmt) = stmt {
            if let Some(Lit::Str(ref lit_str)) = expr_stmt.expr.as_lit() {
                if lit_str.value.as_bytes() == b"use strict" {
                    builder.cur_fnb().enable_strict_mode();
                }
            }
        }
    }

    let mut decls = Vec::new();
    for stmt in stmts {
        find_hdecls_recursive(stmt, &mut decls);
    }
    compile_func_scope_namedefs(builder, &decls)?;

    compile_block(builder, stmts)?;

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
fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Result<VReg> {
    use swc_ecma_ast::{CallExpr, Expr};

    let mut builder = builder.with_span(&expr.span());

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
                        let var = compile_expr(&mut builder, &arg.expr)?;
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
                } else if sym == "require" {
                    if args.len() != 1 {
                        return Err(error!("`require` takes a single argument only"));
                    }

                    let import_path = compile_expr(&mut builder, &args[0].expr)?;
                    let ret = builder.new_vreg();
                    builder.emit(Instr::ImportModule(ret, import_path));
                    return Ok(ret);
                }
            }

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
            let func = compile_function(&mut builder, name.clone(), func)?;

            if let Some(name) = name {
                builder.define_var(name, func);
            }

            Ok(func)
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
            let mut result = compile_expr(&mut builder, &*head)?;
            for expr in tail {
                result = compile_expr(&mut builder, &*expr)?;
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

fn compile_assignment(asmt: &swc_ecma_ast::AssignExpr, builder: &mut Builder) -> Result<VReg> {
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
    use swc_common::{
        errors::{emitter::EmitterWriter, Handler},
        sync::Lrc,
        FileName, SourceMap,
    };
    use swc_ecma_ast::EsVersion;
    use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

    let err_handler = Handler::with_emitter(
        true,  // can_emit_warnings
        false, // treat_err_as_bug
        Box::new(EmitterWriter::new(
            Box::new(std::io::stderr()),
            Some(source_map.clone()),
            false, // short_message
            true,  // teach
        )),
    );

    let path = std::path::PathBuf::from(filename);
    let source_file = source_map.new_source_file(FileName::Real(path), content);

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

    parser.parse_module().map_err(|e| {
        e.into_diagnostic(&err_handler).emit();
        error!("parse error")
    })
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
