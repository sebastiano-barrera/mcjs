/// The Processed AST.
///
/// An intermediate representation that results from an initial processing of
/// swc_ecma_ast's AST, designed expressly to facilitate compilation to bytecode
/// and compile-time checks.
use std::collections::HashSet;
use swc_atoms::JsWord;

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
}

pub struct Function {
    pub parameters: Vec<JsWord>,
    pub unbound_names: Vec<JsWord>,
    pub body: Block,
    pub span: swc_common::Span,
    pub declares_use_strict: bool,
}
impl std::fmt::Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lo = self.span.lo.0;
        let hi = self.span.hi.0;
        write!(f, "[{}-{}] func (", lo, hi)?;
        write_comma_sep(f, self.parameters.iter())?;
        write!(f, ") unbound[")?;
        write_comma_sep(f, self.unbound_names.iter())?;
        write!(f, "] {{")?;
        self.body.fmt(f)?;
        write!(f, "}}")
    }
}

#[derive(Debug)]
pub struct Block {
    pub id: BlockID,
    pub decls: Vec<Decl>,
    pub stmts: Vec<Stmt>,
    pub exprs: Vec<Expr>,
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
impl DeclName {
    pub fn expect_js_word(&self) -> &JsWord {
        match self {
            DeclName::Js(js_word) => js_word,
            DeclName::Tmp(_) => panic!("assertion failed: DeclName expected to be JS word"),
        }
    }
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

pub struct Stmt {
    pub span: swc_common::Span,
    pub op: StmtOp,
}

impl std::fmt::Debug for Stmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}-{}] ", self.span.lo.0, self.span.hi.0)?;
        self.op.fmt(f)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StmtID(u16);

#[derive(Clone, Copy, Debug)]
pub struct StmtsDistance(i16);

impl std::ops::Sub for StmtID {
    type Output = StmtsDistance;

    fn sub(self, rhs: Self) -> Self::Output {
        let a: i16 = self.0.try_into().unwrap();
        let b: i16 = rhs.0.try_into().unwrap();
        StmtsDistance(a - b)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExprID(u16);

#[derive(Debug)]
pub enum StmtOp {
    Invalid,
    Break(BlockID),
    Continue(BlockID),
    Unshare(DeclName),
    IfNot {
        test: ExprID,
        distance: StmtsDistance,
    },
    Jump(StmtsDistance),

    Evaluate(ExprID),

    Assign(DeclName, ExprID),

    ArrayPush(DeclName, ExprID),

    ObjectSet {
        obj: ExprID,
        key: ExprID,
        value: ExprID,
    },

    Return(ExprID),

    Throw(ExprID),
    TryBegin {
        dist_to_handler: StmtID,
    },
    TryEnd,

    Debugger,
}

#[derive(Debug)]
enum Expr {
    Undefined,
    Null,
    This,
    Read(DeclName),
    ReadArg(u8),

    Unary(swc_ecma_ast::UnaryOp, ExprID),
    Binary(swc_ecma_ast::BinaryOp, ExprID, ExprID),

    StringLiteral(JsWord),
    NumberLiteral(f64),
    BoolLiteral(bool),

    ArrayCreate,
    ArrayNth {
        arr: ExprID,
        index: ExprID,
    },
    ArrayLen(ExprID),

    ObjectCreate,
    ObjectGet {
        obj: ExprID,
        key: ExprID,
    },
    ObjectGetKeys(ExprID),

    CreateClosure {
        func: Box<Function>,
    },
    Call {
        callee: ExprID,
        args: Vec<ExprID>,
        is_new: bool,
    },

    CurrentException,
}

const ZERO: Expr = Expr::NumberLiteral(0.0);
const ONE: Expr = Expr::NumberLiteral(1.0);

mod builder {
    use super::{Block, Decl, DeclName, Expr, ExprID, StmtID, StmtOp};

    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BlockID(u32);

    impl std::fmt::Debug for BlockID {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "block{}", self.0)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TmpID(u32);
    impl TmpID {
        pub fn numeric(&self) -> u32 {
            self.0
        }
    }

    pub(super) struct Builder {
        next_id: u32,
    }

    pub(super) struct FnBuilder<'a> {
        builder: &'a mut Builder,

        /// The stack of currently open nested blocks.
        ///
        /// Compilation works by appending Stmt and Expr nodes into the last (innermost) block. As
        /// compilation progresses, the last block is merged into the previous/parent block, until
        /// we finish with a single block that constitutes the function's body.
        blocks: Vec<Block>,

        /// At any given time, the last element of this vector is the ID of the block that would
        /// be repeated/exited by a continue/break statement.
        breakable_blocks: Vec<BlockID>,
    }

    impl Builder {
        pub fn new() -> Self {
            Builder { next_id: 0 }
        }

        pub fn new_function(&mut self) -> FnBuilder {
            FnBuilder {
                builder: self,
                blocks: Vec::new(),
                breakable_blocks: Vec::new(),
            }
        }
    }

    fn merge_block(outer: &mut Block, mut inner: Block) {
        outer.decls.append(&mut inner.decls);
        outer.stmts.append(&mut inner.stmts);
        outer.exprs.append(&mut inner.exprs);
    }

    impl<'a> FnBuilder<'a> {
        pub(super) fn build(self) -> Block {
            assert!(self.breakable_blocks.is_empty());
            assert_eq!(self.blocks.len(), 1);
            self.blocks[0]
        }

        pub(super) fn suspend(&mut self) -> &mut Builder {
            self.builder
        }

        pub(super) fn break_target(&self) -> BlockID {
            self.breakable_blocks.last().copied().unwrap()
        }
        pub(super) fn push_break_target(&mut self) {
            self.breakable_blocks.push(self.cur_block_id());
        }

        pub(crate) fn cur_block_id(&self) -> BlockID {
            self.blocks.last().unwrap().id
        }
        fn cur_block_mut(&mut self) -> &mut Block {
            self.blocks.last_mut().unwrap()
        }

        pub(super) fn block<T>(&mut self, builder: impl FnOnce(&mut Self) -> T) -> T {
            let blkid = BlockID(self.builder.next_id);
            self.builder.next_id += 1;

            // TODO Reuse allocations and get new Block out of a pool
            self.blocks.push(Block {
                id: blkid,
                decls: Vec::new(),
                stmts: Vec::new(),
                exprs: Vec::new(),
            });

            let ret = builder(self);

            let inner_block = self.blocks.pop().unwrap();
            if let Some(breakable_blkid) = self.breakable_blocks.last().copied() {
                if breakable_blkid == inner_block.id {
                    self.breakable_blocks.pop();
                }
            }

            merge_block(self.cur_block_mut(), inner_block);

            // TODO Check redeclarations here

            ret
        }

        fn gen_tmp(&mut self) -> TmpID {
            let tmpid = TmpID(self.builder.next_id);
            self.builder.next_id += 1;
            tmpid
        }

        pub(super) fn add_decl(&mut self, decl: Decl) {
            self.cur_block_mut().decls.push(decl);
        }

        pub(super) fn decl_tmp(&mut self) -> DeclName {
            let name = DeclName::Tmp(self.gen_tmp());
            self.add_decl(Decl {
                name: name.clone(),
                is_lexical: true,
                is_global: false,
            });
            name
        }

        pub(super) fn peek_stmt_id(&mut self) -> StmtID {
            let block = self.blocks.last().unwrap();
            StmtID(block.stmts.len().try_into().unwrap())
        }

        pub(super) fn add_stmt(&mut self, op: StmtOp) -> StmtID {
            let block = self.cur_block_mut();
            let stmt_id_raw = block.stmts.len().try_into().unwrap();
            block.stmts.push(super::Stmt {
                // TODO Set the actual span here.
                // It should be provided implicitly with a separate API
                span: swc_common::Span::default(),
                op,
            });

            StmtID(stmt_id_raw)
        }

        pub(super) fn set_stmt(&mut self, id: StmtID, op: StmtOp) {
            let block = self.cur_block_mut();
            let stmt = block.stmts.get_mut(id.0 as usize).unwrap();
            *stmt = super::Stmt {
                // TODO Set the actual span here.
                // It should be provided implicitly with a separate API
                span: swc_common::Span::default(),
                op,
            };
        }

        pub(super) fn add_expr(&mut self, expr: Expr) -> ExprID {
            let block = self.cur_block_mut();
            let expr_id_raw = block.exprs.len().try_into().unwrap();
            block.exprs.push(expr);
            ExprID(expr_id_raw)
        }

        pub(super) fn add_unshares_up_to(&mut self, block_id: BlockID) {
            let mut unshared_names = Vec::new();

            let mut block_found = false;
            for block in self.blocks.iter().rev() {
                for decl in &block.decls {
                    unshared_names.push(decl.name.clone());
                }

                if block.id == block_id {
                    block_found = true;
                    break;
                }
            }

            assert!(
                block_found,
                "past compiler bug: no such block: {:?}",
                block_id
            );

            for name in unshared_names {
                self.add_stmt(StmtOp::Unshare(name));
            }
        }
    }
}

pub use builder::{BlockID, TmpID};
use builder::{Builder, FnBuilder};

use crate::{error, util::write_comma_sep};

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
    let mut fnb = builder.new_function();
    for stmt in &stmts {
        compile_stmt(&mut fnb, stmt);
    }
    let block = fnb.build();
    let mut function = compile_function_from_parts(ast_module.span, &[], block);

    // Toplevel var (non-lexical) definitions in a script are allocated in
    // the `globalThis` object. This can be achieved simply by removing those
    // declarations.
    let mut kept_decls = Vec::new();
    for decl in std::mem::replace(&mut function.body.decls, Vec::new()) {
        match decl.name {
            DeclName::Js(name) if !decl.is_lexical => {
                function.unbound_names.push(name);
            }
            _ => {
                kept_decls.push(decl);
            }
        }
    }
    function.body.decls = kept_decls;

    function
}

fn compile_stmt(fnb: &mut FnBuilder, stmt: &swc_ecma_ast::Stmt) {
    // TODO stop using `block`, use the FnBuilder API
    let block: &mut Block = todo!();
    match &stmt {
        swc_ecma_ast::Stmt::Block(block_stmt) => {
            fnb.block(|fnb| {
                for stmt in &block_stmt.stmts {
                    compile_stmt(fnb, stmt);
                }
            });
        }
        swc_ecma_ast::Stmt::Empty(_) => {}
        swc_ecma_ast::Stmt::Debugger(debugger_stmt) => {
            fnb.add_stmt(StmtOp::Debugger);
        }

        swc_ecma_ast::Stmt::Return(return_stmt) => {
            let expr = if let Some(arg) = &return_stmt.arg {
                compile_expr(fnb, arg)
            } else {
                fnb.add_expr(Expr::Undefined)
            };

            fnb.add_stmt(StmtOp::Return(expr));
        }

        swc_ecma_ast::Stmt::Break(break_stmt) => {
            todo!("break");
            if break_stmt.label.is_some() {
                panic!("unsupported: labeled break statement");
            }
            let break_target = fnb.break_target();
            block.stmts.push(Stmt {
                op: StmtOp::Break(break_target),
                span: break_stmt.span,
            });
        }
        swc_ecma_ast::Stmt::Continue(continue_stmt) => {
            todo!("continue");
            if continue_stmt.label.is_some() {
                panic!("unsupported: labeled continue statement");
            }
            let break_target = fnb.break_target();
            block.stmts.push(Stmt {
                op: StmtOp::Continue(break_target),
                span: continue_stmt.span,
            });
        }

        swc_ecma_ast::Stmt::If(if_stmt) => {
            let test = compile_expr(fnb, &if_stmt.test);
            let if_not = fnb.add_stmt(StmtOp::Invalid);

            fnb.block(|fnb| compile_stmt(fnb, &if_stmt.cons));
            let after_cons_jmp = fnb.add_stmt(StmtOp::Invalid);

            let else_target = fnb.peek_stmt_id();
            fnb.set_stmt(
                if_not,
                StmtOp::IfNot {
                    test,
                    distance: (else_target - if_not),
                },
            );

            if let Some(alt) = &if_stmt.alt {
                fnb.block(|fnb| compile_stmt(fnb, &if_stmt.cons));
            }

            let after_if = fnb.peek_stmt_id();
            fnb.set_stmt(after_cons_jmp, StmtOp::Jump(after_if - after_cons_jmp));
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
            let exc_value = compile_expr(fnb, &throw_stmt.arg);
            fnb.add_stmt(StmtOp::Throw(exc_value));
        }
        swc_ecma_ast::Stmt::Try(try_stmt) => {
            let try_begin = fnb.add_stmt(StmtOp::Invalid);
            fnb.block(|fnb| {
                for stmt in &try_stmt.block.stmts {
                    compile_stmt(fnb, stmt);
                }
            });
            fnb.add_stmt(StmtOp::TryEnd);
            let jmp_from_try = fnb.add_stmt(StmtOp::Invalid);

            let jmp_from_catch = try_stmt.handler.map(|handler| {
                fnb.block(|fnb| {
                    if let Some(handler_param) = &handler.param {
                        let name = compile_name_pat(handler_param);
                        fnb.add_decl(Decl {
                            name: name.clone(),
                            is_lexical: true,
                            is_global: false,
                        });
                        let cur_exc = fnb.add_expr(Expr::CurrentException);
                        fnb.add_stmt(StmtOp::Assign(name, cur_exc));
                    }

                    for stmt in &handler.body.stmts {
                        compile_stmt(fnb, stmt);
                    }
                });

                fnb.add_stmt(StmtOp::Invalid)
            });

            let finally_start = fnb.peek_stmt_id();
            fnb.set_stmt(jmp_from_try, StmtOp::Jump(finally_start - jmp_from_try));
            if let Some(jmp) = jmp_from_catch {
                fnb.set_stmt(jmp, StmtOp::Jump(finally_start - jmp));
            }

            if let Some(finalizer_block) = &try_stmt.finalizer {
                fnb.block(|fnb| {
                    for stmt in &finalizer_block.stmts {
                        compile_stmt(fnb, stmt);
                    }
                });
            }
        }
        swc_ecma_ast::Stmt::While(while_stmt) => {
            fnb.block(|fnb| {
                let start_sid = fnb.peek_stmt_id();

                let test = compile_expr(fnb, &while_stmt.test);
                let if_not = fnb.add_stmt(StmtOp::Invalid);

                compile_stmt(fnb, &while_stmt.body);

                let jump_sid = fnb.peek_stmt_id();
                fnb.add_stmt(StmtOp::Jump(start_sid - jump_sid));

                let target = fnb.peek_stmt_id();
                fnb.set_stmt(
                    if_not,
                    StmtOp::IfNot {
                        test,
                        distance: (target - if_not),
                    },
                );
            });
        }
        swc_ecma_ast::Stmt::DoWhile(dowhile_stmt) => {
            fnb.block(|fnb| {
                let block_start = fnb.peek_stmt_id();
                compile_stmt(fnb, &dowhile_stmt.body);

                let test = compile_expr(fnb, &dowhile_stmt.test);
                fnb.add_stmt(StmtOp::IfNot {
                    test,
                    distance: StmtsDistance(2),
                });

                let here = fnb.peek_stmt_id();
                fnb.add_stmt(StmtOp::Jump(block_start - here));
            });
        }
        swc_ecma_ast::Stmt::For(for_stmt) => {
            fnb.block(|fnb| {
                let outer_block_id = fnb.cur_block_id();

                match &for_stmt.init {
                    Some(swc_ecma_ast::VarDeclOrExpr::VarDecl(var_decl)) => {
                        compile_var_decl(fnb, var_decl);
                    }
                    Some(swc_ecma_ast::VarDeclOrExpr::Expr(expr)) => {
                        let expr = compile_expr(fnb, expr);
                        fnb.add_stmt(StmtOp::Evaluate(expr));
                    }
                    None => {}
                }

                fnb.block(|fnb| {
                    let loop_start = fnb.peek_stmt_id();
                    fnb.push_break_target();

                    let test_stmt_expr = for_stmt.test.as_ref().map(|test_expr| {
                        let test = compile_expr(fnb, test_expr);
                        let stmt = fnb.add_stmt(StmtOp::Invalid);
                        (stmt, test)
                    });

                    compile_stmt(fnb, &for_stmt.body);

                    if let Some(update) = &for_stmt.update {
                        let expr = compile_expr(fnb, update);
                        fnb.add_stmt(StmtOp::Evaluate(expr));
                    }

                    fnb.add_unshares_up_to(outer_block_id);
                    let here = fnb.peek_stmt_id();
                    fnb.add_stmt(StmtOp::Jump(loop_start - here));

                    if let Some((stmt, test)) = test_stmt_expr {
                        let outta_here = fnb.peek_stmt_id();
                        fnb.set_stmt(
                            stmt,
                            StmtOp::IfNot {
                                test,
                                distance: (outta_here - stmt),
                            },
                        );
                    }
                });
            });
        }

        swc_ecma_ast::Stmt::ForIn(forin_stmt) => {
            use swc_ecma_ast::{ForHead, Pat};

            let span = forin_stmt.span;

            /* TODO
            if is_strict_mode:
                forbid (element var name == any JS keyword)
                    // use is_identifier_keyword
            */

            let item_var_decl: Decl = match &forin_stmt.left {
                ForHead::UsingDecl(_) => unsupported_node!(&forin_stmt.left),
                ForHead::VarDecl(var_decl) => {
                    assert_eq!(var_decl.decls.len(), 1);
                    Decl {
                        name: compile_name_pat(&var_decl.decls[0].name),
                        is_lexical: match var_decl.kind {
                            swc_ecma_ast::VarDeclKind::Var => false,
                            swc_ecma_ast::VarDeclKind::Let => true,
                            swc_ecma_ast::VarDeclKind::Const => true,
                        },
                        is_global: false,
                    }
                }
                ForHead::Pat(pat) => Decl {
                    name: compile_name_pat(pat),
                    is_lexical: true,
                    is_global: false,
                },
            };

            fnb.block(|fnb| {
                let outer_block_id = fnb.cur_block_id();

                let item_var = item_var_decl.name.clone();
                fnb.add_decl(item_var_decl);

                let key_ndx_tmp = fnb.decl_tmp();
                let key_ndx = fnb.add_expr(Expr::Read(key_ndx_tmp.clone()));
                let zero = fnb.add_expr(ZERO);
                fnb.add_stmt(StmtOp::Assign(key_ndx_tmp.clone(), zero));

                let iteree = compile_expr(fnb, &forin_stmt.right);
                let (_, iteree) = create_tmp(fnb, iteree);

                let keys = fnb.add_expr(Expr::ObjectGetKeys(iteree));
                let (_, keys) = create_tmp(fnb, keys);

                let key_count = fnb.add_expr(Expr::ArrayLen(keys));
                let (_, key_count) = create_tmp(fnb, key_count);

                fnb.block(|fnb| {
                    let mid_block_id = fnb.cur_block_id();

                    let test =
                        fnb.add_expr(Expr::Binary(swc_ecma_ast::BinaryOp::Lt, key_ndx, key_count));

                    // IfNot { test, target: after the body }
                    let if_not = fnb.add_stmt(StmtOp::Invalid);

                    let element = fnb.add_expr(Expr::ArrayNth {
                        arr: keys,
                        index: key_ndx,
                    });
                    fnb.add_stmt(StmtOp::Assign(item_var, element));

                    fnb.push_break_target();
                    fnb.block(|fnb| {
                        compile_stmt(fnb, &forin_stmt.body);
                    });

                    {
                        let one = fnb.add_expr(ONE);
                        let new_val =
                            fnb.add_expr(Expr::Binary(swc_ecma_ast::BinaryOp::Add, key_ndx, one));
                        fnb.add_stmt(StmtOp::Assign(key_ndx_tmp, new_val));
                    }

                    fnb.add_unshares_up_to(outer_block_id);
                    fnb.add_stmt(StmtOp::Continue(mid_block_id));

                    let after_body = fnb.peek_stmt_id();
                    fnb.set_stmt(
                        if_not,
                        StmtOp::IfNot {
                            test,
                            distance: (after_body - if_not),
                        },
                    );
                });
            });
        }
        swc_ecma_ast::Stmt::ForOf(_) => todo!(),

        swc_ecma_ast::Stmt::Decl(decl) => match decl {
            swc_ecma_ast::Decl::Fn(fn_decl) => {
                let name = DeclName::Js(fn_decl.ident.sym.clone());
                fnb.add_decl(Decl {
                    name: name.clone(),
                    is_lexical: false,
                    is_global: false,
                });

                let func = {
                    let builder = fnb.suspend();
                    let func = compile_function(builder, &fn_decl.function);
                    Box::new(func)
                };

                let closure = fnb.add_expr(Expr::CreateClosure { func });
                fnb.add_stmt(StmtOp::Assign(name, closure));
            }
            swc_ecma_ast::Decl::Var(var_decl) => compile_var_decl(fnb, var_decl),
            _ => {
                unsupported_node!(decl)
            }
        },
        swc_ecma_ast::Stmt::Expr(expr_stmt) => {
            let value = compile_expr(fnb, &expr_stmt.expr);
            fnb.add_stmt(StmtOp::Evaluate(value));
        }

        swc_ecma_ast::Stmt::Labeled(_) | swc_ecma_ast::Stmt::With(_) => {
            unsupported_node!(stmt)
        }
    }
}

fn compile_var_decl(fnb: &mut FnBuilder, var_decl: &swc_ecma_ast::VarDecl) {
    let is_lexical = match var_decl.kind {
        swc_ecma_ast::VarDeclKind::Var => false,
        swc_ecma_ast::VarDeclKind::Let => true,
        swc_ecma_ast::VarDeclKind::Const => true,
    };

    for declarator in &var_decl.decls {
        let name = compile_name_pat(&declarator.name);
        fnb.add_decl(Decl {
            name: name.clone(),
            is_lexical,
            is_global: false,
        });

        if let Some(init) = &declarator.init {
            let value = compile_expr(fnb, init);
            fnb.add_stmt(StmtOp::Assign(name, value));
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

fn compile_expr(fnb: &mut FnBuilder, expr: &swc_ecma_ast::Expr) -> ExprID {
    match expr {
        swc_ecma_ast::Expr::This(this_expr) => fnb.add_expr(Expr::This),
        swc_ecma_ast::Expr::Array(array_expr) => fnb.block(|fnb| {
            let array_init = fnb.add_expr(Expr::ArrayCreate);
            let (array_tmp, _) = create_tmp(fnb, array_init);

            for value in &array_expr.elems {
                if let Some(expr_or_spread) = value {
                    if expr_or_spread.spread.is_some() {
                        panic!("unsupported: spread syntax in array literal");
                    }
                    let value = compile_expr(fnb, &expr_or_spread.expr);
                    fnb.add_stmt(StmtOp::ArrayPush(array_tmp, value));
                } else {
                    todo!("What does `None` mean here?")
                }
            }

            fnb.add_expr(Expr::Read(array_tmp))
        }),
        swc_ecma_ast::Expr::Object(object_expr) => {
            let init_value = fnb.add_expr(Expr::ObjectCreate);
            let (obj_tmp, obj) = create_tmp(fnb, init_value);

            for prop in &object_expr.props {
                match prop {
                    swc_ecma_ast::PropOrSpread::Spread(_) => {
                        panic!("unsupported: spread syntax in object literal");
                    }
                    swc_ecma_ast::PropOrSpread::Prop(prop) => {
                        let (key, value) = match prop.as_ref() {
                            swc_ecma_ast::Prop::Shorthand(name) => {
                                let key = fnb.add_expr(Expr::StringLiteral(name.sym.clone()));
                                let value =
                                    fnb.add_expr(Expr::Read(DeclName::Js(name.sym.clone())));
                                (key, value)
                            }
                            swc_ecma_ast::Prop::KeyValue(kv) => {
                                let key = match &kv.key {
                                    swc_ecma_ast::PropName::Ident(ident) => fnb.add_expr(
                                        Expr::StringLiteral(JsWord::from(ident.sym.to_string())),
                                    ),
                                    swc_ecma_ast::PropName::Str(str) => fnb.add_expr(
                                        Expr::StringLiteral(JsWord::from(str.value.to_string())),
                                    ),
                                    swc_ecma_ast::PropName::Num(num) => {
                                        fnb.add_expr(Expr::NumberLiteral(num.value))
                                    }
                                    _ => {
                                        unsupported_node!(kv.key)
                                    }
                                };
                                let value = compile_expr(fnb, &kv.value);

                                (key, value)
                            }

                            swc_ecma_ast::Prop::Method(method_prop) => {
                                let name = method_prop.key.as_ident().expect("object literal: method property syntax, but name is not identifier?");
                                let key = fnb.add_expr(Expr::StringLiteral(name.sym.clone()));
                                let func = {
                                    let builder = fnb.suspend();
                                    let func = compile_function(builder, &method_prop.function);
                                    Box::new(func)
                                };
                                let value = fnb.add_expr(Expr::CreateClosure { func });
                                (key, value)
                            }

                            swc_ecma_ast::Prop::Assign(_)
                            | swc_ecma_ast::Prop::Getter(_)
                            | swc_ecma_ast::Prop::Setter(_) => todo!(),
                        };

                        fnb.add_stmt(StmtOp::ObjectSet { obj, key, value });
                    }
                }
            }

            obj
        }

        swc_ecma_ast::Expr::Fn(fn_expr) => {
            let func = {
                let builder = fnb.suspend();
                let func = compile_function(builder, &fn_expr.function);
                Box::new(func)
            };
            fnb.add_expr(Expr::CreateClosure { func })
        }
        swc_ecma_ast::Expr::Arrow(arrow_expr) => {
            let span = arrow_expr.span;

            let params: Vec<_> = arrow_expr.params.iter().cloned().map(From::from).collect();

            let body = {
                let builder = fnb.suspend();
                let mut fnb = builder.new_function();

                match &*arrow_expr.body {
                    swc_ecma_ast::BlockStmtOrExpr::BlockStmt(block_stmts) => fnb.block(|fnb| {
                        for stmt in &block_stmts.stmts {
                            compile_stmt(fnb, stmt);
                        }
                    }),
                    swc_ecma_ast::BlockStmtOrExpr::Expr(expr) => {
                        fnb.block(|fnb| {
                            let value = compile_expr(fnb, expr);
                            fnb.add_stmt(StmtOp::Return(value));
                        });
                    }
                }

                fnb.build()
            };
            let func = compile_function_from_parts(arrow_expr.span, &params, body);

            let func_expr = fnb.add_expr(Expr::CreateClosure {
                func: Box::new(func),
            });

            // Unlike regular function declarations/expressions, arrow
            // expressions inherit the `this` binding from the surrounding
            // context

            let bind_lit = fnb.add_expr(Expr::StringLiteral("bind".into()));
            let bind_method = fnb.add_expr(Expr::ObjectGet {
                obj: func_expr,
                key: bind_lit,
            });

            let this = fnb.add_expr(Expr::This);
            fnb.add_expr(Expr::Call {
                callee: bind_method,
                args: vec![this],
                is_new: false,
            })
        }
        swc_ecma_ast::Expr::Unary(unary_expr) => {
            let arg = compile_expr(fnb, &unary_expr.arg);
            fnb.add_expr(Expr::Unary(unary_expr.op, arg))
        }
        swc_ecma_ast::Expr::Update(update_expr) => {
            let loc = compile_name(swc_ecma_ast::PatOrExpr::Expr(update_expr.arg.clone()));

            let value = compile_expr(fnb, &update_expr.arg);
            let op = match update_expr.op {
                swc_ecma_ast::UpdateOp::PlusPlus => swc_ecma_ast::BinaryOp::Add,
                swc_ecma_ast::UpdateOp::MinusMinus => swc_ecma_ast::BinaryOp::Sub,
            };
            let one = fnb.add_expr(ONE);
            let new_value = fnb.add_expr(Expr::Binary(op, value, one));
            fnb.add_stmt(StmtOp::Assign(loc, new_value));

            new_value
        }
        swc_ecma_ast::Expr::Bin(bin_expr) => {
            let left = compile_expr(fnb, &bin_expr.left);
            let right = compile_expr(fnb, &bin_expr.right);
            fnb.add_expr(Expr::Binary(bin_expr.op, left, right))
        }
        swc_ecma_ast::Expr::Assign(assign_expr) => {
            if let Some(ident) = assign_expr.left.as_ident() {
                let loc = DeclName::Js(ident.sym.clone());
                let init_value = fnb.add_expr(Expr::Read(loc.clone()));
                let value = compile_assignment(fnb, assign_expr, init_value);
                fnb.add_stmt(StmtOp::Assign(loc, value));
                value
            } else if let Some(target_expr) = assign_expr.left.as_expr() {
                match target_expr {
                    swc_ecma_ast::Expr::Member(member_expr) => {
                        compile_member_assignment(fnb, assign_expr, member_expr)
                    }
                    // We should have already handled this case in the `if let ... = asm.left.as_ident()`
                    // case
                    swc_ecma_ast::Expr::Ident(_) => panic!("assertion failed"),
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
            let obj = compile_expr(fnb, &member_expr.obj);
            let key = compile_object_key(fnb, &member_expr.prop);
            fnb.add_expr(Expr::ObjectGet { obj, key })
        }
        swc_ecma_ast::Expr::Cond(cond_expr) => {
            let tmp = fnb.decl_tmp();

            let test = compile_expr(fnb, &cond_expr.test);
            let if_not = fnb.add_stmt(StmtOp::Invalid);

            let value_cons = compile_expr(fnb, &cond_expr.cons);
            fnb.add_stmt(StmtOp::Assign(tmp, value_cons));
            let jmp_to_end = fnb.add_stmt(StmtOp::Invalid);

            let else_target = fnb.peek_stmt_id();
            fnb.set_stmt(
                if_not,
                StmtOp::IfNot {
                    test,
                    distance: (else_target - if_not),
                },
            );

            let value_alt = compile_expr(fnb, &cond_expr.alt);
            fnb.add_stmt(StmtOp::Assign(tmp, value_alt));

            let end = fnb.peek_stmt_id();
            fnb.set_stmt(jmp_to_end, StmtOp::Jump(end - jmp_to_end));

            fnb.add_expr(Expr::Read(tmp))
        }
        swc_ecma_ast::Expr::Call(call_expr) => {
            let callee = match &call_expr.callee {
                swc_ecma_ast::Callee::Super(_) | swc_ecma_ast::Callee::Import(_) => {
                    unsupported_node!(call_expr.callee)
                }
                swc_ecma_ast::Callee::Expr(expr) => expr,
            };
            compile_call(fnb, callee, &call_expr.args, false)
        }
        swc_ecma_ast::Expr::New(new_expr) => {
            let args = new_expr
                .args
                .as_ref()
                .map(|args| args.as_slice())
                .unwrap_or(&[]);
            compile_call(fnb, &new_expr.callee, args, true)
        }
        swc_ecma_ast::Expr::Seq(seq_expr) => {
            let mut last_value = fnb.add_expr(Expr::Undefined);

            fnb.block(|fnb| {
                for expr in &seq_expr.exprs {
                    last_value = compile_expr(fnb, expr);
                }
            });

            last_value
        }
        swc_ecma_ast::Expr::Ident(ident) => {
            fnb.add_expr(Expr::Read(DeclName::Js(ident.sym.clone())))
        }
        swc_ecma_ast::Expr::Lit(lit) => match lit {
            swc_ecma_ast::Lit::Str(str) => {
                fnb.add_expr(Expr::StringLiteral(str.value.to_string().into()))
            }
            swc_ecma_ast::Lit::Bool(b) => fnb.add_expr(Expr::BoolLiteral(b.value)),
            swc_ecma_ast::Lit::Null(null_node) => fnb.add_expr(Expr::Null),
            swc_ecma_ast::Lit::Num(num) => fnb.add_expr(Expr::NumberLiteral(num.value)),
            swc_ecma_ast::Lit::BigInt(_)
            | swc_ecma_ast::Lit::Regex(_)
            | swc_ecma_ast::Lit::JSXText(_) => unsupported_node!(lit),
        },

        swc_ecma_ast::Expr::Tpl(_) => todo!(),
        swc_ecma_ast::Expr::Paren(paren_expr) => compile_expr(fnb, &paren_expr.expr),

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
    fnb: &mut FnBuilder,
    assign_expr: &swc_ecma_ast::AssignExpr,
    member_expr: &swc_ecma_ast::MemberExpr,
) -> ExprID {
    fnb.block(|fnb| {
        let key = compile_object_key(fnb, &member_expr.prop);
        let obj = compile_expr(fnb, &member_expr.obj);

        let (key_tmp, key) = create_tmp(fnb, key);
        let (obj_tmp, obj) = create_tmp(fnb, obj);

        let init_value = fnb.add_expr(Expr::ObjectGet { obj, key });
        let value = compile_assignment(fnb, assign_expr, init_value);
        fnb.add_stmt(StmtOp::ObjectSet { obj, key, value });

        value
    })
}

fn compile_assignment(
    fnb: &mut FnBuilder,
    assign_expr: &swc_ecma_ast::AssignExpr,
    init_value: ExprID,
) -> ExprID {
    let rhs = compile_expr(fnb, &assign_expr.right);
    match assign_expr.op.to_update() {
        // regular assignment
        None => rhs,
        // updating assignment
        Some(binop) => fnb.add_expr(Expr::Binary(binop, init_value, rhs)),
    }
}

fn create_tmp(fnb: &mut FnBuilder, value: ExprID) -> (DeclName, ExprID) {
    let tmp = fnb.decl_tmp();
    fnb.add_stmt(StmtOp::Assign(tmp, value));
    (tmp, fnb.add_expr(Expr::Read(tmp)))
}

fn compile_object_key(fnb: &mut FnBuilder, prop: &swc_ecma_ast::MemberProp) -> ExprID {
    match prop {
        swc_ecma_ast::MemberProp::Ident(ident) => {
            fnb.add_expr(Expr::StringLiteral(ident.sym.clone()))
        }
        swc_ecma_ast::MemberProp::Computed(computed) => compile_expr(fnb, &computed.expr),
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
    fnb: &mut FnBuilder,
    callee: &swc_ecma_ast::Expr,
    args: &[swc_ecma_ast::ExprOrSpread],
    is_new: bool,
) -> ExprID {
    let callee = compile_expr(fnb, callee);
    let args = args
        .iter()
        .map(|expr_or_spread| {
            if expr_or_spread.spread.is_some() {
                panic!("unsupported: spread syntax in call");
            }
            compile_expr(fnb, &expr_or_spread.expr)
        })
        .collect();

    fnb.add_expr(Expr::Call {
        callee,
        args,
        is_new,
    })
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

    let stmts = &swc_func.body.as_ref().unwrap().stmts;
    let mut fnb = builder.new_function();
    for stmt in stmts {
        compile_stmt(&mut fnb, stmt);
    }
    let body = fnb.build();
    compile_function_from_parts(swc_func.span, &swc_func.params, body)
}

fn compile_function_from_parts(
    span: swc_common::Span,
    swc_params: &[swc_ecma_ast::Param],
    body: Block,
) -> Function {
    let mut parameters = Vec::new();
    for param in swc_params.into_iter() {
        let name = compile_name_pat(&param.pat);
        let name = name.expect_js_word();
        parameters.push(name.clone());
    }

    let declares_use_strict = block_starts_with_use_strict(&body);

    let unbound_names = find_unbound_references(&body);
    Function {
        parameters,
        unbound_names,
        declares_use_strict,
        body,
        span,
    }
}

fn block_starts_with_use_strict(block: &Block) -> bool {
    let expr = match block.stmts.first() {
        Some(Stmt {
            op: StmtOp::Evaluate(expr),
            ..
        }) => expr,
        _ => return false,
    };

    let expr = &block.exprs[expr.0 as usize];
    if let Expr::StringLiteral(atom) = expr {
        return atom == "use strict";
    }

    false
}

fn find_unbound_references(root: &Block) -> Vec<JsWord> {
    let mut declared = HashSet::new();
    let mut referenced = HashSet::new();

    for expr in &root.exprs {
        match expr {
            Expr::Read(DeclName::Js(name)) => {
                referenced.insert(name.clone());
            }
            Expr::CreateClosure { func } => {
                for name in &func.unbound_names {
                    referenced.insert(name.clone());
                }
            }
            _ => (),
        }
    }

    for stmt in &root.stmts {
        if let StmtOp::Assign(DeclName::Js(name), _) = &stmt.op {
            referenced.insert(name.clone());
        }
    }

    for decl in &root.decls {
        if let DeclName::Js(js_name) = &decl.name {
            declared.insert(js_name.clone());
        }
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

    #[test]
    fn test_function_before_let() {
        let function = quick_compile(
            "
                (function() {
                    function f() { x = 3 }
                    let x = 2;
                })()
                "
            .to_string(),
        );
        insta::assert_debug_snapshot!(function);
    }

    #[test]
    fn test_func_decl() {
        let function = quick_compile(
            "function myFunction() {
                    return 3
                }"
            .to_string(),
        );

        insta::assert_debug_snapshot!(function);
    }

    fn quick_compile(src: String) -> super::Function {
        let source_map = Rc::new(SourceMap::default());
        let swc_ast =
            super::super::parse_file("<input>".to_string(), src, source_map).expect("parse error");
        let function = super::compile_script(swc_ast);
        function
    }
}
