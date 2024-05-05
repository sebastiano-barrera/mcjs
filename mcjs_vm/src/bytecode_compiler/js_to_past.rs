//! The Processed AST.
//!
//! An intermediate representation that results from an initial processing of
//! swc_ecma_ast's AST, designed expressly to facilitate compilation to bytecode
//! and compile-time checks.

use std::fmt::Write;
use std::{collections::HashSet, rc::Rc};
use swc_atoms::JsWord;
use swc_common::Spanned;

use crate::define_flag;
use crate::{common::MultiErrResult, error, impl_debug_via_dump, tracing, util};

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
    pub strict_mode: StrictMode,
}
impl_debug_via_dump!(Function);

impl util::Dump for Function {
    fn dump<W: std::fmt::Write>(&self, f: &mut util::Formatter<W>) -> std::fmt::Result {
        let lo = self.span.lo.0;
        let hi = self.span.hi.0;

        write!(f, "[{}-{}] func (", lo, hi)?;
        util::write_comma_sep(f, self.parameters.iter())?;
        write!(f, ") unbound[")?;
        util::write_comma_sep(f, self.unbound_names.iter())?;
        write!(f, "] {:?} ", self.strict_mode)?;
        self.body.dump(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrictMode {
    Strict,
    Sloppy,
}

pub struct Block {
    pub id: BlockID,

    is_function_decl_allowed: bool,

    // These are all Vec's, but they have somewhat different semantics. This is
    // reflected in Block's public API
    /// Set of declarations. No ordering.
    decls: Vec<Decl>,

    /// Ordered sequence of function assignments. Not really indexable. Sematics: they
    /// happen after any initialization for `decls`, and before all `stmts`.
    ///
    /// These assignments are distinct from StmtOp::Assign because we want to 'carve out'
    /// a dedicated space at the beginning of each block where function assignments
    /// can be added without shifting every statement forward (therefore invalidating
    /// already emitted StmtID's).
    fn_asmts: Vec<FnAsmt>,

    /// Ordered sequence of statements.  Indexable via StmtID.
    stmts: Vec<Stmt>,

    /// Mapping of ExprID -> Expr. No ordering.
    exprs: Vec<Expr>,
}
impl_debug_via_dump!(Block);

/// A function assignment. See `Block::fn_asmts`.
pub struct FnAsmt {
    pub var_name: DeclName,
    pub expr: ExprID,
}

impl Block {
    fn assert_valid(&self) {
        // no pending instruction remaining
        assert!(
            !self.stmts.iter().any(|stmt| stmt.op.is_pending()),
            "js_to_past bug: unresolved pending stmt"
        );

        #[cfg(debug_assertions)]
        {
            let check_stmt_id = |stmt_id: &StmtID| debug_assert_eq!(stmt_id.1, self.id);
            let check_expr_id = |expr_id: &ExprID| {
                if let ExprID::Node(_, block_id) = expr_id {
                    debug_assert_eq!(*block_id, self.id);
                }
            };

            for stmt in &self.stmts {
                match &stmt.op {
                    StmtOp::Pending
                    | StmtOp::Break(_)
                    | StmtOp::Block(_)
                    | StmtOp::TryEnd
                    | StmtOp::Debugger => {}
                    StmtOp::IfNot { test } => check_expr_id(test),
                    StmtOp::Jump(target) | StmtOp::SetResumePoint(target) => check_stmt_id(target),
                    StmtOp::Return(eid) => check_expr_id(eid),
                    StmtOp::Assign(_, eid) => check_expr_id(eid),
                    StmtOp::Unshare(_) => {}
                    StmtOp::ArrayPush(_, eid) => check_expr_id(eid),
                    StmtOp::ObjectSet { obj, key, value } => {
                        check_expr_id(obj);
                        check_expr_id(key);
                        check_expr_id(value);
                    }
                    StmtOp::Throw(eid) => check_expr_id(eid),
                    StmtOp::TryBegin { handler } => check_stmt_id(handler),
                    StmtOp::StrAppend(_, eid) => check_expr_id(eid),
                }
            }

            for expr in &self.exprs {
                match expr {
                    Expr::Undefined
                    | Expr::Null
                    | Expr::This
                    | Expr::Read(_)
                    | Expr::StringLiteral(_)
                    | Expr::NumberLiteral(_)
                    | Expr::BoolLiteral(_)
                    | Expr::ArrayCreate
                    | Expr::ObjectCreate
                    | Expr::CreateClosure { func: _ }
                    | Expr::CurrentException
                    | Expr::ImportModule { import_path: _ }
                    | Expr::StringCreate
                    | Expr::RegexLiteral { .. }
                    | Expr::Error => {}

                    Expr::Unary(_, eid) => check_expr_id(eid),
                    Expr::Binary(_, l, r) => {
                        check_expr_id(l);
                        check_expr_id(r);
                    }
                    Expr::ArrayNth { arr, index } => {
                        check_expr_id(arr);
                        check_expr_id(index);
                    }
                    Expr::ArrayLen(eid) => check_expr_id(eid),
                    Expr::ObjectGet { obj, key } => {
                        check_expr_id(obj);
                        check_expr_id(key);
                    }
                    Expr::ObjectGetKeys(eid) => check_expr_id(eid),
                    Expr::Call { callee, args }
                    | Expr::New {
                        constructor: callee,
                        args,
                    } => {
                        check_expr_id(callee);
                        args.iter().for_each(&check_expr_id);
                    }
                }
            }
        }
    }

    pub fn fn_asmts(&self) -> impl ExactSizeIterator<Item = &FnAsmt> {
        self.fn_asmts.iter()
    }

    pub fn stmts(&self) -> impl ExactSizeIterator<Item = &Stmt> {
        self.stmts.iter()
    }

    pub fn get_expr(&self, expr_id: ExprID) -> &Expr {
        match expr_id {
            ExprID::Node(nid, _) => &self.exprs[nid as usize],
            ExprID::Undefined => &Expr::Undefined,
            ExprID::Null => &Expr::Null,
            ExprID::This => &Expr::This,
            ExprID::ArrayCreate => &Expr::ArrayCreate,
            ExprID::ObjectCreate => &Expr::ObjectCreate,
            ExprID::StringCreate => &Expr::StringCreate,
            ExprID::CurrentException => &Expr::CurrentException,
            ExprID::Error => &Expr::Error,
        }
    }

    pub fn decls(&self) -> impl ExactSizeIterator<Item = &Decl> {
        self.decls.iter()
    }
}

impl util::Dump for Block {
    fn dump<W: std::fmt::Write>(&self, f: &mut util::Formatter<W>) -> std::fmt::Result {
        writeln!(f, "{:?} {{", self.id)?;
        f.indent();

        writeln!(f, "decls:")?;
        f.indent();
        for decl in &self.decls {
            writeln!(f, "{:?}", decl)?;
        }
        f.dedent();

        writeln!(f, "fn asmts:")?;
        f.indent();
        for fa in &self.fn_asmts {
            writeln!(f, "{:?} <- {:?}", fa.var_name, fa.expr)?;
        }
        f.dedent();

        writeln!(f, "exprs:")?;
        f.indent();
        for (ndx, expr) in self.exprs.iter().enumerate() {
            write!(f, "e{}: ", ndx)?;
            match expr {
                Expr::CreateClosure { func } => {
                    writeln!(f, "CreateClosure:")?;
                    f.indent();
                    func.dump(f)?;
                    f.dedent();
                }
                _ => writeln!(f, "{:?}", expr)?,
            }
        }
        f.dedent();

        writeln!(f, "stmts:")?;
        f.indent();
        for stmt in &self.stmts {
            match &stmt.op {
                StmtOp::Block(block) => block.dump(f)?,
                _ => writeln!(f, "{:?}", stmt)?,
            }
        }
        f.dedent();

        f.dedent();
        writeln!(f, "}}")
    }
}

pub struct Decl {
    pub name: DeclName,
    pub init: DeclInit,
    /// When true, the declaration is going to be 'hoisted' to the parent block,
    /// all the way up until the closest enclosing function/module/script. (i.e.
    /// hoisting does not cross function boundaries).
    ///
    /// Redeclaration rules are checked at every 'step' of the way.
    pub is_hoisted: bool,
    /// When true, any other declaration of the same name will result in a
    /// compile error.
    pub is_conflicting: bool,
}
/// Whether to initialize this variable and to what value.
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum DeclInit {
    /// Do not initialize the variable implicitly.  The time until the first
    /// explicit assignment is called Temporal Dead Zone (TDZ) and during it,
    /// any access to the variable will result in a runtime error.
    TDZ,
    /// Implicitly initialize the variable to undefined, before any of the
    /// block's statements.
    Undefined,
}
impl Decl {
    fn from_js_var_decl(name: DeclName, kind: swc_ecma_ast::VarDeclKind) -> Self {
        match kind {
            swc_ecma_ast::VarDeclKind::Var => Decl {
                name,
                init: DeclInit::Undefined,
                is_hoisted: true,
                is_conflicting: false,
            },
            swc_ecma_ast::VarDeclKind::Let | swc_ecma_ast::VarDeclKind::Const => Decl {
                name,
                init: DeclInit::TDZ,
                is_hoisted: false,
                is_conflicting: true,
            },
        }
    }
}
impl std::fmt::Debug for Decl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Decl: {:?} = {:?}", self.name, self.init)?;
        if self.is_hoisted {
            write!(f, " [hoisted]")?;
        }
        if self.is_conflicting {
            write!(f, " [conflicting]")?;
        }
        Ok(())
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

// The BlockID of origin for each StmtID and ExprID in #[cfg(debug_assertions)]
// to check for a common type of bug where the StmtID/ExprID is used in the
// wrong block

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StmtID(u16, #[cfg(debug_assertions)] BlockID);

impl StmtID {
    pub fn numeric(&self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy)]
pub enum ExprID {
    Node(u16, #[cfg(debug_assertions)] BlockID),
    Undefined,
    Null,
    This,
    ArrayCreate,
    ObjectCreate,
    StringCreate,
    CurrentException,
    Error,
}

impl std::fmt::Debug for ExprID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprID::Node(node_id, _) => write!(f, "e{}", node_id),
            ExprID::Undefined => write!(f, "eUndefined"),
            ExprID::Null => write!(f, "eNull"),
            ExprID::This => write!(f, "eThis"),
            ExprID::ArrayCreate => write!(f, "eArrayCreate"),
            ExprID::ObjectCreate => write!(f, "eObjectCreate"),
            ExprID::StringCreate => write!(f, "eStringCreate"),
            ExprID::CurrentException => write!(f, "eCurrentException"),
            ExprID::Error => write!(f, "eError"),
        }
    }
}

#[derive(Debug)]
pub enum StmtOp {
    Pending,

    Break(BlockID),
    Block(Box<Block>),
    /// Semantics: "Do the next statement if not <test>" or, equivalently,
    /// "Skip the next statement if <test>". Evaluate the `test` expression and
    /// convert to boolean; if true, skip 1 statement; if false, proceed to the
    /// next stmt.
    IfNot {
        test: ExprID,
    },
    Jump(StmtID),
    Return(ExprID),
    /// Set the 'resume point', the IID that this generator's next() function (it has to
    /// be!) will resume to
    SetResumePoint(StmtID),

    Assign(Option<DeclName>, ExprID),
    Unshare(DeclName),
    ArrayPush(DeclName, ExprID),
    ObjectSet {
        obj: ExprID,
        key: ExprID,
        value: ExprID,
    },

    Throw(ExprID),
    TryBegin {
        handler: StmtID,
    },
    TryEnd,

    Debugger,
    StrAppend(DeclName, ExprID),
}

impl StmtOp {
    fn is_pending(&self) -> bool {
        matches!(self, StmtOp::Pending)
    }
}

#[derive(Debug)]
pub enum Expr {
    Undefined,
    Null,
    This,
    Read(DeclName),

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
    },
    New {
        constructor: ExprID,
        args: Vec<ExprID>,
    },

    CurrentException,

    ImportModule {
        import_path: JsWord,
    },
    StringCreate,
    RegexLiteral {
        pattern: String,
        flags: String,
    },
    Error,
}

const ZERO: Expr = Expr::NumberLiteral(0.0);
const ONE: Expr = Expr::NumberLiteral(1.0);

pub use builder::{BlockID, TmpID};
use builder::{Builder, FnBuilder};

mod builder {
    use crate::common::{Error, MultiError};
    use crate::util::pop_while;

    use super::*;

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
        source_map: Rc<swc_common::SourceMap>,
    }

    pub(super) struct FnBuilder<'a> {
        builder: &'a mut Builder,

        /// The stack of currently open nested blocks.
        ///
        /// Compilation works by appending Stmt and Expr nodes into the last (innermost)
        /// block. As compilation progresses, the last block is attached into the
        /// previous/parent block, until we finish with a single block that
        /// constitutes the function's body.
        blocks: Vec<Block>,

        /// The stack of 'current spans' that can be set via `FnBuilder::with_span`
        spans: Vec<swc_common::Span>,

        break_exits: Vec<Exit>,
        continue_exits: Vec<Exit>,

        errors: Vec<Error>,

        /// For as long as we're compiling a section of code interested in a completion
        /// value [*], this is the name of the variable that holds the completion value
        /// (typically a tmp).
        ///
        /// [*] Currently, this is only our hacky version of `eval` that only takes a
        /// string literal.
        completion_value_var: Option<DeclName>,
        strict_mode: StrictMode,

        /// Whether this function is the `next()` method of a generator.  This is a
        /// specific function that is emitted as part of the compilation process for
        /// *generator functions*. See the comment in `compile_function` for context.
        is_generator_next: bool,
    }

    #[derive(PartialEq, Eq)]
    struct Exit {
        block_id: BlockID,
        label: Option<JsWord>,
    }

    #[derive(Clone, Copy)]
    pub(super) enum ExitType {
        Break,
        Continue,
    }

    impl Builder {
        pub fn new(source_map: Rc<swc_common::SourceMap>) -> Self {
            Builder {
                source_map,
                next_id: 0,
            }
        }

        /// Create a function with the given StrictMode. (It can change later, for example
        /// if the function's first statement is "use strict").
        pub fn new_function(&mut self, initial_strict_mode: StrictMode) -> FnBuilder {
            FnBuilder::new(self, initial_strict_mode)
        }
    }

    impl<'a> FnBuilder<'a> {
        fn new(builder: &'a mut Builder, initial_strict_mode: StrictMode) -> Self {
            let mut fnb = FnBuilder {
                builder,
                blocks: Vec::new(),
                break_exits: Vec::new(),
                continue_exits: Vec::new(),
                spans: Vec::new(),
                errors: Vec::new(),
                completion_value_var: None,
                strict_mode: initial_strict_mode,
                is_generator_next: false,
            };
            // The function's outermost block. Will be retrieved via `pop_block` by `build`
            fnb.push_block();
            fnb.allow_fn_decl();
            fnb
        }

        pub(super) fn is_generator_next(&self) -> bool {
            self.is_generator_next
        }
        pub(super) fn set_is_generator_next(&mut self) {
            self.is_generator_next = true;
        }

        pub(super) fn source_map(&self) -> &swc_common::SourceMap {
            &self.builder.source_map
        }

        pub(super) fn build(mut self) -> MultiErrResult<Block> {
            let block = self.pop_block();
            assert!(self.blocks.is_empty());

            match self.errors.len() {
                0 => Ok(block),
                _ => Err(MultiError(self.errors)),
            }
        }

        pub(super) fn suspend(&mut self) -> &mut Builder {
            self.builder
        }

        pub(crate) fn cur_block_id(&self) -> BlockID {
            self.blocks.last().unwrap().id
        }
        fn cur_block_mut(&mut self) -> &mut Block {
            self.blocks.last_mut().unwrap()
        }

        fn push_block(&mut self) {
            let blkid = BlockID(self.builder.next_id);
            self.builder.next_id += 1;

            // TODO Reuse allocations and get new Block out of a pool
            self.blocks.push(Block {
                id: blkid,
                is_function_decl_allowed: false,
                decls: Vec::new(),
                fn_asmts: Vec::new(),
                stmts: Vec::new(),
                exprs: Vec::new(),
            });
        }
        fn pop_block(&mut self) -> Block {
            let block = self.blocks.pop().unwrap();
            block.assert_valid();

            // Check redeclarations.
            // Quadratic, but we don't expect so many declarations, and this allows us to avoid
            // memory allocs (due to HashSet for example).
            for i in 0..block.decls.len() {
                let name = &block.decls[i].name;
                let is_conflicting_i = block.decls[i].is_conflicting;

                for j in i + 1..block.decls.len() {
                    let is_conflicting_j = block.decls[j].is_conflicting;

                    if name == &block.decls[j].name && (is_conflicting_i || is_conflicting_j) {
                        self.signal_error(error!("identifier `{:?}` already declared", name));
                    }
                }
            }

            pop_while(&mut self.break_exits, |target| target.block_id == block.id);
            pop_while(&mut self.continue_exits, |target| {
                target.block_id == block.id
            });

            block
        }

        pub(super) fn block<T>(&mut self, builder: impl FnOnce(&mut Self) -> T) -> T {
            self.push_block();
            let ret = builder(self);
            let mut inner_block = self.pop_block();

            hoist_declarations(&mut self.cur_block_mut().decls, &mut inner_block.decls);

            self.add_stmt(StmtOp::Block(Box::new(inner_block)));

            ret
        }
        pub(super) fn blocks_depth(&self) -> usize {
            self.blocks.len()
        }

        pub(super) fn is_at_fn_body_start(&self) -> bool {
            self.blocks.len() == 1 && self.blocks[0].stmts.is_empty()
        }
        pub(super) fn declare_use_strict(&mut self) {
            self.strict_mode = StrictMode::Strict;
        }
        pub(crate) fn strict_mode(&self) -> StrictMode {
            self.strict_mode
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
                init: DeclInit::TDZ,
                is_hoisted: false,
                is_conflicting: true,
            });
            name
        }

        pub(super) fn peek_stmt_id(&mut self) -> StmtID {
            let block = self.blocks.last().unwrap();
            let stmt_ndx = block.stmts.len().try_into().unwrap();
            StmtID(
                stmt_ndx,
                #[cfg(debug_assertions)]
                self.cur_block_id(),
            )
        }

        pub(super) fn add_stmt(&mut self, op: StmtOp) -> StmtID {
            let span = self.spans.last().copied().unwrap_or_default();
            let block = self.cur_block_mut();
            let stmt_id_raw = block.stmts.len().try_into().unwrap();
            block.stmts.push(super::Stmt { span, op });

            StmtID(
                stmt_id_raw,
                #[cfg(debug_assertions)]
                self.cur_block_id(),
            )
        }

        pub(super) fn set_stmt(&mut self, id: StmtID, op: StmtOp) {
            let span = self.spans.last().copied().unwrap_or_default();
            let block = self.cur_block_mut();
            let stmt = block.stmts.get_mut(id.0 as usize).unwrap();
            *stmt = super::Stmt { span, op };
        }

        pub(super) fn add_expr(&mut self, expr: Expr) -> ExprID {
            match expr {
                Expr::Undefined => ExprID::Undefined,
                Expr::Null => ExprID::Null,
                Expr::This => ExprID::This,
                Expr::ArrayCreate => ExprID::ArrayCreate,
                Expr::ObjectCreate => ExprID::ObjectCreate,
                Expr::StringCreate => ExprID::StringCreate,
                Expr::CurrentException => ExprID::CurrentException,
                Expr::Error => ExprID::Error,
                _ => {
                    let block = self.cur_block_mut();
                    let expr_id_raw = block.exprs.len().try_into().unwrap();
                    block.exprs.push(expr);
                    ExprID::Node(
                        expr_id_raw,
                        #[cfg(debug_assertions)]
                        self.cur_block_id(),
                    )
                }
            }
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

        /// Set the current block as the target for StmtOp::Break statements
        /// (added with `add_break_stmt`) appearing within it.
        ///
        /// This setting is independent from the one for StmtOp::Continue.
        ///
        /// Overrides the break target inherited from parent blocks.
        pub(super) fn break_exits_this_block(&mut self, label: Option<JsWord>) {
            let block_id = self.cur_block_id();
            self.break_exits.push(Exit { block_id, label });
        }

        /// Exactly the same as `break_exits_this_block`, but applies to
        /// StmtOp::Continue statements (added with `add_continue_stmt`)
        /// instead.
        ///
        /// This setting is independent from the one for StmtOp::Break.
        pub(super) fn continue_exits_this_block(&mut self, label: Option<JsWord>) {
            let block_id = self.cur_block_id();
            self.continue_exits.push(Exit { block_id, label });
        }

        pub(super) fn find_exit(
            &self,
            exit_type: ExitType,
            label: Option<&JsWord>,
        ) -> Option<BlockID> {
            let targets_stack = match exit_type {
                ExitType::Break => &self.break_exits,
                ExitType::Continue => &self.continue_exits,
            };

            if let Some(label) = label {
                targets_stack
                    .iter()
                    .rev()
                    .filter(|exit| exit.label.as_ref() == Some(label))
                    .map(|exit| exit.block_id)
                    .next()
            } else {
                targets_stack.last().map(|exit| exit.block_id)
            }
        }

        pub(super) fn with_span<T>(
            &mut self,
            span: swc_common::Span,
            action: impl FnOnce(&mut Self) -> T,
        ) -> T {
            self.spans.push(span);
            let ret = action(self);
            self.spans.pop().unwrap();
            ret
        }

        pub(crate) fn signal_error(&mut self, error: Error) {
            self.errors.push(error);
        }
        pub(crate) fn signal_multi_error(&mut self, multi_err: MultiError) {
            self.errors.extend(multi_err.0);
        }

        pub(crate) fn allow_fn_decl(&mut self) {
            self.cur_block_mut().is_function_decl_allowed = true;
        }
        pub(crate) fn is_fn_decl_allowed(&self) -> bool {
            self.blocks.last().unwrap().is_function_decl_allowed
        }

        pub(crate) fn completion_value_var(&self) -> Option<&DeclName> {
            self.completion_value_var.as_ref()
        }
        pub(crate) fn enable_completion_value_var(&mut self, cv_name: DeclName) {
            // It's possible to work with multiple nested ranges of code interested in completion
            // values, but I have no intention of supporting eval 100%, so this is good enough
            assert!(self.completion_value_var.is_none());
            self.completion_value_var = Some(cv_name);
        }
        pub(crate) fn clear_completion_value(&mut self) {
            if let Some(cv_var) = self.completion_value_var.clone() {
                let undefined = self.add_expr(Expr::Undefined);
                self.add_stmt(StmtOp::Assign(Some(cv_var), undefined));
            }
        }
        pub(crate) fn disable_completion_value_var(&mut self) {
            self.completion_value_var = None;
        }

        pub(crate) fn assign_fn(&mut self, var_name: DeclName, expr: ExprID) {
            self.cur_block_mut()
                .fn_asmts
                .push(FnAsmt { var_name, expr })
        }
    }

    fn hoist_declarations(outer: &mut Vec<Decl>, inner: &mut Vec<Decl>) {
        let all_decls = std::mem::take(inner);

        for decl in all_decls {
            if decl.is_hoisted {
                outer.push(decl);
            } else {
                inner.push(decl);
            }
        }
    }
}

pub fn compile_script(
    script_ast: &swc_ecma_ast::Script,
    source_map: Rc<swc_common::SourceMap>,
) -> MultiErrResult<Function> {
    let mut builder = Builder::new(source_map);
    let mut fnb = builder.new_function(StrictMode::Sloppy);
    for stmt in &script_ast.body {
        compile_stmt(&mut fnb, stmt);
    }
    let strict_mode = fnb.strict_mode();
    let block = fnb.build()?;
    let mut function = compile_function_from_parts(script_ast.span, &[], strict_mode, block);

    // In scripts, declarations that are non-conflicting (var, function) and are
    // found at the toplevel (either because they were brought there via
    // hoisting or were already there in the source code) are allocated in the
    // `globalThis` object. In our IR, this can be achieved simply by removing
    // those declarations and transferring them to the "unbound names" list of
    // the script's root function.
    let mut kept_decls = Vec::new();
    for decl in std::mem::take(&mut function.body.decls) {
        match decl.name {
            DeclName::Js(name) if !decl.is_conflicting => {
                function.unbound_names.push(name);
            }
            _ => {
                kept_decls.push(decl);
            }
        }
    }
    function.body.decls = kept_decls;

    Ok(function)
}

pub fn compile_module(
    script_ast: &swc_ecma_ast::Module,
    source_map: Rc<swc_common::SourceMap>,
) -> MultiErrResult<Function> {
    use swc_ecma_ast::ModuleItem;

    let mut builder = Builder::new(source_map);
    // Modules are "use strict" by definition
    let mut fnb = builder.new_function(StrictMode::Strict);

    let lit_named = fnb.add_expr(Expr::StringLiteral("__named".into()));
    let lit_default = fnb.add_expr(Expr::StringLiteral("__default".into()));
    let module_obj = fnb.add_expr(Expr::ObjectCreate);
    let (_, module_obj) = create_tmp(&mut fnb, module_obj);

    let named_exports_obj = fnb.add_expr(Expr::ObjectCreate);
    let (_, named_exports_obj) = create_tmp(&mut fnb, named_exports_obj);
    fnb.add_stmt(StmtOp::ObjectSet {
        obj: module_obj,
        key: lit_named,
        value: named_exports_obj,
    });

    // -- process imports
    for module_item in &script_ast.body {
        if let ModuleItem::ModuleDecl(swc_ecma_ast::ModuleDecl::Import(import_decl)) = module_item {
            if import_decl.type_only {
                // TODO(small feat) make up a system for warnings
                eprintln!("bytecode_compiler: warning: discarding type-only import statement");
                continue;
            }

            let module_path: JsWord = import_decl.src.value.to_string().into();
            let module = fnb.add_expr(Expr::ImportModule {
                import_path: module_path,
            });
            let (_, module) = create_tmp(&mut fnb, module);
            let named_exports = fnb.add_expr(Expr::ObjectGet {
                obj: module,
                key: lit_named,
            });
            let default_export = fnb.add_expr(Expr::ObjectGet {
                obj: module,
                key: lit_default,
            });

            for spec in &import_decl.specifiers {
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
                            continue;
                        }
                        if d.imported.is_some() {
                            todo!("import specifier with rename");
                        }

                        let var_name: JsWord = d.local.sym.clone();

                        let key = fnb.add_expr(Expr::StringLiteral(var_name.clone()));
                        let imported_value = fnb.add_expr(Expr::ObjectGet {
                            obj: named_exports,
                            key,
                        });

                        let decl_name = DeclName::Js(var_name);
                        fnb.add_decl(Decl {
                            name: decl_name.clone(),
                            init: DeclInit::TDZ,
                            is_hoisted: false,
                            is_conflicting: true,
                        });
                        fnb.add_stmt(StmtOp::Assign(Some(decl_name), imported_value));
                    }
                    swc_ecma_ast::ImportSpecifier::Default(d) => {
                        let name = DeclName::Js(d.local.sym.clone());
                        fnb.add_decl(Decl {
                            name: name.clone(),
                            init: DeclInit::TDZ,
                            is_hoisted: false,
                            is_conflicting: true,
                        });
                        fnb.add_stmt(StmtOp::Assign(Some(name), default_export));
                    }
                    swc_ecma_ast::ImportSpecifier::Namespace(d) => {
                        let name = DeclName::Js(d.local.sym.clone());
                        fnb.add_decl(Decl {
                            name: name.clone(),
                            init: DeclInit::TDZ,
                            is_hoisted: false,
                            is_conflicting: true,
                        });
                        fnb.add_stmt(StmtOp::Assign(Some(name), named_exports));
                    }
                }
            }
        }
    }

    // -- process module body (stmts) & exports
    for module_item in &script_ast.body {
        match module_item {
            ModuleItem::Stmt(stmt) => {
                compile_stmt(&mut fnb, stmt);
            }
            ModuleItem::ModuleDecl(module_decl) => match module_decl {
                swc_ecma_ast::ModuleDecl::ExportDecl(export_decl) => {
                    compile_decl(&mut fnb, &export_decl.decl);
                }
                swc_ecma_ast::ModuleDecl::ExportDefaultDecl(export_default_decl) => {
                    match &export_default_decl.decl {
                        swc_ecma_ast::DefaultDecl::Class(_) => {
                            unsupported_node!(export_default_decl)
                        }
                        swc_ecma_ast::DefaultDecl::Fn(fn_expr) => {
                            compile_fn_expr(&mut fnb, fn_expr);
                        }
                        swc_ecma_ast::DefaultDecl::TsInterfaceDecl(_) => {}
                    }
                }

                swc_ecma_ast::ModuleDecl::ExportDefaultExpr(_)
                | swc_ecma_ast::ModuleDecl::ExportNamed(_)
                | swc_ecma_ast::ModuleDecl::ExportAll(_) => unsupported_node!(module_decl),

                swc_ecma_ast::ModuleDecl::Import(_)
                | swc_ecma_ast::ModuleDecl::TsImportEquals(_)
                | swc_ecma_ast::ModuleDecl::TsExportAssignment(_)
                | swc_ecma_ast::ModuleDecl::TsNamespaceExport(_) => {}
            },
        }
    }

    let strict_mode = fnb.strict_mode();
    let block = fnb.build()?;
    let function = compile_function_from_parts(script_ast.span, &[], strict_mode, block);

    Ok(function)
}

fn compile_stmt(fnb: &mut FnBuilder, stmt: &swc_ecma_ast::Stmt) {
    compile_stmt_ex(fnb, stmt, None)
}

fn compile_stmt_ex(fnb: &mut FnBuilder, stmt: &swc_ecma_ast::Stmt, label: Option<&JsWord>) {
    fnb.with_span(stmt.span(), |fnb| {
        match stmt {
            swc_ecma_ast::Stmt::Block(block_stmt) => {
                compile_block(fnb, block_stmt);
            }
            swc_ecma_ast::Stmt::Empty(_) => {}
            swc_ecma_ast::Stmt::Debugger(_) => {
                fnb.add_stmt(StmtOp::Debugger);
            }

            swc_ecma_ast::Stmt::Return(return_stmt) => {
                let expr = if let Some(arg) = &return_stmt.arg {
                    compile_expr(fnb, arg)
                } else {
                    fnb.add_expr(Expr::Undefined)
                };

                if fnb.is_generator_next() {
                    // for a generator's next() function, we want to compile
                    //    return (x)
                    // into
                    //    return { value: x, done: true }
                    // see the comment in `compile_function` for context.
                    compile_yield(fnb, expr, YieldDone::Yes);
                } else {
                    fnb.add_stmt(StmtOp::Return(expr));
                }
            }

            swc_ecma_ast::Stmt::Break(break_stmt) => {
                let label = break_stmt.label.as_ref().map(|ident| &ident.sym);
                if let Some(target) = find_exit(fnb, builder::ExitType::Break, label) {
                    fnb.add_stmt(StmtOp::Break(target));
                }
            }
            swc_ecma_ast::Stmt::Continue(cont_stmt) => {
                let label = cont_stmt.label.as_ref().map(|ident| &ident.sym);
                if let Some(target) = find_exit(fnb, builder::ExitType::Continue, label) {
                    fnb.add_stmt(StmtOp::Break(target));
                }
            }

            swc_ecma_ast::Stmt::If(if_stmt) => {
                let test = compile_expr(fnb, &if_stmt.test);
                fnb.add_stmt(StmtOp::IfNot { test });
                let if_not_jmp = fnb.add_stmt(StmtOp::Pending);

                fnb.block(|fnb| compile_stmt(fnb, &if_stmt.cons));
                let after_cons = fnb.add_stmt(StmtOp::Pending);

                let else_start = fnb.peek_stmt_id();
                if let Some(alt) = &if_stmt.alt {
                    fnb.block(|fnb| compile_stmt(fnb, alt));
                }

                let end = fnb.peek_stmt_id();
                fnb.set_stmt(if_not_jmp, StmtOp::Jump(else_start));
                fnb.set_stmt(after_cons, StmtOp::Jump(end));
            }

            swc_ecma_ast::Stmt::Switch(switch_stmt) => {
                // The tricky bit here is that declarations have to be shared across the whole of
                // the switch stmt's body, so all cases need to be compiled into a single block.
                // The cases are not really separate.
                fnb.clear_completion_value();
                fnb.block(|fnb| {
                    // This block is necessarily brace-delimited
                    fnb.allow_fn_decl();

                    let discriminant = compile_expr(fnb, &switch_stmt.discriminant);
                    let (_, discriminant) = create_tmp(fnb, discriminant);

                    fnb.break_exits_this_block(None);

                    // The switch stmt is compiled into two segments:
                    //
                    // - compare the discriminant against each case value and if it matches,
                    // StmtOp::Jump to the corresponding consequent statement. If none matches, go
                    // to the default case.
                    //      - in this phase, the check for the default case comes last
                    //
                    // - all statements in the switch body without the 'case:' labels. The Jump
                    // stmts jump to somewhere in this block.
                    //      - in this phase, the cases are compiled in the order they appear in the
                    //      source code (including the default case).

                    // TODO Remove this vector? It's always going to contain perfectly sequential
                    // stmt IDs
                    let mut non_default_cases = Vec::new();
                    let mut has_default_case = false;

                    for case in &switch_stmt.cases {
                        match &case.test {
                            Some(test) => {
                                // We know the test value so we can write the comparison stmt, ...
                                let test_value = compile_expr(fnb, test);
                                let test = fnb.add_expr(Expr::Binary(
                                    // Negated, because it's an If*Not*.
                                    //  we generate this code: if not !==, then jmp to X
                                    //  to get these semantics: if ===, then jmp to X
                                    swc_ecma_ast::BinaryOp::NotEqEq,
                                    discriminant,
                                    test_value,
                                ));
                                fnb.add_stmt(StmtOp::IfNot { test });

                                // ... but we don't know the jump target yet, so we
                                // reserve the stmt slot and resolve it later
                                let jmpif_sid = fnb.add_stmt(StmtOp::Pending);
                                non_default_cases.push(jmpif_sid);
                            }
                            None => {
                                if has_default_case {
                                    fnb.signal_error(error!("multiple `default` clauses"));
                                }
                                has_default_case = true;
                            }
                        }
                    }

                    let mut default_case = if has_default_case {
                        // at this point in the code we've failed all the other case tests, so
                        // we just emit a jump (to be resolved later)
                        let stmt_id = fnb.add_stmt(StmtOp::Pending);
                        Some(stmt_id)
                    } else {
                        None
                    };

                    let mut non_default_cases = non_default_cases.into_iter();
                    for case in &switch_stmt.cases {
                        let jmp_target = fnb.peek_stmt_id();

                        let jmpif_sid = if case.test.is_some() {
                            non_default_cases.next().unwrap()
                        } else {
                            default_case.take().unwrap()
                        };
                        assert_ne!(jmpif_sid, jmp_target);
                        fnb.set_stmt(jmpif_sid, StmtOp::Jump(jmp_target));

                        for stmt in &case.cons {
                            compile_stmt(fnb, stmt);
                        }
                    }

                    assert!(non_default_cases.next().is_none());
                    assert!(default_case.is_none());
                });
            }

            swc_ecma_ast::Stmt::Throw(throw_stmt) => {
                let exc_value = compile_expr(fnb, &throw_stmt.arg);
                fnb.add_stmt(StmtOp::Throw(exc_value));
            }
            swc_ecma_ast::Stmt::Try(try_stmt) => {
                let try_begin = fnb.add_stmt(StmtOp::Pending);
                compile_block(fnb, &try_stmt.block);
                fnb.add_stmt(StmtOp::TryEnd);
                let jmp_from_try = fnb.add_stmt(StmtOp::Pending);

                let handler_start = fnb.peek_stmt_id();
                let jmp_from_catch = try_stmt.handler.as_ref().map(|handler| {
                    fnb.block(|fnb| {
                        if let Some(handler_param) = &handler.param {
                            let name = compile_name_pat(handler_param);
                            fnb.add_decl(Decl {
                                name: name.clone(),
                                init: DeclInit::TDZ,
                                is_hoisted: false,
                                is_conflicting: true,
                            });
                            let cur_exc = fnb.add_expr(Expr::CurrentException);
                            fnb.add_stmt(StmtOp::Assign(Some(name), cur_exc));
                        }

                        compile_block(fnb, &handler.body);
                    });

                    fnb.add_stmt(StmtOp::Pending)
                });

                let finally_start = fnb.peek_stmt_id();
                if let Some(finalizer_block) = &try_stmt.finalizer {
                    compile_block(fnb, finalizer_block);
                }

                fnb.set_stmt(
                    try_begin,
                    StmtOp::TryBegin {
                        handler: handler_start,
                    },
                );
                fnb.set_stmt(jmp_from_try, StmtOp::Jump(finally_start));
                if let Some(jmp_from_catch) = jmp_from_catch {
                    fnb.set_stmt(jmp_from_catch, StmtOp::Jump(finally_start));
                }
            }
            swc_ecma_ast::Stmt::While(while_stmt) => {
                fnb.block(|fnb| {
                    fnb.break_exits_this_block(label.cloned());
                    let block_start = fnb.peek_stmt_id();

                    let test = compile_expr(fnb, &while_stmt.test);
                    fnb.add_stmt(StmtOp::IfNot { test });

                    if let Some(target) = fnb.find_exit(builder::ExitType::Break, label) {
                        fnb.add_stmt(StmtOp::Break(target));
                    }

                    fnb.block(|fnb| {
                        fnb.continue_exits_this_block(label.cloned());
                        compile_stmt(fnb, &while_stmt.body);
                    });

                    fnb.add_stmt(StmtOp::Jump(block_start));
                });
            }
            swc_ecma_ast::Stmt::DoWhile(dowhile_stmt) => {
                fnb.block(|fnb| {
                    fnb.break_exits_this_block(label.cloned());

                    let block_start = fnb.peek_stmt_id();
                    fnb.block(|fnb| {
                        fnb.continue_exits_this_block(label.cloned());
                        compile_stmt(fnb, &dowhile_stmt.body);
                    });

                    let test = compile_expr(fnb, &dowhile_stmt.test);
                    fnb.add_stmt(StmtOp::IfNot { test });

                    if let Some(target) = fnb.find_exit(builder::ExitType::Break, None) {
                        fnb.add_stmt(StmtOp::Break(target));
                    }
                    fnb.add_stmt(StmtOp::Jump(block_start));
                });
            }
            swc_ecma_ast::Stmt::For(for_stmt) => {
                fnb.block(|fnb| {
                    let outer_block_id = fnb.cur_block_id();

                    // If this statement is labeled, then this compile_stmt_ex has
                    // been called by another compile_stmt_ex with stmt =
                    // Stmt::Labeled(_).  The caller frame has already taken care of
                    // defining the labeled break exit (in another wrapping block).
                    fnb.break_exits_this_block(None);

                    match &for_stmt.init {
                        Some(swc_ecma_ast::VarDeclOrExpr::VarDecl(var_decl)) => {
                            compile_var_decl(fnb, var_decl);
                        }
                        Some(swc_ecma_ast::VarDeclOrExpr::Expr(expr)) => {
                            let expr = compile_expr(fnb, expr);
                            fnb.add_stmt(StmtOp::Assign(None, expr));
                        }
                        None => {}
                    }

                    fnb.block(|fnb| {
                        let loop_start = fnb.peek_stmt_id();

                        if let Some(test_expr) = &for_stmt.test {
                            let test = compile_expr(fnb, test_expr);
                            fnb.add_stmt(StmtOp::IfNot { test });

                            if let Some(target) = find_exit(fnb, builder::ExitType::Break, label) {
                                fnb.add_stmt(StmtOp::Break(target));
                            }
                        }

                        fnb.block(|fnb| {
                            fnb.continue_exits_this_block(label.cloned());
                            compile_stmt(fnb, &for_stmt.body)
                        });

                        if let Some(update) = &for_stmt.update {
                            let expr = compile_expr(fnb, update);
                            fnb.add_stmt(StmtOp::Assign(None, expr));
                        }

                        fnb.add_unshares_up_to(outer_block_id);
                        fnb.add_stmt(StmtOp::Jump(loop_start));
                    });
                });
            }

            swc_ecma_ast::Stmt::ForIn(forin_stmt) => {
                use swc_ecma_ast::ForHead;

                // let span = forin_stmt.span;

                /* TODO
                if is_strict_mode:
                    forbid (element var name == any JS keyword)
                        // use is_identifier_keyword
                */

                let item_var_decl: Decl = match &forin_stmt.left {
                    ForHead::UsingDecl(_) => unsupported_node!(&forin_stmt.left),
                    ForHead::VarDecl(var_decl) => {
                        assert_eq!(var_decl.decls.len(), 1);
                        let name = compile_name_pat(&var_decl.decls[0].name);
                        Decl::from_js_var_decl(name, var_decl.kind)
                    }
                    ForHead::Pat(pat) => {
                        let name = compile_name_pat(pat);
                        Decl::from_js_var_decl(name, swc_ecma_ast::VarDeclKind::Let)
                    }
                };

                fnb.block(|fnb| {
                    let outer_block_id = fnb.cur_block_id();
                    fnb.break_exits_this_block(label.cloned());

                    let item_var = item_var_decl.name.clone();
                    fnb.add_decl(item_var_decl);

                    let zero = fnb.add_expr(ZERO);
                    let (key_ndx_tmp, key_ndx) = create_tmp(fnb, zero);

                    let iteree = compile_expr(fnb, &forin_stmt.right);
                    let (_, iteree) = create_tmp(fnb, iteree);

                    let keys = fnb.add_expr(Expr::ObjectGetKeys(iteree));
                    let (_, keys) = create_tmp(fnb, keys);

                    let key_count = fnb.add_expr(Expr::ArrayLen(keys));
                    let (_, key_count) = create_tmp(fnb, key_count);

                    let loop_start = fnb.peek_stmt_id();

                    let test =
                        fnb.add_expr(Expr::Binary(swc_ecma_ast::BinaryOp::Lt, key_ndx, key_count));
                    fnb.add_stmt(StmtOp::IfNot { test });

                    if let Some(target) = fnb.find_exit(builder::ExitType::Break, label) {
                        fnb.add_stmt(StmtOp::Break(target));
                    };

                    let element = fnb.add_expr(Expr::ArrayNth {
                        arr: keys,
                        index: key_ndx,
                    });
                    fnb.add_stmt(StmtOp::Assign(Some(item_var), element));

                    fnb.block(|fnb| {
                        fnb.continue_exits_this_block(label.cloned());
                        compile_stmt(fnb, &forin_stmt.body);
                    });

                    {
                        let one = fnb.add_expr(ONE);
                        let new_val =
                            fnb.add_expr(Expr::Binary(swc_ecma_ast::BinaryOp::Add, key_ndx, one));
                        fnb.add_stmt(StmtOp::Assign(Some(key_ndx_tmp), new_val));
                    }

                    fnb.add_unshares_up_to(outer_block_id);
                    fnb.add_stmt(StmtOp::Jump(loop_start));
                });
            }
            swc_ecma_ast::Stmt::ForOf(_) => todo!(),

            swc_ecma_ast::Stmt::Decl(decl) => {
                compile_decl(fnb, decl);
            }
            swc_ecma_ast::Stmt::Expr(expr_stmt) => {
                if fnb.is_at_fn_body_start() {
                    if let swc_ecma_ast::Expr::Lit(swc_ecma_ast::Lit::Str(s)) =
                        expr_stmt.expr.as_ref()
                    {
                        if &s.value == "use strict" {
                            fnb.declare_use_strict();
                            return;
                        }
                    }
                }

                let expr = compile_expr(fnb, &expr_stmt.expr);
                fnb.add_stmt(StmtOp::Assign(fnb.completion_value_var().cloned(), expr));
            }

            swc_ecma_ast::Stmt::Labeled(labeled_stmt) => {
                // we interpret the 'labeled statement' feature as a 'labelled
                // block', actually. then we just, well, label the block. the
                // existing machinery for break/continue is already fit for the
                // purpose.

                let label = labeled_stmt.label.sym.clone();
                fnb.block(move |fnb| {
                    fnb.break_exits_this_block(Some(label.clone()));
                    compile_stmt_ex(fnb, &labeled_stmt.body, Some(&label));
                });
            }

            swc_ecma_ast::Stmt::With(_) => {
                unsupported_node!(stmt)
            }
        }
    })
}

define_flag!(YieldDone);

fn compile_yield(fnb: &mut FnBuilder<'_>, expr: ExprID, done: YieldDone) {
    let iterator_item = fnb.add_expr(Expr::ObjectCreate);
    let (_, iterator_item) = create_tmp(fnb, iterator_item);

    let key = fnb.add_expr(Expr::StringLiteral("value".into()));
    fnb.add_stmt(StmtOp::ObjectSet {
        obj: iterator_item,
        key,
        value: expr,
    });

    let key = fnb.add_expr(Expr::StringLiteral("done".into()));
    let done = fnb.add_expr(Expr::BoolLiteral(done.into()));
    fnb.add_stmt(StmtOp::ObjectSet {
        obj: iterator_item,
        key,
        value: done,
    });

    fnb.add_stmt(StmtOp::Return(iterator_item));
}

fn compile_block(fnb: &mut FnBuilder, block_stmt: &swc_ecma_ast::BlockStmt) {
    // Stupid fucking hack to get around the fact that swc doesn't tell us if the block
    // is actually delimited by curly braces (as opposed to being implicit and
    // single-statement), and JavaScript somehow cares :(
    let swc_common::SourceFileAndBytePos { sf, pos } =
        fnb.source_map().lookup_byte_offset(block_stmt.span().lo);
    let first_char = sf.src.as_bytes()[pos.0 as usize];
    let is_brace_delimited = first_char == b'{';

    fnb.block(|fnb| {
        if is_brace_delimited {
            fnb.allow_fn_decl();
        }

        for stmt in &block_stmt.stmts {
            compile_stmt(fnb, stmt);
        }
    });
}

fn find_exit(
    fnb: &mut FnBuilder,
    exit_type: builder::ExitType,
    label: Option<&JsWord>,
) -> Option<BlockID> {
    let exit = fnb.find_exit(exit_type, label);
    if exit.is_none() {
        let error = if let Some(label) = label {
            error!("undefined block label `{}`", label.as_ref())
        } else {
            error!("break/continue statement outside of loop statement")
        };

        fnb.signal_error(error);
    }

    exit
}

fn compile_decl(fnb: &mut FnBuilder, decl: &swc_ecma_ast::Decl) {
    match decl {
        swc_ecma_ast::Decl::Fn(fn_decl) => {
            if !fnb.is_fn_decl_allowed() {
                fnb.signal_error(error!("function decl not allowed in this position"));
            }

            let name = DeclName::Js(fn_decl.ident.sym.clone());

            // For function declarations, assignment to their value is always done at the beginning
            // of the block.  This allows the function to be called earlier in the block than the
            // declaration site in the source code.
            //
            // In strict mode, the declaration is also hoisted at the beginning of the block. In
            // non-strict mode, instead, it's hoisted to the top of the enclosing function/script,
            // but the assignment still isn't! This can be considered a form of the
            // implmentation-defined "strange behavior" described by MDN for block-scoped function
            // declarations [1].
            //
            // [1] https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/function#description.
            fnb.add_decl(Decl {
                name: name.clone(),
                init: DeclInit::Undefined,
                is_hoisted: match fnb.strict_mode() {
                    StrictMode::Strict => false,
                    StrictMode::Sloppy => true,
                },
                // At the toplevel scope of a script or a function, function declarations can
                // conflict (like `var`) In a block, they can NOT conflict (like
                // let/const).
                is_conflicting: (fnb.blocks_depth() > 1),
            });

            let closure = compile_fn_as_expr(fnb, &fn_decl.function);
            fnb.assign_fn(name, closure);
        }
        swc_ecma_ast::Decl::Var(var_decl) => compile_var_decl(fnb, var_decl),
        _ => {
            unsupported_node!(decl)
        }
    }
}

fn compile_var_decl(fnb: &mut FnBuilder, var_decl: &swc_ecma_ast::VarDecl) {
    for declarator in &var_decl.decls {
        let name = compile_name_pat(&declarator.name);
        fnb.add_decl(Decl::from_js_var_decl(name.clone(), var_decl.kind));

        if let Some(init) = &declarator.init {
            let value = compile_expr(fnb, init);
            fnb.add_stmt(StmtOp::Assign(Some(name), value));
        }
    }
}

fn compile_expr(fnb: &mut FnBuilder, expr: &swc_ecma_ast::Expr) -> ExprID {
    fnb.with_span(expr.span(), |fnb| {
        match expr {
            swc_ecma_ast::Expr::This(_) => fnb.add_expr(Expr::This),
            swc_ecma_ast::Expr::Array(array_expr) => {
                let array_init = fnb.add_expr(Expr::ArrayCreate);
                let (array_tmp, array) = create_tmp(fnb, array_init);

                for value in &array_expr.elems {
                    if let Some(expr_or_spread) = value {
                        if expr_or_spread.spread.is_some() {
                            panic!("unsupported: spread syntax in array literal");
                        }
                        let value = compile_expr(fnb, &expr_or_spread.expr);
                        fnb.add_stmt(StmtOp::ArrayPush(array_tmp.clone(), value));
                    } else {
                        let t = tracing::section("compile_expr");
                        t.log("unsupported_array_expr", &format!("{:?}", array_expr.span));
                        // todo!("What does `None` mean here? {:?}", array_expr.span)
                    }
                }

                array
            }
            swc_ecma_ast::Expr::Object(object_expr) => {
                let init_value = fnb.add_expr(Expr::ObjectCreate);
                let (_, obj) = create_tmp(fnb, init_value);

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
                                    let value = compile_fn_as_expr(fnb, &method_prop.function);
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
                compile_fn_expr(fnb, fn_expr)
            }
            swc_ecma_ast::Expr::Arrow(arrow_expr) => {
                // let span = arrow_expr.span;

                let params: Vec<_> = arrow_expr.params.iter().cloned().map(From::from).collect();

                let (strict_mode, body_res) = {
                    let outer_strict_mode = fnb.strict_mode();
                    let builder = fnb.suspend();
                    let mut fnb = builder.new_function(outer_strict_mode);

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

                    let inner_strict_mode = fnb.strict_mode();
                    let body = fnb.build();
                    (inner_strict_mode, body)
                };

                let body = match body_res {
                    Ok(body) => body,
                    Err(multi_err) => {
                        fnb.signal_multi_error(multi_err);
                        return fnb.add_expr(Expr::Error);
                    },
                };

                let func = compile_function_from_parts(arrow_expr.span, &params, strict_mode, body);

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
                fnb.add_stmt(StmtOp::Assign(Some(loc), new_value));

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
                    fnb.add_stmt(StmtOp::Assign(Some(loc.clone()), value));
                    // Important. If anybody wants to use the value of the expression,
                    // they should read the varable we just assigned (an operation that
                    // has no side effects) instead of evaluating the rhs expr again
                    // (which may contain a call!)
                    fnb.add_expr(Expr::Read(loc))
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
                fnb.add_stmt(StmtOp::IfNot { test });
                let if_not_jmp = fnb.add_stmt(StmtOp::Pending);

                let value_cons = compile_expr(fnb, &cond_expr.cons);
                fnb.add_stmt(StmtOp::Assign(Some(tmp.clone()), value_cons));
                let after_cons_jmp = fnb.add_stmt(StmtOp::Pending);

                let else_target = fnb.peek_stmt_id();
                let value_alt = compile_expr(fnb, &cond_expr.alt);
                fnb.add_stmt(StmtOp::Assign(Some(tmp.clone()), value_alt));

                let end = fnb.peek_stmt_id();

                fnb.set_stmt(after_cons_jmp, StmtOp::Jump(end));
                fnb.set_stmt(if_not_jmp, StmtOp::Jump(else_target));

                fnb.add_expr(Expr::Read(tmp))
            }
            swc_ecma_ast::Expr::Call(call_expr) => {
                let callee = match &call_expr.callee {
                    swc_ecma_ast::Callee::Super(_) | swc_ecma_ast::Callee::Import(_) => {
                        unsupported_node!(call_expr.callee)
                    }
                    swc_ecma_ast::Callee::Expr(expr) => expr,
                };
                compile_call(fnb, callee, &call_expr.args)
            }
            swc_ecma_ast::Expr::New(new_expr) => {
                let args = new_expr.args.as_deref().unwrap_or(&[]);
                compile_new(fnb, &new_expr.callee, args)
            }
            swc_ecma_ast::Expr::Seq(seq_expr) => {
                if let Some((last, non_last)) = seq_expr.exprs.split_last() {
                    for expr in non_last {
                         let expr_id = compile_expr(fnb, expr);
                        fnb.add_stmt(StmtOp::Assign(None, expr_id));
                    }

                    let last_value = compile_expr(fnb, last);
                    fnb.add_stmt(StmtOp::Assign(fnb.completion_value_var().cloned(), last_value));
                    last_value
                } else {
                    fnb.add_expr(Expr::Undefined)
                }
            }
            swc_ecma_ast::Expr::Ident(ident) => {
                match ident.sym.as_ref() {
                    "undefined" => fnb.add_expr(Expr::Undefined),
                    _ => fnb.add_expr(Expr::Read(DeclName::Js(ident.sym.clone())))
                }
            }
            swc_ecma_ast::Expr::Lit(lit) => match lit {
                swc_ecma_ast::Lit::Str(str) => {
                    fnb.add_expr(Expr::StringLiteral(str.value.to_string().into()))
                }
                swc_ecma_ast::Lit::Bool(b) => fnb.add_expr(Expr::BoolLiteral(b.value)),
                swc_ecma_ast::Lit::Null(_) => fnb.add_expr(Expr::Null),
                swc_ecma_ast::Lit::Num(num) => fnb.add_expr(Expr::NumberLiteral(num.value)),
                swc_ecma_ast::Lit::Regex(re) => {
                    fnb.add_expr(Expr::RegexLiteral {
                        pattern: re.exp.to_string(),
                        flags: re.flags.to_string(),
                    })
                },

                swc_ecma_ast::Lit::BigInt(_)
                | swc_ecma_ast::Lit::JSXText(_) => unsupported_node!(lit),
            },

            swc_ecma_ast::Expr::Tpl(tpl) => {
                assert_eq!(tpl.quasis.len(), tpl.exprs.len() + 1);

                let create_str = fnb.add_expr(Expr::StringCreate);
                let (tmp_var, read_tmp) = create_tmp(fnb, create_str);

                for (quasi, expr) in tpl.quasis.iter().zip(tpl.exprs.iter()) {
                    let quasi = fnb.add_expr(Expr::StringLiteral(quasi.raw.to_string().into()));
                    fnb.add_stmt(StmtOp::StrAppend(tmp_var.clone(), quasi));

                    let value = compile_expr(fnb, expr);
                    fnb.add_stmt(StmtOp::StrAppend(tmp_var.clone(), value));
                }

                let last_quasi = tpl.quasis.last().unwrap().raw.to_string();
                let quasi = fnb.add_expr(Expr::StringLiteral(last_quasi.into()));
                fnb.add_stmt(StmtOp::StrAppend(tmp_var, quasi));

                read_tmp
            },
            swc_ecma_ast::Expr::Paren(paren_expr) => compile_expr(fnb, &paren_expr.expr),

            swc_ecma_ast::Expr::Yield(yield_expr) => {
		if yield_expr.delegate {
		    // `yield*` currently unsupported
		    unsupported_node!(expr);
		}

		let expr = match &yield_expr.arg {
		    Some(arg) => compile_expr(fnb, &*arg),
		    None => fnb.add_expr(Expr::Undefined),
		};


		let set_resume = fnb.add_stmt(StmtOp::Pending);
		compile_yield(fnb, expr, YieldDone::No);

		let after_yield = fnb.peek_stmt_id();
		fnb.set_stmt(set_resume, StmtOp::SetResumePoint(after_yield));

		// TODO Implement sending value
		fnb.add_expr(Expr::Undefined)
	    },


            swc_ecma_ast::Expr::SuperProp(_)
            | swc_ecma_ast::Expr::TaggedTpl(_)
            | swc_ecma_ast::Expr::Class(_)
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
    })
}

fn compile_fn_expr(fnb: &mut FnBuilder, fn_expr: &swc_ecma_ast::FnExpr) -> ExprID {
    compile_fn_as_expr(fnb, &fn_expr.function)
}

fn compile_fn_as_expr(fnb: &mut FnBuilder<'_>, func_ast: &swc_ecma_ast::Function) -> ExprID {
    let res = {
        let strict_mode = fnb.strict_mode();
        let builder = fnb.suspend();
        compile_function(builder, func_ast, strict_mode)
    };
    let func = match res {
        Ok(func) => func,
        Err(multi_err) => {
            fnb.signal_multi_error(multi_err);
            return fnb.add_expr(Expr::Error);
        }
    };
    fnb.add_expr(Expr::CreateClosure {
        func: Box::new(func),
    })
}

fn compile_member_assignment(
    fnb: &mut FnBuilder,
    assign_expr: &swc_ecma_ast::AssignExpr,
    member_expr: &swc_ecma_ast::MemberExpr,
) -> ExprID {
    let key = compile_object_key(fnb, &member_expr.prop);
    let obj = compile_expr(fnb, &member_expr.obj);

    let (_, key) = create_tmp(fnb, key);
    let (_, obj) = create_tmp(fnb, obj);

    let init_value = fnb.add_expr(Expr::ObjectGet { obj, key });
    let value = compile_assignment(fnb, assign_expr, init_value);
    fnb.add_stmt(StmtOp::ObjectSet { obj, key, value });

    value
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
    fnb.add_stmt(StmtOp::Assign(Some(tmp.clone()), value));
    let expr = fnb.add_expr(Expr::Read(tmp.clone()));
    (tmp, expr)
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
) -> ExprID {
    // We special-case the eval('string literal') syntax to remove the call to eval, because
    // several tests in test262 rely on it, and this is simpler than alternatives.
    //
    // Any form of 'eval' that doesn't match these if's trickles through and gets compiled as
    // a regular function call, where it will fail at the PAST stage.
    if let swc_ecma_ast::Expr::Ident(ident) = callee {
        if &ident.sym == "eval" {
            if let Some(arg) = args.first() {
                if let Some(swc_ecma_ast::Lit::Str(str_lit)) = arg.expr.as_lit() {
                    return compile_eval_literal_arg(fnb, str_lit.value.to_string());
                }
            }
        }
    }

    let callee = compile_expr(fnb, callee);
    let args = compile_args(fnb, args);
    fnb.add_expr(Expr::Call { callee, args })
}

fn compile_eval_literal_arg(fnb: &mut FnBuilder, src: String) -> ExprID {
    use swc_common::SourceMap;
    use swc_ecma_ast::EsVersion;
    use swc_ecma_parser::{lexer::Lexer, Parser, Syntax};

    let source_map = Rc::new(SourceMap::default());
    let err_handler = crate::bytecode_compiler::mk_error_handler(&source_map);

    let filename = swc_common::FileName::Anon;

    let source_file = source_map.new_source_file(filename, src);
    let input = swc_ecma_parser::StringInput::from(source_file.as_ref());

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2015,
        input,
        None,
    );
    let mut parser = Parser::new_from(lexer);

    // error management not particularly refined here, but we don't care about eval
    // that much
    let swc_ast = match parser.parse_script() {
        Ok(swc_ast) => swc_ast,
        Err(err) => {
            err.into_diagnostic(&err_handler).emit();
            panic!("parse error");
        }
    };

    let (_, read_cv) = compile_stmt_get_completion(fnb, |fnb| {
        for stmt in &swc_ast.body {
            compile_stmt(fnb, stmt);
        }
    });

    read_cv
}

fn compile_stmt_get_completion(
    fnb: &mut FnBuilder,
    action: impl FnOnce(&mut FnBuilder),
) -> (DeclName, ExprID) {
    let init = fnb.add_expr(Expr::Undefined);
    let (cv_name, read_cv) = create_tmp(fnb, init);

    fnb.enable_completion_value_var(cv_name.clone());
    action(fnb);
    fnb.disable_completion_value_var();

    (cv_name, read_cv)
}

fn compile_new(
    fnb: &mut FnBuilder,
    callee: &swc_ecma_ast::Expr,
    args: &[swc_ecma_ast::ExprOrSpread],
) -> ExprID {
    let callee = compile_expr(fnb, callee);
    let args = compile_args(fnb, args);
    fnb.add_expr(Expr::New {
        constructor: callee,
        args,
    })
}

fn compile_args(fnb: &mut FnBuilder, args: &[swc_ecma_ast::ExprOrSpread]) -> Vec<ExprID> {
    args.iter()
        .map(|expr_or_spread| {
            if expr_or_spread.spread.is_some() {
                panic!("unsupported: spread syntax in call");
            }
            compile_expr(fnb, &expr_or_spread.expr)
        })
        .collect()
}

fn compile_function(
    builder: &mut Builder,
    swc_func: &swc_ecma_ast::Function,
    initial_strict_mode: StrictMode,
) -> MultiErrResult<Function> {
    if !swc_func.decorators.is_empty() {
        panic!("unsupported: function decorators");
    }
    if swc_func.is_async {
        panic!("unsupported: async functions");
    }
    if swc_func.return_type.is_some() || swc_func.type_params.is_some() {
        eprintln!("bytecode_compiler: warning: discarding types in function literal");
    }

    let stmts = &swc_func.body.as_ref().unwrap().stmts;

    if swc_func.is_generator {
        // we generate somewhat augmented PAST like this (using pseudo-syntax for the
        // SetResumePoint primitive).  this is the function we're compiling:
        //
        //   function* makeGenerator() {
        //       for (let i=0; i < 5; ++i) {
        // 	  yield i * 2;
        //       }
        //   }
        //
        // and this is what we want to translate it into:
        //
        //   function makeGenerator() {
        //     return {
        //       next() {
        //         for (let i=0; i < 5; ++i) {
        //           $resumeAfter {
        //             return { value: (i * 2), done: false };
        //           }
        //         }
        //         return { value: undefined, done: true };
        //       }
        //     }
        //   }
        //
        // $resumeAfter takes a snapshot of the current stack frame and writes it into the
        // current closure so that, the next time it is called it does not start with a
        // fresh new stack frame, but by restoring the snapshot. This way, the function
        // will resume execution right at the end of $resumeAfter, with all of its state
        // restored.
        //
        // to get the result above, some tweaks are also made to the compilation of
        // `return` statements.

        let next_fn = {
            let mut fnb = builder.new_function(initial_strict_mode);
            fnb.set_is_generator_next();

            // statments go into 'next()'; yield statements are going to be compiled by
            // `compile_yield` which *expects* to be in the inner function!
            for stmt in stmts {
                compile_stmt(&mut fnb, stmt);
            }

            // append an artificial `return undefined` statement to the function, so that
            // even if the function doesn't end with a `return` statement, we can stil
            // guarantee that if the function terminates, it does so with a `{ done: true
            // }` item
            let undef = fnb.add_expr(Expr::Undefined);
            compile_yield(&mut fnb, undef, YieldDone::Yes);

            let strict_mode = fnb.strict_mode();
            let next_fn_body = fnb.build()?;
            compile_function_from_parts(swc_func.span, &[], strict_mode, next_fn_body)
        };

        let mut fnb = builder.new_function(initial_strict_mode);
        let next_fn = fnb.add_expr(Expr::CreateClosure {
            func: Box::new(next_fn),
        });
        let iterator_wrapper = fnb.add_expr(Expr::ObjectCreate);
        let (_, iterator_wrapper) = create_tmp(&mut fnb, iterator_wrapper);
        let key = fnb.add_expr(Expr::StringLiteral("next".into()));
        fnb.add_stmt(StmtOp::ObjectSet {
            obj: iterator_wrapper,
            key,
            value: next_fn,
        });
        fnb.add_stmt(StmtOp::Return(iterator_wrapper));

        let strict_mode = fnb.strict_mode();
        let body = fnb.build()?;
        let func = compile_function_from_parts(swc_func.span, &swc_func.params, strict_mode, body);
        Ok(func)
    } else {
        let mut fnb = builder.new_function(initial_strict_mode);
        for stmt in stmts {
            compile_stmt(&mut fnb, stmt);
        }
        let strict_mode = fnb.strict_mode();
        let body = fnb.build()?;
        let func = compile_function_from_parts(swc_func.span, &swc_func.params, strict_mode, body);
        Ok(func)
    }
}

fn compile_function_from_parts(
    span: swc_common::Span,
    swc_params: &[swc_ecma_ast::Param],
    strict_mode: StrictMode,
    body: Block,
) -> Function {
    let mut parameters = Vec::new();
    for param in swc_params.iter() {
        let name = compile_name_pat(&param.pat);
        let name = name.expect_js_word();
        parameters.push(name.clone());
    }

    let unbound_names = find_unbound_references(&body, &parameters);
    Function {
        parameters,
        unbound_names,
        body,
        span,
        strict_mode,
    }
}

fn find_unbound_references(root: &Block, param_names: &[JsWord]) -> Vec<JsWord> {
    type Names = HashSet<JsWord>;

    fn process_block(block: &Block) -> Names {
        let mut referenced = Names::new();

        for expr in &block.exprs {
            match expr {
                Expr::Read(DeclName::Js(name)) => {
                    referenced.insert(name.clone());
                }
                Expr::CreateClosure { func } => {
                    for name in &func.unbound_names {
                        referenced.insert(name.clone());
                    }
                }
                _ => {}
            }
        }

        for stmt in &block.stmts {
            match &stmt.op {
                StmtOp::Assign(Some(DeclName::Js(name)), _) => {
                    referenced.insert(name.clone());
                }
                StmtOp::Block(block) => {
                    referenced.extend(process_block(block).into_iter());
                }
                _ => {}
            }
        }

        for decl in &block.decls {
            if let DeclName::Js(js_name) = &decl.name {
                referenced.remove(js_name);
            }
        }

        referenced
    }

    let mut referenced = process_block(root);

    for js_name in param_names {
        referenced.remove(js_name);
    }

    #[allow(unused_mut)]
    let mut unbound: Vec<_> = referenced.into_iter().collect();

    // Helps with insta tests
    #[cfg(test)]
    unbound.sort();

    unbound
}

#[cfg(test)]
mod tests {
    use crate::util::DumpExt;

    #[test]
    fn test_simple_for() {
        let function = quick_compile(
            r#"
                let closures = []
                for (let i=0; closures.push(() => i), i < 5; ++i) {}
                "#
            .to_string(),
        );
        // TODO fix: the seq expr (closures.push(() => i), i < 5) is compiled to a block, but the
        // block does not 'return' its value as an expression in the parent block
        insta::assert_snapshot!(function.dump_to_string());
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
        insta::assert_snapshot!(function.dump_to_string());
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
        insta::assert_snapshot!(function.dump_to_string());
    }

    #[test]
    #[should_panic]
    fn test_function_in_non_allowed_position() {
        const CODE: &str = r#""use strict"; do function g() {} while (false)"#;
        quick_compile(CODE.to_string());
    }
    #[test]
    fn test_function_in_allowed_position() {
        const CODE: &str = r#""use strict"; do { function g() {} } while (false)"#;
        quick_compile(CODE.to_string());
    }

    #[test]
    fn test_func_decl() {
        let src = "function myFunction() { return 3 }".to_string();
        let function = quick_compile(src);
        insta::assert_snapshot!(function.dump_to_string());
    }

    #[test]
    #[should_panic]
    fn test_redecl_let() {
        const CODE: &str = "let i = 1; let i = 3;";
        quick_compile(CODE.to_string());
    }
    #[test]
    fn test_redecl_let_negative() {
        const CODE: &str = "let i = 1; let j = 3;";
        quick_compile(CODE.to_string());
    }
    #[test]
    #[should_panic]
    fn test_redecl_const() {
        const CODE: &str = "const i = 1; const i = 3;";
        quick_compile(CODE.to_string());
    }
    #[test]
    fn test_redecl_const_negative() {
        const CODE: &str = "const i = 1; const j = 3;";
        quick_compile(CODE.to_string());
    }
    #[test]
    #[should_panic]
    fn test_redecl_switch() {
        const CODE: &str = "switch(0) { case 1: const i = 1; default: const i = 3; }";
        quick_compile(CODE.to_string());
    }
    #[test]
    fn test_redecl_switch_negative() {
        const CODE: &str = "switch(0) { case 1: const i = 1; default: const j = 3; }";
        quick_compile(CODE.to_string());
    }

    fn quick_compile(src: String) -> super::Function {
        let (swc_ast, source_map) = crate::bytecode_compiler::quick_parse_script(src);

        super::compile_script(&swc_ast, source_map).unwrap()
    }
}
