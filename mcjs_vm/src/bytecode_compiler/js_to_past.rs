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

#[derive(Debug)]
pub enum StmtOp {
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
    Return(Box<Stmt>),

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
    let block = compile_block(&mut builder, None, &stmts);
    let mut function = compile_function_from_parts(&mut builder, ast_module.span, &[], block);

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

fn compile_block(
    builder: &mut Builder,
    outer_block: Option<&mut Block>,
    swc_stmts: &[swc_ecma_ast::Stmt],
) -> Block {
    let block_id = builder.gen_block_id();
    let mut block = Block {
        id: block_id,
        decls: Vec::new(),
        stmts: Vec::new(),
    };

    for stmt in swc_stmts {
        compile_stmt(builder, &mut block, stmt);
    }

    if let Some(outer_block) = outer_block {
        hoist_declarations(outer_block, &mut block.decls);
    }
    block
}

fn compile_block_single_stmt(
    builder: &mut Builder,
    outer_block: Option<&mut Block>,
    swc_stmt: &swc_ecma_ast::Stmt,
) -> Block {
    let block_id = builder.gen_block_id();
    let mut block = Block {
        id: block_id,
        decls: Vec::new(),
        stmts: Vec::new(),
    };

    compile_stmt(builder, &mut block, swc_stmt);
    if let Some(outer_block) = outer_block {
        hoist_declarations(outer_block, &mut block.decls);
    }
    block
}

fn compile_stmt(builder: &mut Builder, block: &mut Block, stmt: &swc_ecma_ast::Stmt) {
    match &stmt {
        swc_ecma_ast::Stmt::Block(block_stmt) => {
            let inner_block = compile_block(builder, Some(block), &block_stmt.stmts);
            block.stmts.push(Stmt {
                span: block_stmt.span,
                op: StmtOp::Block(inner_block),
            });
        }
        swc_ecma_ast::Stmt::Empty(_) => {}
        swc_ecma_ast::Stmt::Debugger(debugger_stmt) => {
            block.stmts.push(Stmt {
                op: StmtOp::Debugger,
                span: debugger_stmt.span,
            });
        }

        swc_ecma_ast::Stmt::Return(return_stmt) => {
            let expr = return_stmt
                .arg
                .as_ref()
                .map(|expr| compile_expr(builder, &expr))
                .unwrap_or(Stmt {
                    op: StmtOp::Undefined,
                    span: return_stmt.span,
                });
            block.stmts.push(Stmt {
                op: StmtOp::Return(Box::new(expr)),
                span: return_stmt.span,
            });
        }

        swc_ecma_ast::Stmt::Break(break_stmt) => {
            if break_stmt.label.is_some() {
                panic!("unsupported: labeled break statement");
            }
            let break_target = builder.break_target();
            block.stmts.push(Stmt {
                op: StmtOp::Break(break_target),
                span: break_stmt.span,
            });
        }
        swc_ecma_ast::Stmt::Continue(continue_stmt) => {
            if continue_stmt.label.is_some() {
                panic!("unsupported: labeled continue statement");
            }
            let break_target = builder.break_target();
            block.stmts.push(Stmt {
                op: StmtOp::Continue(break_target),
                span: continue_stmt.span,
            });
        }

        swc_ecma_ast::Stmt::If(if_stmt) => {
            let test = compile_expr(builder, &if_stmt.test);

            // TODO Avoid these ephemeral Vec's

            let block_cons = compile_block_single_stmt(builder, Some(block), &if_stmt.cons);

            let block_alt = if let Some(alt) = &if_stmt.alt {
                compile_block_single_stmt(builder, Some(block), alt)
            } else {
                Block::empty(builder.gen_block_id())
            };

            block.stmts.push(Stmt {
                op: StmtOp::If {
                    test: Box::new(test),
                    cons: Box::new(Stmt {
                        op: StmtOp::Block(block_cons),
                        span: Default::default(),
                    }),
                    alt: Box::new(Stmt {
                        op: StmtOp::Block(block_alt),
                        span: Default::default(),
                    }),
                },
                span: if_stmt.span,
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
            block.stmts.push(Stmt {
                op: StmtOp::Throw(Box::new(value)),
                span: throw_stmt.span,
            });
        }
        swc_ecma_ast::Stmt::Try(try_stmt) => {
            let main_block = Stmt {
                op: StmtOp::Block(compile_block(builder, Some(block), &try_stmt.block.stmts)),
                span: try_stmt.block.span,
            };

            let finalizer_block = try_stmt
                .finalizer
                .as_ref()
                .map(|block_stmt| Stmt {
                    op: StmtOp::Block(compile_block(builder, Some(block), &block_stmt.stmts)),
                    span: block_stmt.span,
                })
                .unwrap_or(Stmt {
                    op: StmtOp::Undefined,
                    span: swc_common::Span::default(),
                });

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
                        stmts.push(Stmt {
                            op: StmtOp::Assign(
                                name,
                                Box::new(Stmt {
                                    op: StmtOp::GetCurrentException,
                                    span: swc_common::Span::default(),
                                }),
                            ),
                            span: swc_common::Span::default(),
                        });
                    }

                    // The catch clause is converted into two nested blocks:
                    //  ... catch (exc) { handler_block; }
                    //  => ... { exc = currentException(); { handler_block; } }
                    //  - outer block only defines the exception var;
                    //  - inner block corresponds to the catch clause in the source code
                    let handler_block = compile_block(builder, Some(block), &handler.body.stmts);
                    stmts.push(Stmt {
                        op: StmtOp::Block(handler_block),
                        span: handler.body.span,
                    });

                    Stmt {
                        op: StmtOp::Block(Block {
                            id: handler_block_id,
                            decls,
                            stmts,
                        }),
                        span: handler.body.span,
                    }
                } else {
                    Stmt {
                        op: StmtOp::Undefined,
                        span: swc_common::Span::default(),
                    }
                }
            };

            block.stmts.push(Stmt {
                op: StmtOp::Try {
                    main_block: Box::new(main_block),
                    handler_block: Box::new(handler_block),
                    finalizer_block: Box::new(finalizer_block),
                },
                span: try_stmt.span,
            })
        }
        swc_ecma_ast::Stmt::While(while_stmt) => {
            let mut inner_block = Block::empty(builder.gen_block_id());

            let test_expr = compile_expr(builder, &while_stmt.test);
            let test_expr_span = test_expr.span;
            let test_expr = Stmt {
                op: StmtOp::Unary(swc_ecma_ast::UnaryOp::Bang, Box::new(test_expr)),
                span: test_expr_span,
            };
            inner_block.stmts.push(Stmt {
                op: StmtOp::If {
                    test: Box::new(test_expr),
                    cons: Box::new(Stmt {
                        op: StmtOp::Break(inner_block.id),
                        span: swc_common::Span::default(),
                    }),
                    alt: Box::new(Stmt {
                        op: StmtOp::Undefined,
                        span: swc_common::Span::default(),
                    }),
                },
                span: swc_common::Span::default(),
            });

            compile_stmt(builder, &mut inner_block, &while_stmt.body);

            inner_block.stmts.push(Stmt {
                op: StmtOp::Continue(inner_block.id),
                span: swc_common::Span::default(),
            });

            block.stmts.push(Stmt {
                op: StmtOp::Block(inner_block),
                span: swc_common::Span::default(),
            });
        }
        swc_ecma_ast::Stmt::DoWhile(dowhile_stmt) => {
            let mut inner_block = Block::empty(builder.gen_block_id());

            compile_stmt(builder, &mut inner_block, &dowhile_stmt.body);

            let test_expr = compile_expr(builder, &dowhile_stmt.test);
            let test_expr_span = test_expr.span;
            let test_expr = Stmt {
                op: StmtOp::Unary(swc_ecma_ast::UnaryOp::Bang, Box::new(test_expr)),
                span: test_expr_span,
            };
            inner_block.stmts.push(Stmt {
                op: StmtOp::If {
                    test: Box::new(test_expr),
                    cons: Box::new(Stmt {
                        op: StmtOp::Continue(inner_block.id),
                        span: swc_common::Span::default(),
                    }),
                    alt: Box::new(Stmt {
                        op: StmtOp::Undefined,
                        span: swc_common::Span::default(),
                    }),
                },
                span: swc_common::Span::default(),
            });

            block.stmts.push(Stmt {
                op: StmtOp::Block(inner_block),
                span: dowhile_stmt.span,
            });
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
                None => {}
            }

            let mut inner_block = Block::empty(builder.gen_block_id());

            inner_block.stmts.push(Stmt {
                op: StmtOp::If {
                    test: Box::new(
                        for_stmt
                            .test
                            .as_ref()
                            .map(|expr| compile_expr(builder, &expr))
                            .unwrap_or_else(|| Stmt {
                                op: StmtOp::BoolLiteral(true),
                                span: for_stmt.span,
                            }),
                    ),
                    cons: Box::new(Stmt {
                        op: StmtOp::Undefined,
                        span: swc_common::Span::default(),
                    }),
                    alt: Box::new(Stmt {
                        op: StmtOp::Break(outer_block.id),
                        span: swc_common::Span::default(),
                    }),
                },
                span: for_stmt.span,
            });

            compile_stmt(builder, &mut inner_block, &for_stmt.body);
            if let Some(update) = &for_stmt.update {
                inner_block.stmts.push(compile_expr(builder, update));
            }
            inner_block.stmts.push(Stmt {
                op: StmtOp::Unshare(outer_block.id),
                span: swc_common::Span::default(),
            });
            inner_block.stmts.push(Stmt {
                op: StmtOp::Continue(inner_block.id),
                span: swc_common::Span::default(),
            });

            outer_block.stmts.push(Stmt {
                op: StmtOp::Block(inner_block),
                span: for_stmt.span,
            });
            block.stmts.push(Stmt {
                op: StmtOp::Block(outer_block),
                span: for_stmt.span,
            });
        }

        swc_ecma_ast::Stmt::ForIn(_) => todo!(),
        swc_ecma_ast::Stmt::ForOf(_) => todo!(),

        swc_ecma_ast::Stmt::Decl(decl) => match decl {
            swc_ecma_ast::Decl::Fn(fn_decl) => {
                let name = DeclName::Js(fn_decl.ident.sym.clone());
                block.decls.push(Decl {
                    name: name.clone(),
                    is_lexical: false,
                    is_global: false,
                });
                let func = compile_function(builder, &fn_decl.function);
                block.stmts.push(Stmt {
                    op: StmtOp::Assign(
                        name,
                        Box::new(Stmt {
                            op: StmtOp::CreateClosure {
                                func: Box::new(func),
                            },
                            span: fn_decl.function.span,
                        }),
                    ),
                    span: fn_decl.function.span,
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

fn compile_var_decl(builder: &mut Builder, block: &mut Block, var_decl: &swc_ecma_ast::VarDecl) {
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
            block.stmts.push(Stmt {
                op: StmtOp::Assign(name, Box::new(value)),
                span: declarator.span,
            });
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
        swc_ecma_ast::Expr::This(this_expr) => Stmt {
            op: StmtOp::This,
            span: this_expr.span,
        },
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

            stmts.push(Stmt {
                op: StmtOp::Assign(
                    tmp.clone(),
                    Box::new(Stmt {
                        op: StmtOp::ArrayCreate,
                        span: Default::default(),
                    }),
                ),
                span: Default::default(),
            });
            for value in &array_expr.elems {
                // TODO What does `None` mean here?
                if let Some(expr_or_spread) = value {
                    if expr_or_spread.spread.is_some() {
                        panic!("unsupported: spread syntax in array literal");
                    }

                    let value = compile_expr(builder, &expr_or_spread.expr);
                    stmts.push(Stmt {
                        op: StmtOp::ArrayPush(tmp.clone(), Box::new(value)),
                        span: Default::default(),
                    });
                }
            }
            stmts.push(Stmt {
                op: StmtOp::Read(tmp),
                span: Default::default(),
            });

            Stmt {
                op: StmtOp::Block(Block {
                    id: block_id,
                    decls,
                    stmts,
                }),
                span: array_expr.span,
            }
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

            stmts.push(Stmt {
                op: StmtOp::Assign(
                    tmp.clone(),
                    Box::new(Stmt {
                        op: StmtOp::ObjectCreate,
                        span: Default::default(),
                    }),
                ),
                span: Default::default(),
            });

            for prop in &object_expr.props {
                match prop {
                    swc_ecma_ast::PropOrSpread::Spread(_) => {
                        panic!("unsupported: spread syntax in object literal");
                    }
                    swc_ecma_ast::PropOrSpread::Prop(prop) => {
                        let (key, value) = match prop.as_ref() {
                            swc_ecma_ast::Prop::Shorthand(name) => {
                                let key = Stmt {
                                    op: StmtOp::StringLiteral(name.sym.clone()),
                                    span: name.span,
                                };
                                let value = Stmt {
                                    op: StmtOp::Read(DeclName::Js(name.sym.clone())),
                                    span: name.span,
                                };
                                (key, value)
                            }
                            swc_ecma_ast::Prop::KeyValue(kv) => {
                                let key = match &kv.key {
                                    swc_ecma_ast::PropName::Ident(ident) => Stmt {
                                        op: StmtOp::StringLiteral(JsWord::from(
                                            ident.sym.to_string(),
                                        )),
                                        span: ident.span,
                                    },
                                    swc_ecma_ast::PropName::Str(str) => Stmt {
                                        op: StmtOp::StringLiteral(JsWord::from(
                                            str.value.to_string(),
                                        )),
                                        span: str.span,
                                    },
                                    swc_ecma_ast::PropName::Num(num) => Stmt {
                                        op: StmtOp::NumberLiteral(num.value),
                                        span: num.span,
                                    },
                                    _ => {
                                        unsupported_node!(kv.key)
                                    }
                                };
                                let value = compile_expr(builder, &kv.value);

                                (key, value)
                            }

                            swc_ecma_ast::Prop::Method(method_prop) => {
                                let name = method_prop.key.as_ident().expect("object literal: method property syntax, but name is not identifier?");
                                let key = Stmt {
                                    span: name.span,
                                    op: StmtOp::StringLiteral(name.sym.clone()),
                                };
                                let func =
                                    Box::new(compile_function(builder, &method_prop.function));
                                let value = Stmt {
                                    span: func.span,
                                    op: StmtOp::CreateClosure { func },
                                };
                                (key, value)
                            }

                            swc_ecma_ast::Prop::Assign(_)
                            | swc_ecma_ast::Prop::Getter(_)
                            | swc_ecma_ast::Prop::Setter(_) => todo!(),
                        };

                        stmts.push(Stmt {
                            op: StmtOp::ObjectSet {
                                obj: Box::new(Stmt {
                                    op: StmtOp::Read(tmp.clone()),
                                    span: Default::default(),
                                }),
                                key: Box::new(key),
                                value: Box::new(value),
                            },
                            span: object_expr.span,
                        });
                    }
                }
            }

            stmts.push(Stmt {
                op: StmtOp::Read(tmp),
                span: Default::default(),
            });

            Stmt {
                op: StmtOp::Block(Block {
                    id: block_id,
                    decls,
                    stmts,
                }),
                span: object_expr.span,
            }
        }

        swc_ecma_ast::Expr::Fn(fn_expr) => {
            let func = compile_function(builder, &fn_expr.function);
            Stmt {
                op: StmtOp::CreateClosure {
                    func: Box::new(func),
                },
                span: fn_expr.function.span,
            }
        }
        swc_ecma_ast::Expr::Arrow(arrow_expr) => {
            let span = arrow_expr.span;

            let params: Vec<_> = arrow_expr.params.iter().cloned().map(From::from).collect();
            let body = match &*arrow_expr.body {
                swc_ecma_ast::BlockStmtOrExpr::BlockStmt(block_stmts) => {
                    compile_block(builder, None, &block_stmts.stmts)
                }
                swc_ecma_ast::BlockStmtOrExpr::Expr(expr) => {
                    let mut block = Block::empty(builder.gen_block_id());
                    block.stmts.push(compile_expr(builder, expr));
                    block
                }
            };
            let func = compile_function_from_parts(builder, arrow_expr.span, &params, body);
            let func_expr = Stmt {
                op: StmtOp::CreateClosure {
                    func: Box::new(func),
                },
                span,
            };

            // Unlike regular function declarations/expressions, arrow
            // expressions inherit the `this` binding from the surrounding
            // context

            let bind_method = Stmt {
                op: StmtOp::ObjectGet {
                    obj: Box::new(func_expr),
                    key: Box::new(Stmt {
                        span,
                        op: StmtOp::StringLiteral("bind".into()),
                    }),
                },
                span,
            };

            let bind_call = Stmt {
                op: StmtOp::Call {
                    is_new: false,
                    callee: Box::new(bind_method),
                    args: vec![Stmt {
                        op: StmtOp::This,
                        span,
                    }],
                },
                span,
            };

            bind_call
        }
        swc_ecma_ast::Expr::Unary(unary_expr) => {
            let arg = compile_expr(builder, &unary_expr.arg);
            Stmt {
                op: StmtOp::Unary(unary_expr.op, Box::new(arg)),
                span: unary_expr.span,
            }
        }
        swc_ecma_ast::Expr::Update(update_expr) => {
            let loc = compile_name(swc_ecma_ast::PatOrExpr::Expr(update_expr.arg.clone()));
            let new_value = {
                let value = Box::new(compile_expr(builder, &update_expr.arg));
                let one = Box::new(Stmt {
                    op: StmtOp::NumberLiteral(1.0),
                    span: Default::default(),
                });
                match update_expr.op {
                    swc_ecma_ast::UpdateOp::PlusPlus => Stmt {
                        op: StmtOp::Binary(swc_ecma_ast::BinaryOp::Add, value, one),
                        span: update_expr.span,
                    },
                    swc_ecma_ast::UpdateOp::MinusMinus => Stmt {
                        op: StmtOp::Binary(swc_ecma_ast::BinaryOp::Sub, value, one),
                        span: update_expr.span,
                    },
                }
            };
            Stmt {
                op: StmtOp::Assign(loc, Box::new(new_value)),
                span: update_expr.span,
            }
        }
        swc_ecma_ast::Expr::Bin(bin_expr) => {
            let left = compile_expr(builder, &bin_expr.left);
            let right = compile_expr(builder, &bin_expr.right);
            Stmt {
                op: StmtOp::Binary(bin_expr.op, Box::new(left), Box::new(right)),
                span: bin_expr.span,
            }
        }
        swc_ecma_ast::Expr::Assign(assign_expr) => {
            if let Some(ident) = assign_expr.left.as_ident() {
                let loc = DeclName::Js(ident.sym.clone());
                let init_value = Stmt {
                    op: StmtOp::Read(loc.clone()),
                    span: ident.span,
                };
                let value = compile_assignment(builder, assign_expr, init_value);
                Stmt {
                    op: StmtOp::Assign(loc, Box::new(value)),
                    span: assign_expr.span,
                }
            } else if let Some(target_expr) = assign_expr.left.as_expr() {
                match target_expr {
                    swc_ecma_ast::Expr::Member(member_expr) => {
                        compile_member_assignment(builder, assign_expr, member_expr)
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
            let obj = compile_expr(builder, &member_expr.obj);
            let key = compile_object_key(builder, &member_expr.prop);
            Stmt {
                op: StmtOp::ObjectGet {
                    obj: Box::new(obj),
                    key: Box::new(key),
                },
                span: member_expr.span,
            }
        }
        swc_ecma_ast::Expr::Cond(cond_expr) => {
            let test = Box::new(compile_expr(builder, &cond_expr.test));
            let cons = Box::new(compile_expr(builder, &cond_expr.cons));
            let alt = Box::new(compile_expr(builder, &cond_expr.alt));
            Stmt {
                op: StmtOp::If { test, cons, alt },
                span: cond_expr.span,
            }
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
            Stmt {
                op: StmtOp::Block(Block {
                    id: block_id,
                    stmts,
                    decls: Vec::new(),
                }),
                span: seq_expr.span,
            }
        }
        swc_ecma_ast::Expr::Ident(ident) => Stmt {
            op: StmtOp::Read(DeclName::Js(ident.sym.clone())),
            span: ident.span,
        },
        swc_ecma_ast::Expr::Lit(lit) => match lit {
            swc_ecma_ast::Lit::Str(str) => Stmt {
                op: StmtOp::StringLiteral(str.value.to_string().into()),
                span: str.span,
            },
            swc_ecma_ast::Lit::Bool(b) => Stmt {
                op: StmtOp::BoolLiteral(b.value),
                span: b.span,
            },
            swc_ecma_ast::Lit::Null(null_node) => Stmt {
                op: StmtOp::Null,
                span: null_node.span,
            },
            swc_ecma_ast::Lit::Num(num) => Stmt {
                op: StmtOp::NumberLiteral(num.value),
                span: num.span,
            },
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

    let init_value = Stmt {
        op: StmtOp::ObjectGet {
            obj: Box::new(Stmt {
                op: StmtOp::Read(tmp_obj.clone()),
                span: Default::default(),
            }),
            key: Box::new(Stmt {
                op: StmtOp::Read(tmp_key.clone()),
                span: Default::default(),
            }),
        },
        span: member_expr.span,
    };

    let value = compile_assignment(builder, assign_expr, init_value);
    block.stmts.push(Stmt {
        op: StmtOp::ObjectSet {
            obj: Box::new(Stmt {
                op: StmtOp::Read(tmp_obj),
                span: Default::default(),
            }),
            key: Box::new(Stmt {
                op: StmtOp::Read(tmp_key),
                span: Default::default(),
            }),
            value: Box::new(value),
        },
        span: assign_expr.span,
    });

    Stmt {
        op: StmtOp::Block(block),
        span: assign_expr.span,
    }
}

fn compile_assignment(
    builder: &mut Builder,
    assign_expr: &swc_ecma_ast::AssignExpr,
    init_value: Stmt,
) -> Stmt {
    let rhs = compile_expr(builder, &assign_expr.right);
    let value = match assign_expr.op.to_update() {
        // regular assignment
        None => rhs,
        // updating assignment
        Some(binop) => Stmt {
            op: StmtOp::Binary(binop, Box::new(init_value), Box::new(rhs)),
            span: assign_expr.span,
        },
    };
    value
}

fn create_tmp(builder: &mut Builder, block: &mut Block, value: Stmt) -> DeclName {
    let tmp = DeclName::Tmp(builder.gen_tmp());
    block.decls.push(Decl {
        is_lexical: true,
        name: tmp.clone(),
        is_global: false,
    });
    block.stmts.push(Stmt {
        op: StmtOp::Assign(tmp.clone(), Box::new(value)),
        span: Default::default(),
    });
    tmp
}

fn compile_object_key(builder: &mut Builder, prop: &swc_ecma_ast::MemberProp) -> Stmt {
    match prop {
        swc_ecma_ast::MemberProp::Ident(ident) => Stmt {
            op: StmtOp::StringLiteral(ident.sym.clone()),
            span: ident.span,
        },
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

    let span = callee.span;
    Stmt {
        op: StmtOp::Call {
            is_new,
            callee,
            args,
        },
        span,
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

    let stmts = &swc_func.body.as_ref().unwrap().stmts;
    let body = compile_block(builder, None, stmts);
    compile_function_from_parts(builder, swc_func.span, &swc_func.params, body)
}

fn compile_function_from_parts(
    builder: &mut Builder,
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

    let declares_use_strict = match body.stmts.first() {
        Some(Stmt {
            op: StmtOp::StringLiteral(atom),
            ..
        }) => atom == "use strict",
        _ => false,
    };

    let unbound_names = find_unbound_references(&body);
    Function {
        parameters,
        unbound_names,
        declares_use_strict,
        body,
        span,
    }
}

fn visit_stmts(root: &Stmt, mut handler: impl FnMut(&Stmt)) {
    let mut queue = vec![root];

    while let Some(stmt) = queue.pop() {
        handler(stmt);
        match &stmt.op {
            StmtOp::Block(block) => {
                queue.extend(block.stmts.iter());
            }
            StmtOp::Break(_) => {}
            StmtOp::Continue(_) => {}
            StmtOp::Unshare(_) => {}
            StmtOp::If { test, cons, alt } => {
                queue.push(test);
                queue.push(cons);
                queue.push(alt);
            }
            StmtOp::Undefined => {}
            StmtOp::Null => {}
            StmtOp::This => {}
            StmtOp::Read(_) => {}
            StmtOp::AssignParam(_, _) => {}
            StmtOp::Assign(_, value_stmt) => {
                queue.push(value_stmt);
            }
            StmtOp::Unary(_, arg) => {
                queue.push(arg);
            }
            StmtOp::Binary(_, left, right) => {
                queue.push(left);
                queue.push(right);
            }
            StmtOp::StringLiteral(_) => {}
            StmtOp::NumberLiteral(_) => {}
            StmtOp::BoolLiteral(_) => {}
            StmtOp::ArrayCreate => {}
            StmtOp::ArrayPush(_, value_stmt) => {
                queue.push(value_stmt);
            }
            StmtOp::ObjectCreate => {}
            StmtOp::ObjectGet { obj, key } => {
                queue.push(obj);
                queue.push(key);
            }
            StmtOp::ObjectSet { obj, key, value } => {
                queue.push(obj);
                queue.push(key);
                queue.push(value);
            }
            StmtOp::CreateClosure { .. } => {}
            StmtOp::Call { callee, args, .. } => {
                queue.push(callee.as_ref());
                queue.extend(args.iter());
            }
            StmtOp::Return(arg) => {
                queue.push(arg.as_ref());
            }
            StmtOp::Throw(arg) => {
                queue.push(arg);
            }
            StmtOp::GetCurrentException => {}
            StmtOp::Try {
                main_block,
                handler_block,
                finalizer_block,
            } => {
                queue.push(main_block.as_ref());
                queue.push(handler_block.as_ref());
                queue.push(finalizer_block.as_ref());
            }
            StmtOp::Debugger => {}
        }
    }
}

fn visit_stmts_mut(root: &mut Stmt, mut handler: impl FnMut(&mut Stmt)) {
    let mut queue = vec![root];

    while let Some(stmt) = queue.pop() {
        handler(stmt);
        match &mut stmt.op {
            StmtOp::Block(block) => {
                queue.extend(block.stmts.iter_mut());
            }
            StmtOp::Break(_) => {}
            StmtOp::Continue(_) => {}
            StmtOp::Unshare(_) => {}
            StmtOp::If { test, cons, alt } => {
                queue.push(test);
                queue.push(cons);
                queue.push(alt);
            }
            StmtOp::Undefined => {}
            StmtOp::Null => {}
            StmtOp::This => {}
            StmtOp::Read(_) => {}
            StmtOp::AssignParam(_, _) => {}
            StmtOp::Assign(_, value_stmt) => {
                queue.push(value_stmt);
            }
            StmtOp::Unary(_, arg) => {
                queue.push(arg);
            }
            StmtOp::Binary(_, left, right) => {
                queue.push(left);
                queue.push(right);
            }
            StmtOp::StringLiteral(_) => {}
            StmtOp::NumberLiteral(_) => {}
            StmtOp::BoolLiteral(_) => {}
            StmtOp::ArrayCreate => {}
            StmtOp::ArrayPush(_, value_stmt) => {
                queue.push(value_stmt);
            }
            StmtOp::ObjectCreate => {}
            StmtOp::ObjectGet { obj, key } => {
                queue.push(obj);
                queue.push(key);
            }
            StmtOp::ObjectSet { obj, key, value } => {
                queue.push(obj);
                queue.push(key);
                queue.push(value);
            }
            StmtOp::CreateClosure { .. } => {}
            StmtOp::Call { callee, args, .. } => {
                queue.push(callee.as_mut());
                queue.extend(args.iter_mut());
            }
            StmtOp::Return(arg) => {
                queue.push(arg.as_mut());
            }
            StmtOp::Throw(arg) => {
                queue.push(arg);
            }
            StmtOp::GetCurrentException => {}
            StmtOp::Try {
                main_block,
                handler_block,
                finalizer_block,
            } => {
                queue.push(main_block.as_mut());
                queue.push(handler_block.as_mut());
                queue.push(finalizer_block.as_mut());
            }
            StmtOp::Debugger => {}
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
        visit_stmts(block_stmt, |stmt| match &stmt.op {
            StmtOp::Block(block) => {
                add_block_decls(block);
            }
            StmtOp::Read(DeclName::Js(js_name)) => {
                referenced.insert(js_name.clone());
            }
            StmtOp::CreateClosure { func } => {
                for name in &func.unbound_names {
                    referenced.insert(name.clone());
                }
            }
            StmtOp::Assign(DeclName::Js(name), _) => {
                referenced.insert(name.clone());
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
