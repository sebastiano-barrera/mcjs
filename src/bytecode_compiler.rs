use std::collections::HashMap;

use swc_atoms::JsWord;
use swc_ecma_ast::{BinaryOp, Decl, Lit};

use crate::common::{Error, Result};
use crate::interpreter::{self, ArithOp, CmpOp, Instr, Operand, Value, IID};

pub fn compile_file(filename: String, content: String) -> Result<interpreter::Module> {
    let ast_module = parse_file(filename, content)?;
    compile_module(&ast_module)
}

struct Builder {
    scopes: Vec<Scope>,
    instrs: Vec<Instr>,
}

impl Builder {
    fn new() -> Self {
        Builder {
            scopes: vec![Scope::new()],
            instrs: vec![],
        }
    }

    fn build(self) -> interpreter::Module {
        interpreter::Module::new(self.instrs)
    }

    fn reserve(&mut self) -> IID {
        self.emit(Instr::Nop)
    }

    fn emit(&mut self, instr: Instr) -> IID {
        let iid = self.peek_iid();
        self.instrs.push(instr);
        iid
    }

    fn get_mut(&mut self, iid: IID) -> Option<&mut Instr> {
        let IID(ndx) = iid;
        self.instrs.get_mut(ndx as usize)
    }

    fn define_var(&mut self, name: JsWord, var: Var) {
        self.scopes
            .last_mut()
            .expect("no scopes!")
            .define(name, var);
    }

    fn peek_iid(&self) -> IID {
        IID(self.instrs.len() as u32)
    }

    fn get_var(&self, sym: &JsWord) -> Option<&Var> {
        // Lookup sym in each scope, inner to outer
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get_var(sym))
    }
}

struct Scope {
    vars: HashMap<JsWord, Var>,
}
impl Scope {
    fn new() -> Self {
        Scope {
            vars: HashMap::new(),
        }
    }

    fn define(&mut self, name: JsWord, var: Var) {
        if self.vars.contains_key(&name) {
            panic!(
                "definition of var `{}` shadows previous definition (compiler limitation)",
                name
            );
        }
        self.vars.insert(name, var);
    }

    fn get_var(&self, sym: &JsWord) -> Option<&Var> {
        self.vars.get(sym)
    }
}

struct Var {
    iid: IID,
    is_const: bool,
}

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
}

fn compile_module(ast_module: &swc_ecma_ast::Module) -> Result<interpreter::Module> {
    use swc_ecma_ast::{ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    let mut builder = Builder::new();

    for item in &ast_module.body {
        match item {
            // ModuleItem::ModuleDecl(decl) => match decl {
            //     ModuleDecl::Import(_) => todo!(),
            //     ModuleDecl::ExportDecl(_) => todo!(),
            //     ModuleDecl::ExportNamed(_) => todo!(),
            //     ModuleDecl::ExportDefaultDecl(_) => todo!(),
            //     ModuleDecl::ExportDefaultExpr(_) => todo!(),
            //     ModuleDecl::ExportAll(_) => todo!(),
            //     ModuleDecl::TsImportEquals(_) => todo!(),
            //     ModuleDecl::TsExportAssignment(_) => todo!(),
            //     ModuleDecl::TsNamespaceExport(_) => todo!(),
            // },
            ModuleItem::Stmt(stmt) => {
                compile_stmt(&mut builder, stmt)?;
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
            for stmt in &block.stmts {
                compile_stmt(builder, stmt)?;
            }
        }
        // Stmt::Empty(_) => todo!(),
        // Stmt::Debugger(_) => todo!(),
        // Stmt::With(_) => todo!(),
        // Stmt::Return(_) => todo!(),
        // Stmt::Labeled(_) => todo!(),
        // Stmt::Break(_) => todo!(),
        // Stmt::Continue(_) => todo!(),
        // Stmt::Switch(_) => todo!(),
        // Stmt::Throw(_) => todo!(),
        // Stmt::Try(_) => todo!(),
        // Stmt::While(_) => todo!(),
        // Stmt::DoWhile(_) => todo!(),
        // Stmt::For(_) => todo!(),
        // Stmt::ForIn(_) => todo!(),
        // Stmt::ForOf(_) => todo!(),
        Stmt::Decl(decl) => match decl {
            // Decl::Class(_) => todo!(),
            // Decl::Fn(_) => todo!(),
            Decl::Var(var_decl) => {
                compile_var_decl(builder, var_decl)?;
            }
            Decl::TsInterface(_) | Decl::TsTypeAlias(_) | Decl::TsEnum(_) | Decl::TsModule(_) => {
                panic!("TypeScript syntax not supported (for now!)")
            }
            other => unsupported_node!(other),
        },
        Stmt::Expr(expr) => {
            compile_expr(builder, &expr.expr)?;
        }

        Stmt::If(if_stmt) => {
            let condition = compile_expr(builder, &if_stmt.test)?;

            if let Some(else_blk) = &if_stmt.alt {
                // with an else block:
                //      jmpif (cond) -> CONS
                //          < if false >
                //      jmp END
                // CONS     < if true >
                // END      < rest... >

                let jmpif = builder.reserve();
                compile_stmt(builder, &else_blk)?;
                let jmp_end = builder.reserve();
                let if_true_start: IID = builder.peek_iid();
                compile_stmt(builder, &*if_stmt.cons)?;
                let end: IID = builder.peek_iid();

                *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                    cond: condition.into(),
                    dest: if_true_start,
                };
                *builder.get_mut(jmp_end).unwrap() = Instr::Jmp(end);

                // TODO builder should fail if any reserved instruction is left untouched
            } else {
                // without an else block:
                //      jmpif (not cond) -> END
                //          < if true >
                // END      < rest... >

                let neg_condition = builder.emit(Instr::Not(condition.into()));
                let jmpif = builder.reserve();
                compile_stmt(builder, &*if_stmt.cons);
                let end: IID = builder.peek_iid();

                *builder.get_mut(jmpif).unwrap() = Instr::JmpIf {
                    cond: neg_condition.into(),
                    dest: end,
                };
            }
        }
        other => unsupported_node!(other),
    }

    Ok(())
}

fn compile_var_decl(builder: &mut Builder, var_decl: &swc_ecma_ast::VarDecl) -> Result<()> {
    use swc_ecma_ast::VarDeclKind;

    let is_const = match var_decl.kind {
        VarDeclKind::Var => panic!("limitation: `var` bindings not supported"),
        VarDeclKind::Let => false,
        VarDeclKind::Const => true,
    };

    for decl in &var_decl.decls {
        let iid = match &decl.init {
            Some(expr) => compile_expr(builder, expr)?,
            None => builder.emit(Instr::Const(Value::Undefined)),
        };

        if let Some(ident) = decl.name.as_ident() {
            let name: JsWord = ident.id.to_id().0;
            let var = Var { iid, is_const };
            builder.define_var(name, var);
        } else {
            unsupported_node!(decl)
        }
    }

    Ok(())
}

fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Result<IID> {
    use swc_ecma_ast::{CallExpr, Expr};

    match expr {
        Expr::Call(CallExpr { callee, args, .. }) => {
            if let Some(callee) = callee.as_expr() {
                if let Some(callee) = callee.as_ident() {
                    if callee.sym.as_ref() == "sink" {
                        for arg in args {
                            let value = compile_expr(builder, &arg.expr)?.into();
                            builder.emit(Instr::PushSink(value));
                        }

                        return Ok(builder.emit(Instr::Const(Value::Null)));
                    }
                }
            }

            Err(Error::UnsupportedItem)
        }

        Expr::Bin(bin_expr) => {
            let a = compile_expr(builder, &bin_expr.left)?.into();
            let b = compile_expr(builder, &bin_expr.right)?.into();

            let arith = |op, a, b| Instr::Arith { a, b, op };
            let cmp = |op, a, b| Instr::Cmp { a, b, op };
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
                _ => panic!("unsupported binary op: {:?}", bin_expr.op),
            };

            Ok(builder.emit(instr))
        }

        Expr::Lit(lit) => match lit {
            Lit::Num(number) => {
                let result = builder.emit(Instr::Const(Value::Number(number.value)));
                Ok(result)
            }
            Lit::Str(s) => Ok(builder.emit(Instr::Const(Value::String(s.value.to_string())))),
            // Lit::Bool(_) => todo!(),
            // Lit::Null(_) => todo!(),
            // Lit::BigInt(_) => todo!(),
            // Lit::Regex(_) => todo!(),
            // Lit::JSXText(_) => todo!(),
            other => unsupported_node!(other),
        },

        Expr::Ident(ident) => match builder.get_var(&ident.sym) {
            Some(var) => Ok(var.iid),
            None => Err(Error::UnboundVariable(ident.sym.to_string())),
        },

        Expr::Assign(asmt) => {
            let ident = asmt
                .left
                .as_ident()
                .expect("compiler limitation: assignment is only supported to identifiers");
            let var_id = builder
                .get_var(&ident.sym)
                .ok_or_else(|| Error::UnboundVariable(ident.to_string()))?
                .iid;

            let value = compile_expr(builder, &asmt.right)?.into();

            Ok(builder.emit(Instr::Set { var_id, value }))
        }

        // Expr::This(_) => todo!(),
        // Expr::Array(_) => todo!(),
        // Expr::Object(_) => todo!(),
        // Expr::Fn(_) => todo!(),
        // Expr::Unary(_) => todo!(),
        // Expr::Update(_) => todo!(),
        // Expr::Member(_) => todo!(),
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
        // Expr::Paren(_) => todo!(),
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

fn parse_file(filename: String, content: String) -> Result<swc_ecma_ast::Module> {
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
        Ok(ast_module) => Ok(ast_module),
        Err(e) => {
            e.into_diagnostic(&err_handler).emit();
            Err(Error::ParseError)
        }
    }
}
