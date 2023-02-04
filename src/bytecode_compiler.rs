use std::collections::HashMap;

use swc_atoms::JsWord;
use swc_common::{sync::Lrc, SourceMap};
use swc_ecma_ast::{AssignOp, BinaryOp, Decl, Function, Lit, Pat, UpdateOp};

use crate::common::{Error, Result};
use crate::interpreter::{self, ArithOp, CmpOp, FnId, Instr, Operand, Value, IID};

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
}

fn err_unsupported_node(span: swc_common::Span) -> Error {
    Error::UnsupportedItem {
        span,
        details: "unsupported syntax!",
    }
}

pub struct Compiler {
    builtins: HashMap<JsWord, Operand>,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            builtins: HashMap::new(),
        }
    }

    pub fn bind_native(&mut self, ident: JsWord, operand: Operand) {
        self.builtins.insert(ident, operand);
    }

    pub fn compile_file(
        &mut self,
        filename: String,
        content: String,
    ) -> Result<interpreter::Module> {
        let (source_map, ast_module) = parse_file(filename, content)?;
        let builder = Builder::new(self.builtins.clone());
        let res = compile_module(builder, &ast_module);
        if let Err(err) = &res {
            eprintln!("error: {}", err.message(&*source_map));
        }

        res
    }
}

struct Builder {
    fns: HashMap<FnId, FnBuilder>,
    fn_stack: Vec<FnBuilder>,
    next_fnid: u32,
}

struct FnBuilder {
    fnid: FnId,
    scopes: Vec<Scope>,
    instrs: Vec<Instr>,
}
impl FnBuilder {
    fn new(id: FnId) -> Self {
        FnBuilder {
            fnid: id,
            scopes: vec![Scope::new()],
            instrs: Vec::new(),
        }
    }
    fn build(self) -> interpreter::Function {
        interpreter::Function::new(self.instrs.into_boxed_slice())
    }

    fn peek_iid(&self) -> IID {
        IID(self.instrs.len() as u32)
    }
}

impl Builder {
    fn new(builtins: HashMap<JsWord, Operand>) -> Self {
        let next_fnid = 1;
        assert_ne!(FnId(next_fnid), FnId::ROOT_FN);

        let mut root_fn = FnBuilder::new(FnId::ROOT_FN);
        let outermost_scope = root_fn.scopes.last_mut().unwrap();
        for (name, operand) in builtins.into_iter() {
            let var = Var {
                is_const: true,
                operand,
            };
            outermost_scope.define(name.clone(), var);
        }

        Builder {
            fns: HashMap::new(),
            fn_stack: vec![root_fn],
            next_fnid,
        }
    }

    fn build(mut self) -> interpreter::Module {
        self.end_function();
        assert!(self.fn_stack.is_empty());

        let fns = self
            .fns
            .drain()
            .map(|(fn_id, fn_builder)| (fn_id, fn_builder.build()))
            .collect();
        interpreter::Module::new(fns)
    }

    fn reserve(&mut self) -> IID {
        self.emit(Instr::Nop)
    }

    fn cur_fnb(&mut self) -> &mut FnBuilder {
        self.fn_stack.last_mut().expect("no FnBuilder!")
    }

    fn emit(&mut self, instr: Instr) -> IID {
        let fnb = self.cur_fnb();
        let iid = fnb.peek_iid();
        fnb.instrs.push(instr);
        iid
    }

    fn peek_iid(&mut self) -> IID {
        self.cur_fnb().peek_iid()
    }

    fn get_mut(&mut self, iid: IID) -> Option<&mut Instr> {
        self.cur_fnb().instrs.get_mut(iid.0 as usize)
    }

    fn define_var(&mut self, name: JsWord, var: Var) {
        // let depth = self.cur_fnb().scopes.len() - 1;
        // eprintln!("define_var {name} [{depth}]");
        self.cur_fnb()
            .scopes
            .last_mut()
            .expect("no scopes!")
            .define(name, var);
    }

    fn get_var(&mut self, sym: &JsWord) -> Option<&Var> {
        // Lookup sym in each scope, inner to outer

        self.fn_stack
            .iter()
            .rev()
            .find_map(|fnb| fnb.scopes.iter().rev().find_map(|scope| scope.get_var(sym)))
    }

    fn start_function(&mut self, name: Option<JsWord>, params: &[swc_ecma_ast::Param]) {
        let fnid = FnId(self.next_fnid);
        self.next_fnid += 1;

        self.fn_stack.push(FnBuilder::new(fnid));

        if let Some(name) = name {
            self.define_var(
                name,
                Var {
                    operand: Value::SelfFunction.into(),
                    is_const: true,
                },
            );
        }

        for (param_ndx, param) in params.iter().enumerate() {
            if !param.decorators.is_empty() {
                panic!("unsupported: decorators on function parameters");
            }

            match &param.pat {
                Pat::Ident(ident) => {
                    let iid = self.emit(Instr::GetArg(param_ndx));
                    self.define_var(
                        ident.sym.clone(),
                        Var {
                            operand: iid.into(),
                            is_const: false,
                        },
                    );
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
}

#[derive(Debug)]
struct Scope {
    // TODO Take advantage of identifier interning!
    vars: HashMap<String, Var>,
}
impl Scope {
    fn new() -> Self {
        Scope {
            vars: HashMap::new(),
        }
    }

    fn define(&mut self, name: JsWord, var: Var) {
        let name = String::from_utf8(name.as_bytes().to_owned())
            .expect("only UTF-8 identifiers are supporeted!");
        if self.vars.contains_key(&name) {
            panic!(
                "definition of var `{}` shadows previous definition (compiler limitation)",
                name
            );
        }
        self.vars.insert(name, var);
    }

    fn get_var(&self, sym: &JsWord) -> Option<&Var> {
        let name = String::from_utf8_lossy(sym.as_bytes());
        self.vars.get(name.as_ref())
    }
}

#[derive(Debug)]
struct Var {
    operand: Operand,
    is_const: bool,
}

fn compile_module(
    mut builder: Builder,
    ast_module: &swc_ecma_ast::Module,
) -> Result<interpreter::Module> {
    use swc_ecma_ast::{ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    let exports_object = builder.emit(Instr::ObjNew).into();
    builder.define_var(
        "module".into(),
        Var {
            is_const: true,
            operand: exports_object,
        },
    );

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
            let neg_condition = builder.emit(Instr::Not(condition.into()));
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
        // Stmt::ForIn(_) => todo!(),
        // Stmt::ForOf(_) => todo!(),
        Stmt::Decl(decl) => compile_decl(builder, decl),

        Stmt::Expr(expr) => {
            compile_expr(builder, &expr.expr)?;
            Ok(())
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
                compile_stmt(builder, else_blk)?;
                let jmp_end = builder.reserve();
                let if_true_start: IID = builder.peek_iid();
                compile_stmt(builder, &if_stmt.cons)?;
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

fn compile_decl(builder: &mut Builder, decl: &Decl) -> Result<()> {
    match decl {
        // Decl::Class(_) => todo!(),
        Decl::Fn(fn_decl) => {
            if fn_decl.declare {
                panic!("unsupported case: fn_decl.declare");
            }

            let (name, _) = fn_decl.ident.to_id();
            let func = &fn_decl.function;

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

            builder.start_function(Some(name.clone()), func.params.as_slice());
            let stmts = &func.body.as_ref().expect("function without body?!").stmts;
            for stmt in stmts {
                compile_stmt(builder, stmt)?;
            }
            let fnid = builder.end_function();

            let iid = builder.emit(Instr::Const(Value::LocalFn(fnid)));
            builder.define_var(
                name,
                Var {
                    operand: iid.into(),
                    is_const: true,
                },
            );
        }

        Decl::Var(var_decl) => {
            compile_var_decl(builder, var_decl)?;
        }

        Decl::TsInterface(_) | Decl::TsTypeAlias(_) | Decl::TsEnum(_) | Decl::TsModule(_) => {
            panic!("TypeScript syntax not supported (for now!)")
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
        let operand = match &decl.init {
            Some(expr) => compile_expr(builder, expr)?,
            None => Value::Undefined.into(),
        };

        if let Some(ident) = decl.name.as_ident() {
            let name: JsWord = ident.id.to_id().0;
            let var = Var { operand, is_const };
            builder.define_var(name, var);
        } else {
            unsupported_node!(decl)
        }
    }

    Ok(())
}

fn compile_expr(builder: &mut Builder, expr: &swc_ecma_ast::Expr) -> Result<Operand> {
    use swc_ecma_ast::{CallExpr, Expr};

    match expr {
        Expr::Call(call_expr @ CallExpr { callee, args, .. }) => {
            if let Some(callee) = callee.as_expr() {
                if let Some(callee) = callee.as_ident() {
                    if callee.sym.as_ref() == "sink" {
                        for arg in args {
                            let value = compile_expr(builder, &arg.expr)?.into();
                            builder.emit(Instr::PushSink(value));
                        }

                        return Ok(builder.emit(Instr::Const(Value::Null)).into());
                    }
                }

                let fn_iid = compile_expr(builder, callee)?;
                let mut args_iids = vec![];
                for arg in args {
                    if arg.spread.is_some() {
                        panic!("unsupported: spread function parameter: function(a, b, ...)");
                    }
                    let iid = compile_expr(builder, &arg.expr)?;
                    args_iids.push(iid.into());
                }
                return Ok(builder
                    .emit(Instr::Call {
                        callee: fn_iid.into(),
                        args: args_iids,
                    })
                    .into());
            }

            Err(Error::UnsupportedItem {
                span: call_expr.span,
                details: "only calls to simple identifiers are supported for now",
            })
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

            Ok(builder.emit(instr).into())
        }

        Expr::Lit(lit) => match lit {
            Lit::Num(number) => {
                let result = builder
                    .emit(Instr::Const(Value::Number(number.value)))
                    .into();
                Ok(result)
            }
            Lit::Str(s) => Ok(builder
                .emit(Instr::Const(Value::String(s.value.to_string())))
                .into()),
            // Lit::Bool(_) => todo!(),
            Lit::Null(_) => Ok(builder.emit(Instr::Const(Value::Null)).into()),
            // Lit::BigInt(_) => todo!(),
            // Lit::Regex(_) => todo!(),
            // Lit::JSXText(_) => todo!(),
            other => unsupported_node!(other),
        },

        Expr::Ident(ident) => get_var(builder, ident),

        Expr::Assign(asmt) => {
            use swc_ecma_ast::{MemberExpr, MemberProp, PatOrExpr};

            let target_operand = if let Some(ident) = asmt.left.as_ident() {
                get_var(builder, ident)?.clone()
            } else if let Some(target_expr) = asmt.left.as_expr() {
                match target_expr {
                    Expr::Member(member_expr) => {
                        let obj = compile_expr(builder, member_expr.obj.as_ref())?;
                        let key = match &member_expr.prop {
                            MemberProp::Ident(prop_ident) => {
                                Operand::Value(Value::String(prop_ident.sym.to_string()))
                            }
                            _ => {
                                return Err(Error::UnsupportedItem {
                                    span: asmt.span,
                                    details: "assignment to an expression is unsupported",
                                })
                            }
                        };
                        builder.emit(Instr::ObjGet { obj, key }).into()
                    }
                    // We should have already handled this case in the `if let ... = asm.left.as_ident()` case
                    Expr::Ident(_) => unreachable!(),
                    _ => {
                        return Err(Error::UnsupportedItem {
                            span: asmt.span,
                            details: "assignment to an expression is unsupported",
                        })
                    }
                }
            } else {
                panic!("unsupported pattern as assignment target: {:?}", asmt.left)
            };

            let var_id = match target_operand {
                // Reuse the variable ID, if there is one
                Operand::IID(iid) => iid,
                Operand::Value(_) => return Err(Error::IllegalAssignment { span: asmt.span }),
            };

            let right = compile_expr(builder, &asmt.right)?.into();

            let value = match asmt.op {
                AssignOp::Assign => right,
                AssignOp::AddAssign => builder
                    .emit(Instr::Arith {
                        op: ArithOp::Add,
                        a: target_operand.clone(),
                        b: right,
                    })
                    .into(),
                AssignOp::SubAssign => builder
                    .emit(Instr::Arith {
                        op: ArithOp::Sub,
                        a: target_operand.clone(),
                        b: right,
                    })
                    .into(),
                AssignOp::MulAssign => builder
                    .emit(Instr::Arith {
                        op: ArithOp::Mul,
                        a: target_operand.clone(),
                        b: right,
                    })
                    .into(),
                AssignOp::DivAssign => builder
                    .emit(Instr::Arith {
                        op: ArithOp::Div,
                        a: target_operand.clone(),
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

            Ok(builder.emit(Instr::Set { var_id, value }).into())
        }

        Expr::Object(obj_expr) => {
            let obj: Operand = builder.emit(Instr::ObjNew).into();

            for prop_or_spread in obj_expr.props.iter() {
                match prop_or_spread {
                    swc_ecma_ast::PropOrSpread::Spread(spread_elm) => {
                        return Err(Error::UnsupportedItem {
                            span: spread_elm.dot3_token,
                            details: "spread syntax (...) is currently unsupported",
                        })
                    }
                    swc_ecma_ast::PropOrSpread::Prop(prop) => match prop.as_ref() {
                        swc_ecma_ast::Prop::KeyValue(kv_expr) => {
                            let value = compile_expr(builder, &kv_expr.value)?;
                            let key = match &kv_expr.key {
                                swc_ecma_ast::PropName::Ident(ident) => {
                                    Value::String(ident.sym.to_string())
                                }
                                swc_ecma_ast::PropName::Str(s) => {
                                    Value::String(s.value.to_string())
                                }
                                swc_ecma_ast::PropName::Num(num) => Value::Number(num.value),
                                swc_ecma_ast::PropName::Computed(x) => {
                                    return Err(err_unsupported_node(x.span))
                                }
                                swc_ecma_ast::PropName::BigInt(x) => {
                                    return Err(err_unsupported_node(x.span))
                                }
                            }
                            .into();

                            builder.emit(Instr::ObjSet {
                                obj: obj.clone(),
                                key,
                                value,
                            });
                        }
                        swc_ecma_ast::Prop::Shorthand(expr) => {
                            return Err(err_unsupported_node(expr.span))
                        }
                        swc_ecma_ast::Prop::Assign(expr) => {
                            return Err(err_unsupported_node(expr.key.span))
                        }
                        swc_ecma_ast::Prop::Getter(expr) => {
                            return Err(err_unsupported_node(expr.span))
                        }
                        swc_ecma_ast::Prop::Setter(expr) => {
                            return Err(err_unsupported_node(expr.span))
                        }
                        swc_ecma_ast::Prop::Method(expr) => {
                            return Err(err_unsupported_node(expr.function.span))
                        }
                    },
                }
            }

            Ok(obj)
        }

        // Expr::This(_) => todo!(),
        // Expr::Array(_) => todo!(),
        Expr::Fn(fn_expr) => {
            // TODO Refactor this with Decl::Fn
            let name = fn_expr.ident.as_ref().map(|ident| ident.to_id().0);
            let func = &fn_expr.function;

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

            builder.start_function(name.clone(), func.params.as_slice());
            let stmts = &func.body.as_ref().expect("function without body?!").stmts;
            for stmt in stmts {
                compile_stmt(builder, stmt)?;
            }
            let fnid = builder.end_function();

            let iid = builder.emit(Instr::Const(Value::LocalFn(fnid)));
            if let Some(name) = name {
                builder.define_var(
                    name,
                    Var {
                        operand: iid.into(),
                        is_const: true,
                    },
                );
            }
            Ok(iid.into())
        }

        Expr::Unary(unary_expr) => {
            match unary_expr.op {
                swc_ecma_ast::UnaryOp::Bang => {
                    let arg = compile_expr(builder, &unary_expr.arg)?;
                    Ok(builder.emit(Instr::Not(arg.into())).into())
                }
                other => unsupported_node!(other),
                // swc_ecma_ast::UnaryOp::Minus => todo!(),
                // swc_ecma_ast::UnaryOp::Plus => todo!(),
                // swc_ecma_ast::UnaryOp::Tilde => todo!(),
                // swc_ecma_ast::UnaryOp::TypeOf => todo!(),
                // swc_ecma_ast::UnaryOp::Void => todo!(),
                // swc_ecma_ast::UnaryOp::Delete => todo!(),
            }
        }
        Expr::Update(update_expr) => {
            if let Expr::Ident(ident) = &*update_expr.arg {
                // NOTE: update_expr.prefix does not matter in this case, but
                // it will matter when this code is extended to other types of args
                let var_id = match get_var(builder, ident)? {
                    Operand::Value(_) => {
                        return Err(Error::IllegalAssignment {
                            span: update_expr.span,
                        })
                    }
                    Operand::IID(iid) => iid,
                };
                let value: Operand = builder
                    .emit(Instr::Arith {
                        op: match update_expr.op {
                            UpdateOp::PlusPlus => ArithOp::Add,
                            UpdateOp::MinusMinus => ArithOp::Sub,
                        },
                        a: var_id.into(),
                        // TODO Use integers here, when they get implemented
                        b: Value::Number(1.0).into(),
                    })
                    .into();
                builder.emit(Instr::Set {
                    var_id,
                    value: value.clone(),
                });
                Ok(value)
            } else {
                todo!("unsupported: UpdateExpr on anything other than an identifier")
            }
        }
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

fn get_var(builder: &mut Builder, ident: &swc_ecma_ast::Ident) -> Result<Operand> {
    match builder.get_var(&ident.sym) {
        Some(var) => Ok(var.operand.clone()),
        None => {
            // eprintln!("unbound variable; scopes info:");
            // for (ndx, fnb) in builder.fn_stack.iter().enumerate() {
            //     eprintln!("  [{ndx}]: {:#?}", fnb.scopes);
            // }

            Err(Error::UnboundVariable {
                span: ident.span,
                ident: ident.sym.to_string(),
            })
        }
    }
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
            Err(Error::ParseError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_compile(code: &str) -> interpreter::Module {
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

        let root_fn = module.functions().get(&FnId::ROOT_FN).unwrap();
        let instrs = &root_fn.instrs();
        let obj_id = instrs
            .iter()
            .find_map(|instr| match instr {
                Instr::PushSink(Operand::IID(iid)) => Some(*iid),
                _ => None,
            })
            .unwrap();
        let keys: Vec<_> = instrs
            .iter()
            .filter_map(|instr| match instr {
                Instr::ObjSet {
                    obj: Operand::IID(iid),
                    key: Operand::Value(key),
                    ..
                } if *iid == obj_id => Some(key.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(
            keys.as_slice(),
            &[
                Value::String("aString".to_owned()),
                Value::String("aNumber".to_owned()),
                Value::String("anotherObject".to_owned()),
                Value::String("aFunction".to_owned()),
            ]
        );
    }
}
