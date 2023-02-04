use std::collections::HashMap;

use swc_atoms::JsWord;
use swc_common::{sync::Lrc, SourceMap, Span};
use swc_ecma_ast::{AssignOp, BinaryOp, Decl, Function, Lit, Pat, UpdateOp};

pub use crate::common::{Context, Error, Result};
use crate::error;
use crate::interpreter::{
    self, ArithOp, CmpOp, FnId, Instr, Operand, StackFrameId, Value, VarId, VarIndex, IID,
};

macro_rules! unsupported_node {
    ($value:expr) => {{
        todo!("unsupported AST node: {:#?}", $value);
    }};
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
        use crate::common::Context;
        let (source_map, ast_module) = parse_file(filename.clone(), content)
            .with_context(error!("while parsing file: {filename}"))?;
        let builder = Builder::new(self.builtins.clone());
        let res = compile_module(builder, &ast_module)
            .with_context(error!("while compiling module: {filename}"));
        if let Err(err) = &res {
            eprintln!("\nbytecode compiler error: {}\n", err.message(&*source_map));
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
    last_index: u16,
}
#[derive(Debug)]
struct Scope {
    // TODO Take advantage of identifier interning!
    vars: HashMap<String, VarIndex>,
}
impl FnBuilder {
    fn new(id: FnId) -> Self {
        FnBuilder {
            fnid: id,
            scopes: vec![Self::new_scope()],
            instrs: Vec::new(),
            last_index: 0,
        }
    }

    fn build(self) -> interpreter::Function {
        let n_slots = self.last_index;
        interpreter::Function::new(self.instrs.into_boxed_slice(), n_slots)
    }

    fn new_scope() -> Scope {
        Scope {
            vars: HashMap::new(),
        }
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

    fn define(&mut self, name: JsWord) -> VarIndex {
        let name = String::from_utf8(name.as_bytes().to_owned())
            .expect("only UTF-8 identifiers are supporeted!");

        let var_index = VarIndex(self.last_index);
        let scope = self.inner_scope_mut();
        if scope.vars.contains_key(&name) {
            panic!(
                "definition of var `{}` shadows previous definition (compiler limitation)",
                name
            );
        }
        scope.vars.insert(name, var_index);

        self.last_index += 1;
        var_index
    }

    fn get_var(&self, sym: &JsWord) -> Option<VarIndex> {
        let name = String::from_utf8_lossy(sym.as_bytes());
        self.inner_scope().vars.get(name.as_ref()).copied()
    }
}

impl Builder {
    fn new(builtins: HashMap<JsWord, Operand>) -> Self {
        let next_fnid = 1;
        assert_ne!(FnId(next_fnid), FnId::ROOT_FN);

        let mut builder = Builder {
            fns: HashMap::new(),
            fn_stack: vec![FnBuilder::new(FnId::ROOT_FN)],
            next_fnid,
        };

        for (name, operand) in builtins.into_iter() {
            let var_id = builder.define_var(name.clone());
            builder.emit(Instr::Set {
                var_id,
                value: operand,
            });
        }

        builder
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

    fn define_var(&mut self, name: JsWord) -> VarId {
        // let depth = self.cur_fnb().scopes.len() - 1;
        // eprintln!("define_var {name} [{depth}]");
        let var_ndx = self.cur_fnb().define(name);
        VarId {
            stack_fid: StackFrameId(0),
            var_ndx,
        }
    }

    fn get_var(&mut self, sym: &JsWord) -> Option<VarId> {
        // Lookup sym in each scope, inner to outer
        self.fn_stack
            .iter()
            .rev()
            .enumerate()
            .find_map(|(stack_frame_offset, fnb)| {
                fnb.get_var(sym).map(|var_ndx| VarId {
                    stack_fid: StackFrameId(stack_frame_offset as u16),
                    var_ndx,
                })
            })
    }

    fn start_function(&mut self, name: Option<JsWord>, params: &[swc_ecma_ast::Param]) {
        let fnid = FnId(self.next_fnid);
        self.next_fnid += 1;

        self.fn_stack.push(FnBuilder::new(fnid));

        if let Some(name) = name {
            let var_id = self.define_var(name);
            self.emit(Instr::Set {
                var_id,
                value: Value::SelfFunction.into(),
            });
        }

        for (param_ndx, param) in params.iter().enumerate() {
            if !param.decorators.is_empty() {
                panic!("unsupported: decorators on function parameters");
            }

            match &param.pat {
                Pat::Ident(ident) => {
                    let iid = self.emit(Instr::GetArg(param_ndx));
                    let var_id = self.define_var(ident.sym.clone());
                    self.emit(Instr::Set {
                        var_id,
                        value: iid.into(),
                    });
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

fn compile_module(
    mut builder: Builder,
    ast_module: &swc_ecma_ast::Module,
) -> Result<interpreter::Module> {
    use swc_ecma_ast::{ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    // Exports object.  TODO Actually make it accessible via `require`!
    {
        let obj = builder.emit(Instr::ObjNew).into();
        let var_id = builder.define_var("module".into());
        builder.emit(Instr::Set { var_id, value: obj });
    }

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
            for stmt in &block.stmts {
                compile_stmt(builder, stmt)
                    .with_context(error!("in block").with_span(block.span))?;
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
                compile_stmt(builder, stmt)
                    .with_context(error!("in function declaration").with_span(func.span))?;
            }
            let fnid = builder.end_function();

            let var_id = builder.define_var(name);
            builder.emit(Instr::Set {
                var_id,
                value: Value::LocalFn(fnid).into(),
            });
        }

        Decl::Var(var_decl) => {
            compile_var_decl(builder, var_decl)
                .with_context(error!("in variable declaration").with_span(var_decl.span))?;
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
            let var_id = builder.define_var(name);
            builder.emit(Instr::Set {
                var_id,
                value: operand,
            });
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

            Err(
                error!("only calls to simple identifiers are supported for now")
                    .with_span(call_expr.span),
            )
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

        Expr::Ident(ident) => {
            let var_id = get_var(builder, ident)?;
            Ok(Operand::Var(var_id))
        }

        Expr::Assign(asmt) => compile_assignment(asmt, builder)
            .with_context(error!("in assignment").with_span(asmt.span)),

        Expr::Object(obj_expr) => {
            let obj: Operand = builder.emit(Instr::ObjNew).into();

            for prop_or_spread in obj_expr.props.iter() {
                match prop_or_spread {
                    swc_ecma_ast::PropOrSpread::Spread(spread_elm) => {
                        return Err(error!("spread syntax (...) is currently unsupported")
                            .with_span(spread_elm.dot3_token))
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
                                    return Err(
                                        error!("unsupported node: {:?}", x).with_span(x.span)
                                    )
                                }
                                swc_ecma_ast::PropName::BigInt(x) => {
                                    return Err(
                                        error!("unsupported node: {:?}", x).with_span(x.span)
                                    )
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
                        swc_ecma_ast::Prop::Method(expr) => {
                            return Err(error!("unsupported node: {:?}", expr)
                                .with_span(expr.function.span));
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
                let var_id = builder.define_var(name);
                builder.emit(Instr::Set {
                    var_id,
                    value: iid.into(),
                });
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
                let var_id = get_var(builder, ident)?;
                let value: Operand = builder
                    .emit(Instr::Arith {
                        op: match update_expr.op {
                            UpdateOp::PlusPlus => ArithOp::Add,
                            UpdateOp::MinusMinus => ArithOp::Sub,
                        },
                        a: Operand::Var(var_id),
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
        Expr::Member(member_expr) => {
            let (_, _, value) = compile_member_access(builder, member_expr)?;
            Ok(value)
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

fn compile_assignment(asmt: &swc_ecma_ast::AssignExpr, builder: &mut Builder) -> Result<Operand> {
    use swc_ecma_ast::{Expr, MemberExpr, MemberProp, PatOrExpr};

    if let Some(ident) = asmt.left.as_ident() {
        let var_id = get_var(builder, ident)?;
        let operand = Operand::Var(var_id);
        let new_value = compile_assignment_rhs(builder, asmt, &operand)?;
        builder.emit(Instr::Set {
            var_id,
            value: new_value,
        });
        Ok(operand)
    } else if let Some(target_expr) = asmt.left.as_expr() {
        match target_expr {
            Expr::Member(member_expr) => {
                let (obj, key, old_value) = compile_member_access(builder, member_expr)?;
                let new_value = compile_assignment_rhs(builder, asmt, &old_value)?;
                builder.emit(Instr::ObjSet {
                    obj,
                    key,
                    value: new_value.clone(),
                });
                Ok(new_value)
            }
            // We should have already handled this case in the `if let ... = asm.left.as_ident()` case
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
) -> Result<(Operand, Operand, Operand)> {
    use swc_ecma_ast::MemberProp;

    let obj = compile_expr(builder, member_expr.obj.as_ref())?;
    let key = match &member_expr.prop {
        MemberProp::Ident(prop_ident) => Operand::Value(Value::String(prop_ident.sym.to_string())),
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
    target_operand: &Operand,
) -> Result<Operand> {
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
    Ok(value)
}

fn get_var(builder: &mut Builder, ident: &swc_ecma_ast::Ident) -> Result<VarId> {
    builder
        .get_var(&ident.sym)
        .ok_or_else(|| error!("unbound variable `{}`", ident.sym).with_span(ident.span))
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
