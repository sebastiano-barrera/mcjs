//! PAST->Bytecode compiler.
//!
//! This is the second and last phase of the bytecode compiler, after the JavaScript->PAST step.
//!
//! It works pretty much as you'd expect, taking full advantage of knowing the full set of
//! declarations of each scope/block.
//!

// ## Error handling
//
// Compilation errors arising in this phase are to be considered fully unexpected, and are handled
// exclusively by panicking. (The panic is only caught at the topmost level in `compile_module` so
// that the entire application does not have to crash.) The motivation for this is twofold.
//
// First, a malformed PAST can come *exclusively* from a bug in the JavaScript->PAST phase, not
// from any mistake made by our user in their JavaScript code. Such user errors should be caught
// and reported by the JavaScript->PAST phase, in which case that phase terminates with an error,
// without producing any PAST.  If a PAST comes out of that phase, then it's supposed to be fully
// correct. There is no point in continuing the compilation in the presence of a certified bug.
//
// Second, interrupting the compilation 'in the middle' would leave our internal data structures in
// a logically invalid state, missing some invariants (details in the `builder` module). There is
// no easy way to prevent the ensuing problems, only hard ones (e.g. putting changes in an
// auxiliary structure to be committed/rolled-back afterwards; copying the FnBuilder, changing only
// one of the two copies and then discarding one of them).  But there is no point: if a compiler
// bug is detected, we might as well throw the entire thing out with just a message.

use std::collections::HashSet;

use swc_atoms::JsWord;

use super::js_to_past::{self, Block, DeclName, Stmt, StmtID, StmtOp};
use super::CompiledModule;

use crate::bytecode::{self, Instr, Literal};
use crate::common::Result;
use crate::{error, tracing};

use builder::{FnBuilder, Loc, ModuleBuilder};

pub fn compile_module(func: &js_to_past::Function, min_lfnid: u16) -> Result<CompiledModule> {
    let res = std::panic::catch_unwind(move || {
        let mut module_builder = builder::ModuleBuilder::new(min_lfnid);

        let globals: HashSet<_> = func.unbound_names.iter().cloned().collect();

        // Similar to `compile_function`, but we have no parameters, no captures,
        // and we define 'globalThis' at the root scope
        let mut fnb = builder::FnBuilder::new(&globals, &mut module_builder);
        fnb.set_strict_mode(func.declares_use_strict);

        let stmts_count = func.body.stmts().count();
        fnb.block(func.body.id, stmts_count, |fnb| {
            let reg = fnb.gen_reg();
            fnb.emit(Instr::GetGlobalThis(reg));
            fnb.define_name(DeclName::Js("globalThis".into()), reg);

            compile_block_internal(fnb, &func.body)
        });

        let root_lfnid = fnb.build(func.span);
        module_builder.build(root_lfnid)
    });

    match res {
        Ok(module) => {
            trace_dump_module(&module);
            Ok(module)
        }
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<String>() {
                message
            } else {
                "<not a string>"
            };
            Err(error!(
                "past->bytecode internal module compilation error: {}",
                message
            ))
        }
    }
}

// TODO Refactor into trace_dump and impl util::Dump for CompiledModule
fn trace_dump_module(module: &CompiledModule) {
    let t = tracing::section("bytecode");
    for (fnid, func) in &module.functions {
        use std::fmt::Write;

        let mut buf = String::new();

        let mode_name = if func.is_strict_mode() {
            "strict"
        } else {
            "sloppy"
        };
        writeln!(buf, "mode: {}", mode_name).unwrap();

        writeln!(buf).unwrap();
        writeln!(buf, "-- consts").unwrap();
        for (ndx, lit) in func.consts().iter().enumerate() {
            writeln!(buf, "  k{:<4} {:?}", ndx, lit).unwrap();
        }

        writeln!(buf, "-- instrs").unwrap();
        for (ndx, instr) in func.instrs().iter().enumerate() {
            writeln!(buf, "  {:4} {:?}", ndx, instr).unwrap();
        }

        t.log(&format!("{:?}", fnid), &buf);
    }
}

fn compile_function<'a>(
    globals: &'a HashSet<JsWord>,
    module_builder: &'a mut ModuleBuilder,
    captures: Vec<DeclName>,
    func: &js_to_past::Function,
    force_strict: bool,
) -> bytecode::LocalFnId {
    let mut fnb = builder::FnBuilder::new(globals, module_builder);
    fnb.set_strict_mode(force_strict || func.declares_use_strict);

    let stmts_count = func.body.stmts().count();
    fnb.block(func.body.id, stmts_count, |fnb| {
        // We currently only support a limited number of arguments.
        // They're placed in the first (bytecode::ARGS_COUNT_MAX) vregs
        // automatically by the interpreter. We just prepare our data structures so
        // that the generated bytecode makes use of this fact.
        for (ndx, name) in func
            .parameters
            .iter()
            .take(bytecode::ARGS_COUNT_MAX as usize)
            .enumerate()
        {
            let reg = bytecode::VReg(ndx.try_into().unwrap());
            let decl_name = DeclName::Js(name.clone());
            fnb.define_name(decl_name, reg);
        }

        for (cap_ndx, cap_decl) in captures.into_iter().enumerate() {
            let cap_ndx = cap_ndx.try_into().unwrap();

            let reg = fnb.gen_reg();
            fnb.emit(Instr::LoadCapture(reg, bytecode::CaptureIndex(cap_ndx)));
            fnb.define_name(cap_decl, reg);
        }

        compile_block_internal(fnb, &func.body)
    });

    fnb.build(func.span)
}

fn compile_one_stmt<'a>(
    fnb: &mut FnBuilder,
    block: &Block,
    stmts: &mut impl Iterator<Item = &'a Stmt>,
) -> bool {
    let stmt = match stmts.next() {
        Some(stmt) => stmt,
        None => return false,
    };

    let iid_start = fnb.peek_iid();
    fnb.mark_stmt_start();

    match &stmt.op {
        StmtOp::Pending => {
            panic!("compiler bug: pending stmt left over from previous stage")
        }
        StmtOp::Break(block_id) => {
            let iid = fnb.reserve_instr();

            let block_id = *block_id;
            fnb.on_function_completion(move |fnb| {
                let (_, iid_end) = fnb.block_boundaries(block_id);
                fnb.set_instr(iid, Instr::Jmp(iid_end));
            });
        }
        StmtOp::Unshare(name) => {
            let loc = fnb.resolve_name(name);
            let reg = match loc {
                builder::Loc::Reg(reg) => reg,
                builder::Loc::Global(_) => {
                    panic!("malformed PAST: Unshared name must be local, not global")
                }
            };

            fnb.emit(Instr::Unshare(reg));
        }
        StmtOp::IfNot { test } => {
            // IfNot can be understood as "skip the next op if <test>"
            let cond = fnb.gen_reg();
            compile_expr(fnb, Some(cond), block, *test);
            let jmpif = fnb.reserve_instr();

            compile_one_stmt(fnb, block, stmts);

            let dest = fnb.peek_iid();
            fnb.set_instr(jmpif, Instr::JmpIf { cond, dest });
        }
        StmtOp::Assign(dest, value_eid) => {
            if let Some(dest) = dest {
                compile_assignment(fnb, dest, block, *value_eid);
            } else {
                // the dest register is then forgotten
                let _ = compile_expr(fnb, None, block, *value_eid);
            }
        }
        StmtOp::ArrayPush(arr, value_eid) => {
            let arr = compile_read(fnb, arr);
            let value = compile_expr(fnb, None, block, *value_eid);
            fnb.emit(Instr::ArrayPush { arr, value });
        }
        StmtOp::ObjectSet { obj, key, value } => {
            let obj_reg = compile_expr(fnb, None, block, *obj);
            let key_reg = compile_expr(fnb, None, block, *key);
            let value_reg = compile_expr(fnb, None, block, *value);
            fnb.emit(Instr::ObjSet {
                obj: obj_reg,
                key: key_reg,
                value: value_reg,
            });
        }
        StmtOp::Return(arg) => {
            let arg_reg = compile_expr(fnb, None, block, *arg);
            fnb.emit(Instr::Return(arg_reg));
        }

        StmtOp::Throw(arg) => {
            let arg_reg = compile_expr(fnb, None, block, *arg);
            fnb.emit(Instr::Throw(arg_reg));
        }
        StmtOp::Debugger => {
            fnb.emit(Instr::Breakpoint);
        }
        StmtOp::Block(child_block) => {
            compile_block(fnb, child_block);
        }
        StmtOp::Jump(target_sid) => {
            let iid = fnb.reserve_instr();
            let target_sid = *target_sid;
            fnb.on_block_completion(move |fnb| {
                let target = fnb.iid_of_stmt(target_sid);
                fnb.set_instr(iid, Instr::Jmp(target));
            });
        }
        StmtOp::TryBegin { handler } => {
            let iid = fnb.reserve_instr();

            let handler = *handler;
            fnb.on_block_completion(move |fnb| {
                let handler_iid = fnb.iid_of_stmt(handler);
                fnb.set_instr(iid, Instr::PushExcHandler(handler_iid));
            });
        }
        StmtOp::TryEnd => {
            fnb.emit(Instr::PopExcHandler);
        }

        StmtOp::StrAppend(buf, chunk) => {
            let buf = compile_read(fnb, buf);
            let chunk = compile_expr(fnb, None, block, *chunk);
            fnb.emit(Instr::StrAppend(buf, chunk));
        }
    };

    if !stmt.span.is_dummy() {
        let iid_end = fnb.peek_iid();
        fnb.add_breakable_range(stmt.span, iid_start, iid_end);
    }

    true
}

fn compile_assignment(
    fnb: &mut FnBuilder,
    var_name: &DeclName,
    block: &Block,
    value_eid: js_to_past::ExprID,
) {
    match fnb.resolve_name(var_name) {
        Loc::Reg(reg) => {
            compile_expr(fnb, Some(reg), block, value_eid);
        }
        Loc::Global(name) => {
            let global_this = fnb.gen_reg();
            fnb.emit(Instr::GetGlobalThis(global_this));

            let key = fnb.gen_reg();
            compile_load_const(fnb, key, Literal::JsWord(name));

            let value = compile_expr(fnb, None, block, value_eid);

            fnb.emit(Instr::ObjSet {
                obj: global_this,
                key,
                value,
            });
        }
    };
}

fn compile_read(fnb: &mut FnBuilder<'_>, name: &DeclName) -> bytecode::VReg {
    match fnb.resolve_name(name) {
        Loc::Reg(reg) => reg,
        Loc::Global(name) => compile_read_global(fnb, name),
    }
}

fn compile_read_global(fnb: &mut FnBuilder<'_>, name_lit: JsWord) -> bytecode::VReg {
    let dest = fnb.gen_reg();
    let name = fnb.add_const(Literal::JsWord(name_lit));
    fnb.emit(Instr::GetGlobal { dest, name });
    dest
}
fn compile_load_const(fnb: &mut FnBuilder, dest: bytecode::VReg, lit: Literal) {
    // Avoid string clone?
    let const_ndx = fnb.add_const(lit);
    fnb.emit(Instr::LoadConst(dest, const_ndx));
}

fn compile_expr(
    fnb: &mut FnBuilder,
    forced_dest: Option<bytecode::VReg>,
    block: &Block,
    expr_id: js_to_past::ExprID,
) -> bytecode::VReg {
    use js_to_past::Expr;

    let get_dest = move |fnb: &mut FnBuilder| forced_dest.unwrap_or_else(|| fnb.gen_reg());

    let expr = block.get_expr(expr_id);
    match expr {
        Expr::Undefined => {
            let dest = get_dest(fnb);
            fnb.emit(Instr::LoadUndefined(dest));
            dest
        }
        Expr::Null => {
            let dest = get_dest(fnb);
            fnb.emit(Instr::LoadNull(dest));
            dest
        }
        Expr::This => {
            let dest = get_dest(fnb);
            fnb.emit(Instr::LoadThis(dest));
            dest
        }
        Expr::Read(DeclName::Js(name)) if name == "Infinity" => {
            let dest = get_dest(fnb);
            compile_load_const(fnb, dest, Literal::Number(f64::INFINITY));
            dest
        }
        Expr::Read(name) => {
            let src = compile_read(fnb, name);
            if let Some(forced_dest) = forced_dest {
                fnb.emit(Instr::Copy {
                    dst: forced_dest,
                    src,
                });
                forced_dest
            } else {
                src
            }
        }
        Expr::Unary(op, arg_eid) => {
            let dest = compile_expr(fnb, forced_dest, block, *arg_eid);
            match op {
                swc_ecma_ast::UnaryOp::Minus => {
                    fnb.emit(Instr::UnaryMinus { dest, arg: dest });
                }
                swc_ecma_ast::UnaryOp::Plus => {}
                swc_ecma_ast::UnaryOp::Bang => {
                    fnb.emit(Instr::BoolNot { dest, arg: dest });
                }
                swc_ecma_ast::UnaryOp::TypeOf => {
                    fnb.emit(Instr::TypeOf { dest, arg: dest });
                }
                swc_ecma_ast::UnaryOp::Tilde
                | swc_ecma_ast::UnaryOp::Void
                | swc_ecma_ast::UnaryOp::Delete => panic!("unsupported unary op: {:?}", op),
            }
            dest
        }
        Expr::Binary(op, left, right) => {
            let dest = get_dest(fnb);
            let left = compile_expr(fnb, None, block, *left);
            let right = compile_expr(fnb, None, block, *right);

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

            fnb.emit(instr);
            dest
        }
        Expr::StringLiteral(lit) => {
            let dest = get_dest(fnb);
            compile_load_const(fnb, dest, bytecode::Literal::String(lit.to_string()));
            dest
        }
        Expr::NumberLiteral(lit) => {
            let dest = get_dest(fnb);
            compile_load_const(fnb, dest, bytecode::Literal::Number(*lit));
            dest
        }
        Expr::BoolLiteral(lit) => {
            let dest = get_dest(fnb);
            compile_load_const(fnb, dest, bytecode::Literal::Bool(*lit));
            dest
        }
        Expr::ArrayCreate => {
            let array = get_dest(fnb);
            let constructor: bytecode::VReg = compile_read_global(fnb, "Array".into());
            compile_new(fnb, array, constructor, &[]);
            array
        }
        Expr::ArrayNth { arr, index } => {
            let arr = compile_expr(fnb, None, block, *arr);
            let index = compile_expr(fnb, None, block, *index);
            let dest = get_dest(fnb);
            fnb.emit(Instr::ArrayNth { dest, arr, index });
            dest
        }
        Expr::ArrayLen(arr) => {
            let arr = compile_expr(fnb, None, block, *arr);
            let dest = get_dest(fnb);
            fnb.emit(Instr::ArrayLen { dest, arr });
            dest
        }
        Expr::ObjectCreate => {
            let obj = get_dest(fnb);
            fnb.emit(Instr::ObjCreateEmpty(obj));
            obj
        }

        Expr::ObjectGet { obj, key } => {
            let obj = compile_expr(fnb, None, block, *obj);
            let key = compile_expr(fnb, None, block, *key);
            let dest = get_dest(fnb);
            fnb.emit(Instr::ObjGet { dest, obj, key });
            dest
        }
        Expr::ObjectGetKeys(obj) => {
            let obj = compile_expr(fnb, None, block, *obj);
            let dest = get_dest(fnb);
            fnb.emit(Instr::ObjGetKeys { dest, obj });
            dest
        }
        Expr::CreateClosure { func } => {
            let mut cap_names = Vec::new();
            let mut cap_regs = Vec::new();

            for name in func.unbound_names.iter() {
                let name = DeclName::Js(name.clone());
                match fnb.resolve_name(&name) {
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

            let force_strict = fnb.is_strict_mode();
            let lfnid = {
                let (globals, module_builder) = fnb.suspend();
                compile_function(globals, module_builder, cap_names, func, force_strict)
            };
            let dest = get_dest(fnb);
            fnb.emit(Instr::ClosureNew {
                dest,
                fnid: lfnid,
                forced_this: None,
            });

            for reg in cap_regs {
                fnb.emit(Instr::ClosureAddCapture(reg));
            }

            {
                let key = fnb.gen_reg();

                // <function_object>.prototype = {}
                compile_load_const(fnb, key, Literal::String("prototype".to_string()));
                let proto = fnb.gen_reg();
                fnb.emit(Instr::ObjCreateEmpty(proto));
                fnb.emit(Instr::ObjSet {
                    obj: dest,
                    key,
                    value: proto,
                });

                // <function_object>.prototype = globalThis.Function
                let ctor = compile_read_global(fnb, "Function".into());
                compile_load_const(fnb, key, Literal::String("constructor".to_string()));
                fnb.emit(Instr::ObjSet {
                    obj: dest,
                    key,
                    value: ctor,
                });
            }

            dest
        }
        Expr::Call {
            callee: callee_eid,
            args,
        } => {
            match block.get_expr(*callee_eid) {
                // Some things expressed in the `f(...)` call syntax are not actually calls to
                // anything, but have a special meaning
                Expr::Read(DeclName::Js(name)) if name == "sink" => {
                    for arg in args {
                        let var = compile_expr(fnb, None, block, *arg);
                        fnb.emit(Instr::PushToSink(var));
                    }

                    let dest = get_dest(fnb);
                    compile_load_const(fnb, dest, Literal::Undefined);
                    dest
                }
                Expr::Read(DeclName::Js(name)) if name == "eval" => {
                    panic!("`eval` not supported")
                }
                callee => {
                    let mut arg_regs = Vec::new();
                    for arg in args {
                        let reg = compile_expr(fnb, None, block, *arg);
                        arg_regs.push(reg);
                    }

                    let (this, callee) = match callee {
                        Expr::ObjectGet { obj, key } => {
                            let this = compile_expr(fnb, None, block, *obj);
                            let key = compile_expr(fnb, None, block, *key);
                            let callee = fnb.gen_reg();
                            fnb.emit(Instr::ObjGet {
                                dest: callee,
                                obj: this,
                                key,
                            });
                            (this, callee)
                        }
                        _ => {
                            let this = fnb.gen_reg();
                            let callee = compile_expr(fnb, None, block, *callee_eid);
                            // TODO 'this substitution'
                            fnb.emit(Instr::LoadUndefined(this));
                            (this, callee)
                        }
                    };

                    let return_value = get_dest(fnb);
                    fnb.emit(Instr::Call {
                        return_value,
                        this,
                        callee,
                    });

                    for reg in arg_regs {
                        fnb.emit(Instr::CallArg(reg));
                    }
                    return_value
                }
            }
        }

        Expr::New { constructor, args } => {
            let constructor = compile_expr(fnb, None, block, *constructor);
            let arg_regs: Vec<_> = args
                .iter()
                .map(|arg| compile_expr(fnb, None, block, *arg))
                .collect();

            let obj = get_dest(fnb);
            compile_new(fnb, obj, constructor, &arg_regs);

            obj
        }

        Expr::CurrentException => {
            let dest = get_dest(fnb);
            fnb.emit(Instr::GetCurrentException(dest));
            dest
        }

        Expr::ImportModule { import_path } => {
            let dest = get_dest(fnb);
            let import_path_reg = fnb.gen_reg();
            compile_load_const(fnb, import_path_reg, Literal::JsWord(import_path.clone()));
            fnb.emit(Instr::ImportModule(dest, import_path_reg));
            dest
        }

        Expr::StringCreate => {
            let dest = get_dest(fnb);
            fnb.emit(Instr::StrCreateEmpty(dest));
            dest
        }

        Expr::RegexLiteral { pattern, flags } => {
            let obj = get_dest(fnb);
            let constructor = compile_read_global(fnb, "RegExp".into());

            let pattern_reg = fnb.gen_reg();
            compile_load_const(fnb, pattern_reg, Literal::String(pattern.clone()));

            let flags_reg = fnb.gen_reg();
            compile_load_const(fnb, flags_reg, Literal::String(flags.clone()));

            compile_new(fnb, obj, constructor, &[pattern_reg, flags_reg]);

            obj
        }
        Expr::Error => panic!("malformed PAST: Expr::Error left behind by previous phase"),
    }
}

fn compile_new(
    fnb: &mut FnBuilder,
    obj: bytecode::VReg,
    constructor: bytecode::VReg,
    arg_regs: &[bytecode::VReg],
) {
    fnb.emit(Instr::ObjCreateEmpty(obj));

    {
        // return value is discarded
        let return_value = fnb.gen_reg();
        fnb.emit(Instr::Call {
            return_value,
            this: obj,
            callee: constructor,
        });
    }

    for reg in arg_regs {
        fnb.emit(Instr::CallArg(*reg));
    }

    let key = fnb.gen_reg();

    let prototype = fnb.gen_reg();
    compile_load_const(fnb, key, Literal::String("prototype".into()));
    fnb.emit(Instr::ObjGet {
        dest: prototype,
        obj: constructor,
        key,
    });

    compile_load_const(fnb, key, Literal::String("__proto__".into()));
    fnb.emit(Instr::ObjSet {
        obj,
        key,
        value: prototype,
    });

    compile_load_const(fnb, key, Literal::String("constructor".into()));
    fnb.emit(Instr::ObjSet {
        obj,
        key,
        value: constructor,
    });
}

fn compile_block(fnb: &mut FnBuilder, block: &Block) {
    let stmts_count = block.stmts().count();
    fnb.block(block.id, stmts_count, |fnb| {
        compile_block_internal(fnb, block)
    })
}

fn compile_block_internal(fnb: &mut FnBuilder, block: &Block) {
    for decl in block.decls() {
        let reg = fnb.gen_reg();
        fnb.define_name(decl.name.clone(), reg);

        if !decl.is_lexical {
            // `var` declarations are implicitly initialized as `undefined` at the beginning of the
            // block (assuming hoisting has been done already)
            fnb.emit(Instr::LoadUndefined(reg));
        }
    }

    for fn_asmt in block.fn_asmts() {
        compile_assignment(fnb, &fn_asmt.var_name, block, fn_asmt.expr);
    }

    let mut iter = block.stmts();
    while compile_one_stmt(fnb, block, &mut iter) {}
}

mod builder {
    use std::collections::{HashMap, HashSet};

    use swc_atoms::JsWord;

    use super::{DeclName, StmtID};
    use crate::{
        bytecode,
        bytecode_compiler::{js_to_past, CompiledModule},
    };

    use bytecode::IID;
    use js_to_past::BlockID;

    pub(super) struct ModuleBuilder {
        fns: HashMap<bytecode::LocalFnId, bytecode::Function>,
        next_lfnid: u16,
        breakable_ranges: Vec<bytecode::BreakRange>,
    }
    impl ModuleBuilder {
        pub(super) fn new(min_fnid: u16) -> Self {
            ModuleBuilder {
                fns: HashMap::new(),
                next_lfnid: min_fnid,
                breakable_ranges: Vec::new(),
            }
        }

        pub(super) fn gen_id(&mut self) -> bytecode::LocalFnId {
            let lfnid = bytecode::LocalFnId(self.next_lfnid);
            self.next_lfnid += 1;
            lfnid
        }

        pub(super) fn put_fn(&mut self, lfnid: bytecode::LocalFnId, function: bytecode::Function) {
            self.fns.insert(lfnid, function);
        }

        pub(super) fn build(self, root_fnid: bytecode::LocalFnId) -> CompiledModule {
            CompiledModule {
                root_fnid,
                functions: self.fns,
                breakable_ranges: self.breakable_ranges,
            }
        }
    }

    pub(super) struct FnBuilder<'a> {
        instrs: Vec<bytecode::Instr>,
        consts: Vec<bytecode::Literal>,
        n_regs: u8,
        blocks: Vec<Block>,
        block_boundaries: HashMap<BlockID, (IID, IID)>,
        deferred_actions: Vec<BoxedAction>,

        is_strict_mode: bool,

        globals: &'a HashSet<JsWord>,
        module_builder: &'a mut ModuleBuilder,
        lfnid: bytecode::LocalFnId,
    }

    type BoxedAction = Box<dyn FnOnce(&mut FnBuilder)>;

    struct Block {
        id: BlockID,
        names: HashMap<DeclName, bytecode::VReg>,
        iid_of_stmt: Box<[bytecode::IID]>,
        n_started_stmts: usize,
        deferred_actions: Vec<BoxedAction>,
    }

    pub enum Loc {
        Reg(bytecode::VReg),
        Global(JsWord),
    }

    impl<'a> FnBuilder<'a> {
        pub(super) fn new(
            globals: &'a HashSet<JsWord>,
            module_builder: &'a mut ModuleBuilder,
        ) -> FnBuilder<'a> {
            let lfnid = module_builder.gen_id();

            FnBuilder {
                instrs: Vec::new(),
                consts: Vec::new(),
                // The first ARGS_COUNT_MAX register are reserved for the first few function
                // arguments
                n_regs: bytecode::ARGS_COUNT_MAX,
                blocks: Vec::new(),
                block_boundaries: HashMap::new(),
                is_strict_mode: false,
                globals,
                module_builder,
                deferred_actions: Vec::new(),
                lfnid,
            }
        }

        pub(super) fn set_strict_mode(&mut self, value: bool) {
            self.is_strict_mode = value;
        }
        pub(super) fn is_strict_mode(&self) -> bool {
            self.is_strict_mode
        }

        fn push_block(&mut self, block_id: BlockID, stmts_count: usize) {
            self.blocks.push(Block {
                id: block_id,
                names: HashMap::new(),
                // '+ 1' because we also map the one-past-the-end StmtID
                iid_of_stmt: (0..stmts_count + 1)
                    .map(|_| bytecode::IID(u16::MAX))
                    .collect(),
                n_started_stmts: 0,
                deferred_actions: Vec::new(),
            })
        }
        fn pop_block(&mut self) {
            let block = self.blocks.pop().unwrap();
            // check that every stmt has been marked (with `mark_stmt_start`)
            assert_eq!(block.n_started_stmts, block.iid_of_stmt.len());
        }
        fn cur_block_mut(&mut self) -> &mut Block {
            self.blocks.last_mut().unwrap()
        }
        pub(super) fn block<R>(
            &mut self,
            block_id: BlockID,
            stmts_count: usize,
            action: impl FnOnce(&mut FnBuilder) -> R,
        ) -> R {
            self.push_block(block_id, stmts_count);

            let iid_start = self.peek_iid();
            let ret = action(self);
            // Start one more statement and consider a StmtID that points to
            // one-past-the-end the list of stmts; it's still valid and its
            // corresponding IID is iid_end, which is one-past-the-end the list
            // of instructions.
            self.mark_stmt_start();
            let iid_end = self.peek_iid();

            let block = self.cur_block_mut();
            let deferred_actions = std::mem::take(&mut block.deferred_actions);
            let block_id = block.id;

            self.block_boundaries.insert(block_id, (iid_start, iid_end));
            for action in deferred_actions {
                action(self);
            }

            self.pop_block();
            ret
        }
        pub(super) fn define_name(&mut self, decl_name: DeclName, reg: bytecode::VReg) {
            let block = self.blocks.last_mut().unwrap();
            block.names.insert(decl_name, reg);
        }

        pub(super) fn gen_reg(&mut self) -> bytecode::VReg {
            let reg = bytecode::VReg(self.n_regs);
            self.n_regs += 1;
            reg
        }

        pub(super) fn add_const(&mut self, lit: bytecode::Literal) -> bytecode::ConstIndex {
            // TODO deduplicate consts
            let index = self.consts.len();
            self.consts.push(lit);
            bytecode::ConstIndex(index.try_into().unwrap())
        }

        pub(super) fn peek_iid(&self) -> bytecode::IID {
            bytecode::IID(self.instrs.len().try_into().unwrap())
        }
        pub(super) fn emit(&mut self, instr: bytecode::Instr) {
            self.instrs.push(instr);
        }
        pub(super) fn reserve_instr(&mut self) -> bytecode::IID {
            let iid = self.peek_iid();
            self.instrs.push(bytecode::Instr::Nop);
            iid
        }
        pub(super) fn set_instr(&mut self, iid: bytecode::IID, instr: bytecode::Instr) {
            self.instrs[iid.0 as usize] = instr;
        }
        pub(super) fn mark_stmt_start(&mut self) {
            let iid = self.peek_iid();
            let block = self.blocks.last_mut().unwrap();

            let slot = &mut block.iid_of_stmt[block.n_started_stmts];
            // check that we haven't set the same slot twice
            debug_assert_eq!(slot.0, u16::MAX);
            *slot = iid;

            block.n_started_stmts += 1;
        }
        pub(super) fn iid_of_stmt(&mut self, stmt_id: StmtID) -> bytecode::IID {
            let block = self.blocks.last().unwrap();
            block.iid_of_stmt[stmt_id.numeric() as usize]
        }
        pub(super) fn block_boundaries(
            &mut self,
            block_id: js_to_past::BlockID,
        ) -> (bytecode::IID, bytecode::IID) {
            self.block_boundaries.get(&block_id).copied().unwrap()
        }

        pub(super) fn add_breakable_range(
            &mut self,
            span: swc_common::Span,
            iid_start: bytecode::IID,
            iid_end: bytecode::IID,
        ) {
            self.module_builder
                .breakable_ranges
                .push(bytecode::BreakRange {
                    lo: span.lo,
                    hi: span.hi,
                    local_fnid: self.lfnid,
                    iid_start,
                    iid_end,
                });
        }

        pub(super) fn on_block_completion(
            &mut self,
            action: impl 'static + FnOnce(&mut FnBuilder),
        ) {
            self.blocks
                .last_mut()
                .unwrap()
                .deferred_actions
                .push(Box::new(action));
        }

        pub(super) fn on_function_completion(
            &mut self,
            action: impl 'static + FnOnce(&mut FnBuilder),
        ) {
            self.deferred_actions.push(Box::new(action));
        }

        pub(super) fn build(mut self, span: swc_common::Span) -> bytecode::LocalFnId {
            assert!(self.blocks.is_empty());

            let deferred_actions = std::mem::take(&mut self.deferred_actions);
            for action in deferred_actions {
                action(&mut self);
            }

            let bc_func = bytecode::FunctionBuilder {
                instrs: self.instrs.into_boxed_slice(),
                consts: self.consts.into_boxed_slice(),
                n_regs: self.n_regs,
                // TODO TODO This is all yet to be implemented:
                ident_history: Vec::new(),
                trace_anchors: HashMap::new(),
                is_strict_mode: self.is_strict_mode,
                span,
            }
            .build();

            self.module_builder.put_fn(self.lfnid, bc_func);
            self.lfnid
        }

        pub(crate) fn resolve_name(&self, name: &DeclName) -> Loc {
            let local_reg = self
                .blocks
                .iter()
                .rev()
                .find_map(|scope| scope.names.get(name).copied().map(Loc::Reg));

            if let Some(reg) = local_reg {
                return reg;
            }

            if let DeclName::Js(name) = name {
                if self.globals.contains(name) {
                    return Loc::Global(name.clone());
                }
            }
            panic!("malformed PAST: undeclared name: {:?}", name)
        }

        // this sucks lol
        pub(super) fn suspend(&mut self) -> (&mut &'a HashSet<JsWord>, &mut &'a mut ModuleBuilder) {
            (&mut self.globals, &mut self.module_builder)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{bytecode, bytecode_compiler::CompiledModule};

    #[test]
    fn test_global_var() {
        let CompiledModule {
            functions,
            root_fnid,
            ..
        } = quick_compile(
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
                if lfnid == &root_fnid { "[root]" } else { "" }
            );

            func.dump();
        }
    }

    #[test]
    fn test_let_and_var() {
        quick_compile(
            r#"(function() {
                    function f() { x = 1 }
                    let x = 5;
                    return f()
                })()
                "#
            .to_string(),
        );
    }

    #[test]
    fn test_function_before_let() {
        let compiled_module = quick_compile(
            "
                (function() {
                    function f() { x = 3 }
                    let x = 2;
                })()
                "
            .to_string(),
        );

        insta::assert_snapshot!(dump_functions(&compiled_module.functions));
    }

    #[test]
    fn test_func_decl() {
        let compiled_module = quick_compile(
            "function myFunction() {
                    return 3
                }"
            .to_string(),
        );

        insta::assert_snapshot!(dump_functions(&compiled_module.functions));
    }

    fn dump_functions(functions: &HashMap<bytecode::LocalFnId, bytecode::Function>) -> String {
        let mut ids: Vec<_> = functions.keys().copied().collect();
        ids.sort();

        let mut buf = String::new();
        for fnid in ids {
            use std::fmt::Write;

            let func = functions.get(&fnid).unwrap();
            writeln!(buf, "fn {:?}", fnid).unwrap();
            for (ndx, literal) in func.consts().iter().enumerate() {
                writeln!(buf, "  k{} {:?}", ndx, literal).unwrap();
            }
            for instr in func.instrs() {
                writeln!(buf, "  {:?}", instr).unwrap();
            }
        }

        buf
    }

    fn quick_compile(src: String) -> CompiledModule {
        let (swc_ast, source_map) = crate::bytecode_compiler::quick_parse_script(src);
        let past_function = super::js_to_past::compile_script(&swc_ast, source_map).unwrap();

        let mut module_builder = super::ModuleBuilder::new(0);
        let globals = past_function.unbound_names.iter().cloned().collect();
        let force_strict = false;
        let root_lfnid = super::compile_function(
            &globals,
            &mut module_builder,
            Vec::new(),
            &past_function,
            force_strict,
        );

        module_builder.build(root_lfnid)
    }
}
