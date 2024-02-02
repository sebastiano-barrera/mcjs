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
use crate::util::Mask;
use crate::{error, tracing};

pub use swc_common::SourceMap;

use std::rc::Rc;

mod js_to_past;
mod legacy;
mod past_to_bytecode;

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

    let t = tracing::section("compile_file");
    t.log("source", &content);

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

fn parse_file(
    filename: String,
    content: String,
    source_map: Lrc<SourceMap>,
) -> Result<swc_ecma_ast::Module> {
    let err_handler = mk_error_handler(&source_map);

    let path = std::path::PathBuf::from(filename);
    let source_file = source_map.new_source_file(swc_common::FileName::Real(path), content);

    let mut parser = make_parser(&source_file, &err_handler);
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

    let input = StringInput::from(source_file);
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2015,
        input,
        None,
    );
    let mut parser = Parser::new_from(lexer);

    for e in parser.take_errors() {
        e.into_diagnostic(err_handler).emit();
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
    legacy::compile_module(ast_module, source_map, flags)
}

fn compile_script(
    ast_module: swc_ecma_ast::Module,
    source_map: Rc<SourceMap>,
    flags: CompileFlags,
) -> Result<CompiledModule> {
    use swc_ecma_ast::{ExportDecl, Expr, ModuleDecl, ModuleItem, Stmt, VarDeclKind};

    let t = tracing::section("compile_script");

    let function = js_to_past::compile_script(ast_module);
    t.log_value("PAST", &function);
    assert!(function.parameters.is_empty());

    // At this level, function.unbound_names contains the list of variables that should be accessed
    // via `globalThis`.
    #[cfg(any())]
    {
        let mut mod_builder = past_to_bytecode::ModuleBuilder::new(flags.min_fnid);

        let globals = function.unbound_names.iter().cloned().collect();
        let root_fnid = past_to_bytecode::compile_function(
            &mut mod_builder,
            &globals,
            Vec::new(),
            &function,
            false, // force_strict
        )?;

        let module = mod_builder.build(root_fnid);

        trace_dump_module(&module);
        Ok(module)
    }

    Ok(todo!())
}
