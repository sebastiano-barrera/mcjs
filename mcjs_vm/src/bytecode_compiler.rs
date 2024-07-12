use std::collections::HashMap;

use swc_common::sync::Lrc;
use swc_ecma_ast::EsVersion;
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

use crate::bytecode::{self, FnID};
use crate::common::{MultiError, Result};
use crate::loader::FileIDRef;
use crate::{define_flag, error, error_item, tracing};

pub use swc_common::SourceMap;

use std::rc::Rc;

mod js_to_past;
mod past_to_bytecode;

pub struct CompiledChunk {
    pub root_fnid: FnID,
    pub functions: HashMap<FnID, bytecode::Function>,
    pub breakable_ranges: Vec<bytecode::BreakRange>,
}

define_flag!(pub AllowDirectForms);

pub struct CompileFlags {
    pub min_fnid: u32,
    pub source_type: SourceType,

    /// Whether to allow the use of "direct forms".
    ///
    /// These are special forms (bits of syntax) that result in emitting a
    /// single bytecode instruction (or a very small sequence).
    ///
    /// They're designed to only be used with the bootstrap script, NEVER for
    /// regular user script.
    pub allow_direct_forms: AllowDirectForms,
}

/// Whether we're compiling an actual module or a 'classic script'.
///
/// Influences the way that declarations work in the toplevel.  The distinction
/// is made largely at compile time, so we have to know here.
#[derive(PartialEq, Eq, Debug)]
pub enum SourceType {
    Script,
    Module,
}

/// Compile the given chunk of source code into executable bytecode.
///
/// The given `file_id` is used *exclusively* for composing error messages and
/// to initialize the returned source map with a significant identifier for the
/// compiled file.  It does not have to reflect an existing entity in any file
/// system, and is not checked in any way.
///
/// See `CompiledChunk` for details on the executable bytecode's representation.
pub fn compile_file(
    file_id: FileIDRef,
    content: String,
    source_map: Lrc<SourceMap>,
    flags: CompileFlags,
) -> Result<CompiledChunk> {
    use crate::common::Context;

    let t = tracing::section("compile_file");
    t.log("source", &content);

    let swc_path = match file_id {
        FileIDRef::Anon(_) => swc_common::FileName::Anon,
        FileIDRef::File(path) => swc_common::FileName::Real(path.to_owned()),
    };
    let display_filename = match file_id {
        FileIDRef::Anon(_) => "<input>".to_string(),
        FileIDRef::File(path) => path.to_string_lossy().into_owned(),
    };

    let source_file = source_map.new_source_file(swc_path, content);
    let input = StringInput::from(source_file.as_ref());

    let err_handler = mk_error_handler(&source_map);
    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2022,
        input,
        None,
    );
    let mut parser = Parser::new_from(lexer);

    let compiled_mod = match flags.source_type {
        SourceType::Module => {
            let module_ast = parser
                .parse_module()
                .map_err(|e| {
                    e.into_diagnostic(&err_handler).emit();
                    error!("parse error")
                })
                .with_context(error_item!("while parsing file: {}", display_filename))?;

            compile_module(&module_ast, flags, Lrc::clone(&source_map))
                .with_context(error_item!("while compiling module: {}", display_filename))
                .with_source_map(Lrc::clone(&source_map))?
        }
        SourceType::Script => {
            let script_ast = parser
                .parse_script()
                .map_err(|e| {
                    e.into_diagnostic(&err_handler).emit();
                    error!("parse error")
                })
                .with_context(error_item!("while parsing file: {}", display_filename))
                .with_source_map(Lrc::clone(&source_map))?;

            compile_script(script_ast, flags, Lrc::clone(&source_map))
                .with_context(error_item!("while compiling script: {}", display_filename))
                .with_source_map(Lrc::clone(&source_map))?
        }
    };

    Ok(CompiledChunk {
        root_fnid: compiled_mod.root_fnid,
        functions: compiled_mod.functions,
        breakable_ranges: compiled_mod.breakable_ranges,
    })
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

#[cfg(test)]
pub(crate) fn quick_parse_script(src: String) -> (swc_ecma_ast::Script, Rc<SourceMap>) {
    let source_map = Rc::new(SourceMap::default());
    let err_handler = crate::bytecode_compiler::mk_error_handler(&source_map);

    let filename = swc_common::FileName::Anon;

    let source_file = source_map.new_source_file(filename, src);
    let input = StringInput::from(source_file.as_ref());

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::Es2015,
        input,
        None,
    );
    let mut parser = Parser::new_from(lexer);

    let res = parser.parse_script();
    let swc_ast = match res {
        Ok(swc_ast) => swc_ast,
        Err(err) => {
            err.into_diagnostic(&err_handler).emit();
            panic!("parse error");
        }
    };
    (swc_ast, source_map)
}

struct CompiledModule {
    root_fnid: FnID,
    functions: HashMap<FnID, bytecode::Function>,
    breakable_ranges: Vec<bytecode::BreakRange>,
}

fn compile_module(
    ast_module: &swc_ecma_ast::Module,
    flags: CompileFlags,
    source_map: Rc<SourceMap>,
) -> Result<CompiledModule> {
    let t = tracing::section("compile_script");

    let function = js_to_past::compile_module(ast_module, source_map, flags.allow_direct_forms)
        .map_err(MultiError::into_single)?;
    t.log_value("PAST", &function);
    assert!(function.parameters.is_empty());

    // At this level, function.unbound_names contains the list of variables that should be
    // accessed via `globalThis`.
    let module = past_to_bytecode::compile_module(&function, flags.min_fnid)?;
    Ok(module)
}

fn compile_script(
    script_ast: swc_ecma_ast::Script,
    flags: CompileFlags,
    source_map: Rc<SourceMap>,
) -> Result<CompiledModule> {
    let t = tracing::section("compile_script");

    let function = js_to_past::compile_script(&script_ast, source_map, flags.allow_direct_forms)
        .map_err(MultiError::into_single)?;
    t.log_value("PAST", &function);
    assert!(function.parameters.is_empty());

    // At this level, function.unbound_names contains the list of variables that should be
    // accessed via `globalThis`.
    let module = past_to_bytecode::compile_module(&function, flags.min_fnid)?;
    Ok(module)
}
