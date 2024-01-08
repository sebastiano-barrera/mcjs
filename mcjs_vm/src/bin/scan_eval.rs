use swc_ecma_ast::{Expr, Lit};
use swc_ecma_visit::{Visit, VisitWith};

fn main() {
    use std::io::Read;

    let filenames: Vec<_> = std::env::args().skip(1).collect();
    if filenames.is_empty() {
        panic!("usage: scan_eval <filename.js>");
    }

    for filename in filenames {
        println!("FILE {}", filename);

        let mut content = String::new();
        std::fs::File::open(filename.clone())
            .expect("could not open")
            .read_to_string(&mut content)
            .expect("could not read");

        let module = parse_file(filename, content);
        let mut scanner = Scanner;
        VisitWith::visit_with(&module, &mut scanner);
    }
}

struct Scanner;

impl Visit for Scanner {
    fn visit_call_expr(&mut self, n: &swc_ecma_ast::CallExpr) {
        if let Some(ident) = n.callee.as_expr().and_then(|expr| expr.as_ident()) {
            if &ident.sym == "eval" {
                match n.args.len() {
                    0 => {
                        println!("eval noargs");
                    }
                    1 => {
                        let arg = &n.args[0];
                        if let Expr::Lit(Lit::Str(str_lit)) = arg.expr.as_ref() {
                            println!("eval str {:?}", str_lit.value.to_string());
                        } else {
                            println!("eval singlenonstr {:?}", arg);
                        }
                    }
                    _ => {
                        for arg in &n.args {
                            println!("eval multi {:?}", arg);
                        }
                    }
                }
            }
        }
    }
}

fn parse_file(filename: String, content: String) -> swc_ecma_ast::Module {
    use swc_common::{
        errors::{emitter::EmitterWriter, Handler},
        sync::Lrc,
        FileName, SourceMap,
    };
    use swc_ecma_ast::EsVersion;
    use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};

    let source_map = Lrc::new(SourceMap::default());
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

    parser.parse_module().unwrap_or_else(|e| {
        e.into_diagnostic(&err_handler).emit();
        panic!("parse error")
    })
}
