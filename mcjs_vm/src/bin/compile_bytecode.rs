use mcjs_vm::FileLoader;
use std::collections::HashMap;
use swc_atoms::JsWord;

fn main() {
    let filename = std::env::args()
        .nth(1)
        .expect("usage: compile_bytecode <filename.js>");

    let cwd = std::env::current_dir().unwrap();
    let loader = FileLoader::new(std::iter::once(cwd.as_path()));

    let mut native_fns = HashMap::new();
    let words = [
        "RegExp",
        "String",
        "Number",
        "Boolean",
        "Array",
        "Object",
        "TypeError",
        "Math",
    ];
    for word in words {
        native_fns.insert(JsWord::from(word), 123);
    }
    let mut builder = mcjs_vm::BuilderParams {
        loader: Box::new(loader),
    }
    .to_builder();

    let res = builder.compile_file(filename);
    if let Err(err) = res {
        eprintln!("error:");
        eprintln!("{}", err.message());
        eprintln!();
    } else {
        let codebase = builder.build();
        codebase.dump();
    }
}
