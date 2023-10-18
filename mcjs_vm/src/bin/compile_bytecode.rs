use std::{collections::HashMap, path::Path};
use swc_atoms::JsWord;

fn main() {
    let filename = std::env::args()
        .nth(1)
        .expect("usage: compile_bytecode <filename.js>");
    let content = std::fs::read_to_string(Path::new(&filename)).expect("error while reading file");

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

    let mut loader = mcjs_vm::Loader::new(None);
    match loader.load_script(Some(filename.clone()), content) {
        Err(err) => {
            eprintln!("error:");
            eprintln!("{}", err.message());
            eprintln!();
        }
        Ok(fnid) => {
            let _func = loader.get_function(fnid).unwrap();
            todo!("<dump function bytecode>");
        }
    }
}
