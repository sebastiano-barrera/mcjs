use std::collections::HashMap;
use swc_atoms::JsWord;

fn main() {
    let filename = std::env::args()
        .nth(1)
        .expect("usage: compile_bytecode <filename.js>");
    let filename = std::path::PathBuf::from(filename);

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

    let mut loader = mcjs_vm::Loader::new_cwd();
    match loader.load_script_file(&filename) {
        Err(err) => {
            eprintln!("error:");
            eprintln!("{}", err);
            eprintln!();
        }
        Ok(fnid) => {
            let _func = loader.get_function(fnid).unwrap();
            todo!("<dump function bytecode>");
        }
    }
}
