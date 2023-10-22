extern crate mcjs_vm;

use mcjs_vm::Literal;
use std::path::PathBuf;

#[test]
fn test_load_json5_stringify() {
    test_integration_script("test_stringify.mjs");
}

#[test]
fn test_load_json5_parse() {
    test_integration_script("test_parse.mjs");
}

fn test_integration_script(filename: &str) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_path = manifest_dir.join("test-resources/test-scripts/json5/");

    let mut loader = mcjs_vm::Loader::new(Some(root_path.clone()));

    let script_path = root_path.join(filename);
    assert!(script_path.starts_with(&root_path));

    let content = std::fs::read_to_string(&script_path).expect("error while reading file");
    let fnid = loader
        .load_script(Some(filename.to_string()), content)
        .expect("error while compiling script");

    let mut realm = mcjs_vm::Realm::new();
    let sink = mcjs_vm::Interpreter::new(&mut realm, &mut loader, fnid)
        .run()
        .unwrap_or_else(|err| panic!("runtime error:\n{:?}", err))
        .expect_finished()
        .sink;

    assert_eq!(
        &sink,
        &[
            Some(Literal::String("null".into())),
            Some(Literal::String("123".into())),
            Some(Literal::String("456.78".into())),
            Some(Literal::String("true".into())),
            Some(Literal::String("false".into())),
        ]
    );
}
