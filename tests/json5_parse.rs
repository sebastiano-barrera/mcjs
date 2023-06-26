extern crate mcjs;

use mcjs::InterpreterValue;
use std::path::PathBuf;

#[test]
fn test_load_json5_stringify() {
    test_integration_script("test_stringify.mjs".to_owned());
}

#[test]
fn test_load_json5_parse() {
    test_integration_script("test_parse.mjs".to_owned());
}

fn test_integration_script(filename: String) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_paths = vec![
        manifest_dir.join("test-resources/modules/json5/dist"),
        manifest_dir.join("test-resources/test-scripts/json5/"),
    ];
    let file_loader = Box::new(mcjs::FileLoader::new(include_paths));
    let mut builder = mcjs::BuilderParams {
        loader: file_loader,
    }
    .to_builder();
    let test_mod_id = builder
        .compile_file(filename)
        .unwrap_or_else(|err| panic!("compile error: {:?}", err));
    let codebase = builder.build();
    let mut vm = mcjs::Interpreter::new(&codebase);
    vm.run_module(test_mod_id)
        .unwrap_or_else(|err| panic!("runtime error: {:?}", err));
    let sink = vm.take_sink();
    assert_eq!(
        &sink,
        &[
            InterpreterValue::String("null".into()),
            InterpreterValue::String("123".into()),
            InterpreterValue::String("456.78".into()),
            InterpreterValue::String("true".into()),
            InterpreterValue::String("false".into()),
        ]
    );
}
