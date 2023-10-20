extern crate mcjs_vm;

use mcjs_vm::Literal;
use std::path::PathBuf;

use serde::Serialize;

#[test]
fn test_load_json5_stringify() {
    test_integration_script("test_stringify.mjs");
}

#[test]
fn test_load_json5_parse() {
    test_integration_script("test_parse.mjs");
}

fn test_integration_script(filename: &str) {
    // TODO Refactor the inspector case file export logic, so that it can be reused across all
    // test cases

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_path = manifest_dir.join("test-resources/test-scripts/json5/");

    let res = std::panic::catch_unwind(|| {
        let mut loader = mcjs_vm::Loader::new(Some(root_path.clone()));

        let script_path = root_path.join(filename);
        assert!(script_path.starts_with(&root_path));

        let content = std::fs::read_to_string(&script_path).expect("error while reading file");
        let fnid = loader
            .load_script(Some(filename.to_string()), content)
            .expect("error while compiling script");

        mcjs_vm::Interpreter::new(&mut loader)
            .run_function(fnid)
            .unwrap_or_else(|err| panic!("runtime error:\n{:?}", err))
            .sink
    });

    match res {
        Ok(sink) => {
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

        Err(_) => {
            let root = mcjs_vm::inspector_case::Root::ModuleImport(filename.to_string());
            mcjs_vm::inspector_case::export_inspector_case(vec![root_path], root);
            panic!("Test failed.");
        }
    }
}
