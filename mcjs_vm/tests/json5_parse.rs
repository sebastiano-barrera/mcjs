extern crate mcjs_vm;

use mcjs_vm::Literal;
use std::path::PathBuf;

use serde::Serialize;

#[test]
fn test_load_json5_stringify() {
    test_integration_script("test_stringify.mjs".to_owned());
}

#[test]
fn test_load_json5_parse() {
    test_integration_script("test_parse.mjs".to_owned());
}

fn test_integration_script(filename: String) {
    // TODO Refactor the inspector case file export logic, so that it can be reused across all
    // test cases

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_paths = vec![
        manifest_dir.join("test-resources/modules/json5/dist"),
        manifest_dir.join("test-resources/test-scripts/json5/"),
    ];
    let file_loader = Box::new(mcjs_vm::FileLoader::new(
        include_paths.iter().map(|pb| pb.as_path()),
    ));
    let mut builder = mcjs_vm::BuilderParams {
        loader: file_loader,
    }
    .to_builder();
    let test_mod_id = builder
        .compile_file(filename.clone())
        .unwrap_or_else(|err| panic!("compile error: {:?}", err));
    let codebase = builder.build().codebase;

    let res = std::panic::catch_unwind(|| {
        let output = mcjs_vm::Interpreter::new(&codebase)
            .run_module(test_mod_id)
            .unwrap_or_else(|err| panic!("runtime error: {:?}", err));
        output.sink
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
            let root = mcjs_vm::inspector_case::Root::ModuleImport(filename);
            mcjs_vm::inspector_case::export_inspector_case(include_paths, root);
            panic!("Test failed.");
        }
    }
}
