extern crate mcjs;

use std::path::PathBuf;

// TODO(test): enable these tests once the implementation is mature enough
#[test]
fn test_load_json5_parse() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_paths = vec![manifest_dir.join("test-resources/modules/json5/dist")];
    let file_loader = Box::new(mcjs::FileLoader::new(include_paths));

    let code = r#"
        import mod from 'index.mjs';

        sink(mod.stringify({
            a: 2345.123,
            b: 'asdlol',
            pt: {x: 123, y: 456}
        }));
    "#;
    let mut mock_loader = Box::new(mcjs::MockLoader::new());
    mock_loader.add_module("test.mjs".to_owned(), mcjs::ModuleId(1), code.to_owned());

    let mut builder = mcjs::BuilderParams {
        loader: Box::new(mcjs::CombinedLoader::new(vec![file_loader, mock_loader])),
    }
    .to_builder();
    let test_mod_id = builder
        .compile_file("test.mjs".to_owned())
        .unwrap_or_else(|err| panic!("compile error: {:?}", err));
    let codebase = builder.build();

    let mut vm = mcjs::Interpreter::new(&codebase);
    vm.run_module(test_mod_id)
        .unwrap_or_else(|err| panic!("runtime error: {:?}", err));

    let sink = vm.take_sink();
    assert!(false, "{:?}", sink);
}
