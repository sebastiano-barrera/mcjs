extern crate mcjs;

use std::path::PathBuf;

// TODO(test): enable these tests once the implementation is mature enough
#[cfg(not(test))]
#[test]
fn test_load_json5_parse() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_path = manifest_dir.join("test-resources/modules/json5/dist");

    let mut vm = mcjs::VM::new();
    vm.add_include_path(include_path.to_owned()).unwrap();

    let code = &r#"
        import * as util from 'index.mjs';

        sink(!util.isDigit(''));
        sink(util.isDigit('12394'));
        sink(!util.isDigit('asdlol'));

        sink(util.isIdStartChar('a'));
        sink(util.isIdStartChar('b'));
        sink(util.isIdStartChar('_'));
        sink(!util.isIdStartChar('8'));
        sink(!util.isIdStartChar(''));
    "#;

    vm.run_script(code.to_string(), Default::default()).unwrap();

    let sink = vm.take_sink();
    assert!(false, "{:?}", sink);
}
