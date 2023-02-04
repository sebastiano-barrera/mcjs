extern crate mcjs;

use std::path::PathBuf;

#[test]
fn test_load_json5_parse() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include_path = manifest_dir.join("test-resources/modules/json5/lib");

    let mut vm = mcjs::VM::new();
    vm.add_include_path(include_path.to_owned()).unwrap();
    vm.run_script("require('cli')".to_string(), Default::default()).unwrap();

    assert!(false);
}
