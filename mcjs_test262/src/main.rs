use std::path::Path;

fn main() {
    let test262_root = match std::env::var("TEST262_ROOT") {
        Ok(value) => Path::new(&std::ffi::OsString::from(value))
            .canonicalize()
            .unwrap(),
        Err(err) => {
            eprintln!("Couldn't get required env var TEST262_ROOT: {}", err);
            return;
        }
    };

    let mut loader = mcjs_vm::Loader::new(None);

    let chunk_paths = [
        test262_root.join("harness/assert.js"),
        test262_root.join("harness/sta.js"),
        test262_root.join("test/language/function-code/10.4.3-1-1-s.js"),
    ];

    let mut realm = mcjs_vm::Realm::new();

    for file_path in chunk_paths {
        let file_path_str = file_path.to_string_lossy().into_owned();
        eprintln!(" (**) file: {}", file_path_str);

        eprintln!("   (..) reading");
        let content = std::fs::read_to_string(file_path).expect("read_to_string");

        eprintln!("   (..) loading");
        let chunk_fnid: mcjs_vm::bytecode::FnId =
            loader.load_script(Some(file_path_str), content).unwrap();

        eprintln!("   (..) running chunk ({:?})", chunk_fnid);
        mcjs_vm::Interpreter::new(&mut realm, &mut loader, chunk_fnid)
            .run()
            .expect("run");
    }
}
