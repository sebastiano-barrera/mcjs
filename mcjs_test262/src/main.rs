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

    let filename = Path::new("test/language/function-code/10.4.3-1-1-s.js");

    let mut loader = mcjs_vm::Loader::new(None);

    let chunk_paths = [
        test262_root.join(Path::new("harness/assert.js")),
        test262_root.join(Path::new("harness/sta.js")),
        filename.to_path_buf(),
    ];

    let mut chunk_fnids = Vec::with_capacity(chunk_paths.len());
    for file_path in chunk_paths {
        let file_path_str = file_path.to_string_lossy().into_owned();
        eprintln!(" (..) reading {}", file_path_str);
        let content = std::fs::read_to_string(file_path).expect("read_to_string");

        eprintln!(" (..) loading {}", file_path_str);
        let chunk_fnid: mcjs_vm::bytecode::FnId =
            loader.load_script(Some(file_path_str), content).unwrap();

        chunk_fnids.push(chunk_fnid);
    }

    let mut realm = mcjs_vm::Realm::new();

    for fnid in chunk_fnids {
        eprintln!(" (..) running {:?}", fnid);
        mcjs_vm::Interpreter::new(&mut realm, &mut loader, fnid)
            .run()
            .expect("run");
    }
}
