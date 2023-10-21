use anyhow::Result;
use std::path::Path;

struct TestEnv<'a> {
    /// Absolute path to the test262 root folder.
    test262_root: &'a Path,
}

struct TestParams<'a> {
    env: &'a TestEnv<'a>,

    /// Path to the test file.  Must be relative.  It will be interpreted
    /// relative to the test262 root folder.
    filename: &'a Path,
}

fn run_test262<'a>(params: &TestParams<'a>) -> Result<()> {
    let mut loader = mcjs_vm::Loader::new(None);

    let chunk_paths = [
        Path::new("harness/assert.js"),
        Path::new("harness/sta.js"),
        params.filename,
    ];

    let mut chunk_fnids = Vec::with_capacity(chunk_paths.len());
    for file_path in chunk_paths {
        let content = std::fs::read_to_string(file_path)?;

        let file_path_str = file_path.to_string_lossy().into_owned();
        let chunk_fnid: mcjs_vm::bytecode::FnId = loader
            .load_script(Some(file_path_str), content)
            .map_err(|err| anyhow::anyhow!("while loading script: {:?}", err))?;

        chunk_fnids.push(chunk_fnid);
    }

    for fnid in chunk_fnids {
        let mut realm = mcjs_vm::Realm::new();
        mcjs_vm::Interpreter::new(&mut realm, &mut loader, fnid)
            .run()
            .map_err(|err| anyhow::anyhow!("error while running code: {:?}", err))?;
    }

    Ok(())
}

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

    let test_env = TestEnv {
        test262_root: &test262_root,
    };
    let test_params = TestParams {
        env: &test_env,
        filename: Path::new("test/language/function-code/10.4.3-1-1-s.js"),
    };

    run_test262(&test_params).expect("test failed");
}
