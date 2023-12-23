use std::{
    fs::File,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

fn main() {
    let config: ConfigFile = {
        let filename = std::env::args()
            .skip(1)
            .next()
            .expect("usage: mcjs_test262 <path/to/tests.yml>");
        let rdr = File::open(filename).expect("could not open config file");
        serde_yaml::from_reader(rdr).expect("could not parse config file")
    };

    let test262_root = PathBuf::from(config.test262Root);
    if !test262_root.is_dir() {
        panic!(
            "test262Root is not a directory: {}",
            test262_root.to_string_lossy()
        );
    }

    let mut outcomes = Vec::new();

    let tests_count = config.testFiles.len();
    for (ndx, file_path) in config.testFiles.iter().enumerate() {
        eprintln!("test {}/{}", ndx, tests_count);
        let chunk_paths = vec![
            test262_root.join("harness/assert.js"),
            test262_root.join("harness/sta.js"),
            test262_root.join(file_path),
        ];

        let params = TestParams { chunk_paths };
        let outcome = run_test(params);
        outcomes.push(outcome);
    }

    eprintln!("writing output...");
    let stdout = std::io::stdout().lock();
    serde_yaml::to_writer(stdout, &outcomes).unwrap();
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct ConfigFile {
    test262Root: String,
    testFiles: Vec<String>,
}

fn run_test(params: TestParams) -> TestOutcome {
    let paths = &params.chunk_paths;
    let res = std::panic::catch_unwind(move || {
        let mut loader = mcjs_vm::Loader::new(None);
        let mut realm = mcjs_vm::Realm::new();

        for file_path in paths {
            process_file(file_path, &mut realm, &mut loader)?;
        }
        Ok(())
    });

    // Flatten a panic into a regular TestError
    let res = match res {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(err),
        Err(panic_value) => {
            let message = match panic_value.downcast_ref::<String>() {
                Some(s) => format!("{}", s),
                None => "<not a string>".to_string(),
            };
            Err(TestError::Panic(message))
        }
    };

    TestOutcome {
        params,
        error: res.err(),
    }
}

fn process_file(
    file_path: &Path,
    realm: &mut mcjs_vm::Realm,
    loader: &mut mcjs_vm::Loader,
) -> Result<(), TestError> {
    let file_path_str = file_path.to_string_lossy().into_owned();
    let content = std::fs::read_to_string(file_path).map_err(|io_err| TestError::Read(io_err))?;

    let chunk_fnid: mcjs_vm::bytecode::FnId = loader
        .load_script(Some(file_path_str), content)
        .map_err(|vm_err| TestError::Load(vm_err))?;

    mcjs_vm::Interpreter::new(realm, loader, chunk_fnid)
        .run()
        .map_err(|intrp_err| TestError::Run(intrp_err.error))?;

    Ok(())
}

#[derive(Serialize)]
struct TestParams {
    chunk_paths: Vec<PathBuf>,
}

#[derive(Serialize)]
struct TestOutcome {
    params: TestParams,
    error: Option<TestError>,
}

#[derive(Debug)]
enum TestError {
    Read(std::io::Error),
    Load(mcjs_vm::Error),
    Run(mcjs_vm::Error),
    Panic(String),
}

impl Serialize for TestError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let (category, message) = match self {
            TestError::Read(err) => ("read", err.to_string()),
            TestError::Load(err) => ("load", err.message()),
            TestError::Run(err) => ("run", err.message()),
            TestError::Panic(err) => ("panic", err.to_string()),
        };

        let mut struct_ser = serializer.serialize_struct("TestError", 2)?;
        struct_ser.serialize_field("category", category)?;
        struct_ser.serialize_field("message", &message)?;
        struct_ser.end()
    }
}

#[derive(Serialize)]
struct OutputFile {
    outcomes: Vec<TestOutcome>,
}
