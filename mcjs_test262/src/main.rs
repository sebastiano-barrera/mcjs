use std::{io::Write, path::PathBuf};

use mcjs_vm::interpreter::debugger::{DebuggingState, Fuel};
use serde::Serialize;

fn main() {
    let opts = parse_options();

    let test_error = run_test(&opts).err();
    let outcome = TestOutcome { error: test_error };

    // one json per line
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &outcome).unwrap();
    stdout.write_all(b"\n").unwrap();
}

#[derive(Default)]
struct CliOptions {
    files: Vec<PathBuf>,
    is_last_strict: bool,
}

fn parse_options() -> CliOptions {
    let mut opts: CliOptions = CliOptions::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg.starts_with("-") {
            match arg.as_str() {
                "--force-last-strict" => {
                    opts.is_last_strict = true;
                }
                _ => panic!("unsupported option: {}", arg),
            }
        } else {
            opts.files.push(PathBuf::from(arg));
        }
    }

    opts
}

fn run_test(params: &CliOptions) -> Result<(), TestError> {
    let res = std::panic::catch_unwind(move || {
        let mut loader = mcjs_vm::Loader::new_cwd();
        let mut realm = mcjs_vm::Realm::new(&mut loader);

        for (ndx, file_path) in params.files.iter().enumerate() {
            let mut content = std::fs::read_to_string(&file_path).map_err(TestError::Read)?;

            if ndx == params.files.len() - 1 && params.is_last_strict {
                content.insert_str(0, "\"use strict\";");
            }

            let chunk_fnid = loader.load_script_anon(content).map_err(TestError::Load)?;

            let mut interpreter = mcjs_vm::Interpreter::new(&mut realm, &mut loader, chunk_fnid);
            let mut dbg = DebuggingState::new();
            dbg.set_fuel(Fuel::Limited(200_000));
            interpreter.set_debugging_state(&mut dbg);
            interpreter
                .run()
                .map_err(|intrp_err| TestError::Run(intrp_err.error))?;
        }
        Ok(())
    });

    // Flatten a panic into a regular TestError
    match res {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(err),
        Err(panic_value) => {
            let message = match panic_value.downcast_ref::<String>() {
                Some(s) => s.to_string(),
                None => "<not a string>".to_string(),
            };
            Err(TestError::Panic(message))
        }
    }
}

#[derive(Serialize)]
struct TestOutcome {
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
