use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use mcjs_vm::interpreter::{debugger::Probe, Fuel};
use serde::{Deserialize, Serialize};

fn main() {
    let opts = parse_options();
    let config: ConfigFile = {
        let rdr = File::open(opts.config_filename).expect("could not open config file");
        serde_json::from_reader(rdr).expect("could not parse config file")
    };

    let test262_root = PathBuf::from(config.test262Root);
    if !test262_root.is_dir() {
        panic!(
            "test262Root is not a directory: {}",
            test262_root.to_string_lossy()
        );
    }

    let tests_count = config.testFiles.len();
    for (ndx, file_path) in config.testFiles.iter().enumerate() {
        if !file_path.contains(&opts.filter) {
            continue;
        }
        eprintln!("test {}/{}", ndx, tests_count);

        let full_path = &test262_root.join(file_path);
        let metadata = match parse_metadata(full_path) {
            // TODO Try instead!
            Ok(metadata) => metadata,
            Err(err) => {
                write_outcome(TestOutcome {
                    file_path: full_path.to_path_buf(),
                    error: Some(err),
                    is_strict: false,
                });
                continue;
            }
        };

        let negative_test = &metadata.negative_test;

        if metadata.is_strict {
            let params = TestParams {
                test262_root: &test262_root,
                file_path: Path::new(file_path),
                is_strict: true,
            };
            run_and_write(&params, negative_test);
        }

        if metadata.is_nostrict {
            let params = TestParams {
                test262_root: &test262_root,
                file_path: Path::new(file_path),
                is_strict: false,
            };
            run_and_write(&params, negative_test);
        }
    }
}

fn run_and_write(params: &TestParams, negative_test: &Option<metadata::NegativeTest>) {
    let test_error = run_test(&params).err();
    let test_error = filter_expected_error(negative_test, test_error);
    let outcome = TestOutcome {
        file_path: params.test262_root.join(params.file_path),
        error: test_error,
        is_strict: params.is_strict,
    };
    write_outcome(outcome);
}

fn write_outcome(outcome: TestOutcome) {
    // one json per line
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &outcome).unwrap();
    stdout.write_all(b"\n").unwrap();
}

// Turn expected errors into Ok's
fn filter_expected_error(
    negative_test: &Option<metadata::NegativeTest>,
    test_error: Option<TestError>,
) -> Option<TestError> {
    let test_error = match (negative_test, test_error) {
        (
            Some(metadata::NegativeTest {
                phase: metadata::Phase::Parse,
                ..
            }),
            Some(TestError::Load(_)),
        ) => None,
        (_, err) => err,
    };
    test_error
}

#[derive(Default)]
struct CliOptions {
    config_filename: String,
    filter: String,
}

fn parse_options() -> CliOptions {
    let mut opts: CliOptions = CliOptions::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--filter" {
            opts.filter = args.next().expect("--filter requires an argument");
        } else if opts.config_filename.is_empty() {
            opts.config_filename = arg;
        } else {
            panic!("multiple arguments for config filename");
        }
    }

    if opts.config_filename.is_empty() {
        panic!("usage: mcjs_test262 [--filter FILTER] <path/to/tests.yml>");
    }

    opts
}

#[allow(non_snake_case)]
#[derive(Deserialize)]
struct ConfigFile {
    test262Root: String,
    testFiles: Vec<String>,
}

fn run_test(params: &TestParams) -> Result<(), TestError> {
    let full_path = params.test262_root.join(params.file_path);

    let paths = [
        params.test262_root.join("harness/assert.js"),
        params.test262_root.join("harness/sta.js"),
        full_path.clone(),
    ];

    let options = mcjs_vm::Options {
        strict: params.is_strict,
    };
    let res = std::panic::catch_unwind(move || {
        let mut loader = mcjs_vm::Loader::new(None);
        let mut realm = mcjs_vm::Realm::new();

        for file_path in paths {
            process_file(&file_path, options.clone(), &mut realm, &mut loader)?;
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

fn parse_metadata(full_path: &PathBuf) -> Result<metadata::Metadata, TestError> {
    let mut f = File::open(full_path).unwrap();
    let mut content = String::new();
    f.read_to_string(&mut content).unwrap();

    let metadata = metadata::parse_header(&content).map_err(|err| {
        let message = format!("{:?}", err);
        TestError::HeaderParse(message)
    })?;
    Ok(metadata)
}

fn process_file(
    file_path: &Path,
    options: mcjs_vm::Options,
    realm: &mut mcjs_vm::Realm,
    loader: &mut mcjs_vm::Loader,
) -> Result<(), TestError> {
    let file_path_str = file_path.to_string_lossy().into_owned();
    let content = std::fs::read_to_string(file_path).map_err(TestError::Read)?;

    let chunk_fnid: mcjs_vm::bytecode::FnId = loader
        .load_script(Some(file_path_str), content)
        .map_err(TestError::Load)?;

    let mut interpreter = mcjs_vm::Interpreter::with_options(options, realm, loader, chunk_fnid);
    let mut probe = Probe::attach(&mut interpreter);
    probe.set_fuel(Fuel::Limited(200_000));

    interpreter
        .run()
        .map_err(|intrp_err| TestError::Run(intrp_err.error))?;

    Ok(())
}

struct TestParams<'a> {
    test262_root: &'a Path,
    file_path: &'a Path,
    is_strict: bool,
}

#[derive(Serialize)]
struct TestOutcome {
    file_path: PathBuf,
    error: Option<TestError>,
    is_strict: bool,
}

#[derive(Debug)]
enum TestError {
    HeaderParse(String),
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
            TestError::HeaderParse(msg) => ("header_parse", msg.clone()),
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

mod metadata {
    use serde::Deserialize;

    #[derive(Debug)]
    pub enum Error {
        Select,
        Parse(String),
    }

    pub fn parse_header(content: &str) -> Result<Metadata, Error> {
        let (_, after_start) = content.split_once("/*---").ok_or(Error::Select)?;
        let (payload, _) = after_start.split_once("---*/").ok_or(Error::Select)?;

        let header: FileHeader =
            serde_yaml::from_str(payload).map_err(|err| Error::Parse(err.to_string()))?;

        let (is_strict, is_nostrict) = match (
            header.flags.contains(&Flag::OnlyStrict),
            header.flags.contains(&Flag::NoStrict),
        ) {
            (true, true) => (true, true),
            (true, false) => (true, false),
            (false, true) => (false, true),
            (false, false) => (true, true),
        };

        Ok(Metadata {
            description: header.description,
            is_module: header.flags.contains(&Flag::Module),
            is_nostrict,
            is_strict,
            negative_test: header.negative,
        })
    }

    pub struct Metadata {
        pub description: String,
        pub is_module: bool,
        pub is_nostrict: bool,
        pub is_strict: bool,
        pub negative_test: Option<NegativeTest>,
    }

    #[derive(Deserialize)]
    struct FileHeader {
        description: String,
        negative: Option<NegativeTest>,
        #[serde(default)]
        flags: Vec<Flag>,
    }

    #[derive(Deserialize)]
    pub struct NegativeTest {
        pub phase: Phase,
        #[serde(rename = "type")]
        pub type_: String,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub enum Phase {
        Parse,
        Resolution,
        Runtime,
    }

    #[derive(Deserialize, PartialEq, Eq)]
    #[serde(rename_all = "camelCase")]
    enum Flag {
        NoStrict,
        OnlyStrict,
        Module,
        Generated,
        Raw,
    }
}
