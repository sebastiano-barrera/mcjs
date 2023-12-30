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

    let mut stdout = std::io::stdout().lock();

    let tests_count = config.testFiles.len();
    for (ndx, file_path) in config.testFiles.iter().enumerate() {
        if !file_path.contains(&opts.filter) {
            continue;
        }

        eprintln!("test {}/{}", ndx, tests_count);

        let outcome = run_test(TestParams {
            test262_root: &test262_root,
            file_path: &Path::new(file_path),
        });

        // one json per line
        serde_json::to_writer(&mut stdout, &outcome).unwrap();
        stdout.write(b"\n").unwrap();
    }
}

struct CliOptions {
    config_filename: String,
    filter: String,
}

impl Default for CliOptions {
    fn default() -> Self {
        CliOptions {
            config_filename: String::new(),
            filter: String::new(),
        }
    }
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

fn run_test(params: TestParams) -> TestOutcome {
    let full_path = params.test262_root.join(params.file_path);

    let metadata = {
        let mut f = File::open(&full_path).unwrap();
        let mut content = String::new();
        f.read_to_string(&mut content).unwrap();
        match metadata::parse_header(&content) {
            Ok(metadata) => metadata,
            Err(err) => {
                let message = format!("{:?}", err);
                return TestOutcome {
                    file_path: full_path,
                    error: Some(TestError::HeaderParse(message)),
                };
            }
        }
    };

    let paths = [
        params.test262_root.join("harness/assert.js"),
        params.test262_root.join("harness/sta.js"),
        full_path.clone(),
    ];

    let res = std::panic::catch_unwind(move || {
        let mut loader = mcjs_vm::Loader::new(None);
        let mut realm = mcjs_vm::Realm::new();

        for file_path in paths {
            process_file(&file_path, &mut realm, &mut loader)?;
        }
        Ok(())
    });

    // Flatten a panic into a regular TestError
    let res = match res {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => match (&metadata.negative_test, err) {
            (
                Some(metadata::NegativeTest {
                    phase: metadata::Phase::Parse,
                    ..
                }),
                TestError::Load(_),
            ) => Ok(()),
            (_, err) => Err(err),
        },
        Err(panic_value) => {
            let message = match panic_value.downcast_ref::<String>() {
                Some(s) => format!("{}", s),
                None => "<not a string>".to_string(),
            };
            Err(TestError::Panic(message))
        }
    };

    TestOutcome {
        file_path: full_path,
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

    let mut interpreter = mcjs_vm::Interpreter::new(realm, loader, chunk_fnid);
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
}

#[derive(Serialize)]
struct TestOutcome {
    file_path: PathBuf,
    error: Option<TestError>,
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
