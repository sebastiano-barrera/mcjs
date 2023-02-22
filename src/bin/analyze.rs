struct Options {
    script_filename: String,
}

impl Options {
    fn parse() -> Option<Self> {
        match std::env::args().skip(1).next() {
            Some(script_filename) => Some(Options { script_filename }),
            None => {
                eprintln!(
                    "Usage: {} <input.js>",
                    std::env::current_exe()
                        .unwrap()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                );
                None
            }
        }
    }
}

pub fn main() {
    let opts = Options::parse().unwrap();

    let mut vm = mcjs::VM::new();
    vm.add_include_path(std::env::current_dir().unwrap())
        .unwrap();
    vm.options_mut().debug_dump_module = true;
    let flags = mcjs::InterpreterFlags {
        indent_level: 0,
        jit_mode: mcjs::JitMode::Compile,
    };
    vm.load_module_ex(&opts.script_filename, flags).unwrap();

    let trace = vm.take_trace();
    match trace {
        Some(trace) => {
            trace.dump();
        }
        None => {
            eprintln!("<none>");
        }
    }

    let sink = vm.take_sink();
    eprintln!("sink = {:#?}", sink);
}
