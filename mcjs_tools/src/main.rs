#![allow(dead_code)]
#![allow(unused_must_use)]

use std::path::PathBuf;
use std::pin::Pin;

use anyhow::Result;
use mcjs_vm::interpreter::debugger::Probe;

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    eprintln!("params = {:?}", params);

    let si = interpreter_manager::StandaloneInterpreter::new(params);
    let app = AppData { si };

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("mcjs tools", native_options, Box::new(|_cc| Box::new(app))).unwrap();
}

fn parse_args() -> Result<interpreter_manager::Params> {
    let mut params = interpreter_manager::Params {
        main_directory: None,
        filenames: Vec::new(),
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "-b" && arg == "--base-path" {
            let path = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("-b|--base-path requires an argument"))?;
            let path = PathBuf::from(path);
            params.main_directory = Some(path);
        } else {
            params.filenames.push(arg.into());
        }
    }

    if params.main_directory.is_none() && params.filenames.len() == 1 {
        params.main_directory = params.filenames[0]
            .parent()
            .map(|p| p.to_path_buf().canonicalize().unwrap());
    }

    Ok(params)
}

struct AppData {
    si: Pin<Box<interpreter_manager::StandaloneInterpreter>>,
}

impl<'a> eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        use interpreter_manager::State;

        egui::SidePanel::left("sidebar").show(ctx, |ui| {
            ui.button("NEXT");
            if ui.button("CONTINUE").clicked() {
                self.si.resume();
            }
            ui.button("RESTART");
            ui.button("DELETE");

            ui.label("Double click on a <source code range> to set a breakpoint");

            ui.heading("SOURCE BREAKPOINTS");
            ui.heading("INSTR. BREAKPOINTS");
            ui.heading("STACK");
        });

        egui::SidePanel::right("source_code").show(ctx, |ui| {
            let state = self.si.state_mut();
            let text = match state {
                State::Ready { filename_ndx } => format!("ready for file #{}", filename_ndx),
                State::Finished => format!("finished"),
                State::Suspended(intrp) => {
                    let probe = Probe::attach(intrp);
                    format!("suspended at {:?}", probe.giid())
                }
                State::Failed(err) => format!("failure: {:?}", err),
            };

            ui.code(text);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            use mcjs_vm::bytecode::{ConstIndex, Instr, VReg};
            let instrs = &[
                Instr::Copy {
                    dst: VReg(1),
                    src: VReg(2),
                },
                Instr::LoadConst(VReg(3), ConstIndex(2)),
                Instr::ObjCreateEmpty(VReg(4)),
            ];

            for (ndx, instr) in instrs.iter().enumerate() {
                ui.monospace(format!("{:4} {:?}", ndx, instr));
            }
        });
    }
}

mod interpreter_manager {
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::{marker::PhantomPinned, path::Path};

    use mcjs_vm::{
        interpreter::{Exit, InterpreterError},
        Interpreter, Loader, Realm,
    };

    use anyhow::{anyhow, Result};

    #[derive(Debug)]
    pub struct Params {
        pub main_directory: Option<PathBuf>,
        pub filenames: Vec<PathBuf>,
    }

    #[derive(Debug)]
    pub enum Error<'a> {
        Generic(anyhow::Error),
        Interpreter(InterpreterError<'a>),
    }

    pub struct StandaloneInterpreter {
        realm: Realm,
        loader: Loader,
        filenames: Vec<PathBuf>,
        state: State<'static>,
        _pin: PhantomPinned,
    }
    pub enum State<'a> {
        /// Ready to process the next file in the sequence
        Ready {
            filename_ndx: usize,
        },

        /// Finished successfully. Won't proceed to any other state.
        Finished,
        Suspended(Interpreter<'a>),
        Failed(Error<'a>),
    }

    impl StandaloneInterpreter {
        pub fn new(params: Params) -> Pin<Box<StandaloneInterpreter>> {
            let mut loader = Loader::new(params.main_directory);
            let realm = Realm::new(&mut loader);
            let si = StandaloneInterpreter {
                realm,
                loader,
                filenames: params.filenames,
                state: State::Ready { filename_ndx: 0 },
                _pin: PhantomPinned,
            };

            Box::pin(si)
        }

        pub fn state_mut<'a>(self: &'a mut Pin<Box<Self>>) -> &'a mut State<'static> {
            // Safe because I only return `interpreter`, which is the
            // part of the struct that doesn't have to stay pinned
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };
            &mut self_.state
        }

        pub fn resume(self: &mut Pin<Box<Self>>) {
            // Safe because nowhere in this function we move neither
            // `realm` or `loader` (`state`, and therefore the
            // contained `Interpreter` should be fine to move
            // temporarily, as long as we restore it correctly after
            // return)
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };

            let state = std::mem::replace(&mut self_.state, State::Finished);

            self_.state = match state {
                State::Ready { filename_ndx } => {
                    let filename = &self_.filenames[filename_ndx];
                    println!();
                    println!(
                        "(starting loop for file #{}: {})",
                        filename_ndx,
                        filename.to_string_lossy().into_owned()
                    );

                    match start_intrp(&filename, &mut self_.realm, &mut self_.loader) {
                        Ok(intrp) => {
                            let intrp = unsafe { std::mem::transmute(intrp) };
                            State::Suspended(intrp)
                        }
                        Err(err) => State::Failed(Error::Generic(err)),
                    }
                }
                cur @ State::Failed(_) | cur @ State::Finished => {
                    // nothing  to do
                    cur
                }
                State::Suspended(intrp) => match intrp.run() {
                    Ok(exit) => match exit {
                        Exit::Finished(_) => State::Finished,
                        Exit::Suspended(new_intrp) => State::Suspended(new_intrp),
                    },
                    Err(err) => State::Failed(Error::Interpreter(err)),
                },
            };
        }
    }

    fn start_intrp<'a>(
        filename: &Path,
        realm: &'a mut Realm,
        loader: &'a mut Loader,
    ) -> Result<Interpreter<'a>> {
        let filename_str = filename.to_string_lossy().into_owned();
        let script_text = std::fs::read_to_string(filename)
            .map_err(|err| anyhow!("read error: {:?}: {:?}", filename, err))?;

        let main_fnid = loader
            .load_script(Some(filename_str.clone()), script_text)
            .map_err(|err| anyhow!("compile error: {:?}: {:?}", filename_str, err))?;

        Ok(Interpreter::new(realm, loader, main_fnid))
    }
}
