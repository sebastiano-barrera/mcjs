#![allow(dead_code)]
#![allow(unused_must_use)]

use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;

use anyhow::Result;
use mcjs_vm::bytecode;
use mcjs_vm::interpreter::debugger::Probe;
use mcjs_vm::interpreter::Fuel;

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    eprintln!("params = {:?}", params);

    let si = interpreter_manager::StandaloneInterpreter::new(params);
    let app = AppData {
        si,
        recent_state_change: false,
    };

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
    recent_state_change: bool,
}

impl<'a> eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        use interpreter_manager::State;

        let recent_state_change = self.recent_state_change;
        self.recent_state_change = false;

        enum Action {
            Next,
            Restart,
            Continue,
            None,
        }

        match self.si.state_mut() {
            State::Ready { filename_ndx } => {
                let should_start = egui::CentralPanel::default()
                    .show(ctx, |ui| {
                        ui.label(format!("Ready to proceed with file #{}", *filename_ndx + 1));
                        ui.button("Start").clicked()
                    })
                    .inner;
                if should_start {
                    self.si.resume();
                }
                return;
            }
            State::Finished => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.label("Interpreter finished successfully. No debugging!");
                });
                return;
            }
            State::Failed(interpreter_manager::Error::Generic(err)) => {
                egui::SidePanel::right("simple_central_panel").show(ctx, |ui| {
                    ui.label(format!("Error: {:?}", err));
                });
                return;
            }
            _ => {}
        };

        let action = egui::SidePanel::left("sidebar")
            .show(ctx, |ui| {
                let mut action = Action::None;

                if let Some(err_message) = self.si.error_message() {
                    ui.heading("interpreter failed");
                    ui.label(&err_message);
                }

                if ui.button("NEXT").clicked() {
                    action = Action::Next;
                }
                if ui.button("CONTINUE").clicked() {
                    action = Action::Continue;
                }
                if ui.button("RESTART").clicked() {
                    action = Action::Restart;
                }
                ui.button("DELETE");

                let probe = self.si.probe_mut().unwrap();
                let status_text = format!("suspended at {:?}", probe.giid());

                ui.separator();
                ui.label(status_text);

                ui.separator();
                ui.label("Double click on a <source code range> to set a breakpoint");

                ui.heading("SOURCE BREAKPOINTS");
                let loader = probe.loader();
                ui.vertical(|ui| {
                    for (brid, _) in probe.source_breakpoints() {
                        let break_range = loader.get_break_range(brid).unwrap();
                        let source_map = loader.get_source_map(brid.module_id()).unwrap();
                        let loc = source_map.lookup_char_pos(break_range.lo);
                        let filename = loc.file.name.to_string();
                        ui.label(format!("{}:{}", filename, loc.line));
                    }
                });

                ui.heading("INSTR. BREAKPOINTS");
                ui.vertical(|ui| {
                    for giid in probe.instr_breakpoints() {
                        ui.label(format!("{:?}", giid));
                    }
                });

                ui.heading("STACK");
                ui.vertical(|ui| {
                    for (ndx, frame) in probe.frames().enumerate() {
                        let iid = if ndx == 0 {
                            probe.giid().1
                        } else {
                            frame.header().return_target.unwrap().0
                        };
                        ui.label(format!("{:?}:{:?}", frame.header().fn_id, iid));
                    }
                });

                action
            })
            .inner;

        if let State::Suspended(_) = self.si.state_mut() {
            let probe = self.si.probe_mut().unwrap();

            egui::SidePanel::right("source_code").show(ctx, |ui| {
                source_code_view(ui, &probe);
            });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let probe = self.si.probe_mut().unwrap();

            let giid = probe.giid();
            let fnid = giid.0;
            let func = probe.loader().get_function(fnid).unwrap();

            egui::ScrollArea::vertical().show(ui, |ui| {
                for (ndx, instr) in func.instrs().iter().enumerate() {
                    let is_current = ndx == giid.1 .0 as usize;
                    let res = ui.selectable_label(is_current, format!("{:4} {:?}", ndx, instr));
                    if recent_state_change && is_current {
                        res.scroll_to_me(None);
                    }
                }
            });
        });

        match action {
            Action::None => {}
            Action::Next => {
                let mut probe = self.si.probe_mut().unwrap();
                probe.set_fuel(Fuel::Limited(1));
                self.si.resume();
                self.recent_state_change = true;
            }
            Action::Continue => {
                self.si.resume();
                self.recent_state_change = true;
            }
            Action::Restart => {
                self.si.restart();
                self.si.resume();
                self.recent_state_change = true;
            }
        }
    }
}

fn source_code_view(ui: &mut egui::Ui, probe: &Probe) {
    let giid = probe.giid();
    egui::ScrollArea::vertical().show(ui, |ui| match fetch_source_code(probe, giid) {
        Some(text) => ui.code(&*text),
        None => ui.label("(No source code)"),
    });
}

fn fetch_source_code<'a>(
    probe: &'a Probe<'a, '_>,
    giid: bytecode::GlobalIID,
) -> Option<Rc<String>> {
    let fnid = giid.0;

    let loader = probe.loader();
    let source_map = loader.get_source_map(fnid.0)?;
    let mut break_ranges = loader.function_breakranges(fnid).unwrap().peekable();

    // All break ranges must belong to the same file, so we just peek one and use it to get a
    // ptr to that swc_common::SourceFile.
    // Then we use source file's offset in the source map to make sure that the markers are
    // expressed in file-local offsets.
    let (_, brange) = break_ranges.peek()?;
    let source_file = source_map.lookup_byte_offset(brange.lo).sf;

    Some(Rc::clone(&source_file.src))
}

mod interpreter_manager {
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::{marker::PhantomPinned, path::Path};

    use mcjs_vm::interpreter::debugger::Probe;
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

        pub fn restart(self: &mut Pin<Box<Self>>) {
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };
            self_.state = State::Ready { filename_ndx: 0 };
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

        pub fn probe_mut<'a>(self: &'a mut Pin<Box<Self>>) -> Option<Probe<'a, 'static>> {
            // Safe because the Probe allows mutating the Interpreter
            // or accessing the Loader and/or Realm read-only (as &T)
            match self.state_mut() {
                State::Suspended(intrp) => Some(Probe::attach(intrp)),
                State::Failed(Error::Interpreter(intrp_err)) => Some(intrp_err.probe()),
                _ => None,
            }
        }

        pub fn error_message(self: &mut Pin<Box<Self>>) -> Option<String> {
            match self.state_mut() {
                State::Suspended(_) => None,
                State::Failed(Error::Interpreter(intrp_err)) => {
                    Some(format!("{:?}", intrp_err.error))
                }
                _ => None,
            }
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
