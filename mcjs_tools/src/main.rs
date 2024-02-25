use std::path::PathBuf;

use anyhow::Result;
use mcjs_vm::{interpreter, stack};

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    eprintln!("params = {:?}", params);

    if params.filenames.is_empty() {
        eprintln!("No script files in input params!");
        return;
    }

    let app = AppData::new(params);

    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport = native_options
        .viewport
        .with_inner_size(egui::Vec2::new(1000.0, 800.0))
        .with_app_id("mcjs_tools")
        .with_title("mcjs Dev Tools");
    native_options.follow_system_theme = true;
    eframe::run_native("mcjs tools", native_options, Box::new(|_cc| Box::new(app))).unwrap();
}

fn parse_args() -> Result<manager::Params> {
    let mut params = manager::Params {
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
        params.main_directory = params.filenames[0].parent().map(|p| p.to_path_buf());
    }

    Ok(params)
}

struct AppData {
    params: manager::Params,
    intrp: manager::ManagedInterpreter,
    tree: egui_tiles::Tree<Pane>,
}

impl AppData {
    fn new(params: manager::Params) -> Self {
        let tree = egui_tiles::Tree::new_tabs(
            "main_tree",
            vec![
                Pane::Stack,
                Pane::SourceCode,
                Pane::Bytecode,
                Pane::PAST,
                Pane::Heap,
            ],
        );

        let intrp = init_interpreter(&params);

        AppData {
            params,
            intrp,
            tree,
        }
    }

    fn restart(&mut self) {
        self.intrp = init_interpreter(&self.params);
    }

    fn resume(&mut self) {
        // Skip the initialized "Ready" state, as it's not relevant for the user
        // to see
        loop {
            self.intrp.resume();
            match self.intrp.state() {
                manager::State::Ready(_) => {}
                _ => break,
            }
        }
    }
}

fn init_interpreter(params: &manager::Params) -> manager::ManagedInterpreter {
    let mut intrp =
        manager::ManagedInterpreter::new(params).expect("could not initialize interpreter");
    // skip the initial Ready state
    debug_assert!(matches!(intrp.state(), manager::State::Ready(_)));
    intrp.resume();
    intrp
}

enum Pane {
    SourceCode,
    Bytecode,
    PAST,
    Heap,
    Stack,
}

impl eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        enum Action {
            None,
            Resume,
            Restart,
            Next,
        }

        let mut action = Action::None;

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.intrp.state() {
                manager::State::Ready(script_ndx) => {
                    ui.vertical_centered(|ui| {
                        ui.label(format!("Ready to go with file #{}", script_ndx));
                        if ui.button("Start").clicked() {
                            action = Action::Resume;
                        }
                    });
                }
                manager::State::Suspended {
                    script_ndx,
                    intrp_state,
                    cause,
                } => {
                    ui.horizontal(|ui| {
                        if ui.button("Continue").clicked() {
                            action = Action::Resume;
                        }
                        if ui.button("Next").clicked() {
                            action = Action::Next;
                        }

                        ui.label(format!(
                            "{}/{}  {}",
                            script_ndx + 1,
                            self.params.filenames.len(),
                            self.params.filenames[*script_ndx].display()
                        ));
                    });

                    let mut behavior = TreeBehavior {
                        intrp_state: &intrp_state,
                        cause: &cause,
                    };
                    self.tree.ui(&mut behavior, ui);
                }
                manager::State::Finished => {
                    ui.vertical_centered(|ui| {
                        ui.label("Interpreter finished successfully. No more debugging!");
                        if ui.button("Restart").clicked() {
                            action = Action::Restart;
                        }
                    });
                }
                manager::State::Failed(err) => {
                    ui.heading("Interpreter failed:");
                    ui.label(format!("{:?}", err));
                    if ui.button("Restart").clicked() {
                        action = Action::Restart;
                    }
                }
            };
        });

        match action {
            Action::None => {}
            Action::Resume => {
                self.resume();
            }
            Action::Restart => {
                self.restart();
            }
            Action::Next => {
                self.intrp.next();
            }
        }
    }
}

struct TreeBehavior<'a> {
    intrp_state: &'a stack::InterpreterData,
    cause: &'a interpreter::SuspendCause,
}

impl<'a> egui_tiles::Behavior<Pane> for TreeBehavior<'a> {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        match pane {
            Pane::SourceCode => "Source code".into(),
            Pane::Bytecode => "Bytecode".into(),
            Pane::PAST => "PAST".into(),
            Pane::Heap => "Heap".into(),
            Pane::Stack => "Stack".into(),
        }
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        let text = match pane {
            Pane::SourceCode => "Source code",
            Pane::Bytecode => "Bytecode",
            Pane::PAST => "PAST",
            Pane::Heap => "Heap",
            Pane::Stack => return ui_stack_pane(ui, self.intrp_state, self.cause),
        };

        if ui
            .add(egui::Button::new(text).sense(egui::Sense::drag()))
            .drag_started()
        {
            egui_tiles::UiResponse::DragStarted
        } else {
            egui_tiles::UiResponse::None
        }
    }
}

fn ui_stack_pane(
    ui: &mut egui::Ui,
    _intrp_state: &stack::InterpreterData,
    _cause: &interpreter::SuspendCause,
) -> egui_tiles::UiResponse {
    let drag_button_res = ui.add(egui::Button::new("Stack").sense(egui::Sense::drag()));

    if drag_button_res.drag_started() {
        egui_tiles::UiResponse::DragStarted
    } else {
        egui_tiles::UiResponse::None
    }
}

mod manager {
    use std::{io::Read, path::PathBuf};
    use thiserror::Error;

    use mcjs_vm::{
        bytecode,
        interpreter::{self, debugger::Fuel, Exit, Interpreter},
        stack, Loader, Realm,
    };

    #[derive(Debug)]
    pub struct Params {
        pub main_directory: Option<PathBuf>,
        pub filenames: Vec<PathBuf>,
    }

    pub struct ManagedInterpreter {
        script_fnids: Vec<bytecode::FnId>,
        realm: Realm,
        loader: Loader,
        state: State,
        dbg: interpreter::debugger::DebuggingState,
    }

    pub enum State {
        /// Ready to go with the next file
        Ready(usize),
        Suspended {
            script_ndx: usize,
            intrp_state: stack::InterpreterData,
            cause: interpreter::SuspendCause,
        },
        Finished,
        Failed(interpreter::InterpreterError),
    }

    type Result<T> = std::result::Result<T, Error>;

    #[derive(Debug, Error)]
    pub enum Error {
        #[error("no script files")]
        NoFiles,

        #[error("I/O error: {0}")]
        Io(#[from] std::io::Error),

        #[error("VM error: {0}")]
        Vm(#[from] mcjs_vm::Error),

        #[error("debugger error: {0}")]
        Debugger(#[from] interpreter::debugger::BreakpointError),
    }

    impl ManagedInterpreter {
        pub fn new(params: &Params) -> Result<Self> {
            if params.filenames.is_empty() {
                return Err(Error::NoFiles);
            }

            let mut loader = Loader::new(params.main_directory.clone());
            let realm = Realm::new(&mut loader);

            let mut dbg = interpreter::debugger::DebuggingState::new();

            let mut script_fnids = Vec::new();
            for filename in &params.filenames {
                let mut content = String::new();
                std::fs::File::open(filename)?.read_to_string(&mut content)?;

                let filename = filename.to_string_lossy().into_owned();
                let main_fnid = loader.load_script(Some(filename), content)?;
                script_fnids.push(main_fnid);

                // Place a breakpoint at the start of each file
                dbg.set_instr_breakpoint(bytecode::GlobalIID(main_fnid, bytecode::IID(0)))?;
            }

            Ok(ManagedInterpreter {
                script_fnids,
                realm,
                loader,
                state: State::Ready(0),
                dbg,
            })
        }

        pub fn state(&self) -> &State {
            match &self.state {
                State::Ready(script_ndx) | State::Suspended { script_ndx, .. } => {
                    assert!(*script_ndx < self.script_fnids.len());
                }
                _ => {}
            }

            &self.state
        }

        pub fn resume(&mut self) {
            let state = std::mem::replace(&mut self.state, State::Finished);
            let (script_ndx, mut intrp) = match state {
                State::Ready(script_ndx) => {
                    let main_fnid = self.script_fnids[script_ndx];
                    let intrp = Interpreter::new(&mut self.realm, &mut self.loader, main_fnid);
                    (script_ndx, intrp)
                }
                State::Suspended {
                    script_ndx,
                    intrp_state,
                    ..
                } => {
                    let intrp = Interpreter::resume(&mut self.realm, &mut self.loader, intrp_state);
                    (script_ndx, intrp)
                }
                State::Finished | State::Failed(_) => {
                    return;
                }
            };

            intrp.set_debugging_state(&mut self.dbg);

            let next_script_ndx = script_ndx + 1;
            assert!(next_script_ndx <= self.script_fnids.len());

            self.state = match intrp.run() {
                Ok(Exit::Finished(_)) => {
                    if next_script_ndx == self.script_fnids.len() {
                        State::Finished
                    } else {
                        State::Ready(next_script_ndx)
                    }
                }
                Ok(Exit::Suspended { intrp_state, cause }) => State::Suspended {
                    script_ndx,
                    intrp_state,
                    cause,
                },
                Err(err) => State::Failed(err),
            }
        }

        pub fn next(&mut self) {
            self.dbg.set_fuel(Fuel::Limited(1));
            self.resume();
        }
    }
}
