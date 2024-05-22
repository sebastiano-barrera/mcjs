use std::{io::Write, path::PathBuf};

use anyhow::Result;
use mcjs_vm::{
    heap,
    interpreter::{self, debugger::InstrBreakpoint, stack},
};

use jemallocator::Jemalloc;

#[global_allocator]
static MY_ALLOC: Jemalloc = Jemalloc;

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    if params.filenames.is_empty() {
        eprintln!("params error: No script files in input params!");
        return;
    }

    let app = match AppData::new(params) {
        Ok(app) => app,
        Err(err) => {
            eprintln!("could not init application: {}", err);
            return;
        }
    };

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

    focus_frame_ndx: usize,
    stack_view: stack_view::State,
    source_view: source_view::State,
    heap_view: heap_view::State,
    open_oids: Vec<heap::ObjectId>,
    highlight: widgets::Highlight,

    save_error_dialog: Option<String>,
    bkpt_error_dialog: Option<String>,
    toast: widgets::Toast,
    error_dialog_toast: widgets::Toast,
}

impl AppData {
    fn new(params: manager::Params) -> AppResult<Self> {
        let tree = egui_tiles::Tree::new_tabs(
            "main_tree",
            vec![
                Pane::Bytecode,
                Pane::Stack,
                Pane::SourceCode,
                Pane::PAST,
                Pane::Heap,
            ],
        );

        let intrp = init_interpreter(&params)?;

        let mut app = AppData {
            params,
            intrp,
            tree,
            focus_frame_ndx: 0,
            stack_view: stack_view::State::default(),
            source_view: source_view::State::default(),
            heap_view: heap_view::State::default(),
            open_oids: Vec::new(),
            highlight: widgets::Highlight::None,
            save_error_dialog: None,
            bkpt_error_dialog: None,
            toast: widgets::Toast::default(),
            error_dialog_toast: widgets::Toast::default(),
        };

        // Purposefully ignore the error, not a time to show it
        let _ = app.load_tree_layout();

        Ok(app)
    }

    fn restart(&mut self) {
        self.intrp.restart();
        self.intrp.resume();
    }

    fn resume(&mut self) {
        self.intrp.resume();
    }

    fn save_tree_layout(&mut self) -> AppResult<()> {
        let path = state_file_path()?;

        let state = StateFileData {
            tree_layout: Some(self.tree.clone()),
        };
        let encoded_content = rmp_serde::to_vec(&state).expect("bug: msgpack serialization error!");

        let mut file = std::fs::File::create(path)?;
        file.write_all(&encoded_content)?;

        self.toast.start("Layout saved.".to_string());
        Ok(())
    }

    fn load_tree_layout(&mut self) -> AppResult<()> {
        let path = state_file_path()?;
        let file = std::fs::File::open(path)?;
        let config: StateFileData =
            rmp_serde::from_read(file).map_err(|err| AppError::Format(err.to_string()))?;

        if let Some(tree) = &config.tree_layout {
            self.tree = tree.clone();
        }

        self.toast.start("Layout restored.".to_string());
        Ok(())
    }
}

fn state_file_path() -> Result<PathBuf, AppError> {
    const DIR_SUFFIX: &str = "mcjs_tools";
    const FILENAME: &str = "state";

    let dir = dirs::preference_dir()
        .map(|p| p.join(DIR_SUFFIX))
        .or_else(|| {
            std::env::current_exe()
                .ok()
                .map(|exe_path| exe_path.parent().unwrap().join(DIR_SUFFIX))
        })
        .ok_or_else(|| AppError::Env("can't determine configuration file location!".to_string()))?;

    std::fs::create_dir_all(&dir)?;

    Ok(dir.join(FILENAME))
}

type AppResult<T> = std::result::Result<T, AppError>;
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("environment error: {0}")]
    Env(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("format error: {0}")]
    Format(String),

    #[error("VM error: {0}")]
    VmManager(manager::Error),
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StateFileData {
    tree_layout: Option<egui_tiles::Tree<Pane>>,
}

fn init_interpreter(params: &manager::Params) -> AppResult<manager::ManagedInterpreter> {
    let mut intrp = manager::ManagedInterpreter::new(params).map_err(AppError::VmManager)?;
    // skip the initial Ready state
    debug_assert!(matches!(intrp.state(), manager::State::Ready(_)));
    intrp.resume();
    Ok(intrp)
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
enum Pane {
    SourceCode,
    Bytecode,
    PAST,
    Heap,
    Stack,
}

#[derive(Debug, PartialEq, Eq, Default)]
enum Action {
    #[default]
    None,
    Resume,
    Restart,
    Next,
    Into,
    SetHighlight(widgets::Highlight),
    OpenObject(heap::ObjectId),
    SaveLayout,
    LoadLayout,
    SetInstrBreakpoint(mcjs_vm::GlobalIID),
    ClearInstrBreakpoint(mcjs_vm::GlobalIID),
    SetBreakOnThrow(bool),
    SetBreakOnUnhandledThrow(bool),
}
impl Action {
    fn set_if_none(&mut self, other: Action) {
        if let Action::None = self {
            *self = other;
        }
    }
}

impl eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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
                        if ui.button("Into").clicked() {
                            action = Action::Into;
                        }

                        ui.menu_button("Break⏷", |ui| {
                            let mut on = self.intrp.debugging_state().should_break_on_throw();
                            if ui.checkbox(&mut on, "on throw").changed() {
                                action = Action::SetBreakOnThrow(on);
                            }

                            let mut on = self
                                .intrp
                                .debugging_state()
                                .should_break_on_unhandled_throw();
                            if ui.checkbox(&mut on, "on unhandled throw").changed() {
                                action = Action::SetBreakOnUnhandledThrow(on);
                            }
                        });

                        match cause {
                            interpreter::SuspendCause::Breakpoint => ui.label("On breakpoint"),
                            // TODO Allow "Open object" here as well
                            interpreter::SuspendCause::Exception(_) => ui.label("On exception"),
                        };

                        ui.label(format!(
                            "{}/{}  {}",
                            script_ndx + 1,
                            self.params.filenames.len(),
                            self.params.filenames[*script_ndx].display()
                        ));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Restart").clicked() {
                                action = Action::Restart;
                            }

                            ui.menu_button("View⏷", |ui| {
                                if ui.button("Save").clicked() {
                                    action = Action::SaveLayout;
                                }
                                if ui.button("Load").clicked() {
                                    action = Action::LoadLayout;
                                }
                            });

                            self.toast.update(ui);
                        });
                    });

                    let mut behavior = TreeBehavior {
                        intrp_state,
                        frame_focus_ndx: &mut self.focus_frame_ndx,
                        loader: self.intrp.loader(),
                        heap: self.intrp.heap(),
                        stack_view: &mut self.stack_view,
                        source_view: &mut self.source_view,
                        action: &mut action,
                        highlight: self.highlight,
                        dbg: self.intrp.debugging_state(),
                        heap_view: &mut self.heap_view,
                        open_object_ids: &self.open_oids,
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
                    ui.label(err.message());
                    ui.horizontal(|ui| {
                        if ui.button("Restart").clicked() {
                            action = Action::Restart;
                        }

                        let giid = err.giid();
                        // if the stack is empty, we don't have enough info to locate the
                        // 'failed instruction'
                        if ui
                            .add_enabled(
                                giid.is_some(),
                                egui::Button::new("Place breakpoint at failed instruction"),
                            )
                            .clicked()
                        {
                            self.error_dialog_toast
                                .start(format!("Set breakpoint at {:?}", giid));
                            action = Action::SetInstrBreakpoint(giid.unwrap());
                        }

                        let mut on = self.intrp.debugging_state().should_break_on_throw();
                        if ui.checkbox(&mut on, "Break on throw").changed() {
                            action = Action::SetBreakOnThrow(on);
                        }

                        let mut on = self
                            .intrp
                            .debugging_state()
                            .should_break_on_unhandled_throw();
                        if ui.checkbox(&mut on, "Break on unhandled throw").changed() {
                            action = Action::SetBreakOnUnhandledThrow(on);
                        }
                    });
                    self.error_dialog_toast.update(ui);
                }
            };
        });

        simple_dialog(ctx, "Save/Load Layout: Error!", &mut self.save_error_dialog);
        simple_dialog(ctx, "Breakpoint error", &mut self.bkpt_error_dialog);

        match action {
            Action::None => {}
            Action::Resume => {
                self.resume();
                self.source_view.request_scroll();
            }
            Action::Restart => {
                self.restart();
                self.source_view.request_scroll();
            }
            Action::Next => {
                if let Err(err) = self.intrp.next() {
                    self.bkpt_error_dialog = Some(err.to_string());
                } else {
                    self.source_view.request_scroll();
                }
            }
            Action::Into => {
                self.intrp.next_into();
                self.source_view.request_scroll();
            }
            Action::SetHighlight(highlight) => {
                self.highlight = highlight;
            }
            Action::OpenObject(oid) => {
                self.open_oids.push(oid);
            }
            Action::SaveLayout => {
                self.save_error_dialog = self.save_tree_layout().err().map(|err| err.to_string());
            }
            Action::LoadLayout => {
                self.save_error_dialog = self.load_tree_layout().err().map(|err| err.to_string());
            }
            Action::SetInstrBreakpoint(giid) => {
                let dbg = self.intrp.debugging_state_mut();
                if let Err(err) = dbg.set_instr_bkpt(giid, InstrBreakpoint::default()) {
                    self.bkpt_error_dialog = Some(err.to_string());
                }
            }
            Action::ClearInstrBreakpoint(giid) => {
                let dbg = self.intrp.debugging_state_mut();
                dbg.clear_instr_bkpt(giid);
            }

            Action::SetBreakOnThrow(value) => {
                self.intrp.debugging_state_mut().set_break_on_throw(value);
            }
            Action::SetBreakOnUnhandledThrow(value) => {
                self.intrp
                    .debugging_state_mut()
                    .set_break_on_unhandled_throw(value);
            }
        }
    }
}

fn simple_dialog(ctx: &egui::Context, title: &str, message: &mut Option<String>) {
    let clear_error = if let Some(err_message) = &message {
        egui::Window::new(title)
            .resizable(false)
            .show(ctx, |ui| {
                ui.monospace(err_message);
                ui.button("OK").clicked()
            })
            .and_then(|response| response.inner)
            .unwrap_or(false)
    } else {
        false
    };

    if clear_error {
        *message = None;
    }
}

struct TreeBehavior<'a> {
    intrp_state: &'a stack::InterpreterData,
    loader: &'a mcjs_vm::Loader,
    heap: &'a heap::Heap,
    dbg: &'a interpreter::debugger::DebuggingState,

    /// The currently focused frame index.
    ///
    /// `0` is for the stack top. Higher numbers are for frames lower in the stack.
    frame_focus_ndx: &'a mut usize,

    stack_view: &'a mut stack_view::State,
    source_view: &'a mut source_view::State,
    heap_view: &'a mut heap_view::State,
    open_object_ids: &'a [heap::ObjectId],

    action: &'a mut Action,
    highlight: widgets::Highlight,
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
        // frame_focus_ndx is 0=top, while frames() is first=bottom
        let frame = if *self.frame_focus_ndx < self.intrp_state.len() {
            let frame_ndx = self.intrp_state.len() - 1 - *self.frame_focus_ndx;
            Some(self.intrp_state.nth_frame(frame_ndx))
        } else {
            None
        };

        let text = match (pane, frame) {
            (Pane::SourceCode, Some(frame)) => {
                return source_view::show(ui, self.source_view, &frame, self.loader).tiles;
            }
            (Pane::Bytecode, Some(frame)) => {
                let is_breakpoint_set = |giid| self.dbg.instr_bkpt_at(&giid).is_some();
                let res =
                    bytecode_view::show(ui, self.loader, &frame, self.highlight, is_breakpoint_set);
                self.action.set_if_none(res.action);
                return res.tiles;
            }
            (Pane::PAST, _) => "PAST",
            (Pane::Heap, _) => {
                let res =
                    heap_view::show(ui, &mut self.heap_view, &self.open_object_ids, &self.heap);
                return res.tiles;
            }
            (Pane::Stack, _) => {
                use stack_view::Action;
                let action = stack_view::show(
                    self.stack_view,
                    ui,
                    self.intrp_state,
                    self.loader,
                    *self.frame_focus_ndx,
                );
                return match action {
                    Action::SetFrameIndex(ndx) => {
                        *self.frame_focus_ndx = ndx;
                        egui_tiles::UiResponse::None
                    }
                    Action::TabDragStarted => egui_tiles::UiResponse::DragStarted,
                    Action::None => egui_tiles::UiResponse::None,
                };
            }
            (_, None) => {
                ui.label(" - No frame - ");
                return egui_tiles::UiResponse::None;
            }
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

mod stack_view {
    use mcjs_vm::interpreter::stack;

    #[derive(Default)]
    pub struct State {}

    pub enum Action {
        SetFrameIndex(usize),
        TabDragStarted,
        None,
    }

    pub fn show(
        _view_state: &mut State,
        ui: &mut egui::Ui,
        intrp_state: &stack::InterpreterData,
        loader: &mcjs_vm::Loader,
        focus_frame_ndx: usize,
    ) -> Action {
        let mut action = Action::None;

        let drag_button_res = ui.add(egui::Button::new("Stack").sense(egui::Sense::drag()));
        if drag_button_res.drag_started() {
            action = Action::TabDragStarted;
        }

        ui.with_layout(
            egui::Layout::top_down(egui::Align::Min).with_cross_justify(true),
            |ui| {
                for (frame_ndx, frame) in intrp_state.frames().enumerate() {
                    let header = frame.header();
                    let lookup = loader.lookup_function(header.fnid).unwrap();
                    let filename = lookup.source_file.name.to_string();

                    let point_str = format!("{:?}:{:?} - {:?}", header.fnid, header.iid, filename);
                    let is_checked = frame_ndx == focus_frame_ndx;
                    if ui.selectable_label(is_checked, point_str).clicked() {
                        action = Action::SetFrameIndex(frame_ndx);
                    }
                }
            },
        );

        action
    }
}

mod bytecode_view {
    use super::{widgets, Action};
    use mcjs_vm::{bytecode, interpreter, interpreter::stack};

    #[derive(Default)]
    pub struct State {}

    #[derive(Default)]
    pub struct Response {
        pub tiles: egui_tiles::UiResponse,
        pub action: Action,
    }

    pub fn show(
        ui: &mut egui::Ui,
        loader: &mcjs_vm::Loader,
        frame: &stack::Frame,
        highlight: widgets::Highlight,
        is_breakpoint_set: impl Fn(bytecode::GlobalIID) -> bool,
    ) -> Response {
        let mut action = Action::None;

        let drag_start = ui
            .add(egui::Button::new("Bytecode").sense(egui::Sense::drag()))
            .drag_started();

        let header = frame.header();
        let fnid = header.fnid;
        let cur_iid = header.iid;

        let func = match loader.get_function(fnid) {
            Some(func) => func,
            None => {
                ui.label(format!("No such function {:?}", fnid));
                return Response::default();
            }
        };

        let instrs = func.instrs();
        let row_height = ui.spacing().interact_size.y;
        egui::ScrollArea::both().show_rows(ui, row_height, instrs.len(), |ui, ndx_range| {
            let cur_iid_ndx = cur_iid.0 as usize; // copy to move into closure
            egui::Grid::new("bytecode->instrs")
                .num_columns(3)
                .with_row_color(move |ndx, style| {
                    if ndx + ndx_range.start == cur_iid_ndx {
                        Some(style.visuals.extreme_bg_color)
                    } else {
                        None
                    }
                })
                .show(ui, |ui| {
                    for iid in ndx_range {
                        use mcjs_vm::bytecode::InstrDescriptor;

                        let instr = &instrs[iid];
                        let iid = bytecode::IID(iid.try_into().unwrap());

                        ui.horizontal(|ui| {
                            ui.set_width(40.0);
                            let giid = bytecode::GlobalIID(fnid, iid);
                            let checked = is_breakpoint_set(giid);
                            let text = format!("i{}", iid.0);
                            if ui.selectable_label(checked, text).clicked() {
                                if checked {
                                    action = Action::ClearInstrBreakpoint(giid);
                                } else {
                                    action = Action::SetInstrBreakpoint(giid);
                                }
                            }
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                            ui.label(egui::RichText::new(instr.opcode()));
                        });

                        let mut description = None;
                        ui.horizontal(|ui| {
                            instr.analyze(|descr| {
                                match descr {
                                    InstrDescriptor::Description(descr) => {
                                        description = Some(descr);
                                    }
                                    InstrDescriptor::VRegRead(vreg) => {
                                        let this_action = widgets::value_button(
                                            ui,
                                            vreg,
                                            description,
                                            widgets::Mode::Read,
                                            frame,
                                            highlight,
                                        );
                                        action.set_if_none(this_action);
                                    }
                                    InstrDescriptor::VRegWrite(vreg) => {
                                        let this_action = widgets::value_button(
                                            ui,
                                            vreg,
                                            description,
                                            widgets::Mode::Write,
                                            frame,
                                            highlight,
                                        );
                                        action.set_if_none(this_action);
                                    }
                                    InstrDescriptor::IID(iid) => {
                                        ui.label(format!("i{}", iid.0));
                                    }
                                    InstrDescriptor::Const(const_ndx) => {
                                        ui.label(format!("k{}", const_ndx.0));
                                    }
                                    InstrDescriptor::Capture(cap_ndx) => {
                                        ui.label(format!("cap[{}]", cap_ndx.0));
                                    }
                                    InstrDescriptor::Arg(arg_ndx) => {
                                        ui.label(format!("arg[{}]", arg_ndx.0));
                                    }
                                    InstrDescriptor::Null => {
                                        ui.label(widgets::richtext_for_value(
                                            interpreter::Value::Null,
                                        ));
                                    }
                                    InstrDescriptor::Undefined => {
                                        ui.label(widgets::richtext_for_value(
                                            interpreter::Value::Undefined,
                                        ));
                                    }
                                    InstrDescriptor::This => {
                                        ui.label("this");
                                    }
                                };
                            });
                        });

                        ui.end_row();
                    }
                });
        });

        Response {
            action,
            tiles: if drag_start {
                egui_tiles::UiResponse::DragStarted
            } else {
                egui_tiles::UiResponse::None
            },
        }
    }
}

mod source_view {
    use std::sync::Arc;

    use mcjs_vm::{bytecode, interpreter::stack, Loader};

    #[derive(Default)]
    pub struct Response {
        pub tiles: egui_tiles::UiResponse,
    }

    #[derive(Default)]
    pub struct State {
        cache: Option<Cache>,
        requested_scroll: bool,
    }
    impl State {
        fn update(&mut self, giid: bytecode::GlobalIID, loader: &Loader, ctx: &egui::Context) {
            if let Some(cache) = &self.cache {
                if cache.is_valid(giid) {
                    return;
                }
            }
            self.cache = Cache::build(giid, loader, ctx);
        }

        pub fn request_scroll(&mut self) {
            self.requested_scroll = true;
        }
    }

    struct Cache {
        giid: bytecode::GlobalIID,
        galley: Arc<egui::Galley>,
        first_hl_line_ndx: usize,
    }
    impl Cache {
        fn is_valid(&self, giid: bytecode::GlobalIID) -> bool {
            self.giid == giid
        }

        fn build(giid: bytecode::GlobalIID, loader: &Loader, ctx: &egui::Context) -> Option<Cache> {
            let bytecode::GlobalIID(fnid, iid) = giid;

            let func_lookup = loader.lookup_function(fnid)?;
            let start_pos = func_lookup.source_file.start_pos;
            // the "tightest" breakrange that still encloses the GIID
            let (hl_start, hl_end) = loader
                .function_breakranges(fnid)?
                .filter(|(_, brange)| brange.iid_start <= iid && iid < brange.iid_end)
                .min_by_key(|(_, brange)| brange.hi - brange.lo)
                .map(|(_, brange)| {
                    let lo = (brange.lo - start_pos).0 as usize;
                    let hi = (brange.hi - start_pos).0 as usize;
                    (lo, hi)
                })
                // no breakrange found => highlight "an empty interval" (nothing)
                .unwrap_or((0, 0));

            let galley = GalleyHighlightParams {
                full_text: &func_lookup.source_file.src,
                hl_start,
                hl_end,
                normal_color: egui::Style::default().visuals.weak_text_color(),
                style: &egui::Style::default(),
                hl_color: egui::Style::default().visuals.strong_text_color(),
                ctx,
            }
            .into_galley();

            let first_hl_line_ndx = func_lookup.source_file.src[0..hl_start]
                .chars()
                .filter(|c| *c == '\n')
                .count();

            Some(Cache {
                giid,
                galley,
                first_hl_line_ndx,
            })
        }
    }

    struct GalleyHighlightParams<'a> {
        full_text: &'a str,

        hl_start: usize,
        hl_end: usize,
        normal_color: egui::Color32,
        hl_color: egui::Color32,

        style: &'a egui::Style,
        ctx: &'a egui::Context,
    }
    impl<'a> GalleyHighlightParams<'a> {
        fn into_galley(self) -> Arc<egui::Galley> {
            let pre = &self.full_text[0..self.hl_start];
            let hl = &self.full_text[self.hl_start..self.hl_end];
            let post = &self.full_text[self.hl_end..];

            let default_valign = egui::Align::Min;

            let mut layout_job = egui::text::LayoutJob::default();
            egui::RichText::new(pre)
                .monospace()
                .color(self.normal_color)
                .append_to(
                    &mut layout_job,
                    self.style,
                    egui::FontSelection::default(),
                    default_valign,
                );
            egui::RichText::new(hl)
                .monospace()
                .color(self.hl_color)
                .append_to(
                    &mut layout_job,
                    self.style,
                    egui::FontSelection::default(),
                    default_valign,
                );
            egui::RichText::new(post)
                .monospace()
                .color(self.normal_color)
                .append_to(
                    &mut layout_job,
                    self.style,
                    egui::FontSelection::default(),
                    default_valign,
                );

            self.ctx.fonts(|fonts| fonts.layout_job(layout_job))
        }
    }

    pub fn show(
        ui: &mut egui::Ui,
        state: &mut State,
        frame: &stack::Frame,
        loader: &Loader,
    ) -> Response {
        let mut button_drag_started = false;
        ui.horizontal(|ui| {
            button_drag_started = ui
                .add(egui::Button::new("Source code").sense(egui::Sense::drag()))
                .drag_started();

            if ui.button("Focus current").clicked() {
                state.requested_scroll = true;
            }
        });

        let fnid = frame.header().fnid;
        let iid = frame.header().iid;
        let giid = bytecode::GlobalIID(fnid, iid);
        state.update(giid, loader, ui.ctx());

        let cache = match &state.cache {
            Some(cache) => cache,
            None => {
                ui.vertical_centered(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("No such function with ID");
                        ui.label(format!("{:?}", fnid));
                    });
                });
                return Response::default();
            }
        };
        assert_eq!(cache.giid, giid);

        let mut scrollarea = egui::ScrollArea::both();
        if std::mem::take(&mut state.requested_scroll) {
            let offset = 11.1 * cache.first_hl_line_ndx as f32;
            scrollarea = scrollarea.vertical_scroll_offset(offset);
        }
        scrollarea.show(ui, |ui| {
            let text = egui::WidgetText::Galley(Arc::clone(&cache.galley));
            let label = egui::Label::new(text).sense(egui::Sense::click());

            let res = ui.add(label);
            if res.secondary_clicked() {
                let click_pos = res.interact_pointer_pos().unwrap() - res.rect.min;
                let offset = cache.galley.cursor_from_pos(click_pos).ccursor.index;
                eprintln!(
                    "TODO! Listing the relevant break ranges (at offset {})",
                    offset
                );
            }
        });

        Response {
            tiles: if button_drag_started {
                egui_tiles::UiResponse::DragStarted
            } else {
                egui_tiles::UiResponse::None
            },
        }
    }
}

mod heap_view {
    use mcjs_vm::heap;
    use mcjs_vm::InterpreterValue;

    #[derive(Default)]
    pub struct State {
        oids: Vec<heap::ObjectId>,
        tree: Vec<TreeNode>,
    }

    #[derive(Default)]
    struct TreeNode {
        key: String,
        value: String,
        expanded: bool,
        children: Vec<TreeNode>,
    }

    fn read_object(heap: &heap::Heap, oid: heap::ObjectId, depth: usize) -> TreeNode {
        if depth >= 5 {
            return TreeNode {
                value: "too deep!".to_string(),
                ..Default::default()
            };
        }

        let obj = match heap.get(oid) {
            Some(obj) => obj,
            None => {
                return TreeNode {
                    value: format!("{:?} = MISSING!", oid),
                    ..Default::default()
                }
            }
        };

        let mut props = Vec::new();
        obj.own_properties(false, &mut props);

        let mut node = TreeNode {
            key: "".to_string(),
            value: format!("{:?} [{} properties]", oid, props.len()),
            expanded: false,
            children: Vec::new(),
        };

        for prop in props {
            let mut child = match obj.get_own(heap::IndexOrKey::Key(&prop)) {
                Some(heap::Property {
                    value: InterpreterValue::Object(child_oid),
                    ..
                }) => read_object(heap, child_oid, depth + 1),
                Some(heap::Property { value, .. }) => TreeNode {
                    value: format!("{:?}", value),
                    ..Default::default()
                },
                None => TreeNode {
                    value: "MISSING!".to_string(),
                    ..Default::default()
                },
            };

            child.key = prop.clone();
            node.children.push(child);
        }

        node
    }

    fn show_tree_node(ui: &mut egui::Ui, node: &mut TreeNode, depth: usize) {
        if depth >= 5 {
            ui.label("too deep!");
            return;
        }

        ui.horizontal(|ui| {
            ui.checkbox(&mut node.expanded, &node.key);
            ui.label(" = ");
            ui.label(&node.value);
        });

        if node.expanded {
            let indent_id = (node as *const TreeNode) as usize;
            ui.indent(indent_id, |ui| {
                for child in &mut node.children {
                    show_tree_node(ui, child, depth + 1);
                }
            });
        }
    }

    #[derive(Default)]
    pub struct Response {
        pub tiles: egui_tiles::UiResponse,
    }

    pub fn show(
        ui: &mut egui::Ui,
        state: &mut State,
        objects: &[heap::ObjectId],
        heap: &heap::Heap,
    ) -> Response {
        let drag_started = ui
            .add(egui::Button::new("Heap").sense(egui::Sense::drag()))
            .drag_started();

        if &state.oids != objects {
            eprintln!("objects set changed");
            state.oids.clear();
            state.oids.extend_from_slice(objects);
            state.tree = objects
                .iter()
                .map(|oid| read_object(heap, *oid, 0))
                .collect();
        }

        debug_assert_eq!(&state.oids, objects);

        for root in &mut state.tree {
            show_tree_node(ui, root, 0);
        }

        Response {
            tiles: if drag_started {
                egui_tiles::UiResponse::DragStarted
            } else {
                egui_tiles::UiResponse::None
            },
        }
    }
}

mod widgets {
    use std::time::{Duration, Instant};

    use super::Action;
    use mcjs_vm::{
        bytecode, heap,
        interpreter::{stack, SlotDebug, Value},
    };

    const COLOR_BLUE: egui::Color32 = egui::Color32::from_rgb(86, 156, 214);
    const COLOR_LIGHT_BLUE: egui::Color32 = egui::Color32::from_rgb(156, 220, 254);
    const _COLOR_ROSE: egui::Color32 = egui::Color32::from_rgb(206, 145, 120);
    const _COLOR_MAGENTA: egui::Color32 = egui::Color32::from_rgb(197, 134, 192);
    const COLOR_GREEN: egui::Color32 = egui::Color32::from_rgb(78, 201, 176);
    const COLOR_YELLOW: egui::Color32 = egui::Color32::from_rgb(220, 220, 170);
    const COLOR_GREY: egui::Color32 = egui::Color32::GRAY;

    const COLOR_VREG_READ: egui::Color32 = COLOR_YELLOW;
    const COLOR_VREG_WRITE: egui::Color32 = egui::Color32::LIGHT_RED;
    const COLOR_HIGHLIGHTED: egui::Color32 = egui::Color32::GOLD;
    const COLOR_NUMBER: egui::Color32 = COLOR_GREEN;
    const COLOR_SINGLETON: egui::Color32 = COLOR_BLUE;
    const COLOR_OBJECT: egui::Color32 = COLOR_LIGHT_BLUE;
    const _COLOR_STRING: egui::Color32 = _COLOR_ROSE;
    const _COLOR_KEYWORD: egui::Color32 = _COLOR_MAGENTA;
    const _COLOR_IID: egui::Color32 = COLOR_GREY;
    const COLOR_INVALID: egui::Color32 = COLOR_GREY;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum Highlight {
        #[default]
        None,
        VReg((bytecode::FnId, bytecode::VReg)),
        Object(heap::ObjectId),
    }

    impl Highlight {
        fn match_vreg(&self, fnid: bytecode::FnId, vreg: bytecode::VReg) -> bool {
            matches!(self, Highlight::VReg(h) if *h == (fnid, vreg))
        }

        fn match_obj_id(&self, obj_id: heap::ObjectId) -> bool {
            *self == Highlight::Object(obj_id)
        }
    }

    pub enum Mode {
        Read,
        Write,
    }

    pub fn value_button(
        ui: &mut egui::Ui,
        vreg: bytecode::VReg,
        description: Option<&'static str>,
        mode: Mode,
        frame: &stack::Frame,
        highlight: Highlight,
    ) -> Action {
        let slot = frame.get_slot(vreg);
        let read_result = frame.get_result(vreg);
        let fnid = frame.header().fnid;

        let is_highlighted = {
            let is_vreg_hl = highlight.match_vreg(fnid, vreg);
            let is_obj_id_hl = matches!(
                read_result,
                Some(Value::Object(value_obj_id)) if highlight.match_obj_id(value_obj_id)
            );
            is_vreg_hl || is_obj_id_hl
        };

        let stroke = if is_highlighted {
            egui::Stroke::new(1.0, COLOR_HIGHLIGHTED)
        } else {
            ui.ctx().style().visuals.window_stroke
        };

        let mut action = Action::None;
        let mut hover_action = Action::None;

        let res = egui::Frame::none()
            .stroke(stroke)
            .rounding(egui::Rounding::same(10.0))
            .inner_margin(egui::Margin::symmetric(10.0, 0.0))
            .show(ui, |ui| {
                if let Some(description) = description {
                    ui.label(description);
                }

                ui.label(richtext_for_vreg(vreg, mode));

                if let SlotDebug::Upvalue(upv_id) = slot {
                    let upv_id = format!("{:?}", upv_id);
                    let upv_id = peel_parens(&upv_id);
                    ui.label(format!("upv{} »", upv_id));
                }

                match read_result {
                    None => {
                        ui.label(
                            egui::RichText::new("TDZ")
                                .color(COLOR_INVALID)
                                .small_raised(),
                        );
                    }
                    Some(value @ Value::Object(obj_id)) => {
                        let value_text = richtext_for_value(value);
                        let res = ui.add(
                            egui::Button::new(value_text)
                                .small()
                                .stroke(egui::Stroke::NONE)
                                .rounding(egui::Rounding::same(10.0)),
                        );

                        // TODO Short text for object (e.g. string)

                        if res.clicked() {
                            action = Action::OpenObject(obj_id);
                        }
                        hover_action = Action::SetHighlight(Highlight::Object(obj_id));
                    }
                    Some(non_obj_value) => {
                        ui.label(richtext_for_value(non_obj_value));
                        hover_action = Action::SetHighlight(Highlight::VReg((fnid, vreg)));
                    }
                }
            });

        if action == Action::None && res.response.hovered() {
            action = hover_action;
        }
        action
    }

    fn richtext_for_vreg(vreg: bytecode::VReg, mode: Mode) -> egui::RichText {
        let text_color = match mode {
            Mode::Read => COLOR_VREG_READ,
            Mode::Write => COLOR_VREG_WRITE,
        };
        egui::RichText::new(format!("v{}", vreg.0)).color(text_color)
    }

    pub fn richtext_for_value(value: Value) -> egui::RichText {
        match value {
            Value::Number(n) => egui::RichText::new(n.to_string()).color(COLOR_NUMBER),
            Value::Bool(true) => egui::RichText::new("true").color(COLOR_SINGLETON),
            Value::Bool(false) => egui::RichText::new("false").color(COLOR_SINGLETON),
            Value::Object(obj_id) => {
                let obj_id = format!("{:?}", obj_id);
                let obj_id = peel_parens(&obj_id);
                let obj_id = format!("obj{}", obj_id);
                egui::RichText::new(obj_id).color(COLOR_OBJECT)
            }
            Value::Null => egui::RichText::new("null").color(COLOR_SINGLETON),
            Value::Undefined => egui::RichText::new("undefined").color(COLOR_SINGLETON),
            Value::SelfFunction => panic!(),
            Value::Internal(_) => panic!(),
            Value::Symbol(sym) => egui::RichText::new(sym).italics().color(COLOR_SINGLETON),
        }
    }

    pub struct Toast {
        pub dur: Duration,
        state: ToastState,
    }
    enum ToastState {
        Off,
        On {
            message: String,
            appear_time: Instant,
        },
    }
    impl Default for Toast {
        fn default() -> Self {
            Toast {
                dur: Duration::from_secs(5),
                state: ToastState::Off,
            }
        }
    }
    impl Toast {
        pub fn start(&mut self, message: String) {
            let appear_time = Instant::now();
            self.state = ToastState::On {
                message,
                appear_time,
            };
        }
        pub fn update(&mut self, ui: &mut egui::Ui) {
            if let ToastState::On {
                appear_time,
                message,
            } = &self.state
            {
                if appear_time.elapsed() > self.dur {
                    self.state = ToastState::Off;
                } else {
                    ui.label(message);
                }
            }
        }
    }

    // Narrows a &str from "xyz(whatever)" into "whatever". (Panics if
    // the string is not in that form).
    //
    // We do this because slotmap doesn't give us visibility into its
    // IDs, so we use this hack to get a more readable string out of
    // their Debug impl.
    fn peel_parens(s: &str) -> &str {
        let (_, s) = s.split_once('(').unwrap();
        let (inside, _) = s.split_once(')').unwrap();
        inside
    }
}

mod manager {
    use std::path::PathBuf;
    use thiserror::Error;

    use mcjs_vm::{
        bytecode,
        interpreter::{
            self,
            debugger::{Fuel, InstrBreakpoint},
            stack, Exit, Interpreter,
        },
        Loader, Realm,
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
        Failed(InterpreterError),
    }

    /// An alternative form of `interpreter::InterpreterError` that is geared towards
    /// being displayed to the user.
    pub struct InterpreterError {
        message: String,
        giid: Option<bytecode::GlobalIID>,
    }
    impl InterpreterError {
        pub fn message(&self) -> &str {
            &self.message
        }
        pub fn giid(&self) -> Option<bytecode::GlobalIID> {
            self.giid
        }

        fn from_vm_error(value: &interpreter::InterpreterError, loader: &Loader) -> Self {
            let mut message = String::new();
            value.error.write_to(&mut message, Some(loader)).unwrap();
            let data = value.interpreter_state();
            let giid = if data.is_empty() {
                None
            } else {
                let hdr = data.top().header();
                let giid = bytecode::GlobalIID(hdr.fnid, hdr.iid);
                Some(giid)
            };
            InterpreterError { message, giid }
        }
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

            let mut loader = match &params.main_directory {
                Some(p) => Loader::new(p.clone()),
                None => Loader::new_cwd(),
            };
            let realm = Realm::new(&mut loader);

            let mut dbg = interpreter::debugger::DebuggingState::new();

            let mut script_fnids = Vec::new();
            for filename in &params.filenames {
                let main_fnid = loader.load_script_file(filename)?;
                script_fnids.push(main_fnid);

                // Place a breakpoint at the start of each file
                let giid = bytecode::GlobalIID(main_fnid, bytecode::IID(0));
                dbg.set_instr_bkpt(giid, InstrBreakpoint::default())?;
            }

            Ok(ManagedInterpreter {
                script_fnids,
                realm,
                loader,
                state: State::Ready(0),
                dbg,
            })
        }

        pub fn restart(&mut self) {
            self.realm = Realm::new(&mut self.loader);
            self.state = State::Ready(0);
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

        pub(crate) fn loader(&self) -> &Loader {
            &self.loader
        }

        pub fn resume(&mut self) {
            loop {
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
                        let intrp =
                            Interpreter::resume(&mut self.realm, &mut self.loader, intrp_state);
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
                    Err(ierr) => {
                        let err = InterpreterError::from_vm_error(&*ierr, &self.loader);
                        State::Failed(err)
                    }
                };

                // Skip the initialized "Ready" state, as it's not relevant for the user
                // to see
                match &self.state {
                    State::Ready(_) => {}
                    _ => break,
                }
            }
        }

        pub fn next_into(&mut self) {
            self.dbg.set_fuel(Fuel::Limited(1));
            self.resume();
        }

        pub fn next(&mut self) -> Result<()> {
            // "Next" without following call/return
            //  == Set a temporary breakpoint on the following instruction, then resume.
            let intrp_state = match self.state() {
                State::Suspended { intrp_state, .. } => intrp_state,
                _ => return Ok(()),
            };

            let header = intrp_state.top().header();
            let next_iid = bytecode::IID(header.iid.0 + 1);
            let giid = bytecode::GlobalIID(header.fnid, next_iid);

            let mut bkpt = InstrBreakpoint::default();
            bkpt.delete_on_hit = true;

            let dbg = self.debugging_state_mut();
            dbg.set_instr_bkpt(giid, bkpt).map_err(Error::Debugger)?;

            self.resume();
            Ok(())
        }

        pub fn debugging_state(&self) -> &interpreter::debugger::DebuggingState {
            &self.dbg
        }
        pub fn debugging_state_mut(&mut self) -> &mut interpreter::debugger::DebuggingState {
            &mut self.dbg
        }

        pub fn heap(&self) -> &mcjs_vm::heap::Heap {
            &self.realm.heap()
        }
    }
}
