use std::{io::Write, path::PathBuf};

use anyhow::Result;
use mcjs_vm::{
    bytecode,
    interpreter::{
        self,
        debugger::{self, InstrBreakpoint},
    },
    stack,
};

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

    focus_frame_ndx: usize,
    stack_view: stack_view::State,
    source_view: source_view::State,
    highlight: widgets::Highlight,

    save_error_dialog: Option<String>,
    bkpt_error_dialog: Option<String>,
    toast: widgets::Toast,
    error_dialog_toast: widgets::Toast,
}

impl AppData {
    fn new(params: manager::Params) -> Self {
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

        let intrp = init_interpreter(&params);

        let mut app = AppData {
            params,
            intrp,
            tree,
            focus_frame_ndx: 0,
            stack_view: stack_view::State::default(),
            source_view: source_view::State::default(),
            highlight: widgets::Highlight::None,
            save_error_dialog: None,
            bkpt_error_dialog: None,
            toast: widgets::Toast::default(),
            error_dialog_toast: widgets::Toast::default(),
        };

        // Purposefully ignore the error, not a time to show it
        let _ = app.load_tree_layout();

        app
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

    #[error("fomat error: {0}")]
    Format(String),
}

#[derive(serde::Serialize, serde::Deserialize)]
struct StateFileData {
    tree_layout: Option<egui_tiles::Tree<Pane>>,
}

fn init_interpreter(params: &manager::Params) -> manager::ManagedInterpreter {
    let mut intrp =
        manager::ManagedInterpreter::new(params).expect("could not initialize interpreter");
    // skip the initial Ready state
    debug_assert!(matches!(intrp.state(), manager::State::Ready(_)));
    intrp.resume();
    intrp
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
    OpenObject(debugger::ObjectId),
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
                    cause: _,
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
                        stack_view: &mut self.stack_view,
                        source_view: &mut self.source_view,
                        action: &mut action,
                        highlight: self.highlight,
                        dbg: self.intrp.debugging_state(),
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
                    ui.horizontal(|ui| {
                        if ui.button("Restart").clicked() {
                            action = Action::Restart;
                        }

                        if ui
                            .button("Place breakpoint at failed instruction")
                            .clicked()
                        {
                            let data = err.interpreter_state();
                            let header = data.top().header();
                            let giid = bytecode::GlobalIID(header.fnid, header.iid);
                            self.error_dialog_toast
                                .start(format!("Set breakpoint at {:?}", giid));
                            action = Action::SetInstrBreakpoint(giid);
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
            }
            Action::Restart => {
                self.restart();
            }
            Action::Next => {
                if let Err(err) = self.intrp.next() {
                    self.bkpt_error_dialog = Some(err.to_string());
                }
            }
            Action::Into => {
                self.intrp.next_into();
            }
            Action::SetHighlight(highlight) => {
                self.highlight = highlight;
            }
            Action::OpenObject(_) => {
                eprintln!("TODO: {:?}", action);
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
    dbg: &'a interpreter::debugger::DebuggingState,

    /// The currently focused frame index.
    ///
    /// `0` is for the stack top. Higher numbers are for frames lower in the stack.
    frame_focus_ndx: &'a mut usize,

    stack_view: &'a mut stack_view::State,
    source_view: &'a mut source_view::State,

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
        let frame_ndx = self.intrp_state.len() - 1 - *self.frame_focus_ndx;
        let frame = self.intrp_state.nth_frame(frame_ndx);
        let text = match pane {
            Pane::SourceCode => {
                return source_view::show(ui, self.source_view, &frame, self.loader).tiles;
            }
            Pane::Bytecode => {
                let is_breakpoint_set = |giid| self.dbg.instr_bkpt_at(&giid).is_some();
                let res =
                    bytecode_view::show(ui, self.loader, &frame, self.highlight, is_breakpoint_set);
                self.action.set_if_none(res.action);
                return res.tiles;
            }
            Pane::PAST => "PAST",
            Pane::Heap => "Heap",
            Pane::Stack => {
                use stack_view::Action;
                let action = stack_view::show(self.stack_view, ui, self.intrp_state, self.loader);
                return match action {
                    Action::SetFrameIndex(ndx) => {
                        *self.frame_focus_ndx = ndx;
                        egui_tiles::UiResponse::None
                    }
                    Action::TabDragStarted => egui_tiles::UiResponse::DragStarted,
                    Action::None => egui_tiles::UiResponse::None,
                };
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
    use mcjs_vm::stack;

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
                    if ui.selectable_label(false, point_str).clicked() {
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
    use mcjs_vm::{bytecode, interpreter, stack};

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
    use std::{ops::Range, sync::Arc};

    use mcjs_vm::{bytecode, stack, Loader};

    #[derive(Default)]
    pub struct Response {
        pub tiles: egui_tiles::UiResponse,
    }

    #[derive(Default)]
    pub struct State {
        cache: Option<Cache>,
    }
    impl State {
        fn update(&mut self, fnid: bytecode::FnId, loader: &Loader, ctx: &egui::Context) {
            let galley_type = GalleyType::Function;
            if let Some(cache) = &self.cache {
                if cache.is_valid(fnid, galley_type) {
                    return;
                }
            }
            self.cache = Cache::build(fnid, loader, ctx, galley_type);
        }
    }

    struct Cache {
        fnid: bytecode::FnId,
        breakranges: Vec<(mcjs_vm::BreakRangeID, Range<usize>)>,
        galley: Arc<egui::Galley>,
        galley_type: GalleyType,
    }
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum GalleyType {
        Function,
        BreakRange(mcjs_vm::BreakRangeID),
    }
    impl Cache {
        fn is_valid(&self, fnid: bytecode::FnId, galley_type: GalleyType) -> bool {
            self.fnid == fnid && self.galley_type == galley_type
        }

        fn build(
            fnid: bytecode::FnId,
            loader: &Loader,
            ctx: &egui::Context,
            galley_type: GalleyType,
        ) -> Option<Cache> {
            let func_lookup = loader.lookup_function(fnid)?;
            let start_pos = func_lookup.source_file.start_pos;
            let breakranges: Vec<_> = loader
                .function_breakranges(fnid)?
                .map(|(brid, brange)| {
                    let lo = (brange.lo - start_pos).0 as usize;
                    let hi = (brange.hi - start_pos).0 as usize;
                    (brid, lo..hi)
                })
                .collect();

            let style = egui::Style::default();
            let hl_start = func_lookup.local_range.start.0 as usize;
            let hl_end = func_lookup.local_range.end.0 as usize;
            let normal_color = style.visuals.weak_text_color();
            let hl_color = style.visuals.strong_text_color();
            let galley = GalleyHighlightParams {
                full_text: &func_lookup.source_file.src,
                hl_start,
                hl_end,
                normal_color,
                style: &style,
                hl_color,
                ctx,
            }
            .into_galley();

            Some(Cache {
                fnid,
                galley,
                breakranges,
                galley_type,
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
        let button_drag_started = ui
            .add(egui::Button::new("Source code").sense(egui::Sense::drag()))
            .drag_started();

        let fnid = frame.header().fnid;
        state.update(fnid, loader, ui.ctx());

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
        assert_eq!(cache.fnid, fnid);

        egui::ScrollArea::both().show(ui, |ui| {
            let text = egui::WidgetText::Galley(Arc::clone(&cache.galley));
            let label = egui::Label::new(text).sense(egui::Sense::click());
            let res = ui.add(label);
            if res.secondary_clicked() {
                let click_pos = res.interact_pointer_pos().unwrap() - res.rect.min;
                let offset = cache.galley.cursor_from_pos(click_pos).ccursor.index;
                todo!(" -- list the relevant break ranges");
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

mod widgets {
    use std::time::{Duration, Instant};

    use super::Action;
    use mcjs_vm::{
        bytecode,
        interpreter::{debugger, Value},
        stack,
    };

    const COLOR_BLUE: egui::Color32 = egui::Color32::from_rgb(86, 156, 214);
    const COLOR_LIGHT_BLUE: egui::Color32 = egui::Color32::from_rgb(156, 220, 254);
    const COLOR_ROSE: egui::Color32 = egui::Color32::from_rgb(206, 145, 120);
    const COLOR_MAGENTA: egui::Color32 = egui::Color32::from_rgb(197, 134, 192);
    const COLOR_GREEN: egui::Color32 = egui::Color32::from_rgb(78, 201, 176);
    const COLOR_YELLOW: egui::Color32 = egui::Color32::from_rgb(220, 220, 170);
    const COLOR_GREY: egui::Color32 = egui::Color32::GRAY;

    const COLOR_VREG_READ: egui::Color32 = COLOR_YELLOW;
    const COLOR_VREG_WRITE: egui::Color32 = egui::Color32::LIGHT_RED;
    const COLOR_HIGHLIGHTED: egui::Color32 = egui::Color32::GOLD;
    const COLOR_NUMBER: egui::Color32 = COLOR_GREEN;
    const COLOR_SINGLETON: egui::Color32 = COLOR_BLUE;
    const COLOR_OBJECT: egui::Color32 = COLOR_LIGHT_BLUE;
    const COLOR_STRING: egui::Color32 = COLOR_ROSE;
    const COLOR_KEYWORD: egui::Color32 = COLOR_MAGENTA;
    const COLOR_IID: egui::Color32 = COLOR_GREY;
    const COLOR_INVALID: egui::Color32 = COLOR_GREY;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum Highlight {
        #[default]
        None,
        VReg((bytecode::FnId, bytecode::VReg)),
        Object(debugger::ObjectId),
    }

    impl Highlight {
        fn match_vreg(&self, fnid: bytecode::FnId, vreg: bytecode::VReg) -> bool {
            matches!(self, Highlight::VReg(h) if *h == (fnid, vreg))
        }

        fn match_obj_id(&self, obj_id: debugger::ObjectId) -> bool {
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

                if let mcjs_vm::SlotDebug::Upvalue(upv_id) = slot {
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
    use std::{io::Read, path::PathBuf};
    use thiserror::Error;

    use mcjs_vm::{
        bytecode,
        interpreter::debugger::{Fuel, InstrBreakpoint},
        interpreter::{self, Exit, Interpreter},
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

            let mut loader = match &params.main_directory {
                Some(p) => Loader::new(p.clone()),
                None => Loader::new_cwd(),
            };
            let realm = Realm::new(&mut loader);

            let mut dbg = interpreter::debugger::DebuggingState::new();

            let mut script_fnids = Vec::new();
            for filename in &params.filenames {
                let mut content = String::new();
                std::fs::File::open(filename)?.read_to_string(&mut content)?;

                let main_fnid = loader.load_script(Some(filename.clone()), content)?;
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
                    Err(err) => State::Failed(err),
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
    }
}
