use std::path::PathBuf;

use anyhow::Result;
use mcjs_vm::{
    interpreter::{self, debugger},
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

    stack_view: stack_view::State,
    source_view: source_view::State,

    highlight: widgets::Highlight,
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

        AppData {
            params,
            intrp,
            tree,
            stack_view: stack_view::State::default(),
            source_view: source_view::State::default(),
            highlight: widgets::Highlight::None,
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

#[derive(Debug, PartialEq, Eq)]
enum Action {
    None,
    Resume,
    Restart,
    Next,
    SetHighlight(widgets::Highlight),
    OpenObject(debugger::ObjectId),
}
impl Action {
    fn set_if_none(&mut self, other: Action) {
        if let Action::None = self {
            *self = other;
        }
    }
}
impl Default for Action {
    fn default() -> Self {
        Action::None
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
                        loader: self.intrp.loader(),
                        stack_view: &mut self.stack_view,
                        source_view: &mut self.source_view,
                        action: &mut action,
                        highlight: self.highlight,
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
            Action::SetHighlight(highlight) => {
                self.highlight = highlight;
            }
            Action::OpenObject(_) => {
                eprintln!("TODO: {:?}", action);
            }
        }
    }
}

struct TreeBehavior<'a> {
    intrp_state: &'a stack::InterpreterData,
    cause: &'a interpreter::SuspendCause,
    loader: &'a mcjs_vm::Loader,

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
        let text = match pane {
            Pane::SourceCode => {
                return source_view::show(
                    ui,
                    self.source_view,
                    &self.intrp_state.top(),
                    self.loader,
                )
                .tiles;
            }
            Pane::Bytecode => {
                let res = bytecode_view::show(ui, self.loader, self.intrp_state, self.highlight);
                self.action.set_if_none(res.action);
                return res.tiles;
            }
            Pane::PAST => "PAST",
            Pane::Heap => "Heap",
            Pane::Stack => return stack_view::show(&mut self.stack_view, ui, self.intrp_state),
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

    pub fn show(
        _view_state: &mut State,
        ui: &mut egui::Ui,
        intrp_state: &stack::InterpreterData,
    ) -> egui_tiles::UiResponse {
        let drag_button_res = ui.add(egui::Button::new("Stack").sense(egui::Sense::drag()));

        for frame in intrp_state.frames() {
            let header = frame.header();
            let point_str = format!("{:?}:{:?}", header.fnid, header.iid);
            ui.selectable_label(false, point_str);
        }

        if drag_button_res.drag_started() {
            egui_tiles::UiResponse::DragStarted
        } else {
            egui_tiles::UiResponse::None
        }
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
        intrp_state: &stack::InterpreterData,
        highlight: widgets::Highlight,
    ) -> Response {
        let mut action = Action::None;

        let drag_start = ui
            .add(egui::Button::new("Bytecode").sense(egui::Sense::drag()))
            .drag_started();

        let frame = intrp_state.top();
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

        egui::ScrollArea::both().show(ui, |ui| {
            egui::Grid::new("bytecode->instrs")
                .num_columns(3)
                .show(ui, |ui| {
                    for (iid, instr) in func.instrs().iter().enumerate() {
                        use mcjs_vm::bytecode::InstrDescriptor;

                        let iid = bytecode::IID(iid.try_into().unwrap());

                        ui.horizontal(|ui| {
                            ui.set_width(40.0);
                            ui.monospace(format!("i{}", iid.0));
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                            let mut text = egui::RichText::new(instr.opcode());
                            if iid == cur_iid {
                                text = text.background_color(egui::Color32::DARK_BLUE);
                            };
                            ui.label(text);
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
                                            &frame,
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
                                            &frame,
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
    use mcjs_vm::{bytecode, stack, FunctionLookup, Loader};

    #[derive(Default)]
    pub struct Response {
        pub tiles: egui_tiles::UiResponse,
    }

    #[derive(Default)]
    pub struct State {
        func_lookup: Option<(bytecode::FnId, FunctionLookup)>,
    }
    impl State {
        fn update(&mut self, fnid: bytecode::FnId, loader: &Loader) {
            if let Some((cached_fnid, _)) = &self.func_lookup {
                if *cached_fnid == fnid {
                    return;
                }
            }

            self.func_lookup = loader
                .lookup_function(fnid)
                .map(|src_lookup| (fnid, src_lookup));
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
        state.update(fnid, loader);

        let (check_fnid, src_lookup) = match &state.func_lookup {
            Some(value) => value,
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
        assert_eq!(*check_fnid, fnid);

        let src = src_lookup.source_file.src.as_str();
        egui::ScrollArea::both().show(ui, |ui| {
            ui.monospace(src);
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Highlight {
        None,
        VReg((bytecode::FnId, bytecode::VReg)),
        Object(debugger::ObjectId),
    }
    impl Default for Highlight {
        fn default() -> Self {
            Highlight::None
        }
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

        pub(crate) fn loader(&self) -> &Loader {
            &self.loader
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
