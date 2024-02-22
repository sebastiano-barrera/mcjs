use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use mcjs_vm::interpreter::debugger::ObjectId;
use mcjs_vm::interpreter::Fuel;
use mcjs_vm::{bytecode, BreakRangeID, GlobalIID};

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    eprintln!("params = {:?}", params);

    let app = AppData::new(params);

    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport = native_options
        .viewport
        .with_inner_size(egui::Vec2::new(1000.0, 800.0))
        .with_app_id("mcjs_tools")
        .with_title("mcjs Dev Tools");
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
        params.main_directory = params.filenames[0].parent().map(|p| p.to_path_buf());
    }

    Ok(params)
}

struct AppData {
    params: interpreter_manager::Params,
    mgr: interpreter_manager::Manager,
    recent_state_change: bool,
    source_code_view: source_code_view::Cache,
    frame_ndx: usize,
    highlight: instr_view::Highlighted,
    object_windows: HashMap<ObjectId, object_view::ObjectWindow>,
}

impl AppData {
    fn new(params: interpreter_manager::Params) -> Self {
        let mgr = interpreter_manager::Manager::new(&params).expect("could not create interpreter");
        // si.resume();

        AppData {
            params,
            mgr,
            recent_state_change: false,
            source_code_view: Default::default(),
            frame_ndx: 0,
            highlight: Default::default(),
            object_windows: HashMap::new(),
        }
    }

    fn open_object_window(&mut self, obj_id: ObjectId) {
        self.object_windows
            .entry(obj_id)
            .or_insert_with(|| object_view::ObjectWindow::new(obj_id));
    }
}

impl eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        use interpreter_manager::State;

        let recent_state_change = self.recent_state_change;
        self.recent_state_change = false;

        enum Action {
            Next,
            Restart,
            Continue,
            None,
            SetStackFrame { index: usize },
            SetSourceBreakpoint { brange_id: BreakRangeID },
            SetInstrBreakpoint { giid: GlobalIID },
            OpenObjectWindow(ObjectId),
        }

        let mut action = Action::None;

        let suspended_state = match self.mgr.state_mut() {
            State::Ready { script_ndx, .. } => {
                let should_start = egui::CentralPanel::default()
                    .show(ctx, |ui| {
                        ui.label(format!("Ready to proceed with file #{}", script_ndx));
                        ui.button("Start").clicked()
                    })
                    .inner;

                if should_start {
                    self.mgr.resume();
                }

                return;
            }
            State::Finished => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.label("Interpreter finished successfully. No debugging!");
                });
                return;
            }
            State::Suspended(suspended_state) => suspended_state,
            State::Failed(err) => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.label(format!("interpreter failed: {:?}", err));
                });
                return;
            }
        };

        {
            let probe = suspended_state.probe_mut();
            let mut window_to_close = None;

            for (obj_id, obj_win) in self.object_windows.iter_mut() {
                let mut stay_open = true;
                obj_win.show(ctx, &probe, &mut stay_open);
                if !stay_open {
                    window_to_close = Some(*obj_id);
                }
            }

            if let Some(obj_id) = window_to_close {
                self.object_windows.remove(&obj_id);
            }
        }

        egui::SidePanel::left("sidebar").show(ctx, |ui| {
            if ui.button("NEXT").clicked() {
                action = Action::Next;
            }
            if ui.button("CONTINUE").clicked() {
                action = Action::Continue;
            }
            if ui.button("RESTART").clicked() {
                action = Action::Restart;
            }
            // TODO implement 'delete'!
            let _ = ui.button("DELETE");

            ui.horizontal(|ui| {
                ui.label("State:");
                let text = format!("Suspended due to {:?}", suspended_state.cause());
                ui.label(text);
            });

            let mut probe = suspended_state.probe_mut();

            ui.separator();
            ui.label("Double click on a <source code range> to set a breakpoint");

            {
                let mut value = probe.break_on_throw();
                let res = ui.checkbox(&mut value, "Break on throw");
                if res.changed() {
                    probe.set_break_on_throw(value);
                }
            }

            {
                let mut value = probe.break_on_unhandled_throw();
                let res = ui.checkbox(&mut value, "Break on unhandled throw");
                if res.changed() {
                    probe.set_break_on_unhandled_throw(value);
                }
            }

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
                    let iid = frame.header().iid;

                    let text = format!("{:?}:{:?}", frame.header().fnid, iid);
                    let bg = if ndx == self.frame_ndx {
                        ctx.style().visuals.selection.bg_fill
                    } else {
                        egui::Color32::TRANSPARENT
                    };

                    let res = ui.add(egui::Button::new(text).frame(false).fill(bg));
                    if res.clicked() {
                        action = Action::SetStackFrame { index: ndx };
                    }
                }
            });
        });

        egui::SidePanel::left("bytecode")
            .min_width(400.0)
            .show(ctx, |ui| {
                let probe = suspended_state.probe_mut();

                if self.frame_ndx >= probe.frames().len() {
                    // invalid frame ndx
                    if self.frame_ndx == 0 {
                        ui.label("Stack frame empty.  This normally happens when an exception goes completely unhandled.");
                    } else {
                        ui.label(&format!("No stack frame at index {}", self.frame_ndx));
                    }

                    return;
                }

                let vm_giid = probe.frame_giid(self.frame_ndx);

                let fnid = vm_giid.0;
                let func = probe.loader().get_function(fnid).unwrap();

                ui.horizontal(|ui| {
                    ui.label("mode: ");
                    let text = if func.is_strict_mode() {
                        "strict"
                    } else {
                        "sloppy"
                    };
                    ui.label(egui::RichText::new(text).strong());
                });

                let mut bkpt_to_set = None;

                let instr_bkpts: Vec<_> = probe.instr_breakpoints().collect();
                let preview_bkpt_iid = self
                    .source_code_view
                    .preview_breakrange()
                    .map(|br| br.iid_start);

                let instrs = func.instrs();
                let mut clicked_obj_id = None;
                egui::ScrollArea::both().auto_shrink(false).show_rows(
                    ui,
                    16.0,
                    instrs.len(),
                    |ui, ndx_range| {
                        for ndx in ndx_range {
                            let instr = &instrs[ndx];
                            let is_current = ndx == vm_giid.1 .0 as usize;
                            let res = ui.horizontal(|ui| {
                                let iid = bytecode::IID(ndx.try_into().unwrap());
                                let giid = bytecode::GlobalIID(fnid, iid);

                                instr_view::show_iid(ui, fnid, iid, &mut self.highlight);

                                if instr_bkpts.contains(&giid) {
                                    ui.label("•");
                                }
                                if preview_bkpt_iid == Some(iid) {
                                    ui.label(" >> ");
                                }

                                let frame = probe.frames().nth(self.frame_ndx).unwrap();
                                let res = instr_view::InstrView {
                                    instr,
                                    frame: &frame,
                                    func,
                                    probe: &probe,
                                    giid,
                                    is_current,
                                    highlighted: &mut self.highlight,
                                }
                                .show(ui);

                                if res.label_clicked {
                                    bkpt_to_set = Some(giid);
                                }

                                clicked_obj_id = clicked_obj_id.or(res.clicked_obj_id);
                            });

                            if recent_state_change && is_current {
                                res.response.scroll_to_me(None);
                            }
                        }
                    },
                );

                if let Some(obj_id) = clicked_obj_id {
                    action = Action::OpenObjectWindow(obj_id);
                }
                if let Some(giid) = bkpt_to_set {
                    action = Action::SetInstrBreakpoint { giid };
                }
            });

        {
            let probe = suspended_state.probe_mut();

            let res = egui::CentralPanel::default()
                .show(ctx, |ui| source_code_view::show(ui, &self.source_code_view))
                .inner;

            ctx.fonts(|fonts| {
                source_code_view::update(&mut self.source_code_view, &probe, &res, fonts)
            });
            if let Some(brange_id) = res.set_source_bpkt {
                action = Action::SetSourceBreakpoint { brange_id };
            }
        }

        match action {
            Action::None => {}
            Action::Next => {
                let mut probe = suspended_state.probe_mut();
                probe.set_fuel(Fuel::Limited(1));
                self.mgr.resume();
                self.recent_state_change = true;
                self.frame_ndx = 0;
            }
            Action::Continue => {
                self.mgr.resume();
                self.recent_state_change = true;
                self.frame_ndx = 0;
            }
            Action::Restart => {
                self.mgr = interpreter_manager::Manager::new(&self.params)
                    .expect("could not create interpreter");
                self.recent_state_change = true;
                self.frame_ndx = 0;
            }
            Action::SetStackFrame { index } => {
                self.frame_ndx = index;
            }
            Action::SetSourceBreakpoint { brange_id } => {
                self.mgr.set_source_breakpoint(brange_id);
            }
            Action::SetInstrBreakpoint { giid } => {
                self.mgr.set_instr_breakpoint(giid);
            }
            Action::OpenObjectWindow(obj_id) => {
                self.open_object_window(obj_id);
            }
        }
    }
}

mod instr_view {
    use mcjs_vm::{
        bytecode,
        interpreter::debugger::{ObjectId, Probe},
        stack::Frame,
        InterpreterValue, Literal,
    };

    pub struct InstrView<'a, 'b, 'c> {
        pub instr: &'a bytecode::Instr,
        pub frame: &'a Frame<'b>,
        pub func: &'a bytecode::Function,
        pub giid: bytecode::GlobalIID,
        pub probe: &'a Probe<'b, 'c>,
        pub is_current: bool,
        pub highlighted: &'a mut Highlighted,
    }

    impl<'a, 'b, 'c> InstrView<'a, 'b, 'c> {
        pub fn show(mut self, ui: &mut egui::Ui) -> Response {
            let bytecode::GlobalIID(fnid, _) = self.giid;

            let mut analyzer = Analyzer::default();
            self.instr.analyze(&mut analyzer);
            let desc: InstrDescriptor = analyzer.describe();

            let res = ui.selectable_label(self.is_current, desc.opcode);
            let label_clicked = res.clicked();

            let mut clicked_obj_id = None;

            for operand in desc.operands {
                match operand {
                    OperandDescriptor::None => {}
                    OperandDescriptor::Description(description) => {
                        ui.label(format!("{}=", description));
                    }
                    OperandDescriptor::VRegRead(vreg) => {
                        let res = self.show_value(ui, vreg, None, Mode::Read);
                        if res.clicked {
                            eprintln!("value clicked");
                            clicked_obj_id = res.obj_id;
                        }
                    }
                    OperandDescriptor::VRegWrite(vreg) => {
                        let res = self.show_value(ui, vreg, None, Mode::Write);
                        if res.clicked {
                            eprintln!("value clicked");
                            clicked_obj_id = res.obj_id;
                        }
                    }
                    OperandDescriptor::IID(iid) => show_iid(ui, fnid, iid, self.highlighted),
                    OperandDescriptor::Const(const_ndx) => {
                        let value = &self.func.consts()[const_ndx.0 as usize];
                        let text = richtext_for_literal(value);
                        ui.label(text);
                    }
                    OperandDescriptor::Capture(cap_ndx) => {
                        // NOTE I assume that Instr::LoadCapture is going to be removed soon, replaced
                        // completely by the implicit inline/upvalue state of stack slots
                        ui.label(format!("{:?}", cap_ndx));
                    }
                    OperandDescriptor::Arg(arg_ndx) => {
                        // NOTE I assume that Instr::LoadArg is going to be removed soon, replaced completely
                        // by the implicit allocation of the first bytecode::ARGS_COUNT_MAX vregs
                        ui.label(format!("{:?}", arg_ndx));
                    }
                    OperandDescriptor::Null => {
                        ui.label(richtext_for_value(InterpreterValue::Null));
                    }
                    OperandDescriptor::Undefined => {
                        ui.label(richtext_for_value(InterpreterValue::Undefined));
                    }
                    OperandDescriptor::This => {
                        ui.label(egui::RichText::new("this").color(COLOR_KEYWORD));
                    }
                }
            }

            Response {
                label_clicked,
                clicked_obj_id,
            }
        }
    }

    fn richtext_for_literal(value: &Literal) -> egui::RichText {
        match value {
            Literal::Number(n) => egui::RichText::new(n.to_string()).color(COLOR_NUMBER),
            Literal::String(s) => egui::RichText::new(format!("{:?}", s)).color(COLOR_STRING),
            Literal::JsWord(jsw) => {
                egui::RichText::new(format!("{:?}", jsw.to_string())).color(COLOR_KEYWORD)
            }
            Literal::Bool(true) => egui::RichText::new("true").color(COLOR_SINGLETON),
            Literal::Bool(false) => egui::RichText::new("false").color(COLOR_SINGLETON),
            Literal::Null => egui::RichText::new("null").color(COLOR_SINGLETON),
            Literal::Undefined => egui::RichText::new("undefined").color(COLOR_SINGLETON),
            Literal::SelfFunction => {
                panic!("I really gotta delete Literal::SelfFunction one of these days")
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

    enum Mode {
        Read,
        Write,
    }

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

    struct ValueResponse {
        obj_id: Option<ObjectId>,
        clicked: bool,
    }

    impl<'a, 'b, 'c> InstrView<'a, 'b, 'c> {
        fn show_value(
            &mut self,
            ui: &mut egui::Ui,
            vreg: bytecode::VReg,
            description: Option<&'static str>,
            mode: Mode,
        ) -> ValueResponse {
            let slot = self.frame.get_slot(vreg);
            let read_result = self.frame.get_result(vreg);

            let fnid = self.probe.giid().0;
            let is_highlighted = {
                let is_vreg_hl = self.highlighted.match_vreg(fnid, vreg);
                let is_obj_id_hl = match read_result {
                    Some(InterpreterValue::Object(value_obj_id)) => {
                        self.highlighted.match_obj_id(fnid, value_obj_id)
                    }
                    _ => false,
                };

                is_vreg_hl || is_obj_id_hl
            };

            let stroke = if is_highlighted {
                egui::Stroke::new(1.0, COLOR_HIGHLIGHTED)
            } else {
                ui.ctx().style().visuals.window_stroke
            };

            let mut clicked = false;

            let res = egui::Frame::none()
                .stroke(stroke)
                .rounding(egui::Rounding::same(10.0))
                .inner_margin(egui::Margin::symmetric(10.0, 0.0))
                .show(ui, |ui| {
                    if let Some(description) = description {
                        ui.label(format!("{}: ", description));
                    }

                    ui.label(richtext_for_vreg(vreg, mode));

                    if let mcjs_vm::SlotDebug::Upvalue(upv_id) = slot {
                        let upv_id = format!("{:?}", upv_id);
                        let upv_id = peel_parens(&upv_id);
                        ui.label(format!("upv{} »", upv_id));
                    }

                    let value = if let Some(value) = read_result {
                        value
                    } else {
                        ui.label(
                            egui::RichText::new("TDZ")
                                .color(COLOR_INVALID)
                                .small_raised(),
                        );
                        return;
                    };

                    let value_text = richtext_for_value(value);

                    if let InterpreterValue::Object(obj_id) = value {
                        let res = ui.add(
                            egui::Button::new(value_text)
                                .small()
                                .stroke(egui::Stroke::NONE)
                                .rounding(egui::Rounding::same(10.0)),
                        );
                        clicked = res.clicked();

                        let probe = &self.probe;

                        if let Some(text) = super::object_view::short_text_for_object(probe, obj_id)
                        {
                            ui.label(text);
                        }
                    } else {
                        ui.label(value_text);
                    }
                });

            let obj_id = match read_result {
                Some(InterpreterValue::Object(obj_id)) => Some(obj_id),
                _ => None,
            };
            if res.response.hovered() {
                if let Some(obj_id) = obj_id {
                    self.highlighted.set_obj_id(fnid, obj_id);
                } else {
                    self.highlighted.set_vreg(fnid, vreg);
                }
            }

            ValueResponse { clicked, obj_id }
        }
    }

    fn richtext_for_vreg(vreg: bytecode::VReg, mode: Mode) -> egui::RichText {
        let text_color = match mode {
            Mode::Read => COLOR_VREG_READ,
            Mode::Write => COLOR_VREG_WRITE,
        };
        egui::RichText::new(format!("v{}", vreg.0)).color(text_color)
    }

    fn richtext_for_value(value: InterpreterValue) -> egui::RichText {
        match value {
            InterpreterValue::Number(n) => egui::RichText::new(n.to_string()).color(COLOR_NUMBER),
            InterpreterValue::Bool(true) => egui::RichText::new("true").color(COLOR_SINGLETON),
            InterpreterValue::Bool(false) => egui::RichText::new("false").color(COLOR_SINGLETON),
            InterpreterValue::Object(obj_id) => {
                let obj_id = format!("{:?}", obj_id);
                let obj_id = peel_parens(&obj_id);
                let obj_id = format!("obj{}", obj_id);
                egui::RichText::new(obj_id).color(COLOR_OBJECT)
            }
            InterpreterValue::Null => egui::RichText::new("null").color(COLOR_SINGLETON),
            InterpreterValue::Undefined => egui::RichText::new("undefined").color(COLOR_SINGLETON),
            InterpreterValue::SelfFunction => panic!(),
            InterpreterValue::Internal(_) => panic!(),
        }
    }

    pub fn richtext_for_iid(iid: bytecode::IID, is_highlighted: bool) -> egui::RichText {
        let text = format!("{:4}", iid);
        let base = egui::RichText::new(text).monospace();
        if is_highlighted {
            base.background_color(COLOR_GREY)
                .color(egui::Color32::BLACK)
        } else {
            base.color(COLOR_IID)
        }
    }

    #[derive(Default)]
    struct Analyzer {
        desc: InstrDescriptor,
        n_operands: usize,
    }
    impl Analyzer {
        fn describe(self) -> InstrDescriptor {
            self.desc
        }
    }

    // TODO Might be a good idea to remove the InstrAnalyzer middleman and just put this 'analysis'
    // resulting into an InstrDescriptor directly into mcjs_vm::bytecode
    struct InstrDescriptor {
        opcode: &'static str,
        operands: [OperandDescriptor; Self::MAX_OPERANDS],
    }
    #[derive(Clone, Copy)]
    #[allow(clippy::upper_case_acronyms)]
    enum OperandDescriptor {
        None,
        Description(&'static str),
        VRegRead(bytecode::VReg),
        VRegWrite(bytecode::VReg),
        IID(bytecode::IID),
        Const(bytecode::ConstIndex),
        Capture(bytecode::CaptureIndex),
        Arg(bytecode::ArgIndex),
        Null,
        Undefined,
        This,
    }

    impl InstrDescriptor {
        const MAX_OPERANDS: usize = 8;
    }
    impl Default for InstrDescriptor {
        fn default() -> Self {
            InstrDescriptor {
                opcode: "",
                operands: [OperandDescriptor::None; Self::MAX_OPERANDS],
            }
        }
    }

    impl Analyzer {
        fn append_operand(&mut self, op_desc: OperandDescriptor) {
            self.desc.operands[self.n_operands] = op_desc;
            self.n_operands += 1;
        }
    }
    impl bytecode::InstrAnalyzer for Analyzer {
        fn start(&mut self, opcode_name: &'static str) {
            self.desc.opcode = opcode_name;
        }

        fn read_vreg_labeled(&mut self, vreg: bytecode::VReg, description: Option<&'static str>) {
            if let Some(description) = description {
                self.append_operand(OperandDescriptor::Description(description));
            }
            self.append_operand(OperandDescriptor::VRegRead(vreg));
        }

        fn write_vreg_labeled(&mut self, vreg: bytecode::VReg, description: Option<&'static str>) {
            if let Some(description) = description {
                self.append_operand(OperandDescriptor::Description(description));
            }
            self.append_operand(OperandDescriptor::VRegWrite(vreg));
        }

        fn jump_target(&mut self, iid: mcjs_vm::IID) {
            self.append_operand(OperandDescriptor::IID(iid));
        }

        fn load_const(&mut self, const_ndx: bytecode::ConstIndex) {
            self.append_operand(OperandDescriptor::Const(const_ndx));
        }

        fn load_null(&mut self) {
            self.append_operand(OperandDescriptor::Null);
        }

        fn load_undefined(&mut self) {
            self.append_operand(OperandDescriptor::Undefined);
        }

        fn load_capture(&mut self, item: bytecode::CaptureIndex) {
            self.append_operand(OperandDescriptor::Capture(item));
        }

        fn load_arg(&mut self, item: bytecode::ArgIndex) {
            self.append_operand(OperandDescriptor::Arg(item));
        }

        fn load_this(&mut self) {
            self.append_operand(OperandDescriptor::This);
        }

        fn end(&mut self, _instr: &bytecode::Instr) {}
    }

    pub struct Response {
        pub label_clicked: bool,
        pub clicked_obj_id: Option<ObjectId>,
    }

    #[allow(clippy::upper_case_acronyms)]
    #[derive(Default)]
    pub enum Highlighted {
        #[default]
        None,
        VReg((bytecode::FnId, bytecode::VReg)),
        Object((bytecode::FnId, ObjectId)),
        IID((bytecode::FnId, bytecode::IID)),
    }
    impl Highlighted {
        fn match_vreg(&self, fnid: bytecode::FnId, vreg: bytecode::VReg) -> bool {
            matches!(self, Highlighted::VReg(h) if *h == (fnid, vreg))
        }
        fn set_vreg(&mut self, fnid: bytecode::FnId, vreg: bytecode::VReg) {
            *self = Highlighted::VReg((fnid, vreg));
        }

        fn match_obj_id(&self, fnid: bytecode::FnId, obj_id: ObjectId) -> bool {
            matches!(self, Highlighted::Object(h) if *h == (fnid, obj_id))
        }
        fn set_obj_id(&mut self, fnid: bytecode::FnId, obj_id: ObjectId) {
            *self = Highlighted::Object((fnid, obj_id));
        }

        fn match_iid(&self, fnid: bytecode::FnId, iid: bytecode::IID) -> bool {
            matches!(self, Highlighted::IID(h) if *h == (fnid, iid))
        }
        fn set_iid(&mut self, fnid: bytecode::FnId, iid: bytecode::IID) {
            *self = Highlighted::IID((fnid, iid));
        }
    }

    pub fn show_iid(
        ui: &mut egui::Ui,
        fnid: bytecode::FnId,
        iid: bytecode::IID,
        highlighted: &mut Highlighted,
    ) {
        let is_highlighted = highlighted.match_iid(fnid, iid);
        let res = ui.label(richtext_for_iid(iid, is_highlighted));

        if res.hovered() {
            highlighted.set_iid(fnid, iid);
        }
    }
}

mod object_view {
    use std::borrow::Cow;

    use mcjs_vm::interpreter::{
        debugger::{ObjectId, Probe},
        Closure,
    };

    pub struct ObjectWindow {
        id: String,
        obj_id: ObjectId,
    }

    impl ObjectWindow {
        pub fn new(obj_id: ObjectId) -> Self {
            let id = format!("{:?}", obj_id);
            ObjectWindow { id, obj_id }
        }

        pub fn show(&mut self, ctx: &egui::Context, probe: &Probe, open_flag: &mut bool) {
            egui::Window::new(&self.id).open(open_flag).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("exotic part");
                    if let Some(text) = short_text_for_object(probe, self.obj_id) {
                        ui.label(text);
                    } else {
                        ui.label("<none>");
                    }
                });

                show_object_attributes(ui, probe, self.obj_id);
            });
        }
    }

    fn show_object_attributes(ui: &mut egui::Ui, probe: &Probe, obj_id: ObjectId) {
        use mcjs_vm::interpreter::debugger::{IndexOrKey, Object};
        use mcjs_vm::interpreter::Value as InterpreterValue;

        let obj = probe.get_object(obj_id).unwrap();
        // TODO Inefficient!
        let props = obj.own_properties();

        for prop in props {
            let value = obj
                .get_own_element_or_property(IndexOrKey::Key(&prop))
                .unwrap();

            match value {
                InterpreterValue::Object(obj_id) => {
                    let header = format!(
                        "{} = {}",
                        prop,
                        short_text_for_object(probe, obj_id).unwrap_or("<object>".into())
                    );
                    ui.collapsing(header, |ui| {
                        show_object_attributes(ui, probe, obj_id);
                    });
                }
                _ => {
                    ui.label(format!("{} = {:?}", prop, value));
                }
            }
        }
    }

    pub fn short_text_for_object<'a>(probe: &'a Probe, obj_id: ObjectId) -> Option<Cow<'a, str>> {
        let obj = probe.get_object(obj_id).unwrap();

        if let Some(str_val) = obj.as_str() {
            Some(format!("{:?}", str_val).into())
        } else if let Some(closure) = obj.as_closure() {
            match closure {
                Closure::Native(_) => Some("[Function Native]".into()),
                Closure::JS(jsc) => Some(format!("[Function {:?}]", jsc.fnid()).into()),
            }
        } else if obj.array_elements().is_some() {
            Some("[Array]".into())
        } else {
            None
        }
    }
}

mod source_code_view {
    use std::{ops::Range, rc::Rc, sync::Arc};

    use mcjs_vm::{bytecode, interpreter::debugger::Probe};

    #[derive(Default)]
    pub struct Cache {
        main: Option<Main>,
        focus: Option<Focus>,
    }
    struct Main {
        fnid: bytecode::FnId,
        ppi: f32,
        galley: Arc<egui::Galley>,
        func_lookup: mcjs_vm::FunctionLookup,
    }
    struct Focus {
        cursor_ofs: u32,
        candidate_breakranges: Vec<BreakRangeCache>,
        preview_breakrange_index: Option<usize>,
    }
    struct BreakRangeCache {
        id: mcjs_vm::BreakRangeID,
        br: bytecode::BreakRange,
        galley: Arc<egui::Galley>,
    }

    impl Cache {
        pub fn preview_breakrange(&self) -> Option<&bytecode::BreakRange> {
            let focus = self.focus.as_ref()?;
            let ndx = focus.preview_breakrange_index?;
            Some(&focus.candidate_breakranges[ndx].br)
        }
    }

    impl Main {
        fn global_range_to_local(
            &self,
            lo: swc_common::BytePos,
            hi: swc_common::BytePos,
        ) -> Option<(u32, u32)> {
            let swc_common::SourceFileAndBytePos { sf: sf_lo, pos: _ } =
                self.func_lookup.source_map.lookup_byte_offset(lo);
            let swc_common::SourceFileAndBytePos { sf: sf_hi, pos: _ } =
                self.func_lookup.source_map.lookup_byte_offset(hi);

            if !Rc::ptr_eq(&sf_lo, &sf_hi) || !Rc::ptr_eq(&sf_lo, &self.func_lookup.source_file) {
                return None;
            }

            let start_ofs = sf_lo.start_pos.0;
            let lo = lo.0 - start_ofs;
            let hi = hi.0 - start_ofs;
            Some((lo, hi))
        }
    }

    #[derive(Default)]
    pub struct Response {
        pub set_source_bpkt: Option<mcjs_vm::BreakRangeID>,
        pub set_focus_cursor: Option<u32>,
        pub close_focus: bool,
        pub set_preview_brange_ndx: Option<usize>,
    }

    pub fn show(ui: &mut egui::Ui, cache: &Cache) -> Response {
        let mut my_response = Response::default();

        let main = match &cache.main {
            Some(main) => main,
            None => {
                ui.label("<no source code available>");
                return my_response;
            }
        };

        egui::ScrollArea::both().show(ui, |ui| {
            let galley = match &cache.focus {
                Some(focus) => match &focus.preview_breakrange_index {
                    Some(ndx) => &focus.candidate_breakranges[*ndx].galley,
                    None => &main.galley,
                },
                None => &main.galley,
            };
            let res = ui.add(egui::Label::new(Arc::clone(galley)).sense(egui::Sense::click()));

            if let Some(pos) = res.interact_pointer_pos() {
                let cursor_ofs: u32 = galley
                    .cursor_from_pos(pos - res.rect.min)
                    .ccursor
                    .index
                    .try_into()
                    .unwrap();
                my_response.set_focus_cursor = Some(cursor_ofs);
            }

            if let Some(focus) = &cache.focus {
                let breakranges = &focus.candidate_breakranges;

                // TODO Make window closable
                egui::Window::new("breakranges at point").show(ui.ctx(), |ui| {
                    ui.label(format!(
                        "{} breakranges at position {}",
                        breakranges.len(),
                        focus.cursor_ofs,
                    ));

                    my_response.set_preview_brange_ndx = None;
                    for (ndx, br_cache) in breakranges.iter().enumerate() {
                        let res = ui.button(format!("{:?}", br_cache.br.lo));

                        if res.hovered() {
                            my_response.set_preview_brange_ndx = Some(ndx);
                        }

                        if res.clicked() {
                            my_response.set_source_bpkt = Some(br_cache.id);
                            my_response.close_focus = true;
                        }
                    }
                });
            }
        });

        my_response
    }

    pub fn update(
        cache: &mut Cache,
        probe: &Probe,
        response: &Response,
        fonts: &egui::epaint::Fonts,
    ) {
        if probe.frames().len() == 0 {
            // stack empty, just reset the whole thing
            cache.main = None;
            cache.focus = None;
            return;
        }

        let giid = probe.giid();
        let fnid = giid.0;

        // if pixels per point changes, we need to re-make the galleys
        let ppi = fonts.pixels_per_point();

        let needs_update =
            !matches!(&cache.main, Some(main) if fnid == main.fnid && ppi == main.ppi);
        if needs_update {
            let loader = probe.loader();
            cache.main = cache_main(loader, fnid, fonts);
            cache.focus = None;
        };

        if response.set_source_bpkt.is_some() {
            cache.focus = None;
        }

        if let Some(cursor_ofs) = response.set_focus_cursor {
            let needs_update =
                !matches!(&cache.focus, Some(focus) if focus.cursor_ofs == cursor_ofs);
            if needs_update {
                let main = cache.main.as_ref().unwrap();
                cache.focus = cache_focus(probe, fnid, cursor_ofs, fonts, main);
            }
        }

        if let (Some(focus), Some(ndx)) = (&mut cache.focus, &response.set_preview_brange_ndx) {
            focus.preview_breakrange_index = Some(*ndx);
        }
    }

    fn cache_focus(
        probe: &Probe<'_, '_>,
        fnid: mcjs_vm::FnId,
        cursor_ofs: u32,
        fonts: &egui::epaint::Fonts,
        main: &Main,
    ) -> Option<Focus> {
        let candidate_breakranges = probe
            .loader()
            .function_breakranges(fnid)
            .unwrap()
            .filter_map(|(brid, br)| {
                let (lo, hi) = main.global_range_to_local(br.lo, br.hi)?;
                if lo <= cursor_ofs && cursor_ofs < hi {
                    Some((brid, br, lo, hi))
                } else {
                    None
                }
            })
            .map(|(brid, br, lo, hi)| {
                let galley = make_highlight_galley(
                    main.func_lookup.full_text(),
                    lo as usize..hi as usize,
                    egui::Color32::GRAY,
                    egui::Color32::RED,
                    fonts,
                );
                BreakRangeCache {
                    id: brid,
                    br: br.clone(),
                    galley,
                }
            })
            .collect();
        Some(Focus {
            cursor_ofs,
            candidate_breakranges,
            preview_breakrange_index: None,
        })
    }

    fn cache_main(
        loader: &mcjs_vm::Loader,
        fnid: mcjs_vm::FnId,
        fonts: &egui::text::Fonts,
    ) -> Option<Main> {
        let func_lookup = loader.lookup_function(fnid)?;
        let abs_span = *loader.get_function(fnid).unwrap().span();
        if abs_span.is_dummy() {
            return None;
        }

        let galley = make_highlight_galley(
            func_lookup.full_text(),
            func_lookup.local_range_usize(),
            egui::Color32::GRAY,
            egui::Color32::WHITE,
            fonts,
        );

        Some(Main {
            fnid,
            ppi: fonts.pixels_per_point(),
            galley,
            func_lookup,
        })
    }

    fn make_highlight_galley(
        text: &str,
        highlight: Range<usize>,
        normal_color: egui::Color32,
        highlight_color: egui::Color32,
        fonts: &egui::epaint::text::Fonts,
    ) -> Arc<egui::Galley> {
        use egui::epaint::text::{FontFamily, FontId, LayoutJob, TextFormat};
        let mut layout_job = LayoutJob::default();

        layout_job.append(
            &text[0..highlight.start],
            0.0,
            TextFormat {
                font_id: FontId::new(14.0, FontFamily::Monospace),
                color: normal_color,
                ..Default::default()
            },
        );

        layout_job.append(
            &text[highlight.start..highlight.end],
            0.0,
            TextFormat {
                font_id: FontId::new(14.0, FontFamily::Monospace),
                color: highlight_color,
                ..Default::default()
            },
        );

        layout_job.append(
            &text[highlight.end..],
            0.0,
            TextFormat {
                font_id: FontId::new(14.0, FontFamily::Monospace),
                color: normal_color,
                ..Default::default()
            },
        );

        fonts.layout_job(layout_job)
    }
}

mod interpreter_manager {
    use std::cmp::Ordering;
    use std::path::PathBuf;
    use std::pin::Pin;

    use mcjs_vm::{
        bytecode,
        interpreter::{
            debugger::{BreakpointError, Probe, SuspendCause},
            InterpreterError,
        },
        BreakRangeID, FnId, GlobalIID, Interpreter, Loader, Realm,
    };

    use anyhow::{anyhow, Result};

    use standalone_interpreter::StandaloneInterpreter;

    #[derive(Debug)]
    pub struct Params {
        pub main_directory: Option<PathBuf>,
        pub filenames: Vec<PathBuf>,
    }

    #[derive(Debug)]
    pub enum Error<'a> {
        Interpreter(InterpreterError<'a>),
    }

    pub struct Manager {
        scripts: Vec<Script>,
        init: InterpreterInit,
        state: State,
    }

    pub enum State {
        /// Ready to start the next file in `self.scripts`.
        Ready {
            // TODO script_ndx really needed?
            script_ndx: usize,
            loader: Pin<Box<Loader>>,
            realm: Pin<Box<Realm>>,
        },

        /// Finished successfully. Won't proceed to any other state.
        Finished,

        Suspended(SuspendedState),

        /// Interpreter has failed with an error.  All ephemeral data
        /// structures, including the Interpreter itself, have been torn down.
        /// Resuming execution or analyzing the interpreter's state is no longer
        /// possible.
        Failed(Error<'static>),
    }
    /// Interpreter suspended.  The reason for suspending (e.g. a
    /// breakpoint) is specified by `cause`.
    pub struct SuspendedState {
        /// The number of scripts (from Manager::scripts) that have been completed.
        n_scripts_done: usize,
        si: StandaloneInterpreter,
        cause: SuspendCause,
    }

    /// This stores the parts of an Interpreter's state that are set every time
    /// a new Interpreter is created.
    ///
    /// (Keep in mind that `Interpreter` is the object that tracks the execution
    /// along a single script or module loading. It gets deleted every time the
    /// script or module is finished running/loading.)
    ///
    /// These include instruction and source breakpoints (so that they survive
    /// across the execution of different scripts).
    #[derive(Default)]
    struct InterpreterInit {
        instr_bkpts: Vec<GlobalIID>,
        source_bkpts: Vec<BreakRangeID>,
    }

    impl InterpreterInit {
        /// Restores the state stored in this object into the given Interpreter.
        ///
        /// This operation should never fail. (It panics otherwise.)
        fn restore(&self, intrp: &mut Interpreter) {
            let mut probe = Probe::attach(intrp);

            for giid in &self.instr_bkpts {
                let res = probe.set_instr_breakpoint(*giid);
                check_breakpoint_result(res);
            }

            for brange_id in &self.source_bkpts {
                let res = probe.set_source_breakpoint(*brange_id);
                check_breakpoint_result(res);
            }
        }
    }

    /// Represents one of the scripts that the interpreter will run.
    ///
    /// Each file is associated to a stable FnId, in order to keep valid those breakpoints that
    /// were placed on scripts (as opposed to modules, which get this property for free, because
    /// they are cached/reused).
    ///
    /// Details other than the FnId should be retrived via the Loader.
    struct Script {
        filename: PathBuf,
        main_fnid: FnId,
    }
    impl Script {
        fn read_script(loader: &mut Loader, filename: PathBuf) -> Result<Script> {
            let filename_str = filename.to_string_lossy().into_owned();
            let script_text = std::fs::read_to_string(filename.clone())
                .map_err(|err| anyhow!("read error: {:?}: {:?}", filename, err))?;

            let main_fnid = loader
                .load_script(Some(filename_str.clone()), script_text)
                .map_err(|err| anyhow!("compile error: {:?}: {:?}", filename_str, err))?;

            Ok(Script {
                filename,
                main_fnid,
            })
        }
    }

    fn check_breakpoint_result<T>(res: Result<T, BreakpointError>) {
        match res {
            Ok(_) | Err(BreakpointError::AlreadyThere) => {}
            Err(err) => panic!("unexpected error while setting breakpoint: {:?}", err),
        }
    }

    impl Manager {
        pub fn new(params: &Params) -> Result<Manager> {
            let mut loader = Box::pin(Loader::new(params.main_directory.clone()));
            let realm = Box::pin(Realm::new(&mut loader));

            let scripts = params
                .filenames
                .iter()
                .map(|filename| Script::read_script(&mut loader, filename.clone()))
                .collect::<Result<Vec<_>, _>>()?;

            let mut init = InterpreterInit::default();
            // Place an instruction breakpoint at the beginning of each script
            for script in &scripts {
                let fnid = script.main_fnid;
                let iid = bytecode::IID(0);
                init.instr_bkpts.push(bytecode::GlobalIID(fnid, iid));
            }

            let state = if scripts.is_empty() {
                State::Finished
            } else {
                State::Ready {
                    loader,
                    realm,
                    script_ndx: 0,
                }
            };

            Ok(Manager {
                scripts,
                init,
                state,
            })
        }

        pub fn state_mut(&mut self) -> &mut State {
            &mut self.state
        }

        pub fn set_instr_breakpoint(&mut self, giid: GlobalIID) {
            self.init.instr_bkpts.push(giid);

            if let State::Suspended(SuspendedState {
                si: ref mut interpreter,
                ..
            }) = &mut self.state
            {
                let mut probe = interpreter.probe();
                let res = probe.set_instr_breakpoint(giid);
                check_breakpoint_result(res);
            }
        }

        pub fn set_source_breakpoint(&mut self, brange_id: BreakRangeID) {
            self.init.source_bkpts.push(brange_id);

            if let State::Suspended(SuspendedState {
                si: ref mut interpreter,
                ..
            }) = &mut self.state
            {
                let mut probe = interpreter.probe();
                let res = probe.set_source_breakpoint(brange_id);
                check_breakpoint_result(res);
            }
        }

        pub fn resume(&mut self) {
            let state = std::mem::replace(&mut self.state, State::Finished);
            self.state = match state {
                State::Ready {
                    script_ndx,
                    loader,
                    realm,
                } => {
                    let script = &self.scripts[script_ndx];
                    println!(
                        "(starting interpreter for file: {})",
                        script.filename.display()
                    );

                    let mut si = StandaloneInterpreter::new(realm, loader, script.main_fnid);
                    self.init.restore(si.interpreter_mut());
                    State::Suspended(SuspendedState {
                        n_scripts_done: script_ndx,
                        si,
                        cause: SuspendCause::Breakpoint,
                    })
                }
                State::Failed(_) | State::Finished => {
                    // nothing to do -- resume does not change state at all
                    return;
                }
                State::Suspended(SuspendedState {
                    si,
                    n_scripts_done,
                    cause: _,
                }) => {
                    use standalone_interpreter::Exit;

                    match si.run() {
                        Ok(Exit::Finished { realm, loader }) => {
                            let script_ndx = n_scripts_done + 1;
                            let script_count = self.scripts.len();
                            match script_ndx.cmp(&script_count) {
                                Ordering::Less => State::Ready {
                                    script_ndx,
                                    loader,
                                    realm,
                                },
                                Ordering::Equal => State::Finished,
                                Ordering::Greater => panic!("assertion failed"),
                            }
                        }
                        Ok(Exit::Suspended { si, cause }) => State::Suspended(SuspendedState {
                            n_scripts_done,
                            si,
                            cause,
                        }),
                        Err(err) => State::Failed(Error::Interpreter(err)),
                    }
                }
            };
        }
    }

    impl SuspendedState {
        pub fn cause(&self) -> &SuspendCause {
            &self.cause
        }

        pub fn probe_mut(&mut self) -> Probe<'_, 'static> {
            self.si.probe()
        }
    }

    mod standalone_interpreter {
        use std::pin::Pin;

        use mcjs_vm::{
            bytecode,
            interpreter::{self, debugger::Probe, InterpreterError, SuspendCause},
            Interpreter, Loader, Realm,
        };

        pub struct StandaloneInterpreter {
            // Keep these 2 still, on the heap:
            realm: Pin<Box<Realm>>,
            loader: Pin<Box<Loader>>,
            // This can be moved in/out
            interpreter: Interpreter<'static>,
            // It's still possible to move the struct itself
        }

        pub enum Exit {
            Finished {
                realm: Pin<Box<Realm>>,
                loader: Pin<Box<Loader>>,
            },
            Suspended {
                si: StandaloneInterpreter,
                cause: SuspendCause,
            },
        }

        impl StandaloneInterpreter {
            pub(super) fn new(
                mut realm: Pin<Box<Realm>>,
                mut loader: Pin<Box<Loader>>,
                fnid: bytecode::FnId,
            ) -> Self {
                let interpreter: Interpreter<'static> = {
                    let realm = Pin::get_mut(Pin::as_mut(&mut realm));
                    let loader = Pin::get_mut(Pin::as_mut(&mut loader));
                    let src = Interpreter::new(realm, loader, fnid);
                    unsafe { std::mem::transmute(src) }
                };

                StandaloneInterpreter {
                    realm,
                    loader,
                    interpreter,
                }
            }

            pub(super) fn run(self) -> std::result::Result<Exit, InterpreterError<'static>> {
                let exit = self.interpreter.run()?;
                match exit {
                    interpreter::Exit::Finished(_) => Ok(Exit::Finished {
                        realm: self.realm,
                        loader: self.loader,
                    }),
                    interpreter::Exit::Suspended { interpreter, cause } => {
                        let si = StandaloneInterpreter {
                            realm: self.realm,
                            loader: self.loader,
                            interpreter,
                        };
                        Ok(Exit::Suspended { si, cause })
                    }
                }
            }

            pub(super) fn probe(&mut self) -> Probe<'_, 'static> {
                Probe::attach(&mut self.interpreter)
            }

            pub(super) fn interpreter_mut(&mut self) -> &mut Interpreter<'static> {
                &mut self.interpreter
            }
        }
    }
}
