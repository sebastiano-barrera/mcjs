use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;

use anyhow::Result;
use mcjs_vm::interpreter::debugger::ObjectId;
use mcjs_vm::interpreter::Fuel;
use mcjs_vm::{bytecode, BreakRangeID, GlobalIID};

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    eprintln!("params = {:?}", params);

    let mut si = interpreter_manager::StandaloneInterpreter::new(params)
        .expect("could not create interpreter");
    si.resume();

    let app = AppData::new(si);

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
    si: Pin<Box<interpreter_manager::StandaloneInterpreter>>,
    recent_state_change: bool,
    source_code_view: source_code_view::Cache,
    frame_ndx: usize,
    highlight: instr_view::Highlighted,
    object_windows: HashMap<ObjectId, object_view::ObjectWindow>,
}

impl AppData {
    fn new(si: Pin<Box<interpreter_manager::StandaloneInterpreter>>) -> Self {
        AppData {
            si,
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
        }

        let mut action = Action::None;

        match self.si.state_mut() {
            State::Ready => {
                let should_start = egui::CentralPanel::default()
                    .show(ctx, |ui| {
                        ui.label(format!(
                            "Ready to proceed with file #{}",
                            self.si.n_filenames_done()
                        ));
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
            _ => {}
        };

        {
            let probe = self.si.probe_mut().unwrap();
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

        egui::SidePanel::left("sidebar")
            .show(ctx, |ui| {
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
                // TODO implement 'delete'!
                let _ = ui.button("DELETE");

                ui.horizontal(|ui| {
                    ui.label("State:");
                    let text = match self.si.state_mut() {
                        State::Ready => "Ready",
                        State::Finished => "Finished",
                        State::Suspended(_) => "Suspended",
                        State::Failed(_) => "Failed",
                    };
                    ui.label(text);
                });

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

                        let text = format!("{:?}:{:?}", frame.header().fn_id, iid);
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
            })
            .inner;

        egui::SidePanel::left("bytecode")
            .min_width(400.0)
            .show(ctx, |ui| {
                let probe = self.si.probe_mut().unwrap();

                let frame = probe.frames().nth(self.frame_ndx).unwrap();
                let vm_giid = probe.frame_giid(self.frame_ndx);
                let fnid = vm_giid.0;
                let func = probe.loader().get_function(fnid).unwrap();

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
                    self.open_object_window(obj_id);
                }
                if let Some(giid) = bkpt_to_set {
                    action = Action::SetInstrBreakpoint { giid };
                }
            });

        if let Some(probe) = self.si.probe_mut() {
            let res = egui::CentralPanel::default()
                .show(ctx, |ui| {
                    source_code_view::show(ui, &mut self.source_code_view)
                })
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
                let mut probe = self.si.probe_mut().unwrap();
                probe.set_fuel(Fuel::Limited(1));
                self.si.resume();
                self.recent_state_change = true;
                self.frame_ndx = 0;
            }
            Action::Continue => {
                self.si.resume();
                self.recent_state_change = true;
                self.frame_ndx = 0;
            }
            Action::Restart => {
                self.si.restart();
                self.si.resume();
                self.recent_state_change = true;
                self.frame_ndx = 0;
            }
            Action::SetStackFrame { index } => {
                self.frame_ndx = index;
            }
            Action::SetSourceBreakpoint { brange_id } => {
                self.si.set_source_breakpoint(brange_id);
            }
            Action::SetInstrBreakpoint { giid } => {
                self.si.set_instr_breakpoint(giid);
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
            let value = self.frame.get_result(vreg);

            let fnid = self.probe.giid().0;
            let is_highlighted = {
                let is_vreg_hl = self.highlighted.match_vreg(fnid, vreg);
                let is_obj_id_hl = match value {
                    InterpreterValue::Object(value_obj_id) => {
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

            let obj_id = match value {
                InterpreterValue::Object(obj_id) => Some(obj_id),
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

    struct Analyzer {
        desc: InstrDescriptor,
        n_operands: usize,
    }
    impl Default for Analyzer {
        fn default() -> Self {
            Analyzer {
                desc: InstrDescriptor::default(),
                n_operands: 0,
            }
        }
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

    pub enum Highlighted {
        None,
        VReg((bytecode::FnId, bytecode::VReg)),
        Object((bytecode::FnId, ObjectId)),
        IID((bytecode::FnId, bytecode::IID)),
    }
    impl Default for Highlighted {
        fn default() -> Self {
            Highlighted::None
        }
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
            });
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
        } else if let Some(_) = obj.array_elements() {
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
        src: Rc<String>,
        source_start_ofs: u32,
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

            if res.double_clicked() {
                let pos = res.interact_pointer_pos().unwrap() - res.rect.min;
                let cursor_ofs: u32 = galley
                    .cursor_from_pos(pos)
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
        let giid = probe.giid();
        let fnid = giid.0;

        // if pixels per point changes, we need to re-make the galleys
        let ppi = fonts.pixels_per_point();

        let needs_update = match &cache.main {
            Some(main) if fnid == main.fnid && ppi == main.ppi => false,
            _ => true,
        };
        if needs_update {
            let loader = probe.loader();
            cache.main = cache_main(loader, fnid, fonts);
            cache.focus = None;
        };

        if response.set_source_bpkt.is_some() {
            cache.focus = None;
        }

        if let Some(cursor_ofs) = response.set_focus_cursor {
            let needs_update = match &cache.focus {
                Some(focus) if focus.cursor_ofs == cursor_ofs => false,
                _ => true,
            };
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
                let lo = br.lo.0 - main.source_start_ofs;
                let hi = br.hi.0 - main.source_start_ofs;
                if lo <= cursor_ofs && cursor_ofs < hi {
                    Some((brid, br, lo, hi))
                } else {
                    None
                }
            })
            .map(|(brid, br, lo, hi)| {
                let galley = make_highlight_galley(
                    main.src.as_str(),
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
        // Can't update with no source map
        let source_map = loader.get_source_map(fnid.0)?;
        let abs_span = *loader.get_function(fnid).unwrap().span();
        if abs_span.is_dummy() {
            return None;
        }

        let source_file = {
            let mut break_ranges = loader.function_breakranges(fnid).unwrap().peekable();

            // All break ranges must belong to the same file, so we just peek
            // one and use it to get a ptr to that swc_common::SourceFile.
            // Then we use source file's offset in the source map to make sure
            // that the markers are expressed in file-local offsets.
            //
            // (There should *always* be at least a single break range in a
            // function, belonging at least to an empty statement.  I *might*
            // be wrong; we'll see.)
            let (_, brange) = break_ranges.peek().expect("no break ranges!");
            source_map.lookup_byte_offset(brange.lo).sf
        };
        let (ofs_start, ofs_end) = source_map.span_to_char_offset(&source_file, abs_span);
        let ofs_start = ofs_start as usize;
        let ofs_end = ofs_end as usize;

        let galley = make_highlight_galley(
            &source_file.src,
            ofs_start..ofs_end,
            egui::Color32::GRAY,
            egui::Color32::WHITE,
            fonts,
        );

        Some(Main {
            fnid,
            ppi: fonts.pixels_per_point(),
            galley,
            src: Rc::clone(&source_file.src),
            source_start_ofs: source_file.start_pos.0,
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
    use std::marker::PhantomPinned;
    use std::path::PathBuf;
    use std::pin::Pin;

    use mcjs_vm::interpreter::debugger::{BreakpointError, Probe};
    use mcjs_vm::{
        interpreter::{Exit, InterpreterError},
        BreakRangeID, FnId, GlobalIID, Interpreter, Loader, Realm,
    };

    use anyhow::{anyhow, Result};

    #[derive(Debug)]
    pub struct Params {
        pub main_directory: Option<PathBuf>,
        pub filenames: Vec<PathBuf>,
    }

    #[derive(Debug)]
    pub enum Error<'a> {
        Interpreter(InterpreterError<'a>),
    }

    pub struct StandaloneInterpreter {
        realm: Realm,
        loader: Loader,
        scripts: Vec<Script>,
        n_scripts_done: usize,
        state: State<'static>,
        saved_state: DebuggingSave,
        _pin: PhantomPinned,
    }
    pub enum State<'a> {
        /// Ready to process the next file in the sequence
        Ready,

        /// Finished successfully. Won't proceed to any other state.
        Finished,
        Suspended(Interpreter<'a>),
        Failed(Error<'a>),
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

    /// Parts of the interpreter's state that are saved and restored across a restart cycle.
    ///
    /// These include instruction and source breakpoints.
    ///
    /// In broad strokes: this state is saved when calling `StandaloneInterpreter::restart`, and
    /// restored as soon as a new interpreter is created with the next call to
    /// `StandaloneInterpreter::next`.
    struct DebuggingSave {
        instr_bkpts: Vec<GlobalIID>,
        source_bkpts: Vec<BreakRangeID>,
    }

    impl Default for DebuggingSave {
        fn default() -> Self {
            DebuggingSave {
                instr_bkpts: Vec::new(),
                source_bkpts: Vec::new(),
            }
        }
    }

    impl DebuggingSave {
        /// Restores the state stored in this object into the given Interpreter.
        ///
        /// Returns true iff every breakpoint was restored successfully. Check it on
        /// return!
        #[must_use]
        fn restore(&self, intrp: &mut Interpreter) -> bool {
            let mut all_ok = true;
            let mut probe = Probe::attach(intrp);

            eprintln!(
                "restoring: {} instr bkpts, {} source bkpts",
                self.instr_bkpts.len(),
                self.source_bkpts.len()
            );

            for giid in &self.instr_bkpts {
                let res = probe.set_instr_breakpoint(*giid);
                match res {
                    Ok(_) | Err(BreakpointError::AlreadyThere) => {}
                    Err(_) => {
                        all_ok = false;
                    }
                }
            }

            for brange_id in &self.source_bkpts {
                let res = probe.set_source_breakpoint(*brange_id);
                match res {
                    Ok(_) | Err(BreakpointError::AlreadyThere) => {}
                    Err(_) => {
                        all_ok = false;
                    }
                }
            }

            all_ok
        }
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

    impl StandaloneInterpreter {
        pub fn new(params: Params) -> Result<Pin<Box<StandaloneInterpreter>>> {
            let mut loader = Loader::new(params.main_directory);
            let realm = Realm::new(&mut loader);

            let scripts = params
                .filenames
                .into_iter()
                .map(|filename| Script::read_script(&mut loader, filename))
                .collect::<Result<Vec<_>, _>>()?;

            let si = StandaloneInterpreter {
                realm,
                loader,
                scripts,
                n_scripts_done: 0,
                state: State::Ready,
                saved_state: DebuggingSave::default(),
                _pin: PhantomPinned,
            };

            Ok(Box::pin(si))
        }

        pub fn n_filenames_done(&self) -> usize {
            self.n_scripts_done
        }

        pub fn state_mut<'a>(self: &'a mut Pin<Box<Self>>) -> &'a mut State<'static> {
            // Safe because I only return `interpreter`, which is the
            // part of the struct that doesn't have to stay pinned
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };
            &mut self_.state
        }

        pub fn set_instr_breakpoint(self: &mut Pin<Box<Self>>, giid: GlobalIID) {
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };

            self_.saved_state.instr_bkpts.push(giid);

            if let Some(intrp) = self_.interpreter_mut() {
                let mut probe = Probe::attach(intrp);
                Self::check_breakpoint_result(probe.set_instr_breakpoint(giid))
            }
        }

        pub fn set_source_breakpoint(self: &mut Pin<Box<Self>>, brange_id: BreakRangeID) {
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };

            self_.saved_state.source_bkpts.push(brange_id);

            if let Some(intrp) = self_.interpreter_mut() {
                let mut probe = Probe::attach(intrp);
                Self::check_breakpoint_result(probe.set_source_breakpoint(brange_id));
            }
        }

        fn check_breakpoint_result<T>(res: Result<T, BreakpointError>) {
            match res {
                Ok(_) | Err(BreakpointError::AlreadyThere) => {}
                Err(err) => panic!("unexpected error while setting breakpoint: {:?}", err),
            }
        }

        pub fn restart(self: &mut Pin<Box<Self>>) {
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };

            self_.n_scripts_done = 0;
            self_.state = State::Ready;
        }

        fn interpreter_mut(&mut self) -> Option<&mut Interpreter<'static>> {
            match &mut self.state {
                State::Suspended(intrp) => Some(intrp),
                _ => None,
            }
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
                State::Ready => {
                    let script = &self_.scripts[self_.n_scripts_done];
                    println!();
                    println!(
                        "(starting loop for file: {})",
                        script.filename.to_string_lossy().into_owned()
                    );

                    let intrp =
                        Interpreter::new(&mut self_.realm, &mut self_.loader, script.main_fnid);
                    // Cast the lifetime: real "trust me" moment
                    let mut intrp: Interpreter<'static> = unsafe { std::mem::transmute(intrp) };

                    let all_ok = self_.saved_state.restore(&mut intrp);
                    if !all_ok {
                        eprintln!("warning: not all breakpoints could be restored after restart");
                    }
                    State::Suspended(intrp)
                }
                cur @ State::Failed(_) | cur @ State::Finished => {
                    // nothing  to do
                    cur
                }
                State::Suspended(intrp) => match intrp.run() {
                    Ok(exit) => match exit {
                        Exit::Finished(_) => {
                            self_.n_scripts_done += 1;
                            match self_.n_scripts_done.cmp(&self_.scripts.len()) {
                                Ordering::Less => {
                                    self_.state = State::Ready;
                                    return self.resume();
                                }
                                Ordering::Equal => State::Finished,
                                Ordering::Greater => panic!("assertion failed"),
                            }
                        }
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
}
