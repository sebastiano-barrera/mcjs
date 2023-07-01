use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use eframe::egui;
use mcjs_vm::{bytecode, inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};

fn open_case(case_file_path: &Path) -> Result<inspector_case::Case> {
    let mut f = std::fs::File::open(&case_file_path)
        .with_context(|| format!("opening case file: {}", case_file_path.display()))?;
    let case: inspector_case::Case = rmp_serde::from_read(&mut f)
        .with_context(|| format!("decoding case file: {}", case_file_path.display()))?;
    Ok(case)
}

fn main() -> eframe::Result<()> {
    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let case = open_case(&case_file_path).expect("could not open case file");

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "mcjs Inspector",
        options,
        Box::new(move |_cc| Box::new(InspectorApp::new(case))),
    )
}

struct InspectorApp {
    state: VMState,

    case: inspector_case::Case,
    root_mod_id: mcjs_vm::ModuleId,
    codebase: Arc<mcjs_vm::Codebase>,

    breakpoints: Vec<Breakpoint>,
}

enum VMState {
    Initial,
    Running {
        // Option because there is an instant where we have a &mut VMState, but we need to take and
        // move the JoinHandle in order to extract the VMResult and switch to the next VMState
        vm_thread: Option<std::thread::JoinHandle<VMResult>>,
    },
    Suspended {
        vm_result: VMResult,
    },
}

struct StackViewItem {
    label_opt: Option<&'static str>,
    val_ndx: String,
    val_str: String,
}
struct InstrView {
    iid: String,
    instr: bytecode::Instr,
}

impl InspectorApp {
    fn new(case: inspector_case::Case) -> Self {
        let include_paths = case.include_paths.iter().map(|pb| pb.as_path());
        let builder_params = mcjs_vm::BuilderParams {
            loader: Box::new(mcjs_vm::FileLoader::new(include_paths)),
        };
        let mut builder = builder_params.to_builder();

        let root_mod_id = match &case.root {
            inspector_case::Root::ModuleImport(path) => builder
                .compile_file(path.clone())
                .unwrap_or_else(|err| panic!("compile error: {:?}", err)),
            inspector_case::Root::InlineScript(_) => {
                todo!("sorry, Root::InlineScript is not supported yet")
            }
        };
        let codebase = Arc::new(builder.build());

        InspectorApp {
            state: VMState::Initial,
            case,
            root_mod_id,
            codebase,
            breakpoints: vec![Breakpoint(GlobalIID(FnId(70), IID(6)))],
        }
    }

    fn ui_initial(&mut self, ui: &mut egui::Ui) -> bool {
        ui.heading("Ready to start");

        let label_layout =
            egui::Layout::top_down(egui::Align::Min).with_cross_align(egui::Align::Max);

        let n_funcs = self.codebase.all_functions().len();
        ui.label(format!("Code loaded.  {} functions.", n_funcs));

        ui.label("Case details:");
        egui::Grid::new("case_details").show(ui, |ui| {
            ui.with_layout(label_layout, |ui| {
                ui.add(egui::Label::new("Include paths").wrap(false));
            });
            ui.vertical(|ui| {
                for path in &self.case.include_paths {
                    ui.add(egui::Label::new(path.to_str().unwrap()).wrap(false));
                }
            });
            ui.end_row();

            ui.with_layout(label_layout, |ui| {
                ui.add(egui::Label::new("Execution root").wrap(false));
            });
            match &self.case.root {
                inspector_case::Root::ModuleImport(module_path) => {
                    ui.label(format!("Load module: {}", module_path));
                }
                inspector_case::Root::InlineScript(script_text) => {
                    let limit = 100;
                    if script_text.len() > limit {
                        ui.label(format!("Run script: {}...", &script_text[..limit]));
                    } else {
                        ui.label(format!("Run script: {}", &script_text));
                    }
                }
            }
            ui.end_row();
        });

        let start_clicked = ui
            .with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                ui.button(egui::WidgetText::RichText(
                    egui::RichText::new("Start").size(20.0),
                ))
                .clicked()
            })
            .inner;

        start_clicked
    }

    fn release_the_kraken(&mut self) -> std::thread::JoinHandle<VMResult> {
        let run_params = RunParams {
            breakpoints: self.breakpoints.clone(),
            codebase: Arc::clone(&self.codebase),
            root_mod_id: self.root_mod_id,
        };
        std::thread::spawn(move || run_vm(run_params))
    }

    fn ui_running(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Running...");
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    fn ui_suspended(&self, ui: &mut egui::Ui, ctx: &egui::Context, vm_result: &VMResult) {
        if let Some(core_dump) = &vm_result.core_dump {
            egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
                ui.heading("VM interrupted");
                ui.label("Core dump received.");

                ui.label(format!("{} stack frames.", core_dump.data.len()));
            });

            egui::SidePanel::left("left_panel").show(ctx, |ui| {
                ui.label(format!(
                    "Executed instructions ({}):",
                    vm_result.instr_history.len()
                ));

                use egui_extras::{Column, TableBuilder};
                ui.push_id("table__instr_history", |ui| {
                    TableBuilder::new(ui)
                        .column(Column::exact(60.0))
                        .column(Column::exact(300.0))
                        .striped(true)
                        .body(|body| {
                            body.rows(20.0, vm_result.instr_history.len(), |row_ndx, mut row| {
                                let GlobalIID(fn_id, iid) = &vm_result.instr_history[row_ndx];

                                let func =
                                    vm_result.run_params.codebase.get_function(*fn_id).unwrap();
                                let instr = &func.instrs()[iid.0 as usize];

                                row.col(|ui| {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(iid.0.to_string());
                                        },
                                    );
                                });
                                row.col(|ui| {
                                    ui.horizontal(|ui| {
                                        render_instruction(ui, instr, func);
                                    });
                                });
                            });
                        });
                });
            });

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading(format!(
                    "Error messages ({})",
                    vm_result.error_messages.len()
                ));
                for msg in &vm_result.error_messages {
                    ui.label(msg);
                }

                ui.heading("Stack");
                let header = core_dump.data.header();

                egui::Grid::new("frame_header").show(ui, |ui| {
                    ui.label("fn ID =");
                    ui.label(format!("{}", header.fn_id.0));
                    ui.end_row();

                    ui.label("this =");
                    ui.label(format!("{:?}", header.this));
                    ui.end_row();

                    ui.label("return value to =");
                    ui.label(format!(
                        "{:?} / {:?}",
                        header.return_value_vreg, header.return_to_iid
                    ));
                    ui.end_row();
                });

                use egui_extras::{Column, TableBuilder};
                ui.push_id("table__stack_view", |ui| {
                    TableBuilder::new(ui)
                        .column(Column::exact(30.0))
                        .column(Column::exact(20.0))
                        .column(Column::remainder())
                        .striped(true)
                        .body(|body| {
                            let items = &vm_result.stack_view;

                            body.rows(20.0, items.len(), |row_ndx, mut row| {
                                let StackViewItem {
                                    label_opt,
                                    val_ndx,
                                    val_str,
                                } = &items[row_ndx];
                                row.col(|ui| {
                                    if let Some(label) = label_opt {
                                        ui.label(*label);
                                    }
                                });
                                row.col(|ui| {
                                    ui.label(format!("{}", val_ndx));
                                });
                                row.col(|ui| {
                                    ui.label(val_str);
                                });
                            });
                        });
                });
            });
        } else {
            ui.heading("VM finished");
            ui.label("No core dump emitted.");

            ui.heading(format!(
                "Error messages ({} total)",
                vm_result.error_messages.len()
            ));
            for msg in &vm_result.error_messages {
                ui.label(msg);
            }
        }
    }
}

fn make_stack_view(intrp_data: &mcjs_vm::stack::InterpreterData) -> Vec<StackViewItem> {
    let mut items = Vec::new();

    let header = intrp_data.header();

    for ndx in 0..header.n_instrs as usize {
        let label_opt = if ndx == 0 { Some("Vars") } else { None };
        let val_ndx = format!("{}", ndx);
        let vreg = bytecode::VReg(ndx.try_into().unwrap());
        let val_str = format!("{:?}", intrp_data.get_result(vreg));
        items.push(StackViewItem {
            label_opt,
            val_ndx,
            val_str,
        });
    }
    for ndx in 0..header.n_args as usize {
        let label_opt = if ndx == 0 { Some("Args") } else { None };
        let argndx = bytecode::ArgIndex(ndx.try_into().unwrap());
        let val_ndx = format!("{}", ndx);
        let val_str = format!("{:?}", intrp_data.get_arg(argndx).unwrap());
        items.push(StackViewItem {
            label_opt,
            val_ndx,
            val_str,
        });
    }
    for ndx in 0..header.n_captures as usize {
        let label_opt = if ndx == 0 { Some("Caps") } else { None };
        let capndx = bytecode::CaptureIndex(ndx.try_into().unwrap());
        let val_ndx = format!("{}", ndx);
        let val_str = format!("{:?}", intrp_data.get_capture(capndx));
        items.push(StackViewItem {
            label_opt,
            val_ndx,
            val_str,
        });
    }

    items
}

fn render_instruction(ui: &mut egui::Ui, instr: &bytecode::Instr, func: &bytecode::Function) {
    use mcjs_vm::bytecode::{Instr, VReg};

    let read1 = |ui: &mut egui::Ui, name: &str, arg: VReg| {
        ui.label(name);
        render_vreg_read(ui, arg);
    };
    let read2 = |ui: &mut egui::Ui, name: &str, arg0: VReg, arg1: VReg| {
        ui.label(name);
        render_vreg_read(ui, arg0);
        render_vreg_read(ui, arg1);
    };
    let read3 = |ui: &mut egui::Ui, name: &str, arg0: VReg, arg1: VReg, arg2: VReg| {
        ui.label(name);
        render_vreg_read(ui, arg0);
        render_vreg_read(ui, arg1);
        render_vreg_read(ui, arg2);
    };
    let simple0 = |ui: &mut egui::Ui, name: &str, dest: VReg| {
        ui.label(name);
        render_vreg_write(ui, dest);
    };
    let simple1 = |ui: &mut egui::Ui, name: &str, dest: VReg, a: VReg| {
        ui.label(name);
        render_vreg_write(ui, dest);
        render_vreg_read(ui, a);
    };
    let simple2 = |ui: &mut egui::Ui, name: &str, dest: VReg, a: VReg, b: VReg| {
        ui.label(name);
        render_vreg_write(ui, dest);
        render_vreg_read(ui, a);
        render_vreg_read(ui, b);
    };

    match instr {
        Instr::Nop => {
            ui.label("Nop");
        }
        Instr::LoadConst(dest, constndx) => {
            ui.label("LoadConst");
            render_vreg_write(ui, *dest);

            let const_value = &func.consts()[constndx.0 as usize];
            let mut text = format!("{:?}", const_value);
            if text.len() > 50 {
                text.truncate(50);
                text.push_str("...");
            }
            ui.label(text);
        }
        Instr::LoadNull(dest) => simple0(ui, "LoadNull", *dest),
        Instr::LoadUndefined(dest) => simple0(ui, "LoadUndefined", *dest),
        Instr::LoadCapture(dest, capndx) => {
            ui.label("LoadCapture");
            render_vreg_write(ui, *dest);
            ui.label(format!("{:?}", capndx));
        }
        Instr::LoadArg(dest, argndx) => {
            ui.label("LoadArg");
            render_vreg_write(ui, *dest);
            ui.label(format!("{:?}", argndx));
        }
        Instr::LoadThis(dest) => simple0(ui, "LoadThis", *dest),
        Instr::Copy { dst, src } => simple1(ui, "Copy", *dst, *src),
        Instr::GetGlobal { dest, key } => simple1(ui, "GetGlobal", *dest, *key),
        Instr::BoolNot { dest, arg } => simple1(ui, "BoolNot", *dest, *arg),
        Instr::UnaryMinus { dest, arg } => simple1(ui, "UnaryMinus", *dest, *arg),
        Instr::ArithAdd(dest, a, b) => simple2(ui, "ArithAdd", *dest, *a, *b),
        Instr::ArithSub(dest, a, b) => simple2(ui, "ArithSub", *dest, *a, *b),
        Instr::ArithMul(dest, a, b) => simple2(ui, "ArithMul", *dest, *a, *b),
        Instr::ArithDiv(dest, a, b) => simple2(ui, "ArithDiv", *dest, *a, *b),
        Instr::ArithInc(dest, arg) => simple1(ui, "ArithInc", *dest, *arg),
        Instr::ArithDec(dest, arg) => simple1(ui, "ArithDec", *dest, *arg),
        Instr::CmpGE(dest, a, b) => simple2(ui, "CmpGE", *dest, *a, *b),
        Instr::CmpGT(dest, a, b) => simple2(ui, "CmpGT", *dest, *a, *b),
        Instr::CmpLT(dest, a, b) => simple2(ui, "CmpLT", *dest, *a, *b),
        Instr::CmpLE(dest, a, b) => simple2(ui, "CmpLE", *dest, *a, *b),
        Instr::CmpEQ(dest, a, b) => simple2(ui, "CmpEQ", *dest, *a, *b),
        Instr::CmpNE(dest, a, b) => simple2(ui, "CmpNE", *dest, *a, *b),
        Instr::BoolOpAnd(dest, a, b) => simple2(ui, "BoolOpAnd", *dest, *a, *b),
        Instr::BoolOpOr(dest, a, b) => simple2(ui, "BoolOpOr", *dest, *a, *b),
        Instr::IsInstanceOf(dest, obj, sup) => simple2(ui, "IsInstanceOf", *dest, *obj, *sup),

        Instr::JmpIf { cond, dest } => {
            ui.label("JmpIf");
            render_vreg_read(ui, *cond);
            render_iid(ui, *dest);
        }
        Instr::Jmp(dest) => {
            ui.label("Jmp");
            render_iid(ui, *dest);
        }

        Instr::PushToSink(arg) => read1(ui, "PushToSink", *arg),
        Instr::Return(arg) => read1(ui, "Return", *arg),
        Instr::Call {
            return_value,
            this,
            callee,
        } => simple2(ui, "Call", *return_value, *this, *callee),
        Instr::CallArg(arg) => read1(ui, "CallArg", *arg),
        Instr::ClosureNew {
            dest,
            fnid,
            forced_this,
        } => {
            ui.label("ClosureNew");
            render_vreg_write(ui, *dest);
            render_fnid(ui, *fnid);
            if let Some(forced_this) = forced_this {
                ui.label("this=");
                render_vreg_read(ui, *forced_this);
            }
        }
        Instr::ClosureAddCapture(arg) => read1(ui, "ClosAddCap", *arg),
        Instr::ObjCreateEmpty(dest) => simple0(ui, "ObjCreateEmpty", *dest),
        Instr::ObjSet { obj, key, value } => simple2(ui, "ObjSet", *obj, *key, *value),
        Instr::ObjGet { dest, obj, key } => simple2(ui, "ObjGet", *dest, *obj, *key),
        Instr::ObjGetKeys { dest, obj } => simple1(ui, "ObjGetKeys", *dest, *obj),
        Instr::ObjDelete { dest, obj, key } => simple2(ui, "ObjDelete", *dest, *obj, *key),
        Instr::ArrayPush { arr, value } => read2(ui, "ArrayPush", *arr, *value),
        Instr::ArrayNth { dest, arr, index } => simple2(ui, "ArrayNth", *dest, *arr, *index),
        Instr::ArraySetNth { arr, index, value } => read3(ui, "ArraySetNth", *arr, *index, *value),
        Instr::ArrayLen { dest, arr } => simple1(ui, "ArrayLen", *dest, *arr),
        Instr::StrCreateEmpty(dest) => simple0(ui, "StrCreateEmpty", *dest),
        Instr::StrAppend(str, other) => read2(ui, "StrAppend", *str, *other),
        Instr::NewIterator { dest, obj } => simple1(ui, "NewIterator", *dest, *obj),
        Instr::IteratorGetCurrent { dest, iter } => simple1(ui, "IteratorGetCurrent", *dest, *iter),
        Instr::IteratorAdvance { iter } => read1(ui, "IteratorAdvance", *iter),
        Instr::JmpIfIteratorFinished { iter, dest } => {
            ui.label("JmpIfIteratorFinished");
            render_vreg_read(ui, *iter);
            render_iid(ui, *dest);
        }
        Instr::TypeOf { dest, arg } => simple1(ui, "TypeOf", *dest, *arg),
        Instr::GetModule(dest, module_id) => {
            ui.label("GetModule");
            render_vreg_write(ui, *dest);
            ui.label(format!("module:{}", module_id.0));
        }
        Instr::Throw(arg) => read1(ui, "Throw", *arg),
    }
}

fn render_fnid(ui: &mut egui::Ui, fnid: FnId) {
    ui.label(format!("#{}", fnid.0));
}

fn render_iid(ui: &mut egui::Ui, dest: IID) {
    ui.button(format!("i{}", dest.0));
}

fn render_vreg_write(ui: &mut egui::Ui, dest: bytecode::VReg) {
    ui.button(format!(">v{}", dest.0));
}

fn render_vreg_read(ui: &mut egui::Ui, dest: bytecode::VReg) {
    ui.button(format!("<v{}", dest.0));
}

impl eframe::App for InspectorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match &mut self.state {
            VMState::Running { vm_thread } => {
                if vm_thread.as_mut().unwrap().is_finished() {
                    let vm_result = vm_thread.take().unwrap().join().unwrap();
                    self.state = VMState::Suspended { vm_result };
                }
            }
            VMState::Initial | VMState::Suspended { .. } => {}
        };

        egui::CentralPanel::default().show(ctx, |ui| match &self.state {
            VMState::Initial => {
                if self.ui_initial(ui) {
                    let vm_thread = self.release_the_kraken();
                    self.state = VMState::Running {
                        vm_thread: Some(vm_thread),
                    };
                }
            }
            VMState::Running { .. } => {
                self.ui_running(ui, ctx);
            }
            VMState::Suspended { vm_result } => {
                self.ui_suspended(ui, ctx, vm_result);
            }
        });
    }
}

struct RunParams {
    breakpoints: Vec<Breakpoint>,
    codebase: Arc<mcjs_vm::Codebase>,
    root_mod_id: mcjs_vm::ModuleId,
}
struct VMResult {
    instr_history: Vec<GlobalIID>,
    error_messages: Vec<String>,
    core_dump: Option<CoreDump>,
    stack_view: Vec<StackViewItem>,
    run_params: RunParams,
}

#[derive(Clone)]
struct Breakpoint(GlobalIID);

fn run_vm(run_params: RunParams) -> VMResult {
    let mut instr_history = Vec::new();
    let mut on_step = |step: &mcjs_vm::InspectorStep| {
        instr_history.push(step.giid);
        if run_params.breakpoints.iter().any(|bp| bp.0 == step.giid) {
            InspectorAction::Fail
        } else {
            InspectorAction::Continue
        }
    };

    let vm = mcjs_vm::Interpreter::new(&run_params.codebase).with_step_handler(&mut on_step);
    let result = vm.run_module(run_params.root_mod_id);

    match result {
        Ok(_) => VMResult {
            instr_history: Vec::new(),
            error_messages: Vec::new(),
            core_dump: None,
            stack_view: Vec::new(),
            run_params,
        },

        Err(intrp_err) => {
            // mcjs_vm::common::Error is not Send due to SourceMap being a forest of Rc<_>.
            // To avoid the problem, we convert the Error chain to another "flatter" type
            // that does not include the SourceMap.  We use the SoureMap during this
            // conversion to retain as much useful info as we can.
            let stack_view = if let Some(core_dump) = &intrp_err.core_dump {
                make_stack_view(&core_dump.data)
            } else {
                Vec::new()
            };
            VMResult {
                instr_history,
                error_messages: intrp_err.error.messages().collect(),
                core_dump: intrp_err.core_dump,
                stack_view,
                run_params,
            }
        }
    }
}
