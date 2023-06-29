use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};

use mcjs_vm::{inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};

fn cli_main() {
    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let case = open_case(&case_file_path).expect("could not open case file");
    eprintln!("Case file content = {:#?}", case);

    let mut builder = mcjs_vm::BuilderParams {
        loader: Box::new(mcjs_vm::FileLoader::new(
            case.include_paths.iter().map(|pb| pb.as_path()),
        )),
    }
    .to_builder();
    let test_mod_id = match case.root {
        inspector_case::Root::ModuleImport(path) => builder
            .compile_file(path)
            .unwrap_or_else(|err| panic!("compile error: {:?}", err)),
        inspector_case::Root::InlineScript(_) => {
            todo!("sorry, Root::InlineScript is not supported yet")
        }
    };
    let codebase = builder.build();

    // -------------------

    let breakpoint = Breakpoint(GlobalIID(FnId(70), IID(6)));
    let mut code_path = Vec::new();
    let mut on_step = |step: &mcjs_vm::InspectorStep| {
        code_path.push(step.giid);
        if step.giid == breakpoint.0 {
            InspectorAction::Fail
        } else {
            InspectorAction::Continue
        }
    };

    let vm = mcjs_vm::Interpreter::new(&codebase).with_step_handler(&mut on_step);
    let result = vm.run_module(test_mod_id);

    match result {
        Ok(output) => {
            println!("interpreter finished OK.");
            println!("sink = {:?}", output.sink);
        }
        Err(error) => {
            eprintln!("code path:");
            for giid in code_path {
                eprintln!(" - {:?}", giid);
            }
            eprintln!("interpreter error: {:?}", error.error);

            if let Some(core_dump) = error.core_dump {
                let header = core_dump.data.header();
                eprintln!("top frame header: {:?}", header);

                eprintln!("results ({}):", header.n_instrs);
                for i in 0u32..header.n_instrs {
                    let vreg = mcjs_vm::bytecode::VReg(i.try_into().unwrap());
                    eprintln!("  [{}] = {:?}", i, core_dump.data.get_result(vreg));
                }

                eprintln!("args ({}):", header.n_args);
                for i in 0..header.n_args {
                    let argndx = mcjs_vm::bytecode::ArgIndex(i);
                    eprintln!("  [{}] = {:?}", i, core_dump.data.get_arg(argndx));
                }

                eprintln!("captures ({}):", header.n_captures);
                for i in 0..header.n_captures {
                    let capndx = mcjs_vm::bytecode::CaptureIndex(i);
                    eprintln!("  [{}] = {:?}", i, core_dump.data.get_capture(capndx));
                }
            } else {
                eprintln!("-- no interpreter core dumped");
            }
        }
    }
}

fn open_case(case_file_path: &Path) -> Result<inspector_case::Case> {
    let mut f = std::fs::File::open(&case_file_path)
        .with_context(|| format!("opening case file: {}", case_file_path.display()))?;
    let case: inspector_case::Case = rmp_serde::from_read(&mut f)
        .with_context(|| format!("decoding case file: {}", case_file_path.display()))?;
    Ok(case)
}

use eframe::egui;

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

struct RunParams {
    breakpoints: Vec<Breakpoint>,
    codebase: Arc<mcjs_vm::Codebase>,
    root_mod_id: mcjs_vm::ModuleId,
}
struct VMResult {
    error_messages: Vec<String>,
    core_dump: Option<CoreDump>,
    run_params: RunParams,
}

#[derive(Clone)]
struct Breakpoint(GlobalIID);

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

    fn release_the_kraken(&mut self) -> VMState {
        let run_params = RunParams {
            breakpoints: self.breakpoints.clone(),
            codebase: Arc::clone(&self.codebase),
            root_mod_id: self.root_mod_id,
        };
        let vm_thread = std::thread::spawn(move || {
            let mut on_step = |step: &mcjs_vm::InspectorStep| {
                if run_params.breakpoints.iter().any(|bp| bp.0 == step.giid) {
                    InspectorAction::Fail
                } else {
                    InspectorAction::Continue
                }
            };

            let vm =
                mcjs_vm::Interpreter::new(&run_params.codebase).with_step_handler(&mut on_step);
            let result = vm.run_module(run_params.root_mod_id);
            match result {
                Ok(_) => VMResult {
                    error_messages: Vec::new(),
                    core_dump: None,
                    run_params,
                },

                Err(intrp_err) => {
                    // mcjs_vm::common::Error is not Send due to SourceMap being a forest of Rc<_>.
                    // To avoid the problem, we convert the Error chain to another "flatter" type
                    // that does not include the SourceMap.  We use the SoureMap during this
                    // conversion to retain as much useful info as we can.
                    VMResult {
                        error_messages: intrp_err.error.messages().collect(),
                        core_dump: intrp_err.core_dump,
                        run_params,
                    }
                }
            }
        });

        VMState::Running {
            vm_thread: Some(vm_thread),
        }
    }

    fn ui_running(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Running...");
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    fn ui_suspended(&self, ui: &mut egui::Ui, vm_result: &VMResult) {
        match &vm_result.core_dump {
            Some(core_dump) => {
                ui.heading("VM interrupted");
                ui.label("Core dump received.");

                ui.label(format!("{} stack frames.", core_dump.data.len()));
            }
            None => {
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
                    self.state = self.release_the_kraken();
                }
            }
            VMState::Running { .. } => {
                self.ui_running(ui, ctx);
            }
            VMState::Suspended { vm_result } => {
                self.ui_suspended(ui, &vm_result);
            }
        });
    }
}
