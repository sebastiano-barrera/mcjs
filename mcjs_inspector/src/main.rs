use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    thread::JoinHandle,
    time::Duration,
};

use anyhow::{Context, Result};
use mcjs_vm::{bytecode, inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};

mod ui {
    slint::include_modules!();
}

fn open_case(case_file_path: &Path) -> Result<inspector_case::Case> {
    let mut f = std::fs::File::open(&case_file_path)
        .with_context(|| format!("opening case file: {}", case_file_path.display()))?;
    let case: inspector_case::Case = rmp_serde::from_read(&mut f)
        .with_context(|| format!("decoding case file: {}", case_file_path.display()))?;
    Ok(case)
}

struct CaseData {
    codebase: Arc<mcjs_vm::Codebase>,
    case: inspector_case::Case,
    root_mod_id: mcjs_vm::ModuleId,
}

impl CaseData {
    fn load(case: inspector_case::Case) -> Self {
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

        CaseData {
            codebase,
            case,
            root_mod_id,
        }
    }
}

fn main() {
    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let case = open_case(&case_file_path).expect("could not open case file");
    let case_data = CaseData::load(case);

    use slint::ComponentHandle;

    let mut root_state = RefCell::new(RootState::new(case_data));

    let mut check_timer = slint::Timer::default();
    let mut hello_world = ui::HelloWorld::new().unwrap();
    show_state(&root_state.borrow(), &mut hello_world);

    {
        let hw = hello_world.as_weak();
        hello_world.on_start(|| {
            let mut root_state_ref = root_state.borrow_mut();
            if !matches!(root_state_ref.state, VMState::Initial) {
                return;
            }

            root_state_ref.start();
            check_timer.start(
                slint::TimerMode::Repeated,
                Duration::from_millis(100),
                || {
                    let mut root_state_ref = root_state.borrow_mut();
                    let vm_thread = if let VMState::Running{vm_thread} = root_state_ref.state {
                        Some(vm_thread)
                    } else {
                        None
                    };
                },
            );
        });
    }

    hello_world.on_quit(|| slint::quit_event_loop().unwrap());
    hello_world.run().unwrap();
}

const SELECTION_MARK: &'static str = "✱";

struct RootState {
    state: VMState,
    case: CaseData,

    // TODO Move to a different state struct?
    breakpoints: Vec<Breakpoint>,
    codebase: Arc<mcjs_vm::Codebase>,
    selection: Cell<Selection>,
    root_mod_id: mcjs_vm::ModuleId,
}

impl RootState {
    fn new(case: CaseData) -> Self {
        let include_paths = case.case.include_paths.iter().map(|pb| pb.as_path());
        let builder_params = mcjs_vm::BuilderParams {
            loader: Box::new(mcjs_vm::FileLoader::new(include_paths)),
        };
        let mut builder = builder_params.to_builder();

        let root_mod_id = match &case.case.root {
            inspector_case::Root::ModuleImport(path) => builder
                .compile_file(path.clone())
                .unwrap_or_else(|err| panic!("compile error: {:?}", err)),
            inspector_case::Root::InlineScript(_) => {
                todo!("sorry, Root::InlineScript is not supported yet")
            }
        };
        let codebase = Arc::new(builder.build());

        RootState {
            state: VMState::Initial,
            codebase,
            selection: Cell::new(Selection::None),
            breakpoints: vec![Breakpoint(GlobalIID(FnId(70), IID(6)))],
            case,
            root_mod_id,
        }
    }

    fn start(&mut self) {
        if !matches!(self.state, VMState::Initial) {
            return;
        }

        let run_params = RunParams {
            breakpoints: self.breakpoints.clone(),
            codebase: Arc::clone(&self.codebase),
            root_mod_id: self.root_mod_id,
        };
        let vm_thread = std::thread::spawn(move || run_vm(run_params));
        self.state = VMState::Running { vm_thread };
    }
}

enum VMState {
    Initial,
    Running { vm_thread: JoinHandle<VMResult> },
    Suspended { vm_result: VMResult },
}

fn show_state(state: &RootState, hw: &mut ui::HelloWorld) {
    let ui_state = match state.state {
        VMState::Initial => ui::VMState::Initial,
        VMState::Running { .. } => ui::VMState::Running,
        VMState::Suspended { .. } => ui::VMState::Suspended,
    };

    hw.set_vm_state(ui_state);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    None,
    VReg(EternalVReg),
    IID(EternalIID),
}

struct CaseDetailsModel<'a>(PhantomData<&'a CaseData>);

struct RunParams {
    breakpoints: Vec<Breakpoint>,
    codebase: Arc<mcjs_vm::Codebase>,
    root_mod_id: mcjs_vm::ModuleId,
}
struct VMResult {
    instr_history: Vec<HistoryItem>,
    error_messages: Vec<String>,
    core_dump: Option<CoreDump>,
    run_params: RunParams,
}

struct HistoryItem {
    stack_depth: usize,
    eiid: EternalIID,
}

struct StackFrameView {
    call_id: CallID,
    header: mcjs_vm::stack_access::FrameHeader,
    items: Vec<StackSlotView>,
}
struct StackSlotView {
    label_opt: Option<&'static str>,
    val_ndx: String,
    val_str: String,
    slot_selection: Option<Selection>,
}

#[derive(Clone)]
struct Breakpoint(GlobalIID);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CallID(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EternalVReg(CallID, bytecode::VReg);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EternalIID(CallID, GlobalIID);

fn run_vm(run_params: RunParams) -> VMResult {
    let mut instr_history = Vec::new();

    let mut call_id_stack = Vec::new();
    let mut call_id = CallID(0);
    let mut next_call_id = move || {
        call_id.0 += 1;
        call_id
    };

    let mut on_step = |step: &mcjs_vm::InspectorStep| {
        let stack_depth = step.intrp_data.len();
        while call_id_stack.len() < stack_depth {
            call_id_stack.push(next_call_id());
        }
        call_id_stack.truncate(stack_depth);

        assert_eq!(call_id_stack.len(), stack_depth);

        let call_id = *call_id_stack.last().unwrap();
        instr_history.push(HistoryItem {
            eiid: EternalIID(call_id, step.giid),
            stack_depth,
        });

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
            run_params,
        },

        Err(intrp_err) => {
            // mcjs_vm::common::Error is not Send due to SourceMap being a forest of Rc<_>.
            // To avoid the problem, we convert the Error chain to another "flatter" type
            // that does not include the SourceMap.  We use the SoureMap during this
            // conversion to retain as much useful info as we can.
            VMResult {
                instr_history,
                error_messages: intrp_err.error.messages().collect(),
                core_dump: intrp_err.core_dump,
                run_params,
            }
        }
    }
}
