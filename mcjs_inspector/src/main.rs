use std::{
    cell::Cell,
    collections::HashMap,
    marker::PhantomData,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

use actix_web::{http::header::ContentType, web, App, HttpResponse, Responder};
use anyhow::{Context, Result};
use askama::Template;
use serde::Deserialize;

use mcjs_vm::{bytecode, inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    use listenfd::ListenFd;

    env_logger::init();

    let mut listenfd = ListenFd::from_env();
    let listener = if let Some(listener) = listenfd.take_tcp_listener(0)? {
        listener
    } else {
        std::net::TcpListener::bind(("127.0.0.1", 10001)).unwrap()
    };

    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let case = open_case(&case_file_path).expect("could not open case file");
    let case_data = CaseData::load(case);
    let state = Arc::new(Mutex::new(RootState::new(case_data)));

    use actix_web::middleware::Logger;
    use actix_web::HttpServer;

    // TODO Remove once this inspector is 'ready'
    if true {
        // Start the session once, so we don't have to click on "start" every time we
        // recompile/restart the program
        let state = Arc::clone(&state);
        let sid = tokio::task::spawn_blocking(move || {
            let state = state.lock().unwrap();
            let sid = state.start_new_session();
            println!();
            sid
        })
        .await
        .unwrap();
        println!(
            " -- initial session created: http://127.0.0.1:10001/sessions/{}/core_dump",
            sid.0
        );
    }

    HttpServer::new(move || {
        let serve_dir = actix_files::Files::new("/assets", "data/assets");
        App::new()
            .service(handle_get_core_dump)
            .service(handle_patch_core_dump)
            .service(handle_start)
            .service(handle_root_page)
            .service(serve_dir)
            .app_data(web::Data::new(Arc::clone(&state)))
            .wrap(Logger::default())
    })
    .listen(listener)
    .unwrap()
    .run()
    .await
}

type StateHandle = Arc<Mutex<RootState>>;

#[actix_web::get("/")]
async fn handle_root_page(state: web::Data<StateHandle>) -> impl Responder {
    #[derive(Template)]
    #[template(path = "index.html")]
    struct RootTemplate<'a> {
        case: &'a mcjs_vm::inspector_case::Case,
    }

    let state = state.lock().unwrap();
    let case = &state.case.case;
    HttpResponse::Ok()
        .content_type(ContentType::html())
        .body(RootTemplate { case }.render().unwrap())
}

#[actix_web::post("/start")]
async fn handle_start(state: web::Data<StateHandle>) -> impl Responder {
    let state = Arc::clone(&state);
    let sid = tokio::task::spawn_blocking(move || {
        let state = state.lock().unwrap();
        let sid = state.start_new_session();
        sid
    })
    .await
    .unwrap();

    web::Redirect::to(format!("/sessions/{}/core_dump", sid.0)).see_other()
}

#[actix_web::get("/sessions/{sid}/core_dump")]
async fn handle_get_core_dump(
    state: web::Data<StateHandle>,
    path: web::Path<SessionID>,
) -> actix_web::Result<HttpResponse> {
    let sid = path.into_inner();

    let state = state.lock().unwrap();
    let sessions = state.sessions();
    let vm_result = sessions.get(sid)?;
    let case = &state.case.case;

    struct HistoryItemView {
        left_padding: String,
        call_id: CallID,
        parts: Vec<InstrPartView>,
    }
    enum InstrPartView {
        Opcode(&'static str),
        Read(bytecode::VReg),
        Write(bytecode::VReg),
        Other(String),
    }
    struct InstrPartCollector(Vec<InstrPartView>);
    impl bytecode::InstrAnalyzer for InstrPartCollector {
        fn start(&mut self, opcode_name: &'static str) {
            self.0.push(InstrPartView::Opcode(opcode_name))
        }

        fn read_vreg(&mut self, vreg: bytecode::VReg) {
            self.0.push(InstrPartView::Read(vreg))
        }

        fn write_vreg(&mut self, vreg: bytecode::VReg) {
            self.0.push(InstrPartView::Write(vreg))
        }

        fn jump_target(&mut self, iid: IID) {
            self.0.push(InstrPartView::Other(format!("j{}", iid.0)))
        }

        fn load_const(&mut self, item: bytecode::ConstIndex) {
            self.0.push(InstrPartView::Other(format!("{:?}", item)))
        }

        fn load_null(&mut self) {
            self.0.push(InstrPartView::Other("null".to_owned()))
        }

        fn load_undefined(&mut self) {
            self.0.push(InstrPartView::Other("undefined".to_owned()))
        }

        fn load_capture(&mut self, item: bytecode::CaptureIndex) {
            self.0.push(InstrPartView::Other(format!("{:?}", item)))
        }

        fn load_arg(&mut self, item: bytecode::ArgIndex) {
            self.0.push(InstrPartView::Other(format!("{:?}", item)))
        }

        fn load_this(&mut self) {
            self.0.push(InstrPartView::Other("this".to_owned()))
        }

        fn end(&mut self, _instr: &bytecode::Instr) {}
    }

    let instr_history = vm_result
        .instr_history
        .iter()
        .map(|item| {
            let EternalIID(call_id, GlobalIID(fnid, iid)) = item.eiid;
            let func = state.codebase.get_function(fnid).unwrap();
            let instr = &func.instrs()[iid.0 as usize];

            let mut collector = InstrPartCollector(Vec::new());
            instr.analyze(&mut collector);

            HistoryItemView {
                left_padding: format!("{}cm", item.stack_depth),
                call_id,
                parts: collector.0,
            }
        })
        .collect();

    #[derive(Template)]
    #[template(path = "core_dump.html")]
    struct CoreDumpTemplate<'a> {
        case: &'a mcjs_vm::inspector_case::Case,
        vm_result: &'a VMResult,
        instr_history: Vec<HistoryItemView>,
        watches: &'a Vec<Watch>,
    }

    let html = CoreDumpTemplate {
        case,
        vm_result,
        instr_history,
        watches: &state.watches,
    }
    .render()
    .unwrap();

    Ok(HttpResponse::Ok()
        .content_type(ContentType::html())
        .body(html))
}

#[derive(Deserialize, Debug)]
struct AddVRegWatchPayload {
    call_id: CallID,
    vreg: u8,
}

#[actix_web::post("/sessions/{sid}/core_dump/add-watch")]
async fn handle_patch_core_dump(
    state: web::Data<StateHandle>,
    path: web::Path<SessionID>,
    form: web::Form<AddVRegWatchPayload>,
) -> actix_web::Result<HttpResponse> {
    let sid = path.into_inner();
    let patch = form.into_inner();

    {
        let mut state = state.lock().unwrap();
        let evreg = EternalVReg(patch.call_id, bytecode::VReg(patch.vreg));
        state.watches.push(Watch::VReg(evreg));
    }

    Ok(HttpResponse::SeeOther()
        .append_header(("Location", format!("/sessions/{}/core_dump", sid.0)))
        .finish())
}

fn open_case(case_file_path: &std::path::Path) -> Result<inspector_case::Case> {
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

struct RootState {
    case: CaseData,

    // TODO Move to a different state struct?
    breakpoints: Vec<Breakpoint>,
    watches: Vec<Watch>,
    codebase: Arc<mcjs_vm::Codebase>,
    selection: Cell<Selection>,
    root_mod_id: mcjs_vm::ModuleId,

    session_mgr: Mutex<SessionMgr>,
}

enum Watch {
    VReg(EternalVReg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
struct SessionID(u32);

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
            codebase,
            selection: Cell::new(Selection::None),
            watches: Vec::new(),
            breakpoints: vec![Breakpoint(GlobalIID(FnId(70), IID(6)))],
            case,
            root_mod_id,
            session_mgr: Mutex::new(SessionMgr::new()),
        }
    }

    fn start_new_session(&self) -> SessionID {
        let run_params = RunParams {
            breakpoints: self.breakpoints.clone(),
            codebase: Arc::clone(&self.codebase),
            root_mod_id: self.root_mod_id,
        };
        let vm_result = run_vm(run_params);
        self.session_mgr.lock().unwrap().put(vm_result)
    }

    fn sessions(&self) -> MutexGuard<SessionMgr> {
        self.session_mgr.lock().unwrap()
    }
}

struct SessionMgr {
    last_session_id: u32,
    sessions: HashMap<SessionID, VMResult>,
}

impl SessionMgr {
    fn new() -> Self {
        SessionMgr {
            last_session_id: 0,
            sessions: HashMap::new(),
        }
    }

    fn put(&mut self, vm_result: VMResult) -> SessionID {
        self.last_session_id += 1;
        let sid = SessionID(self.last_session_id);
        self.sessions.insert(sid, vm_result);
        sid
    }

    fn get(&self, sid: SessionID) -> actix_web::Result<&VMResult> {
        self.sessions.get(&sid).ok_or_else(|| {
            let err: Box<dyn std::error::Error + 'static> =
                anyhow::anyhow!("no such session with ID: {}", sid.0).into();
            err.into()
        })
    }
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
    call_id_stack: Vec<CallID>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
struct CallID(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EternalVReg(CallID, bytecode::VReg);

// Helper to deserialize EternalVReg
#[derive(Debug, Deserialize)]
struct EternalVRegMsg {
    call_id: CallID,
    vreg: u8,
}

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
            call_id_stack: Vec::new(),
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
                call_id_stack,
                error_messages: intrp_err.error.messages().collect(),
                core_dump: intrp_err.core_dump,
                run_params,
            }
        }
    }
}
