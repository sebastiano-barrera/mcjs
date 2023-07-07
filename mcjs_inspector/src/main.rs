use std::{
    cell::Cell,
    collections::HashMap,
    marker::PhantomData,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock},
};

use anyhow::{Context, Result};

use mcjs_vm::{bytecode, inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};
use serde::{ser::SerializeTuple, Deserialize, Serialize};
use tracing::{event, Level};

use axum::{
    extract::{Path, State},
    http::{Request, StatusCode},
    response::{Html, IntoResponse},
};
use tower_http::services::ServeDir;
use tera::Tera;

#[tokio::main]
async fn main() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .init();

    let tmpl = Tera::new("data/templates/**/*").unwrap();
    TEMPLATES.set(RwLock::new(tmpl)).unwrap();

    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let case = open_case(&case_file_path).expect("could not open case file");
    let case_data = CaseData::load(case);
    let state = Arc::new(Mutex::new(RootState::new(case_data)));

    use axum::{
        routing::{get, post},
        Router,
    };

    let serve_dir = ServeDir::new("data/assets");

    let app = Router::new()
        .route("/", get(handle_root_page))
        .route("/start", post(handle_start))
        .route("/sessions/:sid/core_dump", get(handle_get_core_dump))
        .nest_service("/assets/", serve_dir)
        .route_layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            reload_templates,
        ))
        .with_state(state);

    let addr: &SocketAddr = &"127.0.0.1:10001".parse().unwrap();
    eprintln!(" -- Server listening on: http://{}/", addr.to_string());
    axum::Server::bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

type StateHandle = Arc<Mutex<RootState>>;
type Response = Result<Html<String>, AppError>;

#[axum::debug_handler]
async fn handle_root_page(state: State<StateHandle>) -> Response {
    let state = state.lock().unwrap();
    let mut context = tera::Context::new();
    context.insert("case", &state.case.case);
    render_template("index.html", &context)
}

#[axum::debug_handler]
async fn handle_start(state: State<StateHandle>) -> impl IntoResponse {
    let state = Arc::clone(&state);
    let sid = tokio::task::spawn_blocking(move || {
        let state = state.lock().unwrap();
        let sid = state.start_new_session();
        sid
    })
    .await
    .unwrap();

    axum::response::Redirect::to(&format!("/sessions/{}/core_dump", sid.0))
}

#[axum::debug_handler]
async fn handle_get_core_dump(state: State<StateHandle>, Path(sid): Path<SessionID>) -> Response {
    let state = state.lock().unwrap();

    let sessions = state.sessions();
    let vm_result = sessions.get(sid)?;

    #[derive(Serialize)]
    struct HistoryItemView {
        stack_depth: usize,
        instr: String,
    }
    let instr_history: Vec<_> = vm_result
        .instr_history
        .iter()
        .map(|item| {
            let EternalIID(_, GlobalIID(fnid, iid)) = item.eiid;
            let func = state.codebase.get_function(fnid).unwrap();
            let instr = &func.instrs()[iid.0 as usize];
            HistoryItemView {
                stack_depth: item.stack_depth,
                instr: format!("{:?}", instr),
            }
        })
        .collect();

    if let Some(core_dump) = &vm_result.core_dump {
        core_dump.data
    }

    let mut context = tera::Context::new();
    context.insert("instr_history", &instr_history);
    render_template("core_dump.html", &context)
}

async fn reload_templates<B>(
    req: Request<B>,
    next: axum::middleware::Next<B>,
) -> impl IntoResponse {
    {
        let mut tmpl = TEMPLATES.get().unwrap().write().unwrap();
        if let Err(err) = tmpl.full_reload() {
            event!(Level::ERROR, "Could not reload templates: {:?}", err);
        }
    }

    next.run(req).await
}

#[derive(Debug)]
struct AppError {
    status_code: axum::http::StatusCode,
    message: String,
}

impl AppError {
    fn new(message: String) -> Self {
        let status_code = axum::http::StatusCode::INTERNAL_SERVER_ERROR;
        AppError {
            message,
            status_code,
        }
    }
    fn with_status_code(mut self, status_code: StatusCode) -> Self {
        self.status_code = status_code;
        self
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let html = {
            let mut context = tera::Context::new();
            context.insert("message", &self.message);
            render_template("error.html", &context).expect("could not render error template")
        };

        (self.status_code, html).into_response()
    }
}

static TEMPLATES: OnceLock<RwLock<Tera>> = OnceLock::new();
fn render_template(template_name: &str, context: &tera::Context) -> Result<Html<String>, AppError> {
    let html = TEMPLATES
        .get()
        .unwrap()
        .read()
        .unwrap()
        .render(template_name, &context)
        .map_err(|err| AppError::new(format!("error while rendering template: {:?}", err)))?;
    let html = Html(html);
    Ok(html)
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
    codebase: Arc<mcjs_vm::Codebase>,
    selection: Cell<Selection>,
    root_mod_id: mcjs_vm::ModuleId,

    session_mgr: Mutex<SessionMgr>,
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

    fn get(&self, sid: SessionID) -> Result<&VMResult, AppError> {
        self.sessions.get(&sid).ok_or_else(|| {
            AppError::new(format!("no such session with ID: {}", sid.0))
                .with_status_code(axum::http::StatusCode::NOT_FOUND)
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
    error_messages: Vec<String>,
    core_dump: Option<CoreDump>,
    run_params: RunParams,
}

#[derive(Serialize)]
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

impl Serialize for EternalIID {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let EternalIID(CallID(call_id), GlobalIID(FnId(fn_id), IID(iid))) = self;
        let mut tuple = serializer.serialize_tuple(3)?;
        tuple.serialize_element(call_id)?;
        tuple.serialize_element(fn_id)?;
        tuple.serialize_element(iid)?;
        tuple.end()
    }
}

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
