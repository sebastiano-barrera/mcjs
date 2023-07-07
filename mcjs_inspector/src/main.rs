use std::{
    cell::Cell,
    collections::HashMap,
    marker::PhantomData,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

use actix_web::{http::header::ContentType, web, App, HttpResponse, Responder};
use anyhow::{Context, Result};
use maud::{html, Markup};
use serde::Deserialize;

use mcjs_vm::{bytecode, inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let case = open_case(&case_file_path).expect("could not open case file");
    let case_data = CaseData::load(case);
    let state = Arc::new(Mutex::new(RootState::new(case_data)));

    use actix_web::middleware::Logger;
    use actix_web::HttpServer;

    // TODO Automatic template reloading?

    HttpServer::new(move || {
        let serve_dir = actix_files::Files::new("/assets", "data/assets");
        App::new()
            .service(handle_get_core_dump)
            .service(handle_start)
            .service(handle_root_page)
            .service(serve_dir)
            .app_data(web::Data::new(Arc::clone(&state)))
            .wrap(Logger::default())
    })
    .bind(("127.0.0.1", 10001))?
    .run()
    .await
}

type StateHandle = Arc<Mutex<RootState>>;

#[actix_web::get("/")]
async fn handle_root_page(state: web::Data<StateHandle>) -> impl Responder {
    use mcjs_vm::inspector_case::Root;

    let state = state.lock().unwrap();
    let body = html! {
        div class="ring-1 ring-slate-600 bg-slate-700 max-w-lg m-auto p-4 rounded-2xl shadow-lg shadow-slate-700 mt-4 space-y-4" {
            p { "mcjs Inspector ready to go." }

            div {
                h1.text-lg.font-bold { "The case:" }
                p { "Include paths:" }
                ul.list-disc.text-sm."px-4" {
                    // {% for path in case.include_paths %}
                    // <li>{{ path }}</li>
                    // {% endfor %}
                }

                @match &state.case.case.root {
                    Root::ModuleImport(mod_path) => {
                        p {
                            "Start by importing module:"
                            code { (mod_path) }
                        }
                    }
                    other => {
                        p { "Execution root:" ( format!("{:?}", other) ) }
                    }
                }
            }

            div {
                form method="POST" action="/start" {
                    input type="submit"
                          class="block bg-slate-500 rounded-full px-4 py-2 w-1/2 m-auto text-center cursor-pointer hover:bg-slate-600"
                          value="Start";
                }
            }
        }
    };

    HttpResponse::Ok()
        .content_type(ContentType::html())
        .body(page(body).into_string())
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

    struct HistoryItemView {
        left_padding: String,
        instr: String,
    }
    let instr_history = vm_result.instr_history.iter().map(|item| {
        let EternalIID(_, GlobalIID(fnid, iid)) = item.eiid;
        let func = state.codebase.get_function(fnid).unwrap();
        let instr = &func.instrs()[iid.0 as usize];
        HistoryItemView {
            left_padding: format!("{}cm", item.stack_depth),
            instr: format!("{:?}", instr),
        }
    });

    if let Some(core_dump) = &vm_result.core_dump {}

    let body = html! {
        div.grid."grid-cols-2"."inset-0".w-full.h-full {
            div.overflow-y-scroll."inset-0" {
                table.w-full {
                    @for item in instr_history {
                        tr.border-solid."border-b-2"."border-b-slate-700"."hover:bg-slate-800" {
                            td { "-" }
                            td style={"padding-left: " (item.left_padding) } { (item.instr) }
                        }
                    }
                }
            }

            div."p-4" {
                "(Coming later: stack details.)"
            }
        }
    };

    Ok(HttpResponse::Ok()
        .content_type(ContentType::html())
        .body(page(body).into_string()))
}

fn page(body: Markup) -> Markup {
    html! {
        (maud::DOCTYPE)
        html.w-full.h-full {
            head {
                title { "mcjs Inspector" }
                meta charset="UTF-8";
                meta name="viewport" content="width=device-width, initial-scale=1.0";
                script src="/assets/tailwind-cdn.js" {}
            }

            body."bg-slate-900".text-white.w-full.h-full {
                (body)
            }
        }
    }
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
