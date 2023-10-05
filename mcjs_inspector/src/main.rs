use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

use actix_web::{http::header::ContentType, web, App, HttpResponse, Responder};
use anyhow::{Context, Result};
use askama::Template;
use serde::Deserialize;

use mcjs_vm::{bytecode, inspector_case, CoreDump, FnId, GlobalIID, InspectorAction, IID};
use mcjs_vm::{interpreter, ModuleId, SourceMap};

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
        let state = state.lock().unwrap();
        let sid = state.start_new_session();
        println!();
        println!(
            " -- initial session created: http://127.0.0.1:10001/sessions/{}/core_dump",
            sid.0
        );
    }

    HttpServer::new(move || {
        let serve_dir = actix_files::Files::new("/assets", "data/assets");
        App::new()
            .service(handle_get_core_dump)
            .service(handle_add_vreg_watch)
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

    use view_model::InstrPartView;

    #[derive(Template)]
    #[template(path = "core_dump.html")]
    struct CoreDumpTemplate<'a> {
        vm_result: &'a VMResult,
        stack_srcs: Vec<&'a str>,
        functions: HashMap<FnId, view_model::FunctionView>,
        call_instr_ndxs: Vec<Option<usize>>,
        watches: &'a Vec<Watch>,
        watch_values: Vec<view_model::ValueView>,
    }
    impl<'a> CoreDumpTemplate<'a> {
        // Workaround: the template in askama does not support the `&` operator, so I can't do
        // functions[&hdr.fn_id].   This function works around it by taking a value, not a
        // reference.
        pub(crate) fn get_function(&self, fn_id: &FnId) -> &view_model::FunctionView {
            self.functions.get(fn_id).unwrap()
        }
    }

    let mut functions = HashMap::new();
    let mut call_instr_ndxs = Vec::new();
    let mut stack_srcs = Vec::new();
    if let Some(core_dump) = vm_result.core_dump.as_ref() {
        let frames = core_dump.data.frames();
        let codebase = &vm_result.run_params.codebase;
        for frame in frames.iter() {
            let fn_id = frame.header().fn_id;

            functions.entry(fn_id).or_insert_with(|| {
                let func = codebase.get_function(fn_id).unwrap();
                view_model::translate_function(func)
            });
        }

        for frame in frames.iter().rev().skip(1) {
            call_instr_ndxs.push(Some(frame.header().return_to_iid.unwrap().0 as usize));
        }
        call_instr_ndxs.push(None);
        assert_eq!(call_instr_ndxs.len(), frames.len());

        for frame in frames.iter().rev() {
            let fn_id = frame.header().fn_id;
            let src = state.case.source_of_fn.get(&fn_id).map(|s| s.as_str()).unwrap_or("???");
            stack_srcs.push(src);
        }
    }

    let html = CoreDumpTemplate {
        vm_result,
        stack_srcs,
        functions,
        call_instr_ndxs,
        watches: &state.watches,
        watch_values: view_model::translate_watch_values(&vm_result, &state.watches),
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

#[actix_web::post("/sessions/{sid}/core_dump/add-vreg-watch")]
async fn handle_add_vreg_watch(
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

mod view_model {
    use std::collections::HashMap;

    use crate::EternalIID;
    use crate::VMResult;

    use super::CallID;
    use super::EternalVReg;
    use mcjs_vm::bytecode;
    use mcjs_vm::GlobalIID;
    use mcjs_vm::IID;

    pub(crate) struct FunctionView {
        pub(crate) instrs: Vec<InstrView>,
    }

    pub(crate) struct InstrView {
        pub(crate) parts: Vec<InstrPartView>,
    }

    pub(crate) enum InstrPartView {
        Opcode(&'static str),
        Read(bytecode::VReg, Option<String>),
        Write(bytecode::VReg, Option<String>),
        Capture(usize),
        Arg(usize),
        Other(String),
    }

    pub(crate) fn translate_function(func: &bytecode::Function) -> FunctionView {
        struct InstrPartCollector(Vec<InstrPartView>);

        impl bytecode::InstrAnalyzer for InstrPartCollector {
            fn start(&mut self, opcode_name: &'static str) {
                self.0.push(InstrPartView::Opcode(opcode_name))
            }
            fn read_vreg_labeled(&mut self, vreg: bytecode::VReg, description: Option<String>) {
                self.0.push(InstrPartView::Read(vreg, description))
            }
            fn write_vreg_labeled(&mut self, vreg: bytecode::VReg, description: Option<String>) {
                self.0.push(InstrPartView::Write(vreg, description))
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
                self.0.push(InstrPartView::Capture(item.0 as usize))
            }
            fn load_arg(&mut self, item: bytecode::ArgIndex) {
                self.0.push(InstrPartView::Arg(item.0 as usize))
            }
            fn load_this(&mut self) {
                self.0.push(InstrPartView::Other("this".to_owned()))
            }
            fn end(&mut self, _instr: &bytecode::Instr) {}
        }

        let instrs_view = func
            .instrs()
            .iter()
            .map(|instr| {
                let mut collector = InstrPartCollector(Vec::new());
                instr.analyze(&mut collector);
                let parts = collector.0;

                InstrView { parts }
            })
            .collect();

        FunctionView {
            instrs: instrs_view,
        }
    }

    #[derive(Debug)]
    pub(crate) struct ObjectView {
        props: HashMap<String, Box<ValueView>>,
        proto: Option<Box<ObjectView>>,
    }

    #[derive(Debug)]
    pub(crate) enum ValueView {
        Literal(bytecode::Literal),
        Object(ObjectView),
        String(String),
        Closure(bytecode::FnId),
        Past,
    }

    pub(crate) fn translate_watch_values(
        vm_result: &VMResult,
        watches: &[super::Watch],
    ) -> Vec<ValueView> {
        let mut value_views = Vec::new();

        let core_dump = vm_result.core_dump.as_ref().unwrap();
        let stack_frames = core_dump.data.frames();
        for watch in watches {
            match watch {
                crate::Watch::VReg(EternalVReg(call_id, vreg)) => {
                    let frame_ndx = vm_result
                        .call_id_stack
                        .iter()
                        .rev()
                        .enumerate()
                        .find_map(|(ndx, x)| if x == call_id { Some(ndx) } else { None });

                    let view = if let Some(frame_ndx) = frame_ndx {
                        let value = stack_frames[frame_ndx].get_result(*vreg);
                        view_value(value, &core_dump.heap)
                    } else {
                        ValueView::Past
                    };

                    value_views.push(view);
                }
            }
        }

        value_views
    }

    fn view_value(value: mcjs_vm::InterpreterValue, heap: &mcjs_vm::heap::Heap) -> ValueView {
        use mcjs_vm::InterpreterValue;
        match value {
            InterpreterValue::Number(n) => ValueView::Literal(bytecode::Literal::Number(n)),
            InterpreterValue::Bool(b) => ValueView::Literal(bytecode::Literal::Bool(b)),
            InterpreterValue::Null => ValueView::Literal(bytecode::Literal::Null),
            InterpreterValue::Undefined => ValueView::Literal(bytecode::Literal::Undefined),
            InterpreterValue::SelfFunction => ValueView::Literal(bytecode::Literal::SelfFunction),

            InterpreterValue::Object(obj_id) => {
                let obj = heap.get(obj_id).unwrap();

                if matches!(&*obj, mcjs_vm::heap::HeapObject::ClosureObject(_)) {
                    // TODO Add more details
                    return ValueView::String("<function>".to_string());
                }
                match obj.as_str() {
                    Some(str_ref) => ValueView::String(str_ref.to_owned()),
                    None => {
                        let obj = obj.as_object();
                        let object_view = view_object(obj, heap);
                        ValueView::Object(object_view)
                    }
                }
            }

            InterpreterValue::Internal(_) => {
                unreachable!("goddamit, when ever are you going to remove this enum variant")
            }
        }
    }

    fn view_object(obj: &dyn mcjs_vm::heap::Object, heap: &mcjs_vm::heap::Heap) -> ObjectView {
        let mut own_props = HashMap::new();
        for arr_ndx in 0..obj.len() {
            let value = obj.get_element(arr_ndx).unwrap();
            let view = view_value(value, heap);
            own_props.insert(format!("[{}]", arr_ndx), Box::new(view));
        }
        for key in obj.own_properties() {
            // TODO This .clone() shouldn't be there, but the change is more complex
            // than I have patience for right now
            let value = heap.get_property_chained(obj, &key).unwrap();
            let view = view_value(value, heap);
            own_props.insert(key.to_owned(), Box::new(view));
        }

        let proto = obj
            .proto(heap)
            .and_then(|proto_id| heap.get(proto_id))
            .map(|proto| Box::new(view_object(proto.as_object(), heap)));

        let object_view = ObjectView {
            props: own_props,
            proto,
        };
        object_view
    }
}

struct CaseData {
    codebase: Arc<mcjs_vm::Codebase>,
    case: inspector_case::Case,
    root_mod_id: mcjs_vm::ModuleId,
    source_of_fn: HashMap<FnId, String>,
}

impl CaseData {
    fn load(case: inspector_case::Case) -> Self {
        use mcjs_vm::{CombinedLoader, MockLoader};

        const INLINE_FILENAME: &'static str = "__INLINE__.js";

        let include_paths = case.include_paths.iter().map(|pb| pb.as_path());
        let file_loader = Box::new(mcjs_vm::FileLoader::new(include_paths));

        let mut mock_loader = Box::new(MockLoader::new());

        let filename = match &case.root {
            inspector_case::Root::ModuleImport(path) => path.clone(),
            inspector_case::Root::InlineScript(script_code) => {
                mock_loader.add_module(
                    INLINE_FILENAME.to_owned(),
                    bytecode::ModuleId(1),
                    script_code.clone(),
                );
                INLINE_FILENAME.to_owned()
            }
        };

        let loader = Box::new(CombinedLoader::new(vec![mock_loader, file_loader]));

        let builder_params = mcjs_vm::BuilderParams { loader };
        let mut builder = builder_params.to_builder();

        let root_mod_id = builder
            .compile_file(filename)
            .unwrap_or_else(|err| panic!("compile error: {:?}", err));
        let mcjs_vm::Built {
            codebase,
            sourcemap_of_module,
        } = builder.build();
        let codebase = Arc::new(codebase);

        // swc SourceMaps have the lovely property that they're !Send. That
        // means that it's a pain in the ass to use them in this multithreaded
        // program (and actix will force us to have closures and stuff, even if
        // we ditched multithreading) So: we transform them immediately into
        // something less good but nicely shareable.

        let mut source_of_fn = HashMap::new();
        for (fn_id, func) in codebase.all_functions() {
            if let Some(source_span) = func.source_span() {
                let module_id = codebase.module_of_fn(fn_id).unwrap();
                if let Some(sourcemap) = sourcemap_of_module.get(&module_id) {
                    let lo = sourcemap.lookup_char_pos(source_span.lo());
                    let src_bytes = lo.file.src.as_bytes();

                    let lo = source_span.lo().0 as usize;
                    let hi = source_span.hi().0 as usize;
                    let fn_src = src_bytes[lo..hi].to_owned();
                    let fn_src = String::from_utf8(fn_src).unwrap();
                    source_of_fn.insert(fn_id, fn_src);
                }
            }
        }

        CaseData {
            codebase,
            source_of_fn,
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

    session_mgr: Mutex<SessionMgr>,
}

enum Watch {
    VReg(EternalVReg),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
struct SessionID(u32);

impl RootState {
    fn new(case: CaseData) -> Self {
        RootState {
            watches: Vec::new(),
            breakpoints: Vec::new(),
            case,
            session_mgr: Mutex::new(SessionMgr::new()),
        }
    }

    fn start_new_session(&self) -> SessionID {
        let run_params = RunParams {
            breakpoints: self.breakpoints.clone(),
            codebase: Arc::clone(&self.case.codebase),
            root_mod_id: self.case.root_mod_id,
        };
        let vm_result = Tracer::run(run_params);
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

struct RunParams {
    breakpoints: Vec<Breakpoint>,
    codebase: Arc<mcjs_vm::Codebase>,
    root_mod_id: mcjs_vm::ModuleId,
}
struct VMResult {
    call_id_stack: Vec<CallID>,
    error_messages: Vec<String>,
    core_dump: Option<CoreDump>,
    run_params: RunParams,
    last_iid: usize,
    instr_meta: HashMap<GlobalIID, Vec<InstrMeta>>,
}

struct HistoryItem {
    stack_depth: usize,
    eiid: EternalIID,
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

#[derive(Debug)]
enum InstrMeta {
    ClosureInfo {
        upvalues: Vec<interpreter::UpvalueId>,
    },
}

struct Tracer {
    call_id_stack: Vec<CallID>,
    call_id: CallID,
    last_iid: bytecode::IID,
    instr_meta: HashMap<GlobalIID, Vec<InstrMeta>>,
    run_params: RunParams,
}

impl Tracer {
    fn new(run_params: RunParams) -> Self {
        Self {
            call_id_stack: Vec::new(),
            call_id: CallID(0),
            last_iid: bytecode::IID(0),
            instr_meta: HashMap::new(),
            run_params,
        }
    }

    fn next_call_id(&mut self) -> CallID {
        self.call_id.0 += 1;
        self.call_id
    }

    fn run(run_params: RunParams) -> VMResult {
        let mut tracer = Self::new(run_params);

        let codebase = Arc::clone(&tracer.run_params.codebase);
        let module_id = tracer.run_params.root_mod_id;

        let vm = mcjs_vm::Interpreter::new(&codebase).with_step_handler(&mut tracer);
        let result = vm.run_module(module_id);

        let (core_dump, error_messages) = match result {
            Ok(_) => (None, Vec::new()),
            Err(intrp_err) => (intrp_err.core_dump, intrp_err.error.messages().collect()),
        };

        VMResult {
            call_id_stack: tracer.call_id_stack,
            error_messages,
            core_dump,
            last_iid: tracer.last_iid.0 as usize,
            run_params: tracer.run_params,
            instr_meta: tracer.instr_meta,
        }
    }
}

impl interpreter::StepHandler for Tracer {
    fn pre_instr(&mut self, step: &mcjs_vm::InspectorStep) -> InspectorAction {
        let stack_depth = step.intrp_data.len();
        while self.call_id_stack.len() < stack_depth {
            let call_id = self.next_call_id();
            self.call_id_stack.push(call_id);
        }
        self.call_id_stack.truncate(stack_depth);
        assert_eq!(self.call_id_stack.len(), stack_depth);

        self.last_iid = step.giid.1;

        if self
            .run_params
            .breakpoints
            .iter()
            .any(|bp| bp.0 == step.giid)
        {
            InspectorAction::Fail
        } else {
            InspectorAction::Continue
        }
    }

    fn post_instr(&mut self, step: &mcjs_vm::InspectorStep) -> InspectorAction {
        let GlobalIID(fn_id, iid) = step.giid;
        let func = self.run_params.codebase.get_function(fn_id).unwrap();
        let instr = &func.instrs()[iid.0 as usize];
        match instr {
            bytecode::Instr::ClosureNew { dest, .. } => {
                let dest = step.intrp_data.top().get_result(*dest);
                if let interpreter::Value::Object(oid) = dest {
                    let obj_ref = step.heap.get(oid).unwrap();
                    if let Some(interpreter::Closure::JS(jsclo)) = obj_ref.as_closure() {
                        let upvalues = jsclo.upvalues().to_owned();
                        let meta_item = InstrMeta::ClosureInfo { upvalues };
                        eprintln!(" -- meta: {:?} -> {:?}", step.giid, meta_item);
                        self.instr_meta
                            .entry(step.giid)
                            .or_insert_with(|| Vec::new())
                            .push(meta_item);
                    }
                }
            }
            _ => {}
        }

        InspectorAction::Continue
    }
}
