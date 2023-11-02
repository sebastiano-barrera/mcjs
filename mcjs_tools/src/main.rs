use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use actix_web::{web, App, HttpResponse, HttpServer, Responder};

use anyhow::{anyhow, Error, Result};
use handlebars::{handlebars_helper, Handlebars};
use listenfd::ListenFd;
use mcjs_vm::interpreter::debugger::Probe;
use serde_json::json;
use serde_json::Value as JsonValue;
use tokio::sync::broadcast;

handlebars_helper!(lookup_deep: |*args| {
    if args.is_empty() {
        panic!();
    }

    let mut value = args[0];
    for step in &args[1..] {
        value = if let Some(property) = step.as_str() {
            value.get(property).unwrap_or(&JsonValue::Null)
        } else if let Some(index) = step.as_number() {
            let key = index.to_string();
            value.get(key).unwrap_or(&JsonValue::Null)
        } else {
            return Ok(handlebars::ScopedJson::Missing)
        };
    }

    value.clone()
});

#[actix_web::main]
async fn main() -> Result<()> {
    let main_path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: mcjs_tools <filename>   (loader base path is cwd)");
            return Ok(());
        }
    };
    let main_path = PathBuf::from(main_path).canonicalize().unwrap();

    let (model_tx, mut model_rx) = broadcast::channel(5);

    let mut handlebars = Handlebars::new();

    // TODO Make it independent from the cwd
    if let Err(err) = handlebars.register_templates_directory(".html", "./templates") {
        eprintln!("template compile error:\n\n{}", err);
        return Ok(());
    }

    handlebars.register_helper("lookup_deep", Box::new(lookup_deep));
    let data_ref = web::Data::new(AppData {
        handlebars,
        current_state: Mutex::new(State {
            failure: None,
            model: Arc::new(model::VMState::init()),
        }),
    });

    {
        let data_ref = data_ref.clone();
        tokio::spawn(async move {
            loop {
                match model_rx.recv().await {
                    Ok(new_event) => {
                        eprintln!("event: {:?}", new_event);

                        let mut state = data_ref.current_state.lock().unwrap();
                        match new_event {
                            InterpreterEvent::Suspended { new_state } => {
                                state.model = new_state;
                            }
                            InterpreterEvent::Finished => break,
                            InterpreterEvent::Failed {
                                failed_state,
                                message,
                            } => {
                                state.model = failed_state;
                                state.failure = Some(message);
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(err) => panic!("interpreter state latch: unexpected error: {:?} ", err),
                }
            }
        });
    }

    let interpreter_jh = tokio::spawn(interpreter_main_loop(main_path, model_tx));

    tokio::spawn(async move {
        if let Err(err) = interpreter_jh.await {
            eprintln!("interpreter panicked: {:?}", err.to_string());
        }
    });

    let server = HttpServer::new(move || {
        App::new()
            .service(actix_files::Files::new("/assets", "./data/assets/"))
            .service(main_screen)
            .service(events)
            .app_data(data_ref.clone())
    });

    let mut listenfd = ListenFd::from_env();
    let server = match listenfd.take_tcp_listener(0)? {
        Some(listener) => server.listen(listener)?,
        None => server.bind(("127.0.0.1", 10001))?,
    };

    server.run().await.map_err(Error::from)
}

struct AppData<'a> {
    handlebars: Handlebars<'a>,
    current_state: Mutex<State>,
}

struct State {
    failure: Option<String>,
    model: Arc<model::VMState>,
}

#[actix_web::get("/")]
async fn main_screen(app_data: web::Data<AppData<'_>>) -> impl Responder {
    let state = app_data.current_state.lock().unwrap();
    let tmpl_params = json!({
        "failure": state.failure,
        "model": &*state.model,
    });

    eprintln!(
        "main_screen JSON = {}",
        serde_json::to_string_pretty(&tmpl_params).unwrap()
    );

    match app_data.handlebars.render("index", &tmpl_params) {
        Ok(body) => HttpResponse::Ok().body(body),
        Err(err) => {
            let body = format!("template render error:\n\n{}", err);
            HttpResponse::InternalServerError().body(body)
        }
    }
}

#[actix_web::get("/events")]
async fn events(app_data: web::Data<AppData<'static>>) -> impl Responder {
    use actix_web_lab::sse;
    use tokio::sync::mpsc;

    async fn sender_process(app_data: web::Data<AppData<'_>>, tx: mpsc::Sender<sse::Event>) {
        for i in 0..100 {
            let new_body = app_data
                .handlebars
                .render("fragment", &json!({ "event_ndx": i }))
                .unwrap();
            let event = sse::Data::new(new_body).event("content").into();
            tx.send(event).await.expect("could not send event");

            let msecs = rand::random::<u8>() as u64 * 8;
            let dur = Duration::from_millis(msecs);
            tokio::time::sleep(dur).await;
        }
    }

    let (tx, rx) = mpsc::channel(10);
    tokio::spawn(sender_process(app_data.clone(), tx));

    // TODO Figure out *exactly* what this means
    sse::Sse::from_infallible_receiver(rx).with_retry_duration(Duration::from_secs(10))
}

#[derive(Clone)]
enum InterpreterEvent {
    Suspended {
        new_state: Arc<model::VMState>,
    },
    Finished,
    Failed {
        failed_state: Arc<model::VMState>,
        message: String,
    },
}

impl std::fmt::Debug for InterpreterEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpreterEvent::Suspended { .. } => write!(f, "InterpreterEvent::Suspended"),
            InterpreterEvent::Finished => write!(f, "InterpreterEvent::Finished"),
            InterpreterEvent::Failed { message, .. } => {
                write!(f, "InterpreterEvent::Failed: {}", message)
            }
        }
    }
}

async fn interpreter_main_loop(
    main_path: PathBuf,
    model_tx: broadcast::Sender<InterpreterEvent>,
) -> Result<()> {
    use mcjs_vm::interpreter::Exit;
    use mcjs_vm::{Interpreter, Loader, Realm};

    let base_path = main_path.parent().unwrap();
    let filename = main_path
        .file_name()
        .unwrap()
        .to_str()
        .expect("can't convert main filename to UTF-8");
    let import_path = format!("./{}", filename);

    let mut realm = Realm::new();
    let mut loader = Loader::new(Some(base_path.to_owned()));

    let main_fnid = loader
        .load_import(&import_path, mcjs_vm::SCRIPT_MODULE_ID)
        .map_err(|err| anyhow!("compile error: {:?}", err))?;

    // TODO Replace println! with logging

    println!();
    println!("running...");

    let mut intrp = Interpreter::new(&mut realm, &mut loader, main_fnid);

    loop {
        match intrp.run() {
            Ok(Exit::Finished(_)) => {
                let res = model_tx.send(InterpreterEvent::Finished);
                if res.is_err() {
                    println!("channel closed. nobody will know the interpreter finished...");
                }
                break;
            }
            Ok(Exit::Suspended(next_intrp)) => {
                intrp = next_intrp;
                println!("interpreter suspended.  collecting snapshot");

                let probe = Probe::attach(&mut intrp);
                let model = Arc::new(model::VMState::snapshot(&probe));
                let event = InterpreterEvent::Suspended { new_state: model };

                let res = model_tx.send(event);
                if res.is_err() {
                    println!("channel closed; finishing interpreter process, deleting interpreter");
                    break;
                }

                println!("(suspended; resuming immediately)");
            }
            Err(mut err) => {
                println!("interpreter failed.  reporting error");

                let mut message = String::new();
                for msg in err.error.messages() {
                    use std::fmt::Write;
                    writeln!(message, " - {}", msg).unwrap();
                }

                let failed_state = Arc::new(model::VMState::snapshot(&err.probe()));

                let res = model_tx.send(InterpreterEvent::Failed {
                    failed_state,
                    message,
                });
                if res.is_err() {
                    println!("channel closed. failure won't be discovered...");
                }

                break;
            }
        }
    }

    Ok(())
}

#[allow(non_snake_case)]
mod model {
    use std::cmp::{max, min};
    use std::collections::HashMap;

    use mcjs_vm::{bytecode, interpreter::debugger::Probe};
    use serde::Serialize;

    /// A model (a copy, a projection) of the (suspended) interpreter's state,
    /// pre-processed and filtered for easy usage by templates.
    ///
    /// It is also used directly as the template input data for the main screen template.
    #[derive(Clone, Serialize)]
    pub struct VMState {
        pub breakpoints: Vec<Breakpoint>,

        /// Stack frames, in bottom-to-top order
        pub frames: Vec<Frame>,

        pub modules: HashMap<u16, Module>,
        pub objects: HashMap<ObjectID, Object>,
    }

    impl VMState {
        pub fn snapshot(probe: &Probe) -> Self {
            let loader = probe.loader();

            let breakpoints = probe
                .breakpoints()
                .map(|(_, bp)| {
                    let filename = loader
                        .get_abs_path(bp.mod_id)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    let line = bp.loc.line;
                    Breakpoint { line, filename }
                })
                .collect();

            let mut frames = Vec::new();
            let mut prev_return_iid = None;
            for (i, frame) in probe.frames().enumerate() {
                let fnid = frame.header().fn_id;
                let bytecode::FnId(mod_id, lfnid) = fnid;
                // TODO Refactor this elsewhere?
                let giid = {
                    let iid = prev_return_iid.unwrap_or(probe.giid().1);
                    // For the top frame: `iid` is the instruction to *resume* to.
                    // For other frames: `iid` is the instruction to *return* to.
                    // In either case: we actually want the instruction we suspended/called at,
                    // which is the previous one.
                    let iid = mcjs_vm::IID(iid.0 - 1);
                    prev_return_iid = frame.header().return_to_iid;
                    bytecode::GlobalIID(fnid, iid)
                };

                let source_map = loader.get_source_map(mod_id);

                let source = decode_frame_source(source_map, loader, giid);

                let func = loader.get_function(fnid).unwrap();
                let locs = decode_locs_state(&frame, func, giid.1)
                    .into_values()
                    .collect();

                frames.push(Frame {
                    source,
                    // TODO Remove the damn call ID
                    callID: i.try_into().unwrap(),
                    moduleID: mod_id.0,
                    functionID: lfnid.0,
                    iid: giid.1 .0,
                    thisValue: "<todo>".to_string(),
                    returnToInstrID: "<todo>".to_string(),
                    args: Vec::new(),
                    captures: Vec::new(),
                    results: Vec::new(),
                    locs,
                });
            }

            let mut modules = HashMap::new();
            for frame in frames.iter() {
                let fnid = bytecode::FnId(
                    bytecode::ModuleId(frame.moduleID),
                    bytecode::LocalFnId(frame.functionID),
                );

                let module_model = modules.entry(frame.moduleID).or_insert_with(|| Module {
                    functions: HashMap::new(),
                });

                module_model
                    .functions
                    .entry(frame.functionID)
                    .or_insert_with(|| Function {
                        bytecode: loader
                            .get_function(fnid)
                            .unwrap()
                            .instrs()
                            .iter()
                            .map(|instr| format!("{:?}", instr))
                            .collect(),
                    });
            }

            VMState {
                breakpoints,
                frames,
                modules,
                objects: HashMap::new(),
            }
        }

        pub(crate) fn init() -> VMState {
            VMState {
                breakpoints: Vec::new(),
                frames: Vec::new(),
                modules: HashMap::new(),
                objects: HashMap::new(),
            }
        }
    }

    fn decode_frame_source(
        source_map: Option<&swc_common::SourceMap>,
        loader: &mcjs_vm::Loader,
        giid: mcjs_vm::GlobalIID,
    ) -> Option<FrameSource> {
        let source_map = source_map?;

        // NOTE We're making a distinct source map per file, but the API clearly
        // supports a single source map for multiple files.  Should I use it?
        let files = source_map.files();
        let source_file = &files.first().unwrap();

        let filename = {
            let file_name = &source_file.name;
            eprintln!("sourceFile <- {:?}", file_name);
            match file_name {
                swc_common::FileName::Real(path) => path.to_string_lossy().into_owned(),
                _ => return None,
            }
        };

        let line_focus = {
            let break_range = loader.breakrange_at_giid(giid)?;
            source_file.lookup_line(break_range.lo).unwrap()
        };

        let line_start = max(0, line_focus as isize - 150) as usize;
        let line_end = min(source_file.count_lines() - 1, line_focus + 150);

        let lines: Vec<_> = (line_start..line_end)
            .map(|line_ndx| SourceLine {
                ndx: line_ndx.try_into().unwrap(),
                text: source_file.get_line(line_ndx).unwrap().into_owned(),
            })
            .collect();

        Some(FrameSource {
            filename,
            line_focus: line_focus.try_into().unwrap(),
            lines,
        })
    }

    fn decode_locs_state(
        frame: &mcjs_vm::stack::Frame<'_>,
        func: &bytecode::Function,
        iid: mcjs_vm::IID,
    ) -> HashMap<bytecode::Loc, LocState> {
        // We use `ident_history` (the history of how the Identifier->Loc mappings change as the
        // function proceeds) to reconstruct the Identifier->Loc mapping at a specific
        // point of the bytecode (represented by `iid`).

        use bytecode::{ArgIndex, CaptureIndex, VReg};

        let model_value = |value| format!("{:?}", value);

        let mut locs_state = HashMap::new();

        for (vreg_ndx, value) in frame.results().enumerate() {
            let loc = VReg(vreg_ndx as _).into();
            let state = LocState {
                name: format!("{:?}", loc),
                value: model_value(value),
                ident: None,
                prev_idents: Vec::new(),
            };
            locs_state.insert(loc, state);
        }

        for (arg_ndx, value_opt) in frame.args().enumerate() {
            let loc = ArgIndex(arg_ndx as _).into();
            let state = LocState {
                name: format!("{:?}", loc),
                value: value_opt
                    .map(model_value)
                    .unwrap_or_else(|| "???".to_string()),
                ident: None,
                prev_idents: Vec::new(),
            };
            locs_state.insert(loc, state);
        }

        for (cap_ndx, upv_id) in frame.captures().enumerate() {
            let loc = CaptureIndex(cap_ndx as _).into();
            let state = LocState {
                name: format!("{:?}", loc),
                value: format!("{:?}", upv_id),
                ident: None,
                prev_idents: Vec::new(),
            };
            locs_state.insert(loc, state);
        }

        let asmts = func
            .ident_history()
            .iter()
            .take_while(|asmt| asmt.iid.0 <= iid.0);
        for asmt in asmts {
            let state = locs_state.get_mut(&asmt.loc).unwrap();
            let prev_ident = state.ident.replace(asmt.ident.to_string());
            if let Some(prev_ident) = prev_ident {
                state.prev_idents.push(prev_ident);
            }
        }

        locs_state
    }

    #[derive(Clone, Serialize)]
    pub struct Breakpoint {
        pub filename: String,
        pub line: LineNum,
    }

    type LineNum = u32;

    #[derive(Clone, Serialize)]
    pub struct Frame {
        pub source: Option<FrameSource>,

        pub callID: u32,
        pub moduleID: u16,
        pub functionID: u16,
        pub iid: u16,

        pub thisValue: String,
        pub returnToInstrID: String,
        pub args: Vec<Arg>,
        pub captures: Vec<Capture>,
        pub results: Vec<Value>,

        pub locs: Vec<LocState>,
    }

    #[derive(Clone, Serialize)]
    pub struct LocState {
        pub name: String,
        pub ident: Option<String>,
        pub prev_idents: Vec<String>,
        // Just a *model* of the value
        pub value: String,
    }

    #[derive(Clone, Serialize)]
    pub struct FrameSource {
        pub filename: String,
        pub line_focus: LineNum,
        // Yes, whatever, we're copying, I don't care right now
        pub lines: Vec<SourceLine>,
    }

    #[derive(Clone, Serialize)]
    pub struct SourceLine {
        ndx: LineNum,
        text: String,
    }

    #[derive(Clone, Serialize)]
    pub struct Module {
        functions: HashMap<u16, Function>,
    }

    #[derive(Clone, Serialize)]
    pub struct Function {
        bytecode: Vec<String>,
    }

    type ObjectID = u32;
    type Object = HashMap<String, Value>;

    #[derive(Clone, Serialize)]
    pub struct Arg {}

    #[derive(Clone, Serialize)]
    pub struct Capture {}

    #[derive(Clone, Serialize)]
    pub struct Value {}
}
