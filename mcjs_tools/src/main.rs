use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::{path::PathBuf, time::Duration};

use actix_web::{web, App, HttpResponse, HttpServer, Responder};

use anyhow::{Error, Result};
use handlebars::{handlebars_helper, Handlebars};
use listenfd::ListenFd;
use mcjs_vm::bytecode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value as JsonValue;

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

    let mut handlebars = Handlebars::new();

    // TODO Make it independent from the cwd
    if let Err(err) = handlebars.register_templates_directory(".html", "./templates") {
        eprintln!("template compile error:\n\n{}", err);
        return Ok(());
    }
    handlebars.register_helper("lookup_deep", Box::new(lookup_deep));

    let intrp_handle = interpreter_manager::spawn_interpreter(main_path);

    let data_ref = web::Data::new(AppData {
        handlebars,
        intrp_handle,
        cached_snapshot: Mutex::new(Arc::new(Snapshot {
            failure: None,
            model: None,
        })),
    });

    let server = HttpServer::new(move || {
        App::new()
            .service(actix_files::Files::new("/assets", "./data/assets/"))
            .service(main_screen)
            .service(events)
            .service(some_text)
            .service(create_breakpoint)
            .service(frame_view)
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
    intrp_handle: interpreter_manager::Handle,
    cached_snapshot: Mutex<Arc<Snapshot>>,
}

#[derive(Serialize)]
struct Snapshot {
    failure: Option<String>,
    model: Option<model::VMState>,
}

impl<'a> AppData<'a> {
    async fn snapshot(&self) -> Arc<Snapshot> {
        let mut cached_snapshot = self.cached_snapshot.lock().unwrap();

        if cached_snapshot.model.is_some() {
            return Arc::clone(&cached_snapshot);
        }

        let (failure, model) = self
            .intrp_handle
            .query(|state| match state {
                interpreter_manager::State::Finished { version: _ } => (None, None),
                interpreter_manager::State::Suspended { version: _, probe } => {
                    (None, Some(model::VMState::snapshot(probe)))
                }
                interpreter_manager::State::Failed {
                    version: _,
                    probe,
                    error,
                } => (Some(error.clone()), Some(model::VMState::snapshot(probe))),
            })
            .await;

        let snapshot = Arc::new(Snapshot { model, failure });
        *cached_snapshot = Arc::clone(&snapshot);

        snapshot
    }
}

#[actix_web::get("/")]
async fn main_screen(app_data: web::Data<AppData<'_>>) -> actix_web::Result<HttpResponse> {
    let snapshot = app_data.snapshot().await;
    if snapshot.model.is_some() {
        eprintln!(
            "main_screen JSON = {}",
            serde_json::to_string_pretty(&*snapshot).unwrap()
        );

        let body = app_data
            .handlebars
            .render("index", &*snapshot)
            .map_err(actix_web::error::ErrorInternalServerError)?;
        Ok(HttpResponse::Ok().body(body))
    } else {
        Ok(HttpResponse::Ok().body("Interpreter finished successfully.  No debugging!"))
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

#[derive(Deserialize)]
struct BreakpointReq {
    mod_id: u16,
    fn_id: u16,
    iid: u16,
}

#[actix_web::post("/breakpoints")]
async fn create_breakpoint(
    app_data: web::Data<AppData<'_>>,
    params: web::Form<BreakpointReq>,
) -> impl Responder {
    // let breakpoint_req = params.into_inner();

    // app_data.into_inner().intrp_handle.query(|| {});

    //HttpResponse::Ok()
    //    .insert_header(("HX-Trigger", "mcjs_tools_breakpoint_changed"))
    //    .into()

    HttpResponse::NotImplemented().body("not yet implemented!")
}

#[actix_web::get("/frames/{frame_ndx}/view")]
async fn frame_view(
    app_data: web::Data<AppData<'static>>,
    path_params: web::Path<(usize,)>,
) -> actix_web::Result<HttpResponse> {
    let (frame_ndx,) = path_params.into_inner();

    let snapshot = app_data.snapshot().await;
    let frame = snapshot
        .model
        .as_ref()
        .and_then(move |model| model.frames.get(frame_ndx))
        .ok_or_else(|| actix_web::error::ErrorNotFound("no frame at given index"))?;

    #[derive(Serialize)]
    struct TmplParams<'a> {
        frame: &'a model::Frame,
        self_url: String,
    }

    let params = TmplParams {
        frame,
        self_url: format!("/frames/{frame_ndx}/view"),
    };

    let body = app_data
        .handlebars
        .render("frame_view", &params)
        .map_err(|err| actix_web::error::ErrorInternalServerError(err))?;

    Ok(HttpResponse::Ok().body(body))
}

#[actix_web::get("/frames/{frame_ndx}/source_preview")]
async fn some_text(
    app_data: web::Data<AppData<'static>>,
    path: web::Path<(u32,)>,
) -> impl Responder {
    let (frame_ndx,) = path.into_inner();

    let res = app_data
        .intrp_handle
        .query(move |state| {
            let probe = state.probe()?;
            let giid = probe.frame_giid(frame_ndx as usize);
            let fnid = giid.0;

            let source_map = probe.loader().get_source_map(fnid.0)?;

            let loader = probe.loader();
            let markers: Vec<_> = {
                use model::break_range::*;

                let mut markers = Vec::new();

                let break_ranges = loader.function_breakranges(fnid).unwrap();

                for brange in break_ranges {
                    let bytecode::BreakRange {
                        iid_start, iid_end, ..
                    } = *brange;

                    let lo = brange.lo.0;
                    let hi = brange.hi.0;

                    markers.push(Marker {
                        kind: MarkerKind::Start { iid_start, iid_end },
                        offset: lo,
                        length: hi as i32 - lo as i32,
                    });
                    markers.push(Marker {
                        kind: MarkerKind::End,
                        offset: hi,
                        length: lo as i32 - hi as i32, // Will be negative
                    });
                }

                // Longer segments first
                markers.sort_by_key(|m| (m.offset, -m.length));
                markers
            };

            if markers.is_empty() {
                return None;
            }

            let offset_range = {
                let offset_min = markers.iter().map(|m| m.offset).min().unwrap();
                let offset_max = markers.iter().map(|m| m.offset).max().unwrap();
                offset_min..offset_max
            };

            let frame_src = extract_frame_source(source_map, probe.loader(), giid, offset_range)?;
            render_source_code(&markers, &frame_src).ok()
        })
        .await;

    match res {
        Some(text) => HttpResponse::Ok().body(text),
        // TODO Actually expose any errors
        None => HttpResponse::NotFound().body(""),
    }
}

fn render_source_code(
    markers: &[model::break_range::Marker],
    frame_src: &model::FrameSource,
) -> std::result::Result<String, std::fmt::Error> {
    use model::break_range::MarkerKind;
    use std::fmt::Write;

    let mut markers = markers.into_iter().enumerate().peekable();

    let mut buf = String::new();

    writeln!(buf, "<div class=\"grid source-view-grid-cols\">")?;
    writeln!(
        buf,
        "<pre x-init=\"$el.querySelector('.current').scrollIntoView({{ block: 'center' }})\">"
    )?;
    // TODO Is the range inclusive?
    for line_ndx in frame_src.start_line..frame_src.end_line {
        if Some(line_ndx) == frame_src.line_focus {
            writeln!(
                buf,
                "<span class='current bg-white text-black'>{:4}</span>",
                line_ndx,
            )?;
        } else {
            writeln!(buf, "{:4}", line_ndx)?;
        }
    }
    writeln!(buf, "</pre>")?;

    writeln!(buf, "<pre x-data='{{ markerIndex: null }}'>")?;
    for (rel_offset, ch) in frame_src.text.chars().enumerate() {
        let rel_offset: u32 = rel_offset.try_into().unwrap();
        let offset = frame_src.start_offset.0 + rel_offset + 1;

        while let Some((mkr_ndx, mkr)) = markers.peek().filter(|(_, mkr)| mkr.offset == offset) {
            assert!(mkr.offset >= offset);

            match mkr.kind {
                MarkerKind::Start { iid_start, iid_end } => {
                    write!(buf,
                       "<span class='relative cursor-pointer' x-bind:class=\"markerIndex == {} && 'bg-sky-800'\"><span x-on:click='markerIndex = {}; highlight.iidStart = {}; highlight.iidEnd = {};'>◯ </span>",
                       mkr_ndx,
                       mkr_ndx,
                       iid_start.0,
                       iid_end.0,
                   )?;
                }
                MarkerKind::End => {
                    write!(buf, "</span>")?;
                }
            };

            markers.next();
        }

        write!(buf, "{}", ch)?;
    }

    assert!(markers.next().is_none());
    writeln!(buf, "</div>")?;

    Ok(buf)
}

pub fn extract_frame_source(
    source_map: &swc_common::SourceMap,
    loader: &mcjs_vm::Loader,
    giid: mcjs_vm::GlobalIID,
    offset_range: Range<u32>,
) -> Option<model::FrameSource> {
    use swc_common::BytePos;

    // NOTE We're making a distinct source map per file, but the API clearly
    // supports a single source map for multiple files.  Should I use it?
    let files = source_map.files();
    let source_file = &files.first().unwrap();

    let line_focus = line_number_of_giid(loader, giid, source_file);

    use std::cmp::{max, min};
    let Range { mut start, mut end } = offset_range;
    start = max(0, start - 150);
    end = min(source_file.src.len().try_into().unwrap(), end + 150);

    // Snap to line boundaries
    let start_line = source_file.lookup_line(BytePos(start)).unwrap();
    let (start_offset, _) = source_file.line_bounds(start_line);

    let end_line = source_file.lookup_line(BytePos(end)).unwrap();
    let (_, end_offset) = source_file.line_bounds(end_line);

    let text = &source_file.src[start_offset.0 as usize..end_offset.0 as usize];

    Some(model::FrameSource {
        text: text.to_string(),
        line_focus,
        start_line,
        start_offset,
        end_line,
        end_offset,
    })
}

fn line_number_of_giid(
    loader: &mcjs_vm::Loader,
    giid: mcjs_vm::GlobalIID,
    source_file: &swc_common::SourceFile,
) -> Option<usize> {
    let break_range = loader.breakrange_at_giid(giid)?;
    Some(source_file.lookup_line(break_range.lo).unwrap())
}

mod interpreter_manager {
    use mcjs_vm::interpreter::debugger::Probe;

    use anyhow::{anyhow, Result};
    use std::path::PathBuf;
    use std::sync::mpsc;

    pub type Version = u32;

    pub enum State<'a, 'b, 'c> {
        Finished {
            version: Version,
        },
        Suspended {
            version: Version,
            probe: &'a mut Probe<'b, 'c>,
        },
        Failed {
            version: Version,
            probe: &'a mut Probe<'b, 'c>,
            error: String,
        },
    }

    impl<'a, 'b, 'c> State<'a, 'b, 'c> {
        pub fn probe(&'a self) -> Option<&'a Probe<'b, 'c>> {
            match self {
                State::Finished { .. } => None,
                State::Suspended { probe, .. } | State::Failed { probe, .. } => Some(probe),
            }
        }
    }

    pub struct Handle {
        queue_sender: mpsc::Sender<Message>,
    }

    // TODO Remove model_tx?
    pub fn spawn_interpreter(main_path: PathBuf) -> Handle {
        let (queue_sender, queue_recver) = mpsc::channel();
        // TODO Need the JoinHandle?
        std::thread::spawn(move || interpreter_main_loop(main_path, queue_recver));
        Handle { queue_sender }
    }

    impl Handle {
        pub async fn query<T, F>(&self, thunk: F) -> T
        where
            T: 'static + Send,
            F: 'static + Send + FnOnce(&State) -> T,
        {
            let (sender, receiver) = tokio::sync::oneshot::channel();

            let message = Message::Query(Box::new(move |state| {
                let ret_val = thunk(state);
                if let Err(_) = sender.send(ret_val) {
                    panic!("could not send query result from interpreter main loop");
                }
            }));

            self.queue_sender
                .send(message)
                .expect("could not send query to interpreter main loop");

            receiver.await.unwrap()
        }
    }

    enum Message {
        Query(Box<dyn Send + FnOnce(&State)>),
        Resume,
    }

    fn interpreter_main_loop(main_path: PathBuf, queue_rx: mpsc::Receiver<Message>) -> Result<()> {
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
        let mut version = 0;

        let process_messages = move |state: &State| {
            loop {
                let msg = queue_rx
                    .recv()
                    .expect("bug: channel closed before terminating the interpreter");

                match msg {
                    // "Return"
                    Message::Resume => break,
                    Message::Query(query) => {
                        query(state);
                    }
                }
            }
        };

        loop {
            version += 1;

            match intrp.run() {
                Ok(Exit::Finished(_)) => {
                    process_messages(&State::Finished { version });
                    break;
                }
                Ok(Exit::Suspended(next_intrp)) => {
                    intrp = next_intrp;
                    println!("interpreter suspended.  collecting snapshot");

                    let mut probe = Probe::attach(&mut intrp);
                    // let model = Arc::new(model::VMState::snapshot(&probe));
                    let state = State::Suspended {
                        version,
                        probe: &mut probe,
                    };

                    println!("(suspended; handling queries)");
                    process_messages(&state);
                    println!("(resuming)");
                }
                Err(mut err) => {
                    let error = {
                        let mut message = String::new();
                        for msg in err.error.messages() {
                            use std::fmt::Write;
                            writeln!(message, " - {}", msg).unwrap();
                        }

                        message
                    };

                    let probe = &mut err.probe();
                    let state = State::Failed {
                        version,
                        probe,
                        error,
                    };

                    println!("(interpreter error; handling queries)");
                    process_messages(&state);

                    break;
                }
            }
        }

        println!("(main loop quitting)");
        Ok(())
    }
}

#[allow(non_snake_case)]
mod model {
    use std::collections::HashMap;

    use bytecode::{ArgIndex, CaptureIndex, VReg};
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

                let source_filename = loader
                    .get_source_map(mod_id)
                    .and_then(|sm| get_filename(sm.files().first().unwrap()));

                let func = loader.get_function(fnid).unwrap();
                let values = decode_locs_state(&frame, func, giid.1);

                frames.push(Frame {
                    source_filename,
                    // TODO Remove the damn call ID
                    callID: i.try_into().unwrap(),
                    moduleID: mod_id.0,
                    functionID: lfnid.0,
                    iid: giid.1 .0,
                    values,
                    function: Function {
                        bytecode: func
                            .instrs()
                            .iter()
                            .map(|instr| format!("{:?}", instr))
                            .collect(),
                    },
                });
            }

            VMState {
                breakpoints,
                frames,
            }
        }
    }

    pub fn get_filename(source_file: &swc_common::SourceFile) -> Option<String> {
        let file_name = &source_file.name;
        eprintln!("sourceFile <- {:?}", file_name);
        match file_name {
            swc_common::FileName::Real(path) => Some(path.to_string_lossy().into_owned()),
            _ => None,
        }
    }

    fn decode_locs_state(
        frame: &mcjs_vm::stack::Frame<'_>,
        func: &bytecode::Function,
        iid: mcjs_vm::IID,
    ) -> Values {
        use bytecode::Loc;

        // We use `ident_history` (the history of how the Identifier->Loc mappings change as the
        // function proceeds) to reconstruct the Identifier->Loc mapping at a specific
        // point of the bytecode (represented by `iid`).

        let model_value = |value| format!("{:?}", value);
        let mut locs_state = HashMap::new();

        // Merge all 3 categories of values into a single map

        for (vreg_ndx, value) in frame.results().enumerate() {
            let loc: Loc = VReg(vreg_ndx as _).into();
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

        // Use merged map to add identifiers associations
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

        // Split by categories again for viewing

        let registers: Vec<_> = (0..frame.results().len())
            .filter_map(|vreg_ndx| {
                let key = VReg(vreg_ndx as _);
                locs_state.remove(&key.into())
            })
            .collect();

        let arguments: Vec<_> = (0..frame.args().len())
            .filter_map(|arg_ndx| {
                let key = ArgIndex(arg_ndx as _);
                locs_state.remove(&key.into())
            })
            .collect();

        let captures: Vec<_> = (0..frame.captures().len())
            .filter_map(|cap_ndx| {
                let key = CaptureIndex(cap_ndx as _);
                locs_state.remove(&key.into())
            })
            .collect();

        debug_assert!(locs_state.is_empty());

        let this = LocState {
            name: "this".to_string(),
            value: format!("{:?}", frame.header().this),
            ident: Some("this".to_string()),
            prev_idents: Vec::new(),
        };

        Values {
            registers,
            arguments,
            captures,
            this,
        }
    }

    #[derive(Clone, Serialize)]
    pub struct Breakpoint {
        pub filename: String,
        pub line: LineNum,
    }

    type LineNum = u32;

    #[derive(Clone, Serialize)]
    pub struct Frame {
        pub source_filename: Option<String>,

        pub callID: u32,
        pub moduleID: u16,
        pub functionID: u16,
        pub iid: u16,

        pub function: Function,
        pub values: Values,
    }

    pub mod break_range {
        use mcjs_vm::IID;

        #[derive(Debug)]
        pub enum MarkerKind {
            Start { iid_start: IID, iid_end: IID },
            End,
        }
        #[derive(Debug)]
        pub struct Marker {
            pub kind: MarkerKind,
            pub offset: u32,
            pub length: i32, // Only used for ordering
        }
    }

    #[derive(Clone, Serialize)]
    pub struct Values {
        pub registers: Vec<LocState>,
        pub captures: Vec<LocState>,
        pub arguments: Vec<LocState>,
        pub this: LocState,
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
        pub text: String,
        pub start_line: usize,
        pub start_offset: swc_common::BytePos,
        pub end_line: usize,
        pub end_offset: swc_common::BytePos,
        pub line_focus: Option<usize>,
    }

    #[derive(Clone, Serialize)]
    pub struct SourceLine {
        pub ndx1: LineNum,
        pub text: String,
    }

    #[derive(Clone, Serialize)]
    pub struct Function {
        bytecode: Vec<String>,
    }
}
