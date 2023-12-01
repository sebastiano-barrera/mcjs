use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::{path::PathBuf, time::Duration};

use actix_web::{web, App, HttpResponse, HttpServer, Responder};

use anyhow::{Error, Result};
use handlebars::{handlebars_helper, Handlebars};
use listenfd::ListenFd;
use mcjs_vm::bytecode;
use mcjs_vm::interpreter::debugger::BreakRangeID;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value as JsonValue;

use maud::{html, Markup};

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
            .service(frame_view)
            .service(frame_set_breakpoint)
            .service(delete_breakpoint)
            .service(sidebar)
            .service(action_restart)
            .service(action_continue)
            .service(object)
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

    fn invalidate_snapshot(&self) {
        let mut cached_snapshot = self.cached_snapshot.lock().unwrap();
        *cached_snapshot = Arc::new(Snapshot {
            failure: None,
            model: None,
        });
    }
}

#[actix_web::get("/")]
async fn main_screen(app_data: web::Data<AppData<'_>>) -> actix_web::Result<HttpResponse> {
    render_main_screen(app_data.into_inner()).await
}

async fn render_main_screen(app_data: Arc<AppData<'_>>) -> actix_web::Result<HttpResponse> {
    let snapshot = app_data.snapshot().await;
    if snapshot.model.is_some() {
        let body = app_data
            .handlebars
            .render("index", &*snapshot)
            .map_err(actix_web::error::ErrorInternalServerError)?;
        Ok(HttpResponse::Ok().body(body))
    } else {
        Ok(HttpResponse::Ok().body("Interpreter finished successfully.  No debugging!"))
    }
}

#[derive(Deserialize)]
struct FrameViewParams {
    source_visible: bool,
}

#[actix_web::get("/frames/{frame_ndx}/view")]
async fn frame_view(
    app_data: web::Data<AppData<'static>>,
    path_params: web::Path<(usize,)>,
    query_params: web::Query<FrameViewParams>,
) -> actix_web::Result<HttpResponse> {
    let (frame_ndx,) = path_params.into_inner();
    let query_params = query_params.into_inner();
    let body = render_frame_view(&app_data, frame_ndx, query_params.source_visible).await?;
    Ok(HttpResponse::Ok().body(body))
}

async fn render_frame_view(
    app_data: &AppData<'static>,
    frame_ndx: usize,
    source_visible: bool,
) -> actix_web::Result<String> {
    // TODO Possible write-after-read hazard here, after snapshot() 'releases'
    // the interpreter manager and the subsequent query (markers, frame_src)
    // takes it again
    let snapshot = app_data.snapshot().await;
    let frame = snapshot
        .model
        .as_ref()
        .and_then(move |model| model.frames.get(frame_ndx))
        .ok_or_else(|| actix_web::error::ErrorNotFound("no frame at given index"))?;

    let source_raw_markup = {
        let (markers, frame_src) = app_data
            .intrp_handle
            .query(move |state| {
                let probe = state.probe()?;
                let loader = probe.loader();

                let giid = probe.frame_giid(frame_ndx as usize);
                let fnid = giid.0;

                let source_map = loader.get_source_map(fnid.0)?;

                let break_ranges = loader.function_breakranges(fnid).unwrap();
                let markers: Vec<_> = markers_for_breakranges(break_ranges);
                if markers.is_empty() {
                    return None;
                }

                let offset_range = {
                    let offset_min = markers.iter().map(|m| m.offset).min().unwrap();
                    let offset_max = markers.iter().map(|m| m.offset).max().unwrap();
                    offset_min..offset_max
                };

                let frame_src = extract_frame_source(source_map, loader, giid, offset_range)?;
                Some((markers, frame_src))
            })
            .await
            .ok_or_else(|| actix_web::error::ErrorNotFound("Source code not available"))?;

        render_source_code(&markers, &frame_src, frame_ndx as usize)
            .map(|pre_escaped| pre_escaped.into_string())
            .unwrap_or(String::new())
    };

    #[derive(Serialize)]
    struct TmplParams<'a> {
        frame: &'a model::Frame,
        frame_ndx: usize,
        self_url: String,
        source_raw_markup: String,
        source_visible: bool,
    }

    let params = TmplParams {
        frame,
        frame_ndx,
        self_url: format!("/frames/{frame_ndx}/view"),
        source_raw_markup,
        source_visible,
    };

    app_data
        .handlebars
        .render("frame_view", &params)
        .map_err(|err| actix_web::error::ErrorInternalServerError(err))
}

fn markers_for_breakranges<'a>(
    break_ranges: impl Iterator<Item = (BreakRangeID, &'a bytecode::BreakRange)>,
) -> Vec<model::break_range::Marker> {
    use model::break_range::*;

    let mut markers = Vec::new();

    for (brid, brange) in break_ranges {
        let bytecode::BreakRange {
            iid_start, iid_end, ..
        } = *brange;

        let lo = brange.lo.0;
        let hi = brange.hi.0;

        markers.push(Marker {
            kind: MarkerKind::Start {
                brid,
                iid_start,
                iid_end,
            },
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
}

#[actix_web::get("/sidebar")]
async fn sidebar(app_data: web::Data<AppData<'static>>) -> actix_web::Result<HttpResponse> {
    let app_data = app_data.into_inner();
    let snapshot = app_data.snapshot().await;

    let body = app_data
        .handlebars
        .render("sidebar", &*snapshot)
        .map_err(|err| actix_web::error::ErrorInternalServerError(err))?;

    Ok(HttpResponse::Ok().body(body))
}

#[actix_web::post("/frames/{frame_ndx}/break_range/{brange_id}/set")]
async fn frame_set_breakpoint(
    app_data: web::Data<AppData<'static>>,
    path_params: web::Path<(usize, String)>,
    form_params: web::Form<FrameViewParams>,
) -> actix_web::Result<HttpResponse> {
    let (frame_ndx, brange_id_s) = path_params.into_inner();
    let form_params = form_params.into_inner();

    let brange_id = BreakRangeID::parse_string(&brange_id_s)
        .ok_or_else(|| actix_web::error::ErrorBadRequest("invalid break range ID"))?;

    let app_data = app_data.into_inner();
    app_data
        .intrp_handle
        .query(move |state| -> std::result::Result<(), &'static str> {
            let probe = state
                .probe_mut()
                .ok_or_else(|| "interpreter is finished; can't set a breakpoint in this state")?;

            probe
                .set_source_breakpoint(brange_id)
                .map_err(|err| err.message())?;

            Ok(())
        })
        .await
        .map_err(|err_msg| actix_web::error::ErrorBadRequest(err_msg))?;

    app_data.invalidate_snapshot();

    let body = render_frame_view(&app_data, frame_ndx, form_params.source_visible).await?;
    Ok(HttpResponse::Ok()
        .append_header(("HX-Trigger", "breakpoints-changed"))
        .body(body))
}

#[derive(Deserialize, Debug)]
struct DeleteBreakpointParams {
    category: String,
    index: usize,
}

#[actix_web::post("/sidebar/breakpoints/delete")]
async fn delete_breakpoint(
    params: web::Form<DeleteBreakpointParams>,
) -> actix_web::Result<HttpResponse> {
    let params = params.into_inner();

    eprintln!("delete breakpoint request: {:?}", params);

    Ok(HttpResponse::NotImplemented().body("sorry!"))
}

#[actix_web::post("/restart")]
async fn action_restart(app_data: web::Data<AppData<'static>>) -> actix_web::Result<HttpResponse> {
    let app_data = app_data.into_inner();
    app_data.intrp_handle.restart();
    app_data.invalidate_snapshot();
    render_main_screen(app_data).await
}

#[actix_web::post("/continue")]
async fn action_continue(app_data: web::Data<AppData<'static>>) -> actix_web::Result<HttpResponse> {
    let app_data = app_data.into_inner();
    app_data.intrp_handle.resume();
    app_data.invalidate_snapshot();
    render_main_screen(app_data).await
}

#[actix_web::get("/objects/{object_id}")]
async fn object(
    app_data: web::Data<AppData<'static>>,
    path_params: web::Path<u64>,
) -> actix_web::Result<HttpResponse> {
    use mcjs_vm::interpreter::debugger::ObjectId;
    use mcjs_vm::InterpreterValue;

    let object_id_ffi: u64 = path_params.into_inner();
    let object_id: ObjectId = slotmap::KeyData::from_ffi(object_id_ffi).into();

    let markup = app_data
        .intrp_handle
        .query(move |state| {
            use slotmap::Key;
        
            let probe = state.probe()?;
            let obj = probe.get_object(object_id)?;
            let obj = obj.as_object();

            let properties = obj.own_properties();

            // Only get a shallow representation of the object, which the user/client can further expand
            Some(html! {
                table {
                    tr {
                        td { "Key" }
                        td { "Value" }
                    }

                    @for key in properties {
                        tr {
                            td { (key) }
                            td {
                                @match obj.get_own_property(&key).unwrap() {
                                    InterpreterValue::Object(object_id) => {
                                        div.cursor-pointer
                                            hx-get=(format!("/objects/{}", object_id.data().as_ffi()))
                                            hx-swap="outerHTML"
                                            { "<object>" }
                                    },
                                    value => (format!("{:?}", value)),                                    
                                }
                            }
                        }
                    }
                }
            })
        })
        .await
        .ok_or_else(|| actix_web::error::ErrorNotFound("no such object"))?;

    Ok(HttpResponse::Ok().body(markup.into_string()))
}

fn render_source_code(
    markers: &[model::break_range::Marker],
    frame_src: &model::FrameSource,
    frame_ndx: usize,
) -> actix_web::Result<Markup> {
    let pre_escaped_code = {
        use model::break_range::MarkerKind;
        use std::fmt::Write;

        let mut pre_escaped_code = String::new();

        let offset = frame_src.start_offset.0 as usize;
        let mut prev_end = offset;

        for marker in markers {
            let mkr_offset = marker.offset as usize;
            let text = &frame_src.text[prev_end - offset..mkr_offset - offset];
            write!(pre_escaped_code, "{}", text).unwrap();

            match marker.kind {
                MarkerKind::Start { brid, .. } => {
                    write!(
                        pre_escaped_code,
                        "<span class='relative' x-bind:class=\"markedBreakRange == '{}' && 'src-range-selected'\">",
                        brid.to_string(),
                    ).unwrap();

                    let brid_str = brid.to_string();
                    assert!(brid_str.chars().all(|ch| ch == ',' || ch.is_numeric()));

                    write!(
                        pre_escaped_code,
                        "<span class='cursor-pointer' 
                            x-on:click='markedBreakRange = \"{brid_str}\"'
                            hx-trigger='dblclick'
                            hx-post='/frames/{frame_ndx}/break_range/{brid_str}/set'
                        >⚬ </span>",
                    )
                    .unwrap();
                }
                MarkerKind::End => {
                    write!(pre_escaped_code, "</span>").unwrap();
                }
            }

            prev_end = mkr_offset;
        }

        let text_tail = &frame_src.text[prev_end as usize - offset..];
        write!(pre_escaped_code, "{}", text_tail).unwrap();

        maud::PreEscaped(pre_escaped_code)
    };

    let markup = html! {
        div.grid."grid-cols-[1.5cm_1fr]" {
            pre x-init="$el.querySelector('.current-instr').scrollIntoView({ block: 'center' })" {
                @for line_ndx in frame_src.start_line..frame_src.end_line {
                    @if Some(line_ndx) == frame_src.line_focus {
                        span."px-2".current-instr { (format!("{:4}\n", line_ndx)) }
                    } @else {
                        span."px-2" { (format!("{:4}\n", line_ndx)) }
                    }
                }
            }

            div.grid {
                pre x-data="{ markedBreakRange: null }" {
                    (pre_escaped_code)
                }
            }
        }
    };

    Ok(markup)
}

pub fn extract_frame_source(
    source_map: &swc_common::SourceMap,
    loader: &mcjs_vm::Loader,
    giid: mcjs_vm::GlobalIID,
    offset_range: Range<u32>,
) -> Option<model::FrameSource> {
    use std::cmp::{max, min};
    use swc_common::BytePos;

    // NOTE We're making a distinct source map per file, but the API clearly
    // supports a single source map for multiple files.  Should I use it?
    let files = source_map.files();
    let source_file = files.first().unwrap();

    let line_focus = line_number_of_giid(loader, giid, source_file);

    // Snap offset_range to line boundaries
    let start = max(0, offset_range.start - 150);
    let start_line = source_file.lookup_line(BytePos(start)).unwrap();
    let (start_offset, _) = source_file.line_bounds(start_line);
    let start_offset0 = start_offset.0 as usize - 1;

    let end = min(
        (source_file.src.len() - 1).try_into().unwrap(),
        offset_range.end + 150,
    );
    let end_line = source_file.lookup_line(BytePos(end)).unwrap();
    let (_, end_offset) = source_file.line_bounds(end_line);
    let end_offset0 = end_offset.0 as usize - 1;

    let text = &source_file.src[start_offset0..end_offset0];

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
    let (_, break_range) = loader.breakrange_at_giid(giid)?;
    Some(source_file.lookup_line(break_range.lo).unwrap())
}

mod interpreter_manager {
    use mcjs_vm::interpreter::debugger::Probe;

    use anyhow::{anyhow, Result};
    use std::path::PathBuf;
    use std::sync::mpsc;

    pub type Version = u32;

    pub enum State<'a, 'b> {
        Finished {
            version: Version,
        },
        Suspended {
            version: Version,
            probe: Probe<'a, 'b>,
        },
        Failed {
            version: Version,
            probe: Probe<'a, 'b>,
            error: String,
        },
    }

    impl<'a, 'b> State<'a, 'b> {
        pub fn probe(&self) -> Option<&Probe<'a, 'b>> {
            match self {
                State::Finished { .. } => None,
                State::Suspended { probe, .. } | State::Failed { probe, .. } => Some(probe),
            }
        }

        pub fn probe_mut(&mut self) -> Option<&mut Probe<'a, 'b>> {
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
            F: 'static + Send + FnOnce(&mut State) -> T,
        {
            let (sender, receiver) = tokio::sync::oneshot::channel();

            let message = Message::Query(Box::new(move |state| {
                let ret_val = thunk(state);
                if let Err(_) = sender.send(ret_val) {
                    panic!("could not send query result from interpreter main loop");
                }
            }));

            self.send(message);
            receiver.await.unwrap()
        }

        pub fn resume(&self) {
            self.send(Message::Continue);
        }

        pub fn restart(&self) {
            self.send(Message::Restart);
        }

        fn send(&self, message: Message) {
            self.queue_sender
                .send(message)
                .expect("could not send query to interpreter main loop");
        }
    }

    enum Message {
        Query(Box<dyn Send + FnOnce(&mut State)>),
        Continue,
        Restart,
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

        enum ContinueMode {
            Continue,
            Resume,
        }
        let process_messages = move |state: &mut State| loop {
            let msg = queue_rx
                .recv()
                .expect("bug: channel closed before terminating the interpreter");

            match msg {
                Message::Continue => return ContinueMode::Resume,
                Message::Restart => return ContinueMode::Continue,
                Message::Query(query) => {
                    query(state);
                }
            }
        };

        // TODO Replace println! with logging

        let mut intrp = Interpreter::new(&mut realm, &mut loader, main_fnid);
        let mut version = 0;

        // Inner loop:   The state machine in it is used to let clients analyze the state of
        // the interpreter when it is suspended.
        loop {
            version += 1;

            println!();
            println!("(resuming loop)");

            intrp = match intrp.run() {
                Ok(Exit::Finished(_)) => {
                    process_messages(&mut State::Finished { version });

                    // It doesn't really make sense to 'continue' after the interpreter is
                    // finished.  For now, I'll just restart
                    Interpreter::new(&mut realm, &mut loader, main_fnid)
                }
                Ok(Exit::Suspended(next_intrp)) => {
                    intrp = next_intrp;
                    println!("interpreter suspended.  collecting snapshot");

                    let probe = Probe::attach(&mut intrp);
                    // let model = Arc::new(model::VMState::snapshot(&probe));
                    let mut state = State::Suspended { version, probe };

                    println!("(suspended; handling queries)");
                    let continue_mode = process_messages(&mut state);
                    match continue_mode {
                        // Continue with a renewed (reset) interpreter
                        ContinueMode::Continue => intrp.restart(),
                        // Continue with *this* interpreter
                        ContinueMode::Resume => intrp,
                    }
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

                    let mut state = State::Failed {
                        version,
                        probe: err.probe(),
                        error,
                    };

                    println!("(interpreter error; handling queries)");
                    process_messages(&mut state);

                    // Always restarts, independently of the ContinueMode
                    println!("(restarting with a reset interpreter)");
                    err.restart()
                }
            }
        }
    }
}

#[allow(non_snake_case)]
mod model {
    use std::collections::HashMap;

    use bytecode::{ArgIndex, CaptureIndex, VReg};
    use mcjs_vm::{bytecode, interpreter::debugger::Probe, GlobalIID};
    use serde::Serialize;

    /// A model (a copy, a projection) of the (suspended) interpreter's state,
    /// pre-processed and filtered for easy usage by templates.
    ///
    /// It is also used directly as the template input data for the main screen template.
    #[derive(Clone, Serialize)]
    pub struct VMState {
        pub source_breakpoints: Vec<SourceBreakpoint>,
        pub instr_breakpoints: Vec<GlobalIID>,

        /// Stack frames, in bottom-to-top order
        pub frames: Vec<Frame>,
    }

    impl VMState {
        pub fn snapshot(probe: &Probe) -> Self {
            let loader = probe.loader();

            let source_breakpoints = probe
                .source_breakpoints()
                .map(|(brid, _)| {
                    let filename = loader
                        .get_abs_path(brid.module_id())
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    let brange = loader.get_break_range(brid).unwrap();

                    // TODO Reduce number of calls to get_source_map
                    // (Ideally reduce everything to a single source map)
                    let source_map = loader.get_source_map(brid.module_id()).unwrap();
                    let loc = source_map.lookup_char_pos(brange.lo);
                    let line = loc.line.try_into().unwrap();
                    SourceBreakpoint { line, filename }
                })
                .collect();

            let instr_breakpoints: Vec<_> = probe.instr_breakpoints().collect();

            let mut frames = Vec::new();
            let mut prev_return_iid = None;
            for (i, frame) in probe.frames().enumerate() {
                let fnid = frame.header().fn_id;
                let bytecode::FnId(mod_id, lfnid) = fnid;
                // TODO Refactor this elsewhere?
                let giid = {
                    let iid = prev_return_iid.unwrap_or(probe.giid().1);
                    // TODO Point at the instruction that we're currently stuck at.
                    // That would be ideal, but there is a chance that it's the first one in the
                    // function (iid.0 == 0), and I don't want to deal with the underflow right now
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
                            .enumerate()
                            .map(|(ndx, instr)| {
                                let iid = bytecode::IID(ndx.try_into().unwrap());
                                let giid = bytecode::GlobalIID(fnid, iid);
                                Instruction {
                                    textual_repr: format!("{:?}", instr),
                                    has_breakpoint: instr_breakpoints.contains(&giid),
                                }
                            })
                            .collect(),
                    },
                });
            }

            VMState {
                source_breakpoints,
                instr_breakpoints,
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

        let show_value = |value| {
            if let mcjs_vm::InterpreterValue::Object(obj_id) = value {
                use slotmap::Key;
                format!("Object({:?} = {})", obj_id, obj_id.data().as_ffi())
            } else {
                format!("{:?}", value)
            }
        };

        let mut locs_state = HashMap::new();
        let mut locs_order = Vec::new();

        // Merge all 3 categories of values into a single map
        for (cap_ndx, upv_id) in frame.captures().enumerate() {
            let loc = CaptureIndex(cap_ndx as _).into();
            let state = LocState {
                name: format!("{:?}", loc),
                value: format!("{:?}", upv_id),
                ident: None,
                prev_idents: Vec::new(),
            };
            locs_state.insert(loc, state);
            locs_order.push(loc);
        }

        for (arg_ndx, value_opt) in frame.args().enumerate() {
            let loc = ArgIndex(arg_ndx as _).into();
            let state = LocState {
                name: format!("{:?}", loc),
                value: value_opt
                    .map(show_value)
                    .unwrap_or_else(|| "???".to_string()),
                ident: None,
                prev_idents: Vec::new(),
            };
            locs_state.insert(loc, state);
            locs_order.push(loc);
        }

        for (vreg_ndx, value) in frame.results().enumerate() {
            let loc: Loc = VReg(vreg_ndx as _).into();
            let state = LocState {
                name: format!("{:?}", loc),
                value: show_value(value),
                ident: None,
                prev_idents: Vec::new(),
            };
            locs_state.insert(loc, state);
            locs_order.push(loc);
        }

        // Use merged map to add identifiers associations

        let history = func.ident_history();
        let history_limit = history.iter()
            .enumerate()
            .take_while(|(_, asmt)| asmt.iid.0 <= iid.0)
            .last()
            .unwrap().0 + 1;
        for asmt in history[0..history_limit].iter().rev() {
            let ls = locs_state.get_mut(&asmt.loc).unwrap();
            let ident = asmt.ident.to_string();
            if ls.ident.is_some() {
                ls.prev_idents.push(ident);
            } else {
                ls.ident = Some(ident);
            }
        }

        let mut locs = Vec::new();
        locs.push(LocState {
            name: "this".to_string(),
            value: format!("{:?}", frame.header().this),
            ident: Some("this".to_string()),
            prev_idents: Vec::new(),
        });
        locs.extend(locs_order.into_iter().map(|loc| locs_state.remove(&loc).unwrap()));
        assert!(locs_state.is_empty());
  
        Values { locs }
    }

    #[derive(Clone, Serialize)]
    pub struct SourceBreakpoint {
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
        use mcjs_vm::interpreter::debugger::BreakRangeID;
        use mcjs_vm::IID;

        #[derive(Debug)]
        pub enum MarkerKind {
            Start {
                brid: BreakRangeID,
                iid_start: IID,
                iid_end: IID,
            },
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
        // In the same order as they should be shown in the UI
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
        bytecode: Vec<Instruction>,
    }

    #[derive(Clone, Serialize)]
    pub struct Instruction {
        textual_repr: String,
        has_breakpoint: bool,
    }
}
