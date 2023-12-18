use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use actix_web::http::header::{HeaderName, HeaderValue, TryIntoHeaderPair};
use actix_web::middleware::Logger;
use actix_web::{web, App, HttpResponse, HttpServer};
use anyhow::{Error, Result};
use handlebars::Handlebars;
use listenfd::ListenFd;
use serde::{Deserialize, Serialize};

use mcjs_vm::interpreter::debugger::BreakRangeID;
use mcjs_vm::interpreter::Fuel;

#[actix_web::main]
async fn main() -> Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let main_path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: mcjs_tools <filename>   (loader base path is cwd)");
            return Ok(());
        }
    };
    let main_path = PathBuf::from(main_path).canonicalize().unwrap();

    let mut handlebars = Handlebars::new();
    // Important: we use several nested partials, and they contain <pre> tags.
    // Skipping this setting causes handlebars to add a bunch of extra indentation at
    // the beginning of each line of source code while rendering the template.
    handlebars.set_prevent_indent(true);

    // TODO Make it independent from the cwd
    if let Err(err) = handlebars.register_templates_directory(".html", "./templates") {
        eprintln!("template compile error:\n\n{}", err);
        return Ok(());
    }

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
            .wrap(Logger::default())
            .service(actix_files::Files::new("/assets", "./data/assets/"))
            .service(view_main)
            .service(action_set_breakpoint)
            .service(action_delete_breakpoint)
            .service(action_restart)
            .service(action_continue)
            .service(action_next)
            .service(frame_view::view_object)
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

#[derive(Deserialize)]
struct MainViewParams {
    frame_ndx: Option<usize>,
    focus_current_instr: Option<bool>,
}
impl Default for MainViewParams {
    fn default() -> Self {
        MainViewParams {
            frame_ndx: Some(0),
            focus_current_instr: None,
        }
    }
}

#[actix_web::get("/")]
async fn view_main(
    app_data: web::Data<AppData<'_>>,
    query_params: web::Query<MainViewParams>,
) -> actix_web::Result<HttpResponse> {
    let params = query_params.into_inner();
    render_main_screen(app_data.into_inner(), &params).await
}

async fn render_main_screen(
    app_data: Arc<AppData<'_>>,
    params: &MainViewParams,
) -> actix_web::Result<HttpResponse> {
    let snapshot = app_data.snapshot().await;
    if snapshot.model.is_some() {
        let params = serde_json::json!({
            "snapshot": &*snapshot,
            "frame_ndx": params.frame_ndx.unwrap_or(0),
            "focus_current_instr": params.focus_current_instr.unwrap_or(true),
        });
        let body = app_data
            .handlebars
            .render("index", &params)
            .map_err(actix_web::error::ErrorInternalServerError)?;
        Ok(HttpResponse::Ok().body(body))
    } else {
        Ok(HttpResponse::Ok().body("Interpreter finished successfully.  No debugging!"))
    }
}

mod frame_view {
    use std::{collections::HashMap, ops::Range};

    use actix_web::{self, web, HttpResponse};
    use maud::{html, Markup};
    use mcjs_vm::{
        bytecode::{self, ArgIndex, CaptureIndex, VReg},
        interpreter::{
            debugger::{BreakRangeID, Probe},
            UpvalueId,
        },
        GlobalIID, InterpreterValue,
    };
    use serde::Serialize;

    use super::AppData;

    #[actix_web::get("/objects/{object_id}")]
    pub async fn view_object(
        app_data: web::Data<AppData<'static>>,
        path_params: web::Path<u64>,
    ) -> actix_web::Result<HttpResponse> {
        use mcjs_vm::interpreter::debugger::ObjectId;

        let object_id_ffi: u64 = path_params.into_inner();
        let object_id: ObjectId = slotmap::KeyData::from_ffi(object_id_ffi).into();

        let markup = app_data
            .intrp_handle
            .query(move |state| {
                let probe = state.probe()?;
                let obj = probe.get_object(object_id)?;
                let obj = obj.as_object();

                // Only get a shallow representation of the object, which the user/client can further expand
                let properties = obj.own_properties();

                let show_value = |value| {
                    let details_url = get_details_url(Some(value));
                    let header = show_value_header(probe, value, None);
                    html!{
                        div.cursor-pointer x-data="{detailsVisible: false}" { 
                            div {
                                @if let Some(details_url) = details_url {
                                    button
                                        hx-target="next .detailsBox"
                                        hx-swap="innerHTML"
                                        hx-get=(details_url)
                                        x-bind:class="{checked: detailsVisible}"
                                        x-on:click="detailsVisible = !detailsVisible"
                                        class="
                                            relative inline-block align-middle cursor-pointer rounded-sm px-1 
                                            border-b border-zinc-400 dark:border-zinc-500
                                            active:border-b-0 active:mb-[1px] active:top-px

                                            bg-zinc-200 dark:bg-zinc-600
                                            text-zinc-500 dark:text-black
                                            [&.checked]:font-bold [&.checked]:text-orange-500
                                        "
                                    { "—" }
                                }

                                (header)
                            }

                            div.detailsBox x-show="detailsVisible" {}
                        }
                    }
                };

                Some(match properties.len() {
                    0 => html! { div { "~ empty ~" } },
                    1 => {
                        let key = properties.into_iter().next().unwrap();
                        html! {
                            table {
                                tr {
                                    td."align-top" { "{ " }
                                    td."align-top" { (format!("{}:", key)) }
                                    td."align-top" {
                                        (show_value(obj.get_own_property(&key).unwrap()))
                                    }
                                    td."align-top" { "}" }
                                }
                            }
                        }
                    }
                    _ => html! {
                        table {
                            @for (index, key) in properties.iter().enumerate() {
                                tr {
                                    td."align-top" { (if index == 0 { "{ " } else { ", " }) }
                                    td."align-top" { (format!("{}:", key)) }
                                    td."align-top" {
                                        (show_value(obj.get_own_property(&key).unwrap()))
                                    }
                                }
                            }

                            tr {
                                td."align-top" { "}" }
                            }
                        }
                    },
                })
            })
            .await
            .ok_or_else(|| actix_web::error::ErrorNotFound("no such object"))?;

        Ok(HttpResponse::Ok().body(markup.into_string()))
    }

    pub fn snapshot(
        probe: &Probe<'_, '_>,
        loader: &mcjs_vm::Loader,
        has_breakpoint: impl Fn(GlobalIID) -> bool,
    ) -> Snapshot {
        let mut frames = Vec::new();
        let mut prev_return_iid = None;
        for (frame_ndx, frame) in probe.frames().enumerate() {
            let fnid = frame.header().fn_id;
            let bytecode::FnId(mod_id, lfnid) = fnid;
            // TODO Refactor this elsewhere?
            let giid = {
                let iid = prev_return_iid.unwrap_or(probe.giid().1);
                // TODO oint at the instruction that we're currently stuck at.
                // That would be ideal, but there is a chance that it's the first one in the
                // function (iid.0 == 0), and I don't want to deal with the underflow right now
                prev_return_iid = frame.header().return_to_iid;
                bytecode::GlobalIID(fnid, iid)
            };

            let source_filename = loader
                .get_source_map(mod_id)
                .and_then(|sm| get_filename(sm.files().first().unwrap()));

            let func = loader.get_function(fnid).unwrap();
            let values = decode_locs_state(probe, &frame, func, giid.1);
            let source_raw_markup =
                fetch_source_code(probe, giid).unwrap_or_else(|| "???".to_string());

            frames.push(Frame {
                source_filename,
                source_raw_markup,
                // TODO Remove the damn call ID
                callID: frame_ndx.try_into().unwrap(),
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
                                has_breakpoint: has_breakpoint(giid),
                            }
                        })
                        .collect(),
                    consts: func
                        .consts()
                        .iter()
                        .map(|konst| show_literal(konst))
                        .collect(),
                },
            });
        }

        Snapshot { frames }
    }

    fn fetch_source_code(probe: &Probe, giid: bytecode::GlobalIID) -> Option<String> {
        let fnid = giid.0;

        let loader = probe.loader();
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

        let raw_markup = render_source_code(&markers, &frame_src)
            .map(|pre_escaped| pre_escaped.into_string())
            .unwrap_or(String::new());
        Some(raw_markup)
    }

    #[derive(Serialize)]
    pub struct Snapshot {
        frames: Vec<Frame>,
    }

    #[derive(PartialEq, Eq)]
    pub struct StackFingerprint {
        func_ids: Vec<(u16, u16)>,
    }
    impl Snapshot {
        pub fn fingerprint(&self) -> StackFingerprint {
            let func_ids = self
                .frames
                .iter()
                .map(|frame| (frame.moduleID, frame.functionID))
                .collect();
            StackFingerprint { func_ids }
        }
    }

    #[allow(non_snake_case)]
    #[derive(Clone, Serialize)]
    struct Frame {
        source_filename: Option<String>,
        source_raw_markup: String,

        callID: u32,
        moduleID: u16,
        functionID: u16,
        iid: u16,

        function: Function,
        values: Values,
    }

    #[derive(Clone, Serialize)]
    struct Function {
        consts: Vec<String>,
        bytecode: Vec<Instruction>,
    }

    #[derive(Clone, Serialize)]
    struct Instruction {
        textual_repr: String,
        has_breakpoint: bool,
    }

    #[derive(Clone, Serialize)]
    struct Values {
        // In the same order as they should be shown in the UI
        locs: Vec<LocState>,
    }

    #[derive(Clone, Serialize)]
    struct LocState {
        name: String,
        ident: Option<String>,
        prev_idents: Vec<String>,
        value_header: String,
        details_url: Option<String>,
    }

    #[derive(Clone, Serialize)]
    struct FrameSource {
        text: String,
        start_line: usize,
        start_offset: swc_common::BytePos,
        end_line: usize,
        end_offset: swc_common::BytePos,
        line_focus: Option<usize>,
    }

    mod break_range {
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

    fn render_source_code(
        markers: &[break_range::Marker],
        frame_src: &FrameSource,
    ) -> actix_web::Result<Markup> {
        let pre_escaped_code = {
            use break_range::MarkerKind;
            use std::fmt::Write;

            let mut pre_escaped_code = String::new();

            let offset = frame_src.start_offset.0 as usize;
            let mut prev_end = offset;

            for marker in markers {
                let mkr_offset = marker.offset as usize;
                let text = &frame_src.text[prev_end - offset..mkr_offset - offset];
                write!(pre_escaped_code, "{}", text).unwrap();

                match marker.kind {
                    MarkerKind::Start {
                        brid,
                        iid_start,
                        iid_end,
                    } => {
                        write!(
                            pre_escaped_code,
                            "<span class='relative' x-bind:class=\"{{'src-range-selected': (markedBreakRange?.id === '{}')}}\">",
                            brid.to_string(),
                        ).unwrap();

                        let brid_str = brid.to_string();
                        assert!(brid_str.chars().all(|ch| ch == ',' || ch.is_numeric()));

                        write!(
                            pre_escaped_code,
                            "<span class='cursor-pointer' 
                                x-on:click='markedBreakRange = {{id: \"{}\", iidStart: {}, iidEnd: {}}}'
                                hx-trigger='dblclick'
                                hx-post='/break_ranges/{brid_str}/set'
                                hx-swap='none'
                            >⚬ </span>",
                            brid_str, iid_start.0, iid_end.0
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
                ul.font-mono x-init="$el.querySelector('.current-instr').scrollIntoView({ block: 'center' })" {
                    @for line_ndx in frame_src.start_line..frame_src.end_line {
                        @if Some(line_ndx) == frame_src.line_focus {
                            li.current-instr { (format!("{:4}", line_ndx)) }
                        } @else {
                            li { (format!("{:4}", line_ndx)) }
                        }
                    }
                }

                pre {
                    (pre_escaped_code)
                }
            }
        };

        Ok(markup)
    }

    fn line_index_of_giid(
        loader: &mcjs_vm::Loader,
        giid: mcjs_vm::GlobalIID,
        source_file: &swc_common::SourceFile,
    ) -> Option<usize> {
        let (_, break_range) = loader.breakrange_at_giid(giid)?;
        source_file.lookup_line(break_range.lo)
    }

    fn decode_locs_state(
        probe: &Probe,
        frame: &mcjs_vm::stack::Frame<'_>,
        func: &bytecode::Function,
        iid: mcjs_vm::IID,
    ) -> Values {
        use bytecode::Loc;

        // We use `ident_history` (the history of how the Identifier->Loc mappings change as the
        // function proceeds) to reconstruct the Identifier->Loc mapping at a specific
        // point of the bytecode (represented by `iid`).

        let mut locs_state = HashMap::new();
        let mut locs_order = Vec::new();

        {
            let locs_state = &mut locs_state;
            let locs_order = &mut locs_order;

            let mut add = |loc: Loc, value: Option<InterpreterValue>, upv_id: Option<UpvalueId>| {
                let value_str = value
                    .map(|value| show_value_header(probe, value, upv_id))
                    .unwrap_or_else(|| "???".to_string());
                locs_state.insert(
                    loc,
                    LocState {
                        name: format!("{:?}", loc),
                        value_header: value_str,
                        ident: None,
                        prev_idents: Vec::new(),
                        details_url: get_details_url(value),
                    },
                );
                locs_order.push(loc);
            };

            for (cap_ndx, upv_id) in frame.captures().enumerate() {
                let loc = CaptureIndex(cap_ndx as _).into();
                // TODO: Also show the captured value
                let value_opt = frame.deref_upvalue(upv_id);
                add(loc, value_opt, Some(upv_id));
            }

            for (arg_ndx, value_opt) in frame.args().enumerate() {
                let loc = ArgIndex(arg_ndx as _).into();
                add(loc, value_opt, None);
            }

            for (vreg_ndx, value) in frame.results().enumerate() {
                let loc = VReg(vreg_ndx as _).into();
                add(loc, Some(value), None);
            }
        }

        // Use merged map to add identifiers associations

        let history = func.ident_history();
        let history_limit = history
            .iter()
            .enumerate()
            .take_while(|(_, asmt)| asmt.iid.0 <= iid.0)
            .last()
            .map(|(ndx, _)| ndx + 1)
            .unwrap_or(0);
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
        let this_value = frame.header().this;
        locs.push(LocState {
            name: "this".to_string(),
            value_header: show_value_header(probe, this_value, None),
            ident: Some("this".to_string()),
            prev_idents: Vec::new(),
            details_url: get_details_url(Some(this_value)),
        });
        locs.extend(
            locs_order
                .into_iter()
                .map(|loc| locs_state.remove(&loc).unwrap()),
        );
        assert!(locs_state.is_empty());

        Values { locs }
    }

    fn show_value_header(
        probe: &Probe,
        value: InterpreterValue,
        upv_id: Option<UpvalueId>,
    ) -> String {
        use slotmap::Key;
        use std::fmt::Write;

        // A hack, but slotmap does not give us visibility into the internal structure of the ID
        let show_keydata = |kd: slotmap::KeyData| {
            let id_raw: u64 = kd.as_ffi();
            let (ndx, version) = (id_raw & 0xffff_ffff, id_raw >> 32);
            format!("{}v{}", ndx, version)
        };

        let mut buf = String::new();

        if let Some(upv_id) = upv_id {
            // A hack, but slotmap does not give us visibility into the internal structure of the ID
            write!(buf, "<upvalue {}> → ", show_keydata(upv_id.data())).unwrap();
        }

        match value {
            InterpreterValue::Number(num) => write!(buf, "{}", num).unwrap(),
            InterpreterValue::Undefined => write!(buf, "undefined").unwrap(),
            InterpreterValue::Object(obj_id) => {
                let obj = probe.get_object(obj_id);
                let obj_id = show_keydata(obj_id.data());

                if let Some(obj) = obj {
                    if let Some(str) = obj.as_str() {
                        write!(buf, "{:?}", str).unwrap()
                    } else if let Some(_) = obj.as_closure() {
                        write!(buf, "<closure>").unwrap()
                    } else {
                        write!(buf, "<object {}>", obj_id).unwrap()
                    }
                } else {
                    write!(buf, "<object {} → DANGLING REF!>", obj_id).unwrap()
                }
            }
            _ => write!(buf, "{:?}", value).unwrap(),
        }

        buf
    }

    fn show_literal(lit: &mcjs_vm::Literal) -> String {
        match lit {
            mcjs_vm::Literal::Number(num) => format!("{}", num),
            mcjs_vm::Literal::String(s) => format!("{:?}", s),
            mcjs_vm::Literal::Bool(b) => format!("{}", b),
            mcjs_vm::Literal::Null => "null".to_string(),
            mcjs_vm::Literal::Undefined => "undefined".to_string(),
            mcjs_vm::Literal::SelfFunction => "<self function>".to_string(),
        }
    }

    fn get_details_url(value: Option<InterpreterValue>) -> Option<String> {
        use slotmap::Key;

        if let Some(InterpreterValue::Object(obj_id)) = &value {
            Some(format!("/objects/{}", obj_id.data().as_ffi()))
        } else {
            None
        }
    }

    fn extract_frame_source(
        source_map: &swc_common::SourceMap,
        loader: &mcjs_vm::Loader,
        giid: mcjs_vm::GlobalIID,
        offset_range: Range<u32>,
    ) -> Option<FrameSource> {
        use std::cmp::{max, min};
        use swc_common::BytePos;

        // NOTE We're making a distinct source map per file, but the API clearly
        // supports a single source map for multiple files.  Should I use it?
        let files = source_map.files();
        let source_file = files.first().unwrap();

        let line_focus = line_index_of_giid(loader, giid, source_file);

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

        Some(FrameSource {
            text: text.to_string(),
            line_focus,
            start_line,
            start_offset,
            end_line,
            end_offset,
        })
    }

    fn markers_for_breakranges<'a>(
        break_ranges: impl Iterator<Item = (BreakRangeID, &'a bytecode::BreakRange)>,
    ) -> Vec<break_range::Marker> {
        use break_range::*;

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

    fn get_filename(source_file: &swc_common::SourceFile) -> Option<String> {
        let file_name = &source_file.name;
        eprintln!("sourceFile <- {:?}", file_name);
        match file_name {
            swc_common::FileName::Real(path) => Some(path.to_string_lossy().into_owned()),
            _ => None,
        }
    }
}

#[actix_web::post("/break_ranges/{brange_id}/set")]
async fn action_set_breakpoint(
    app_data: web::Data<AppData<'static>>,
    path_params: web::Path<String>,
) -> actix_web::Result<HttpResponse> {
    let brange_id_s = path_params.into_inner();
    let brange_id = BreakRangeID::parse_string(&brange_id_s)
        .ok_or_else(|| actix_web::error::ErrorBadRequest("invalid break range ID"))?;

    let app_data = app_data.into_inner();
    app_data
        .intrp_handle
        .query(move |state| -> std::result::Result<_, &'static str> {
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

    Ok(HttpResponse::Ok()
        .append_header(hx_trigger(&BreakpointAddedEvent))
        .body(""))
}

trait Event: serde::Serialize {
    fn event_name() -> &'static str;
}

#[derive(Serialize)]
struct BreakpointAddedEvent;

impl Event for BreakpointAddedEvent {
    fn event_name() -> &'static str {
        "breakpointAdded"
    }
}

#[derive(Serialize)]
struct BreakpointDeletedEvent;

impl Event for BreakpointDeletedEvent {
    fn event_name() -> &'static str {
        "breakpointDeleted"
    }
}

#[allow(non_snake_case)]
#[derive(Serialize)]
struct VMStateChangedEvent {
    onlyTopFrameChanged: bool,
}

impl Default for VMStateChangedEvent {
    fn default() -> Self {
        VMStateChangedEvent {
            onlyTopFrameChanged: false,
        }
    }
}

impl Event for VMStateChangedEvent {
    fn event_name() -> &'static str {
        "vmStateChanged"
    }
}

fn hx_trigger<E: Event>(event: &E) -> (HeaderName, HeaderValue) {
    let name = E::event_name();
    let events_payload = serde_json::json!({
        name: event,
    })
    .to_string();
    ("HX-Trigger", events_payload).try_into_pair().unwrap()
}

#[derive(Deserialize, Debug)]
struct DeleteBreakpointParams {
    category: String,
    index: usize,
}

#[actix_web::post("/sidebar/breakpoints/delete")]
async fn action_delete_breakpoint(
    app_data: web::Data<AppData<'static>>,
    params: web::Form<DeleteBreakpointParams>,
) -> actix_web::Result<HttpResponse> {
    // TODO Manage the error cases.  This function is full of unwrap's that should be proper HTTP errors

    let params = params.into_inner();

    eprintln!("delete breakpoint request: {:?}", params);
    let was_there = match params.category.as_str() {
        "source" => {
            let res = app_data
                .intrp_handle
                .query(move |state| {
                    let probe = state.probe_mut().unwrap();
                    // TODO Susceptible of "read after write" (ID changing meaning)
                    let (brange_id, _) = probe
                        .source_breakpoints()
                        .nth(params.index)
                        .expect("no such breakpoint");
                    probe.clear_source_breakpoint(brange_id)
                })
                .await;

            match res {
                Ok(was_there) => was_there,
                Err(err) => {
                    return Err(actix_web::error::ErrorInternalServerError(format!(
                        "debugger error: {:?}",
                        err
                    )))
                }
            }
        }
        "instr" => {
            app_data
                .intrp_handle
                .query(move |state| {
                    // TODO Manage the None case
                    let probe = state.probe_mut().unwrap();
                    let giid = probe
                        .instr_breakpoints()
                        .nth(params.index)
                        .expect("no such breakpoint");

                    probe.clear_instr_breakpoint(giid)
                })
                .await
        }
        other => {
            return Err(actix_web::error::ErrorBadRequest(format!(
                "invalid 'category': '{}' (allowed: source, instr)",
                other,
            )))
        }
    };

    // Otherwise the UI wouldn't change, even after a reload
    app_data.invalidate_snapshot();

    match was_there {
        true => Ok(HttpResponse::Ok()
            .append_header(hx_trigger(&BreakpointDeletedEvent))
            .body("")),
        false => Ok(HttpResponse::Ok().body("(nothing deleted; breakpoint did not exist.)")),
    }
}

#[actix_web::post("/restart")]
async fn action_restart(app_data: web::Data<AppData<'static>>) -> actix_web::Result<HttpResponse> {
    let app_data = app_data.into_inner();
    app_data.intrp_handle.restart();
    app_data.invalidate_snapshot();
    Ok(HttpResponse::Ok()
        .append_header(hx_trigger(&VMStateChangedEvent::default()))
        .body(""))
}

#[actix_web::post("/continue")]
async fn action_continue(app_data: web::Data<AppData<'static>>) -> actix_web::Result<HttpResponse> {
    let app_data = app_data.into_inner();
    app_data.intrp_handle.resume();
    app_data.invalidate_snapshot();
    Ok(HttpResponse::Ok()
        .append_header(hx_trigger(&VMStateChangedEvent::default()))
        .body(""))
}

#[actix_web::post("/next")]
async fn action_next(app_data: web::Data<AppData<'static>>) -> actix_web::Result<HttpResponse> {
    let app_data = app_data.into_inner();

    let pre_fingerprint = {
        let snapshot = app_data.snapshot().await;
        snapshot
            .as_ref()
            .model
            .as_ref()
            .map(|model| model.frame_view_snapshot.fingerprint())
    };

    app_data
        .intrp_handle
        .query(|state| {
            // TODO Replace this unwrap with proper error handling
            let probe = state.probe_mut().unwrap();
            probe.set_fuel(Fuel::Limited(1));
        })
        .await;
    app_data.intrp_handle.resume();
    app_data.invalidate_snapshot();

    let post_fingerprint = {
        let snapshot = app_data.snapshot().await;
        snapshot
            .as_ref()
            .model
            .as_ref()
            .map(|model| model.frame_view_snapshot.fingerprint())
    };

    Ok(HttpResponse::Ok()
        .append_header(hx_trigger(&VMStateChangedEvent {
            onlyTopFrameChanged: (pre_fingerprint == post_fingerprint),
        }))
        .body(""))
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
    use mcjs_vm::{interpreter::debugger::Probe, GlobalIID};
    use serde::Serialize;

    /// A model (a copy, a projection) of the (suspended) interpreter's state,
    /// pre-processed and filtered for easy usage by templates.
    ///
    /// It is also used directly as the template input data for the main screen template.
    #[derive(Serialize)]
    pub struct VMState {
        pub source_breakpoints: Vec<SourceBreakpoint>,
        pub instr_breakpoints: Vec<GlobalIID>,
        // This module is destined to disappear in a future refactoring. The
        // awkward `super::frame_view::...`  can stay for now
        pub frame_view_snapshot: super::frame_view::Snapshot,
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

            let has_breakpoint = |giid| instr_breakpoints.contains(&giid);
            let frame_view_snapshot = super::frame_view::snapshot(probe, loader, has_breakpoint);

            VMState {
                source_breakpoints,
                instr_breakpoints,
                frame_view_snapshot,
            }
        }
    }

    #[derive(Clone, Serialize)]
    pub struct SourceBreakpoint {
        pub filename: String,
        pub line: LineNum,
    }

    type LineNum = u32;

    #[derive(Clone, Serialize)]
    pub struct SourceLine {
        pub ndx1: LineNum,
        pub text: String,
    }
}
