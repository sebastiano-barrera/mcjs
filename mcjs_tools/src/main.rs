#![allow(dead_code)]
#![allow(unused_must_use)]

use std::ops::Range;
use std::path::PathBuf;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use mcjs_vm::bytecode;
use mcjs_vm::interpreter::debugger::Probe;
use mcjs_vm::interpreter::Fuel;

fn main() {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let params = parse_args().expect("cli error");
    eprintln!("params = {:?}", params);

    let si = interpreter_manager::StandaloneInterpreter::new(params);
    let app = AppData {
        si,
        recent_state_change: false,
        source_code_view: Default::default(),
    };

    let native_options = eframe::NativeOptions::default();
    eframe::run_native("mcjs tools", native_options, Box::new(|_cc| Box::new(app))).unwrap();
}

fn parse_args() -> Result<interpreter_manager::Params> {
    let mut params = interpreter_manager::Params {
        main_directory: None,
        filenames: Vec::new(),
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "-b" && arg == "--base-path" {
            let path = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("-b|--base-path requires an argument"))?;
            let path = PathBuf::from(path);
            params.main_directory = Some(path);
        } else {
            params.filenames.push(arg.into());
        }
    }

    if params.main_directory.is_none() && params.filenames.len() == 1 {
        params.main_directory = params.filenames[0]
            .parent()
            .map(|p| p.to_path_buf().canonicalize().unwrap());
    }

    Ok(params)
}

struct AppData {
    si: Pin<Box<interpreter_manager::StandaloneInterpreter>>,
    recent_state_change: bool,
    source_code_view: source_code_view::Cache,
}

impl<'a> eframe::App for AppData {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        use interpreter_manager::State;

        let recent_state_change = self.recent_state_change;
        self.recent_state_change = false;

        enum Action {
            Next,
            Restart,
            Continue,
            None,
        }

        match self.si.state_mut() {
            State::Ready { filename_ndx } => {
                let should_start = egui::CentralPanel::default()
                    .show(ctx, |ui| {
                        ui.label(format!("Ready to proceed with file #{}", *filename_ndx + 1));
                        ui.button("Start").clicked()
                    })
                    .inner;
                if should_start {
                    self.si.resume();
                }
                return;
            }
            State::Finished => {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.label("Interpreter finished successfully. No debugging!");
                });
                return;
            }
            State::Failed(interpreter_manager::Error::Generic(err)) => {
                egui::SidePanel::right("simple_central_panel").show(ctx, |ui| {
                    ui.label(format!("Error: {:?}", err));
                });
                return;
            }
            _ => {}
        };

        let action = egui::SidePanel::left("sidebar")
            .show(ctx, |ui| {
                let mut action = Action::None;

                if let Some(err_message) = self.si.error_message() {
                    ui.heading("interpreter failed");
                    ui.label(&err_message);
                }

                if ui.button("NEXT").clicked() {
                    action = Action::Next;
                }
                if ui.button("CONTINUE").clicked() {
                    action = Action::Continue;
                }
                if ui.button("RESTART").clicked() {
                    action = Action::Restart;
                }
                ui.button("DELETE");

                let probe = self.si.probe_mut().unwrap();
                let status_text = format!("suspended at {:?}", probe.giid());

                ui.separator();
                ui.label(status_text);

                ui.separator();
                ui.label("Double click on a <source code range> to set a breakpoint");

                ui.heading("SOURCE BREAKPOINTS");
                let loader = probe.loader();
                ui.vertical(|ui| {
                    for (brid, _) in probe.source_breakpoints() {
                        let break_range = loader.get_break_range(brid).unwrap();
                        let source_map = loader.get_source_map(brid.module_id()).unwrap();
                        let loc = source_map.lookup_char_pos(break_range.lo);
                        let filename = loc.file.name.to_string();
                        ui.label(format!("{}:{}", filename, loc.line));
                    }
                });

                ui.heading("INSTR. BREAKPOINTS");
                ui.vertical(|ui| {
                    for giid in probe.instr_breakpoints() {
                        ui.label(format!("{:?}", giid));
                    }
                });

                ui.heading("STACK");
                ui.vertical(|ui| {
                    for (ndx, frame) in probe.frames().enumerate() {
                        let iid = if ndx == 0 {
                            probe.giid().1
                        } else {
                            frame.header().return_target.unwrap().0
                        };
                        ui.label(format!("{:?}:{:?}", frame.header().fn_id, iid));
                    }
                });

                action
            })
            .inner;

        if let State::Suspended(_) = self.si.state_mut() {
            let mut probe = self.si.probe_mut().unwrap();

            let res = egui::SidePanel::right("source_code")
                .show(ctx, |ui| {
                    source_code_view::show(ui, &probe, &mut self.source_code_view)
                })
                .inner;

            if let Some(brid) = res.set_source_bpkt {
                probe.set_source_breakpoint(brid);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut probe = self.si.probe_mut().unwrap();

            let vm_giid = probe.giid();
            let fnid = vm_giid.0;
            let func = probe.loader().get_function(fnid).unwrap();

            let mut bkpt_to_set = None;

            egui::ScrollArea::both().show(ui, |ui| {
                let instr_bkpts: Vec<_> = probe.instr_breakpoints().collect();
                let preview_bkpt_iid = self
                    .source_code_view
                    .preview_breakrange()
                    .map(|br| br.iid_start);

                for (ndx, instr) in func.instrs().iter().enumerate() {
                    let is_current = ndx == vm_giid.1 .0 as usize;
                    let res = ui.horizontal(|ui| {
                        let iid = bytecode::IID(ndx.try_into().unwrap());
                        let giid = bytecode::GlobalIID(fnid, iid);

                        if instr_bkpts.contains(&giid) {
                            ui.label("•");
                        }
                        if preview_bkpt_iid == Some(iid) {
                            ui.label(" >> ");
                        }

                        let instr_repr = format!("{:4} {:?}", ndx, instr);

                        let res = ui.selectable_label(is_current, instr_repr);
                        if res.clicked() {
                            bkpt_to_set = Some(giid);
                        }
                    });

                    if recent_state_change && is_current {
                        res.response.scroll_to_me(None);
                    }
                }
            });

            if let Some(bkpt_to_set) = bkpt_to_set {
                probe.set_instr_breakpoint(bkpt_to_set);
            }
        });

        match action {
            Action::None => {}
            Action::Next => {
                let mut probe = self.si.probe_mut().unwrap();
                probe.set_fuel(Fuel::Limited(1));
                self.si.resume();
                self.recent_state_change = true;
            }
            Action::Continue => {
                self.si.resume();
                self.recent_state_change = true;
            }
            Action::Restart => {
                self.si.restart();
                self.si.resume();
                self.recent_state_change = true;
            }
        }
    }
}

mod source_code_view {
    use std::{ops::Range, rc::Rc, sync::Arc};

    use mcjs_vm::{bytecode, interpreter::debugger::Probe};

    #[derive(Default)]
    pub struct Cache {
        main: Option<Main>,
        focus: Option<Focus>,
    }
    struct Main {
        fnid: bytecode::FnId,
        galley: Arc<egui::Galley>,
        src: Rc<String>,
        source_start_ofs: u32,
    }
    struct Focus {
        cursor_ofs: u32,
        candidate_breakranges: Vec<BreakRangeCache>,
        preview_breakrange_index: Option<usize>,
    }
    struct BreakRangeCache {
        id: mcjs_vm::BreakRangeID,
        br: bytecode::BreakRange,
        galley: Arc<egui::Galley>,
    }

    impl Cache {
        pub fn preview_breakrange(&self) -> Option<&bytecode::BreakRange> {
            let focus = self.focus.as_ref()?;
            let ndx = focus.preview_breakrange_index?;
            Some(&focus.candidate_breakranges[ndx].br)
        }
    }

    #[derive(Default)]
    pub struct Response {
        pub set_source_bpkt: Option<mcjs_vm::BreakRangeID>,
    }

    pub fn show(ui: &mut egui::Ui, probe: &Probe, cache: &mut Cache) -> Response {
        let mut my_response = Response::default();

        let giid = probe.giid();
        let fnid = giid.0;

        if cache.main.as_ref().map(|m| m.fnid != fnid).unwrap_or(true) {
            let loader = probe.loader();
            let source_map = match loader.get_source_map(fnid.0) {
                Some(sm) => sm,
                None => {
                    ui.label("No source map!");
                    return my_response;
                }
            };

            let abs_span = *loader.get_function(fnid).unwrap().span();

            let source_file = {
                let mut break_ranges = loader.function_breakranges(fnid).unwrap().peekable();

                // All break ranges must belong to the same file, so we just peek
                // one and use it to get a ptr to that swc_common::SourceFile.
                // Then we use source file's offset in the source map to make sure
                // that the markers are expressed in file-local offsets.
                //
                // (There should *always* be at least a single break range in a
                // function, belonging at least to an empty statement.  I *might*
                // be wrong; we'll see.)
                let (_, brange) = break_ranges.peek().expect("no break ranges!");
                source_map.lookup_byte_offset(brange.lo).sf
            };
            let (ofs_start, ofs_end) = source_map.span_to_char_offset(&source_file, abs_span);
            let ofs_start = ofs_start as usize;
            let ofs_end = ofs_end as usize;

            let galley = ui.fonts(|fonts| {
                make_highlight_galley(
                    &source_file.src,
                    ofs_start..ofs_end,
                    egui::Color32::GRAY,
                    egui::Color32::WHITE,
                    fonts,
                )
            });

            cache.main = Some(Main {
                fnid,
                galley,
                src: Rc::clone(&source_file.src),
                source_start_ofs: source_file.start_pos.0,
            });
        }

        let main = cache.main.as_ref().unwrap();

        egui::ScrollArea::both().show(ui, |ui| {
            let galley = match &cache.focus {
                Some(focus) => match &focus.preview_breakrange_index {
                    Some(ndx) => &focus.candidate_breakranges[*ndx].galley,
                    None => &main.galley,
                },
                None => &main.galley,
            };
            let res = ui.add(egui::Label::new(Arc::clone(galley)).sense(egui::Sense::click()));

            if res.double_clicked() {
                let pos = res.interact_pointer_pos().unwrap() - res.rect.min;
                let cursor_ofs: u32 = galley
                    .cursor_from_pos(pos)
                    .ccursor
                    .index
                    .try_into()
                    .unwrap();

                if cache
                    .focus
                    .as_ref()
                    .map(|f| f.cursor_ofs != cursor_ofs)
                    .unwrap_or(true)
                {
                    let candidate_breakranges = probe
                        .loader()
                        .function_breakranges(fnid)
                        .unwrap()
                        .filter_map(|(brid, br)| {
                            let lo = br.lo.0 - main.source_start_ofs;
                            let hi = br.hi.0 - main.source_start_ofs;
                            if lo <= cursor_ofs && cursor_ofs < hi {
                                Some((brid, br, lo, hi))
                            } else {
                                None
                            }
                        })
                        .map(|(brid, br, lo, hi)| {
                            let galley = ui.fonts(|fonts| {
                                make_highlight_galley(
                                    main.src.as_str(),
                                    lo as usize..hi as usize,
                                    egui::Color32::GRAY,
                                    egui::Color32::RED,
                                    fonts,
                                )
                            });
                            BreakRangeCache {
                                id: brid,
                                br: br.clone(),
                                galley,
                            }
                        })
                        .collect();

                    cache.focus = Some(Focus {
                        cursor_ofs,
                        candidate_breakranges,
                        preview_breakrange_index: None,
                    });
                }
            }

            if let Some(focus) = cache.focus.as_mut() {
                let breakranges = &focus.candidate_breakranges;

                egui::Window::new("breakranges at point").show(ui.ctx(), |ui| {
                    ui.label(format!(
                        "{} breakranges at position {}",
                        breakranges.len(),
                        focus.cursor_ofs,
                    ));
                    focus.preview_breakrange_index = None;
                    for (ndx, br_cache) in breakranges.iter().enumerate() {
                        let res = ui.button(format!("{:?}", br_cache.br.lo));

                        if res.hovered() {
                            focus.preview_breakrange_index = Some(ndx);
                        }

                        if res.clicked() {
                            my_response.set_source_bpkt = Some(br_cache.id);
                        }
                    }
                });
            }
        });

        my_response
    }

    fn make_highlight_galley(
        text: &str,
        highlight: Range<usize>,
        normal_color: egui::Color32,
        highlight_color: egui::Color32,
        fonts: &egui::epaint::text::Fonts,
    ) -> Arc<egui::Galley> {
        use egui::epaint::text::{FontFamily, FontId, LayoutJob, TextFormat};
        let mut layout_job = LayoutJob::default();

        layout_job.append(
            &text[0..highlight.start],
            0.0,
            TextFormat {
                font_id: FontId::new(14.0, FontFamily::Monospace),
                color: normal_color,
                ..Default::default()
            },
        );

        layout_job.append(
            &text[highlight.start..highlight.end],
            0.0,
            TextFormat {
                font_id: FontId::new(14.0, FontFamily::Monospace),
                color: highlight_color,
                ..Default::default()
            },
        );

        layout_job.append(
            &text[highlight.end..],
            0.0,
            TextFormat {
                font_id: FontId::new(14.0, FontFamily::Monospace),
                color: normal_color,
                ..Default::default()
            },
        );

        fonts.layout_job(layout_job)
    }

    fn fetch_source_code<'a>(
        probe: &'a Probe<'a, '_>,
        giid: bytecode::GlobalIID,
    ) -> Option<Rc<String>> {
        let fnid = giid.0;

        let loader = probe.loader();
        let source_map = loader.get_source_map(fnid.0)?;
        let mut break_ranges = loader.function_breakranges(fnid).unwrap().peekable();

        // All break ranges must belong to the same file, so we just peek one and use it to get a
        // ptr to that swc_common::SourceFile.
        // Then we use source file's offset in the source map to make sure that the markers are
        // expressed in file-local offsets.
        let (_, brange) = break_ranges.peek()?;
        let source_file = source_map.lookup_byte_offset(brange.lo).sf;

        Some(Rc::clone(&source_file.src))
    }
}

mod interpreter_manager {
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::{marker::PhantomPinned, path::Path};

    use mcjs_vm::interpreter::debugger::Probe;
    use mcjs_vm::{
        interpreter::{Exit, InterpreterError},
        Interpreter, Loader, Realm,
    };

    use anyhow::{anyhow, Result};

    #[derive(Debug)]
    pub struct Params {
        pub main_directory: Option<PathBuf>,
        pub filenames: Vec<PathBuf>,
    }

    #[derive(Debug)]
    pub enum Error<'a> {
        Generic(anyhow::Error),
        Interpreter(InterpreterError<'a>),
    }

    pub struct StandaloneInterpreter {
        realm: Realm,
        loader: Loader,
        filenames: Vec<PathBuf>,
        state: State<'static>,
        _pin: PhantomPinned,
    }
    pub enum State<'a> {
        /// Ready to process the next file in the sequence
        Ready {
            filename_ndx: usize,
        },

        /// Finished successfully. Won't proceed to any other state.
        Finished,
        Suspended(Interpreter<'a>),
        Failed(Error<'a>),
    }

    impl StandaloneInterpreter {
        pub fn new(params: Params) -> Pin<Box<StandaloneInterpreter>> {
            let mut loader = Loader::new(params.main_directory);
            let realm = Realm::new(&mut loader);
            let si = StandaloneInterpreter {
                realm,
                loader,
                filenames: params.filenames,
                state: State::Ready { filename_ndx: 0 },
                _pin: PhantomPinned,
            };

            Box::pin(si)
        }

        pub fn state_mut<'a>(self: &'a mut Pin<Box<Self>>) -> &'a mut State<'static> {
            // Safe because I only return `interpreter`, which is the
            // part of the struct that doesn't have to stay pinned
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };
            &mut self_.state
        }

        pub fn restart(self: &mut Pin<Box<Self>>) {
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };
            let prev_state = std::mem::replace(&mut self_.state, State::Finished);
            self_.state = match prev_state {
                State::Suspended(intrp) => State::Suspended(intrp.restart()),
                State::Ready { .. } | State::Finished | State::Failed(_) => {
                    State::Ready { filename_ndx: 0 }
                }
            };
        }

        pub fn resume(self: &mut Pin<Box<Self>>) {
            // Safe because nowhere in this function we move neither
            // `realm` or `loader` (`state`, and therefore the
            // contained `Interpreter` should be fine to move
            // temporarily, as long as we restore it correctly after
            // return)
            let self_ = unsafe { Pin::get_unchecked_mut(Pin::as_mut(self)) };

            let state = std::mem::replace(&mut self_.state, State::Finished);

            self_.state = match state {
                State::Ready { filename_ndx } => {
                    let filename = &self_.filenames[filename_ndx];
                    println!();
                    println!(
                        "(starting loop for file #{}: {})",
                        filename_ndx,
                        filename.to_string_lossy().into_owned()
                    );

                    match start_intrp(&filename, &mut self_.realm, &mut self_.loader) {
                        Ok(intrp) => {
                            let intrp = unsafe { std::mem::transmute(intrp) };
                            State::Suspended(intrp)
                        }
                        Err(err) => State::Failed(Error::Generic(err)),
                    }
                }
                cur @ State::Failed(_) | cur @ State::Finished => {
                    // nothing  to do
                    cur
                }
                State::Suspended(intrp) => match intrp.run() {
                    Ok(exit) => match exit {
                        Exit::Finished(_) => State::Finished,
                        Exit::Suspended(new_intrp) => State::Suspended(new_intrp),
                    },
                    Err(err) => State::Failed(Error::Interpreter(err)),
                },
            };
        }

        pub fn probe_mut<'a>(self: &'a mut Pin<Box<Self>>) -> Option<Probe<'a, 'static>> {
            // Safe because the Probe allows mutating the Interpreter
            // or accessing the Loader and/or Realm read-only (as &T)
            match self.state_mut() {
                State::Suspended(intrp) => Some(Probe::attach(intrp)),
                State::Failed(Error::Interpreter(intrp_err)) => Some(intrp_err.probe()),
                _ => None,
            }
        }

        pub fn error_message(self: &mut Pin<Box<Self>>) -> Option<String> {
            match self.state_mut() {
                State::Suspended(_) => None,
                State::Failed(Error::Interpreter(intrp_err)) => {
                    Some(format!("{:?}", intrp_err.error))
                }
                _ => None,
            }
        }
    }

    fn start_intrp<'a>(
        filename: &Path,
        realm: &'a mut Realm,
        loader: &'a mut Loader,
    ) -> Result<Interpreter<'a>> {
        let filename_str = filename.to_string_lossy().into_owned();
        let script_text = std::fs::read_to_string(filename)
            .map_err(|err| anyhow!("read error: {:?}: {:?}", filename, err))?;

        let main_fnid = loader
            .load_script(Some(filename_str.clone()), script_text)
            .map_err(|err| anyhow!("compile error: {:?}: {:?}", filename_str, err))?;

        Ok(Interpreter::new(realm, loader, main_fnid))
    }
}
