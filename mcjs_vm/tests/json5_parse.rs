extern crate mcjs_vm;

use mcjs_vm::{bytecode::LocalFnId, interpreter::debugger, FnId, Literal, ModuleId};
use std::path::PathBuf;

#[test]
fn test_load_json5_stringify() {
    test_integration_script("test_stringify.mjs");
}

#[test]
fn test_load_json5_parse() {
    test_integration_script("test_parse.mjs");
}

/// Check that the instruction breakpoint associated to a new source
/// breakpoint is deleted alongside the source breakpoint.
#[test]
fn test_remove_ibkpt_with_sbkpt() {
    // Just load *some* code
    let mut prereq = prepare_vm("test_parse.mjs");

    let fnid = prereq
        .loader
        .load_import("json5", mcjs_vm::bytecode::SCRIPT_MODULE_ID)
        .unwrap();

    let branges: Vec<_> = prereq.loader.function_breakranges(fnid).unwrap().collect();
    let brid = branges[1].0;

    let mut vm = prereq.make_vm();
    let mut probe = debugger::Probe::attach(&mut vm);
    assert_eq!(0, probe.instr_breakpoints().len());
    probe.set_source_breakpoint(brid).unwrap();
    assert_eq!(1, probe.instr_breakpoints().len());

    let was_there = probe.clear_source_breakpoint(brid).unwrap();
    assert!(was_there);
    assert_eq!(0, probe.instr_breakpoints().len());
}

/// Check that a source breakpoint is deleted automatically when its
/// corresponding instruction breakpoint is deleted
#[test]
fn test_remove_sbkpt_with_ibkpt() {
    // Just load *some* code
    let mut prereq = prepare_vm("test_parse.mjs");

    let fnid = prereq
        .loader
        .load_import("json5", mcjs_vm::bytecode::SCRIPT_MODULE_ID)
        .unwrap();

    let branges: Vec<_> = prereq.loader.function_breakranges(fnid).unwrap().collect();
    let brid = branges[1].0;

    let mut vm = prereq.make_vm();
    let mut probe = debugger::Probe::attach(&mut vm);
    assert_eq!(0, probe.source_breakpoints().len());
    assert_eq!(0, probe.instr_breakpoints().len());
    probe.set_source_breakpoint(brid).unwrap();
    assert_eq!(1, probe.source_breakpoints().len());
    assert_eq!(1, probe.instr_breakpoints().len());

    let ibkpt_giid = probe.instr_breakpoints().next().unwrap();
    let was_there = probe.clear_instr_breakpoint(ibkpt_giid);
    assert!(was_there);

    assert_eq!(0, probe.instr_breakpoints().len());
    assert_eq!(0, probe.source_breakpoints().len());
}

fn test_integration_script(filename: &str) {
    let mut prereq = prepare_vm(filename);
    let interpreter = prereq.make_vm();
    let sink = interpreter
        .run()
        .unwrap_or_else(|err| panic!("runtime error:\n{:?}", err))
        .expect_finished()
        .sink;

    assert_eq!(
        &sink,
        &[
            Some(Literal::String("null".into())),
            Some(Literal::String("123".into())),
            Some(Literal::String("456.78".into())),
            Some(Literal::String("true".into())),
            Some(Literal::String("false".into())),
        ]
    );
}

struct VMPrereq {
    loader: mcjs_vm::Loader,
    root_fnid: mcjs_vm::FnId,
    realm: mcjs_vm::Realm,
}
impl VMPrereq {
    fn make_vm(&mut self) -> mcjs_vm::Interpreter {
        mcjs_vm::Interpreter::new(&mut self.realm, &mut self.loader, self.root_fnid)
    }
}

fn prepare_vm(filename: &str) -> VMPrereq {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_path = manifest_dir.join("test-resources/test-scripts/json5/");

    let mut loader = mcjs_vm::Loader::new(Some(root_path.clone()));

    let script_path = root_path.join(filename);
    assert!(script_path.starts_with(&root_path));

    let content = std::fs::read_to_string(&script_path).expect("error while reading file");
    let root_fnid = loader
        .load_script(Some(filename.to_string()), content)
        .expect("error while compiling script");

    let realm = mcjs_vm::Realm::new(&mut loader);

    VMPrereq {
        loader,
        root_fnid,
        realm,
    }
}
