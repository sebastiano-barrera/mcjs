#![cfg(feature = "debugger")]

extern crate mcjs_vm;

use mcjs_vm::interpreter::debugger;
use std::path::PathBuf;

#[test]
fn test_load_json5_stringify() {
    test_integration_script("./test_stringify.mjs");
}

#[test]
fn test_load_json5_parse() {
    test_integration_script("./test_parse.mjs");
}

/// Check that the instruction breakpoint associated to a new source
/// breakpoint is deleted alongside the source breakpoint.
#[test]
fn test_remove_ibkpt_with_sbkpt() {
    // Just load *some* code
    let mut prereq = prepare_vm("./test_parse.mjs");
    let base_path = prereq.loader.base_path().to_owned();

    let fnid = prereq
        .loader
        .load_import_from_dir("json5", &base_path)
        .unwrap();

    let branges: Vec<_> = prereq.loader.function_breakranges(fnid).unwrap().collect();
    let brid = branges[1].0;

    assert_eq!(0, prereq.dbg.instr_bkpts().len());
    prereq
        .dbg
        .set_source_breakpoint(brid, &prereq.loader)
        .unwrap();
    assert_eq!(1, prereq.dbg.instr_bkpts().len());

    let was_there = prereq
        .dbg
        .clear_source_breakpoint(brid, &prereq.loader)
        .unwrap();
    assert!(was_there);
    assert_eq!(0, prereq.dbg.instr_bkpts().len());
}

/// Check that a source breakpoint is deleted automatically when its
/// corresponding instruction breakpoint is deleted
#[test]
fn test_remove_sbkpt_with_ibkpt() {
    // Just load *some* code
    let mut prereq = prepare_vm("./test_parse.mjs");
    let base_path = prereq.loader.base_path().to_owned();

    let fnid = prereq
        .loader
        .load_import_from_dir("json5", &base_path)
        .unwrap();

    let branges: Vec<_> = prereq.loader.function_breakranges(fnid).unwrap().collect();
    let brid = branges[1].0;

    assert_eq!(0, prereq.dbg.source_breakpoints().len());
    assert_eq!(0, prereq.dbg.instr_bkpts().len());
    prereq
        .dbg
        .set_source_breakpoint(brid, &prereq.loader)
        .unwrap();
    assert_eq!(1, prereq.dbg.source_breakpoints().len());
    assert_eq!(1, prereq.dbg.instr_bkpts().len());

    let ibkpt_giid = prereq.dbg.instr_bkpts().next().unwrap();
    let was_there = prereq.dbg.clear_instr_bkpt(ibkpt_giid);
    assert!(was_there);

    assert_eq!(0, prereq.dbg.instr_bkpts().len());
    assert_eq!(0, prereq.dbg.source_breakpoints().len());
}

fn test_integration_script(filename: &str) {
    use mcjs_vm::heap::IndexOrKey;

    let mut prereq = prepare_vm(filename);

    let interpreter = prereq.make_vm();
    interpreter
        .run()
        .unwrap_or_else(|err| panic!("runtime error:\n{:?}", err))
        .expect_finished();

    let g = prereq.realm.global_obj();
    let heap = prereq.realm.heap_mut();

    let json5_encoded = heap
        .get_chained(g, IndexOrKey::Key("json5Encoded"))
        .unwrap()
        .value()
        .unwrap();

    let elms = heap
        .array_elements(json5_encoded)
        .expect("expected an array");
    let strings: Vec<_> = elms
        .iter()
        .map(|val| heap.as_str(*val).unwrap().to_string())
        .collect();

    assert_eq!(&strings, &["null", "123", "456.78", "true", "false"]);
}

struct VMPrereq {
    loader: mcjs_vm::Loader,
    root_fnid: mcjs_vm::FnID,
    realm: mcjs_vm::Realm,
    dbg: debugger::DebuggingState,
}
impl VMPrereq {
    fn make_vm(&mut self) -> mcjs_vm::Interpreter {
        mcjs_vm::Interpreter::new(&mut self.realm, &mut self.loader, self.root_fnid)
    }
}

fn prepare_vm(filename: &str) -> VMPrereq {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_path = manifest_dir.join("test-resources/json5/");

    let mut loader = mcjs_vm::Loader::new(root_path.clone());
    let root_fnid = loader
        .load_import_from_dir(filename, &root_path)
        .expect("error while compiling script");

    let realm = mcjs_vm::Realm::new(&mut loader);

    let dbg = debugger::DebuggingState::new();

    VMPrereq {
        loader,
        root_fnid,
        realm,
        dbg,
    }
}
