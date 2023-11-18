#![cfg(feature = "debugger")]

use std::path::PathBuf;

use mcjs_vm::bytecode;
use mcjs_vm::interpreter::debugger::Probe;
use mcjs_vm::interpreter::{Exit, Interpreter, Realm, Value};

#[test]
fn test_inline_breakpoint() {
    const SOURCE_CODE: &'static str = "
function foo() {
    sink(1);
    debugger;
    sink(2);
}

foo();
        ";

    let mut loader = mcjs_vm::Loader::new(None);
    let main_fnid = loader
        .load_script(Some("foo.js".to_string()), SOURCE_CODE.to_string())
        .unwrap();

    let mut realm = Realm::new();

    let exit = Interpreter::new(&mut realm, &mut loader, main_fnid)
        .run()
        .expect("interpreter failed");

    let mut interpreter = match exit {
        Exit::Finished(_) => panic!("finished instead of interrupting"),
        Exit::Suspended(intrp) => intrp,
    };

    {
        let probe = Probe::attach(&mut interpreter);
        let bytecode::GlobalIID(fnid, _) = probe.giid();
        assert_eq!(
            fnid,
            bytecode::FnId(bytecode::SCRIPT_MODULE_ID, bytecode::LocalFnId(2))
        );

        assert_eq!(probe.sink(), &[Value::Number(1.0)]);

        drop(probe);
    }

    let finish_data = interpreter.run().unwrap().expect_finished();
    assert_eq!(
        &finish_data.sink,
        &[
            Some(bytecode::Literal::Number(1.0)),
            Some(bytecode::Literal::Number(2.0)),
        ]
    );
}

#[test]
fn test_pos_breakpoint() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let base_path = manifest_dir.join("test-resources/test-scripts/debugging/");

    // Simulate `import ... from 'test_pkg'` from a script
    let mut loader = mcjs_vm::Loader::new(Some(base_path));

    let main_fnid = loader
        .load_import("./breakme-0.js", bytecode::SCRIPT_MODULE_ID)
        .unwrap();

    let module_id = main_fnid.0;
    assert_ne!(module_id, bytecode::SCRIPT_MODULE_ID);

    // Hardcoded. Must be updated if breakme-0.js changes
    let pos = swc_common::BytePos(166);

    let mut realm = Realm::new();
    let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);

    let mut probe = Probe::attach(&mut interpreter);

    // Resolve byte offset into line:col
    let loc = {
        let source_map = intrp.loader.get_source_map(module_id).ok_or_else(|| {
            // TODO Report a filename and line:col
            error!(
                "can't set a breakpoint at {:?}:byte#{}: no source map for file",
                mod_id, pos.0
            )
        })?;
        let loc = source_map.lookup_char_pos(pos);
        swc_common::LineCol {
            line: loc.line.try_into().unwrap(),
            col: loc.col_display.try_into().unwrap(),
        }
    };

    // Resolve into the (potentially multiple) GIIDs
    let break_ranges = intrp.loader.resolve_break_loc(module_id, pos)?.into_iter();

    for break_range in break_ranges {
        let bpid = probe.set_source_breakpoint(module_id, pos).unwrap();

        let bp = probe.breakpoint(bpid).unwrap();
        eprintln!("breakpoint set at {}:{}", bp.loc.line, bp.loc.col);

        let mut bp_hit = false;
        let finish_data = loop {
            let exit = interpreter.run().expect("interpreter failed");
            interpreter = match exit {
                Exit::Finished(finish_data) => break finish_data,
                Exit::Suspended(intrp) => intrp,
            };

            bp_hit = true;

            let probe = Probe::attach(&mut interpreter);

            // let mcjs_vm::GlobalIID(fnid, iid) = probe.giid();
            // TODO Check that the interpreter is actually at one of the breakpoint's IIDs.
            // Check: the interpreter's position now points at the position that the
            // interpreter will *resume* at, which is 1 + where it's currently suspended.

            let giid = probe.giid();
            eprintln!("we are at: {:?}; sink = {:?}", giid, probe.sink());
            if giid.0 .1 == bytecode::LocalFnId(2) {
                assert_eq!(probe.sink(), &[Value::Number(1.0)]);
            } else {
                assert_eq!(probe.sink(), &[]);
            }
        };

        assert!(bp_hit);
        assert_eq!(
            &finish_data.sink,
            &[
                Some(bytecode::Literal::Number(1.0)),
                Some(bytecode::Literal::Number(2.0)),
            ]
        );
    }
}
