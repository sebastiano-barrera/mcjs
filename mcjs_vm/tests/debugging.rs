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
    let bpid = probe.set_breakpoint(module_id, pos).unwrap();

    let bp = probe.breakpoint(bpid).unwrap();
    eprintln!("breakpoint set at {}:{}", bp.loc.line, bp.loc.col);

    let exit = interpreter.run().expect("interpreter failed");

    let mut bp_hit = false;
    let finish_data = loop {
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

        assert_eq!(probe.sink(), &[Value::Number(1.0)]);
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
