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

    let mut realm = Realm::new(&mut loader);

    let exit = Interpreter::new(&mut realm, &mut loader, main_fnid)
        .run()
        .expect("interpreter failed");

    let mut interpreter = match exit {
        Exit::Finished(_) => panic!("finished instead of interrupting"),
        Exit::Suspended {
            interpreter: intrp, ..
        } => intrp,
    };

    {
        let probe = Probe::attach(&mut interpreter);
        let bytecode::GlobalIID(fnid, _) = probe.giid();
        let bytecode::FnId(mod_id, _) = fnid;
        assert_eq!(mod_id, bytecode::SCRIPT_MODULE_ID);

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

    let main_fnid = loader.load_import("./breakme-0.js", None).unwrap();

    let module_id = main_fnid.0;
    assert_ne!(module_id, bytecode::SCRIPT_MODULE_ID);

    // Hardcoded. Must be updated if breakme-0.js changes
    let pos = swc_common::BytePos(166);

    let mut realm = Realm::new(&mut loader);

    // Resolve into the (potentially multiple) GIIDs
    let break_range_ids: Vec<_> = loader
        .resolve_break_loc(module_id, pos)
        .unwrap()
        .into_iter()
        .map(|(brid, _)| brid)
        .collect();

    for brid in break_range_ids {
        let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);

        let mut probe = Probe::attach(&mut interpreter);
        probe.set_source_breakpoint(brid).unwrap();

        interpreter = match interpreter.run().expect("interpreter failed") {
            Exit::Finished(_) => panic!("interpreter finished instead of breaking"),
            Exit::Suspended {
                interpreter: intrp, ..
            } => intrp,
        };

        let probe = Probe::attach(&mut interpreter);

        let giid = probe.giid();
        eprintln!("we are at: {:?}; sink = {:?}", giid, probe.sink());
        if giid.0 .1 == bytecode::LocalFnId(2) {
            assert_eq!(probe.sink(), &[Value::Number(1.0)]);
        } else {
            assert_eq!(probe.sink(), &[]);
        }

        let finish_data = interpreter
            .run()
            .expect("interpreter failed")
            .expect_finished();
        assert_eq!(
            &finish_data.sink,
            &[
                Some(bytecode::Literal::Number(1.0)),
                Some(bytecode::Literal::Number(2.0)),
            ]
        );
    }
}
