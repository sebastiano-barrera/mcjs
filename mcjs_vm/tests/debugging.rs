#![cfg(feature = "debugger")]

use std::path::PathBuf;

use mcjs_vm::bytecode;
use mcjs_vm::interpreter::{debugger, Exit, Interpreter, Realm, Value};

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

    let mut loader = mcjs_vm::Loader::new_cwd();
    let main_fnid = loader
        .load_script(Some(PathBuf::from("foo.js")), SOURCE_CODE.to_string())
        .unwrap();

    let mut realm = Realm::new(&mut loader);

    let exit = Interpreter::new(&mut realm, &mut loader, main_fnid)
        .run()
        .expect("interpreter failed");

    let intrp_state = match exit {
        Exit::Finished(_) => panic!("finished instead of interrupting"),
        Exit::Suspended { intrp_state, .. } => intrp_state,
    };

    assert_eq!(&intrp_state.sink, &[Value::Number(1.0)]);

    let finish_data = Interpreter::resume(&mut realm, &mut loader, intrp_state)
        .run()
        .unwrap()
        .expect_finished();
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
    let mut loader = mcjs_vm::Loader::new(base_path);

    let main_fnid = loader.load_import("./breakme-0.js", None).unwrap();

    let module_id = main_fnid.0;

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

    let mut dbg = debugger::DebuggingState::new();

    for brid in break_range_ids {
        dbg.set_source_breakpoint(brid, &loader).unwrap();

        let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);
        interpreter.set_debugging_state(&mut dbg);

        let intrp_state = match interpreter.run().expect("interpreter failed") {
            Exit::Finished(_) => panic!("interpreter finished instead of breaking"),
            Exit::Suspended { intrp_state, .. } => intrp_state,
        };

        let giid = debugger::giid(&intrp_state);
        eprintln!("we are at: {:?}; sink = {:?}", giid, intrp_state.sink);
        if giid.0 .1 == bytecode::LocalFnId(2) {
            assert_eq!(&intrp_state.sink, &[Value::Number(1.0)]);
        } else {
            assert_eq!(&intrp_state.sink, &[]);
        }

        let finish_data = Interpreter::resume(&mut realm, &mut loader, intrp_state)
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
