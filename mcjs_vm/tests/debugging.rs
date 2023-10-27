#![cfg(feature = "debugger")]

use std::path::{Path, PathBuf};

use mcjs_vm::bytecode;
use mcjs_vm::interpreter::debugger::{Position, Probe};
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
        let position: Position = probe.position();
        assert_eq!(
            position.fnid,
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
    let break_ranges: Vec<_> = loader.resolve_break_loc(module_id, pos).unwrap()
        .into_iter()
        .cloned()
        .collect();

    for break_range in break_ranges {
        let mut realm = Realm::new();
        let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);

        let mut probe = Probe::attach(&mut interpreter);
        // TODO Damn this is rough
        let break_giid = bytecode::GlobalIID(
            bytecode::FnId(module_id, break_range.local_fnid),
            break_range.iid,
        );

        if break_giid.0.1 != bytecode::LocalFnId(2) {
            continue
        }
        probe.set_breakpoint(break_giid);
        eprintln!("breakpoint set at {}-{} -> {:?}", break_range.lo.0, break_range.hi.0, break_giid);

        let exit = interpreter.run().expect("interpreter failed");

        let mut interpreter = match exit {
            Exit::Finished(_) => panic!("finished instead of interrupting"),
            Exit::Suspended(intrp) => intrp,
        };

        let probe = Probe::attach(&mut interpreter);
        let position: Position = probe.position();
        // Check: the interpreter's position now points at the position that the
        // interpreter will *resume* at, which is 1 + where it's currently suspended.
        assert_eq!(break_giid, bytecode::GlobalIID(position.fnid, bytecode::IID(position.iid.0 - 1)));
        assert_eq!(probe.sink(), &[Value::Number(1.0)]);

        let finish_data = interpreter.run().unwrap().expect_finished();
        assert_eq!(
            &finish_data.sink,
            &[
                Some(bytecode::Literal::Number(1.0)),
                Some(bytecode::Literal::Number(2.0)),
            ]
        );
    }
}
