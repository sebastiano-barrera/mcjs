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
    const SOURCE_CODE: &'static str = "
function foo() {
    sink(1);
    sink(2);
}

foo();
        ";

    let mut loader = mcjs_vm::Loader::new(None);
    let main_fnid = loader
        .load_script(Some("foo.js".to_string()), SOURCE_CODE.to_string())
        .unwrap();
    
    let module_id = main_fnid.0;
    assert_eq!(module_id, bytecode::SCRIPT_MODULE_ID);

    let (pos, _) = SOURCE_CODE.match_indices("sink").nth(1).unwrap();
    let pos = swc_common::BytePos(pos as u32);
    let break_ranges = loader.resolve_loc(module_id, pos).unwrap();
    assert_eq!(break_ranges.len(), 1);
    let break_range = break_ranges[0].clone();

    let mut realm = Realm::new();

    let mut interpreter = Interpreter::new(&mut realm, &mut loader, main_fnid);

    let mut probe = Probe::attach(&mut interpreter);
    let break_giid = bytecode::GlobalIID(
        bytecode::FnId(module_id, break_range.local_fnid),
        break_range.iid,
    );
    probe.set_breakpoint(break_giid).unwrap();

    let exit = interpreter.run().expect("interpreter failed");

    let mut interpreter = match exit {
        Exit::Finished(_) => panic!("finished instead of interrupting"),
        Exit::Suspended(intrp) => intrp,
    };

    let probe = Probe::attach(&mut interpreter);
    let position: Position = probe.position();
    assert_eq!(break_giid, bytecode::GlobalIID(position.fnid, position.iid));
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
