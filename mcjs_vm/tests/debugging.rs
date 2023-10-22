use mcjs_vm::bytecode;
use mcjs_vm::interpreter::debugger::{Position, Probe};
use mcjs_vm::interpreter::{Exit, Interpreter, Realm, Value};

#[test]
fn test_simple_breakpoint() {
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
