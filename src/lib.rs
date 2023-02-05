// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

mod bytecode_compiler;
mod common;
mod interpreter;
mod jit;
mod regalloc;

pub use interpreter::{InterpreterFlags, TracerFlags, VM};

#[cfg(test)]
mod tests {
    use super::*;

    use interpreter::{Value, VM};

    struct Output {
        sink: Vec<Value>,
    }

    fn quick_run(code: &str) -> Result<Output, interpreter::Error> {
        let mut vm = VM::new();
        vm.run_script(code.to_string(), Default::default())?;
        Ok(Output {
            sink: vm.take_sink(),
        })
    }

    // For the future, for loading code from `test-resources/`:
    // let manifest_dir =
    //     PathBuf::from_str(env!("CARGO_MANIFEST_DIR")).expect("CARGO_MANIFEST_PATH undefined!");
    // let include_path = manifest_dir.join("test-resources").join("test-module");

    #[test]
    fn test_simple_call() {
        let output = quick_run("/* Here is some simple code: */ sink(1 + 4 + 99); ").unwrap();
        assert_eq!(&[Value::Number(104.0)], &output.sink[..]);
    }

    #[test]
    fn test_multiple_calls() {
        let output =
            quick_run("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ").unwrap();
        assert_eq!(
            &[Value::Number(12. * 5.), Value::Number(99. - 15.)],
            &output.sink[..]
        );
    }

    #[test]
    fn test_if() {
        let output = quick_run(
            "
            const x = 123;
            let y = 'a';
            if (x < 200) {
                y = 'b';
            }
            sink(y);
            ",
        )
        .unwrap();

        let val: Value = "b".to_string().into();
        assert_eq!(&[val], &output.sink[..]);
    }

    #[test]
    fn test_simple_fn() {
        let output = quick_run(
            "
            function foo(a, b) { return a + b; }
            sink(foo(1, 2));
            ",
        )
        .unwrap();

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        assert_eq!(&[Value::Number(3.0)], &output.sink[..]);
    }

    #[ignore]
    #[test]
    fn test_fn_with_branch() {
        let output = quick_run(
            "
            function foo(mode, a, b) { 
                if (mode === 'sum')
                    return a + b;
                else if (mode === 'product')
                    return a * b;
                else
                    return 'something else';
            }

            sink(foo('product', 9, 8));
            ",
        )
        .unwrap();

        // eprint!("trace = ");
        // let trace = output.trace.unwrap();
        // trace.dump();

        assert!(false);
    }

    #[ignore]
    #[test]
    fn test_while() {
        let output = quick_run(
            "
            function sum_range(n) {
                let i = 0;
                let ret = 0;
                while (i <= n) {
                    ret += i;
                    i++;
                }
                return ret;
            }
            
            sink(sum_range(2));
            ",
        )
        .unwrap();

        // let trace = output.trace.as_ref().unwrap();
        assert!(false);
    }

    #[ignore]
    #[test]
    fn test_uncastable() {
        let output = quick_run("sink(!'some string');").unwrap();

        // let output = interpreter::interpret_and_trace(&module, 0).unwrap();
        // assert!(output.trace.is_none());
    }
}
