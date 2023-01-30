// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

mod bytecode_compiler;
mod common;
mod interpreter;
mod jit;

#[cfg(test)]
mod tests {
    use super::*;

    use interpreter::{Value, VM};

    fn quick_compile(code: &str) -> interpreter::Module {
        let module =
            bytecode_compiler::compile_file("input.js".to_string(), code.to_string()).unwrap();
        module.dump();
        module
    }

    // For the future, for loading code from `test-resources/`:
    // let manifest_dir =
    //     PathBuf::from_str(env!("CARGO_MANIFEST_DIR")).expect("CARGO_MANIFEST_PATH undefined!");
    // let include_path = manifest_dir.join("test-resources").join("test-module");

    #[test]
    fn test_simple_call() {
        let module = quick_compile("/* Here is some simple code: */ sink(1 + 4 + 99); ");

        // 0   const 1
        // 1   const 4
        // 2   add v0, v1
        // 3   push_sink v2

        let sink = interpreter::interpret(&module).unwrap().sink;
        assert_eq!(&[Value::Number(104.0)], &sink[..]);
    }

    #[test]
    fn test_multiple_calls() {
        let module =
            quick_compile("/* Here is some simple code: */ sink(12 * 5);  sink(99 - 15); ");
        let sink = interpreter::interpret(&module).unwrap().sink;
        assert_eq!(
            &[Value::Number(12. * 5.), Value::Number(99. - 15.)],
            &sink[..]
        );
    }

    #[test]
    fn test_if() {
        let module = quick_compile(
            "
            const x = 123;
            let y = 'a';
            if (x < 200) {
                y = 'b';
            }
            sink(y);
            ",
        );

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        let sink = interpreter::interpret(&module).unwrap().sink;
        let val: Value = "b".to_string().into();
        assert_eq!(&[val], &sink[..]);
    }

    #[test]
    fn test_simple_fn() {
        let module = quick_compile(
            "
            function foo(a, b) { return a + b; }
            sink(foo(1, 2));
            ",
        );

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        let sink = interpreter::interpret(&module).unwrap().sink;
        assert_eq!(&[Value::Number(3.0)], &sink[..]);

        // assert!(false);
    }

    #[test]
    fn test_simple_trace() {
        let module = quick_compile(
            "
            const x = 123;
            let y = 'a';
            if (x < 256) {
                y = 'b';
            }
            sink(y);
            ",
        );

        // 0    const 123
        // 1    const 'a'
        // 2    cmp v0 < v1
        // 3    jmpif v2 -> #5
        // 4    set v2 <- 'b'
        // 5    push_sink v2

        let output = interpreter::interpret_and_trace(&module).unwrap();
        eprint!("trace = ");
        let trace = output.trace.unwrap();
        trace.dump();

        let thunk = trace.compile();
        assert_eq!(123, thunk.run());

        let sink = &output.sink;
        let val: Value = "b".to_string().into();
        assert_eq!(&[val], &sink[..]);
    }

    #[test]
    fn test_while() {
        let module = quick_compile(
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
            
            sink(sum_range(8));
            ",
        );

        let output = interpreter::interpret_and_trace(&module).unwrap();
        assert_eq!(&[Value::Number(28.0)], &output.sink[..]);

        assert!(output.trace.is_some());
    }

    #[test]
    fn test_uncastable() {
        let module = quick_compile("sink(!'some string');");

        let output = interpreter::interpret_and_trace(&module).unwrap();
        assert!(output.trace.is_none());
    }
}
