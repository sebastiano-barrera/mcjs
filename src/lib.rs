// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

mod bytecode_compiler;
mod common;
mod interpreter;
mod jit;
mod stack;
mod util;

pub use interpreter::{InterpreterFlags, JitMode, TracerFlags, VM};

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
}
