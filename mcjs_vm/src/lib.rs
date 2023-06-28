// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

mod bytecode;
mod fs;
mod bytecode_compiler;
mod common;
mod interpreter;
// mod jit;
mod stack;
mod heap;
mod util;

pub use bytecode::{Codebase, ModuleId, Literal};
pub use bytecode_compiler::{BuilderParams, Loader};
pub use fs::{FileLoader, MockLoader, CombinedLoader};
pub use interpreter::{Interpreter, Value as InterpreterValue};

mod stack_access;

// TODO Compile this module and build/link serde only for test builds
pub mod inspector_case;

// pub use interpreter::{InterpreterFlags, JitMode, TracerFlags, VM};
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     use interpreter::{Value, VM};
//
//     struct Output {
//         sink: Vec<Value>,
//     }
//
//     fn quick_run(code: &str) -> Result<Output, interpreter::Error> {
//         let mut vm = VM::new();
//         vm.run_script(code.to_string(), Default::default())?;
//         Ok(Output {
//             sink: vm.take_sink(),
//         })
//     }
// }
