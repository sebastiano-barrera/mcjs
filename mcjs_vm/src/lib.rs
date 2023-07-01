// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

pub mod bytecode;
mod fs;
mod bytecode_compiler;
mod common;
mod interpreter;
// mod jit;
mod heap;
mod util;

#[cfg(not(feature = "inspection"))]
mod stack_access;
#[cfg(feature = "inspection")]
pub mod stack_access;

#[cfg(not(feature = "inspection"))]
mod stack;
#[cfg(feature = "inspection")]
pub mod stack;

pub use bytecode::{Codebase, ModuleId, Literal, IID, FnId, GlobalIID};
pub use bytecode_compiler::{BuilderParams, Loader};
pub use fs::{FileLoader, MockLoader, CombinedLoader};
pub use interpreter::{Interpreter, Value as InterpreterValue};
#[cfg(feature = "inspection")]
pub use interpreter::{InspectorStep, InspectorAction, CoreDump};

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
