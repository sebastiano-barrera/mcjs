// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

pub mod bytecode;
mod bytecode_compiler;
mod common;
mod fs;
// mod jit;
mod util;

// TODO is there a better way to set visibility based on the presence of a feature?
#[cfg(not(feature = "inspection"))]
mod stack_access;
#[cfg(feature = "inspection")]
pub mod stack_access;

#[cfg(not(feature = "inspection"))]
mod stack;
#[cfg(feature = "inspection")]
pub mod stack;

#[cfg(not(feature = "inspection"))]
mod interpreter;
#[cfg(feature = "inspection")]
pub mod interpreter;

#[cfg(not(feature = "inspection"))]
mod heap;
#[cfg(feature = "inspection")]
pub mod heap;

pub use bytecode::{Codebase, FnId, GlobalIID, Literal, ModuleId, IID};
pub use bytecode_compiler::{BuilderParams, Loader};
pub use fs::{CombinedLoader, FileLoader, MockLoader};
#[cfg(feature = "inspection")]
pub use interpreter::{CoreDump, InspectorAction, InspectorStep};
pub use interpreter::{Interpreter, Value as InterpreterValue};

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
