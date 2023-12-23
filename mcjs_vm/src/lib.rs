// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

pub mod bytecode;
mod bytecode_compiler;
mod common;
mod loader;
// mod jit;
mod heap;
pub mod interpreter;
pub mod stack;

mod util;

// TODO Refine the set of re-exported things?
pub use bytecode::{FnId, GlobalIID, Literal, ModuleId, IID, SCRIPT_MODULE_ID};
pub use interpreter::{Interpreter, Realm, Value as InterpreterValue};
pub use loader::Loader;
pub use common::Error;
