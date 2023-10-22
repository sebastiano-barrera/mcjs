// Temporary; delete these directives for cleanup
#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused_imports)]

pub mod bytecode;
mod bytecode_compiler;
mod common;
mod loader;
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

pub mod interpreter;

#[cfg(not(feature = "inspection"))]
mod heap;
#[cfg(feature = "inspection")]
pub mod heap;

pub use bytecode::{FnId, GlobalIID, Literal, ModuleId, IID};
pub use interpreter::{Interpreter, Realm, Value as InterpreterValue};
pub use loader::Loader;

#[cfg(feature = "inspection")]
pub use bytecode_compiler::{CompiledChunk, SourceMap};

// TODO Compile this module and build/link serde only for test builds
pub mod inspector_case;
