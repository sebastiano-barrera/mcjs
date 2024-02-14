pub mod bytecode;
mod bytecode_compiler;
mod common;
mod loader;
// mod jit;
mod heap;
pub mod interpreter;
pub mod stack;

mod tracing;
mod util;

// TODO Refine the set of re-exported things?
pub use bytecode::{FnId, GlobalIID, Literal, ModuleId, IID, SCRIPT_MODULE_ID};
pub use common::Error;
pub use interpreter::{Interpreter, Options, Realm, Value as InterpreterValue};
pub use loader::{BreakRangeID, FunctionLookup, Loader};
#[cfg(feature = "debugger")]
pub use stack::SlotDebug;
