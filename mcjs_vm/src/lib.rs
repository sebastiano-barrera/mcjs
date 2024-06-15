pub mod bytecode;
mod bytecode_compiler;
mod common;
mod loader;
// mod jit;
pub mod heap;
pub mod interpreter;

mod tracing;
mod util;

// TODO Refine the set of re-exported things?
pub use bytecode::{FnID, GlobalIID, Literal, IID};
pub use common::Error;
pub use interpreter::{
    Closure,
    Interpreter,
    JSClosure,
    // The following types are re-exported to allow mcjs clients to define
    // NativeFunction's
    NativeClosure,
    NativeFn,
    Realm,
    RunError,
    RunResult,
    Value as InterpreterValue,
};
pub use loader::{BreakRangeID, FunctionLookup, Loader};
