use dynasm::dynasm;

use super::{BoxedValue, Trace};

pub struct NativeThunk {
    buf: dynasmrt::ExecutableBuffer,
    entry_offset: dynasmrt::AssemblyOffset,
    sink: Vec<BoxedValue>,
}

impl NativeThunk {
    pub(crate) fn run(&self) -> u64 {
        let ptr = self.buf.ptr(self.entry_offset);
        let thunk: extern "C" fn() -> u64 = unsafe { std::mem::transmute(ptr) };
        thunk()
    }
}

pub(super) fn to_native(trace: &Trace) -> NativeThunk {
    use dynasmrt::{DynasmApi, DynasmLabelApi};
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();

    dynasm!(asm
    ; .arch x64
    ; entry:
    ; mov eax, 123
    ; ret
    );

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        sink: Vec::new(),
    }
}
