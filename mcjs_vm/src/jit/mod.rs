use std::collections::HashMap;

use crate::interpreter;

mod builder;
mod codegen;
mod regalloc;
mod tracking;

pub use builder::{CloseMode, InterpreterStep, TraceBuilder};
// TODO(cleanup) Move some of these from `builder` to this module?
pub use codegen::NativeThunk;

use self::builder::{Instr, SnapshotMap, SnapshotUpdate, ValueId};

type BoxedValue = interpreter::Value;

// A JIT trace, in SSA representation.
pub struct Trace {
    hreg_alloc: regalloc::Allocation,
    snapshot_map: SnapshotMap,
    snapshot_final_update: SnapshotUpdate,
    instrs: Vec<Instr>,
    is_loop: bool,
    // phis: std::collections::HashMap<ValueId, ValueId>,
    is_enabled: Vec<bool>,
}

impl Trace {
    pub fn dump(&self) {
        use std::borrow::Cow;

        eprintln!(" === trace");
        eprintln!(" snapshot: {:?}", self.snapshot_map);

        let pr_inst = |vid: ValueId, instr: &Instr| {
            let exp_type: Cow<'static, _> = match instr.result_type() {
                Some(ty) => format!("{:?}", ty).into(),
                None => "--".into(),
            };

            let ndx = vid.0 as usize;
            eprintln!(
                " {} {:4} {:6} {:?}",
                if self.is_enabled[ndx] { ' ' } else { 'X' },
                vid.0,
                exp_type,
                instr
            );
        };

        for (ndx, instr) in self.instrs.iter().enumerate() {
            pr_inst(ValueId(ndx as u32), instr);
        }
        eprintln!(
            "              final snapshot update: {:?}",
            self.snapshot_final_update
        );
    }

    pub(crate) fn compile(&self) -> NativeThunk {
        codegen::to_native(self)
    }

    fn get_instr(&self, vid: ValueId) -> Option<&Instr> {
        self.instrs.get(vid.0 as usize)
    }

    #[allow(clippy::needless_lifetimes)]
    fn iter_vids<'a>(&'a self) -> impl 'a + Iterator<Item = ValueId> {
        (0..self.instrs.len())
            .filter(|ndx| self.is_enabled[*ndx])
            .map(|ndx| ValueId(ndx as u32))
    }
    fn iter_instrs(&self) -> impl Iterator<Item = (ValueId, &Instr)> {
        self.iter_vids()
            .map(|vid| (vid, self.instrs.get(vid.0 as usize).unwrap()))
    }

    #[must_use]
    pub(crate) fn snapshot_map(&self) -> &SnapshotMap {
        &self.snapshot_map
    }
}
