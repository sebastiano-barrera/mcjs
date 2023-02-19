use std::collections::HashMap;

use crate::{interpreter, regalloc};

mod builder;
mod codegen;
mod loop_hoist;
mod tracking;

use builder::{BoxedValue, Instr, TraceParam};
// TODO Move some of these from `builder` to this module?
pub use builder::{InterpreterStep, TraceBuilder, TraceStart};
pub use codegen::NativeThunk;

use self::builder::ValueId;

// A JIT trace, in SSA representation.
pub struct Trace {
    instrs: Vec<Instr>,
    hreg_alloc: regalloc::Allocation,
    snapshot_map: Vec<interpreter::Operand>,
    is_loop: bool,
    pub(crate) phis: std::collections::HashMap<ValueId, ValueId>,
    pub(crate) loop_head_vid: ValueId,
    order: Vec<ValueId>,
}

impl Trace {
    pub fn dump(&self) {
        use std::borrow::Cow;

        eprintln!(" === trace");
        eprintln!(" snapshot: {:?}", self.snapshot_map);

        for (vid, instr) in self.iter_instrs() {
            if vid == self.loop_head_vid {
                eprintln!("  ----- loop");
            }

            let exp_type: Cow<'static, _> = match instr.result_type() {
                Some(ty) => format!("{:?}", ty).into(),
                None => "--".into(),
            };

            let hreg = self.hreg_alloc.hreg_of_instr(vid.clone().into());
            let hreg = hreg
                .map(|x| Cow::Owned(format!("{:?}", x)))
                .unwrap_or_else(|| Cow::Borrowed("???"));
            eprintln!("v{:<4} {:5} {:6} {:?}", vid.0, hreg, exp_type, instr);
        }

        eprintln!("      phis [");
        for (old, new) in self.phis.iter() {
            eprintln!("           {:?} <- {:?}", old, new);
        }
        eprintln!("      ]");
    }

    pub(crate) fn compile(&self) -> NativeThunk {
        codegen::to_native(self)
    }

    fn get_instr(&self, vid: ValueId) -> Option<&Instr> {
        self.instrs.get(vid.0 as usize)
    }

    fn iter_instrs(&self) -> impl Iterator<Item = (ValueId, &Instr)> {
        self.order.iter().map(|vid| {
            let instr = self.instrs.get(vid.0 as usize).unwrap();
            (*vid, instr)
        })
    }
}
