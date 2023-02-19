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
    pub(crate) n_loop_invariant: usize,
}

impl Trace {
    pub fn dump(&self) {
        use std::borrow::Cow;

        let is_enabled = self.enabled_mask();

        eprintln!(" === trace");
        eprintln!(" snapshot: {:?}", self.snapshot_map);

        let skip_disabled = false;

        for (ndx, instr) in self.instrs.iter().enumerate() {
            if skip_disabled && !is_enabled[ndx] {
                continue;
            }

            if ndx == self.n_loop_invariant {
                eprintln!("  ----- loop");
            }

            let enb_prefix = if is_enabled[ndx] { "    " } else { "OFF " };

            let exp_type: Cow<'static, _> = match instr.result_type() {
                Some(ty) => format!("{:?}", ty).into(),
                None => "--".into(),
            };

            let hreg = self.hreg_alloc.hreg_of_instr(ndx);
            let hreg = hreg
                .map(|x| Cow::Owned(format!("{:?}", x)))
                .unwrap_or_else(|| Cow::Borrowed("???"));
            eprintln!(
                " {}{:4?} {:5} {:6} {:?}",
                enb_prefix, ndx, hreg, exp_type, instr
            );
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

    fn enabled_mask(&self) -> Box<[bool]> {
        self.instrs
            .iter()
            .enumerate()
            .map(|(ndx, instr)| {
                self.hreg_alloc.hreg_of_instr(ndx).is_some() || instr.has_side_effects()
            })
            .collect()
    }
}
