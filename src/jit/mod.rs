use std::collections::HashMap;

use crate::{interpreter, regalloc};

mod builder;
mod codegen;
mod loop_hoist;
mod tracking;

pub use builder::{InterpreterStep, TraceBuilder, CloseMode};
// TODO Move some of these from `builder` to this module?
pub use codegen::NativeThunk;

use self::builder::{Instr, ValueId};

type BoxedValue = interpreter::Value;

// A JIT trace, in SSA representation.
pub struct Trace {
    hreg_alloc: regalloc::Allocation,
    snapshot_map: Vec<interpreter::Operand>,
    instrs: Vec<Instr>,
    is_loop: bool,
    // phis: std::collections::HashMap<ValueId, ValueId>,
    is_loop_variant: Vec<bool>,
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

            let hreg_opt = self.hreg_alloc.hreg_of_instr(vid.clone().into());
            let hreg: Cow<'static, str> = if let Some(hreg) = hreg_opt {
                format!("{:?}", hreg).into()
            } else {
                "---".into()
            };

            let ndx = vid.0 as usize;
            eprintln!(
                " {}{} {:4} {:5} {:6} {:?}",
                if self.is_enabled[ndx] { ' ' } else { 'X' },
                if self.is_loop_variant[ndx] { ' ' } else { 'H' },
                vid.0,
                hreg,
                exp_type,
                instr
            );
        };

        for (ndx, instr) in self.instrs.iter().enumerate() {
            pr_inst(ValueId(ndx as u32), instr);
        }
    }

    pub(crate) fn compile(&self) -> NativeThunk {
        codegen::to_native(self)
    }

    fn get_instr(&self, vid: ValueId) -> Option<&Instr> {
        self.instrs.get(vid.0 as usize)
    }

    fn iter_header_vids<'a>(&'a self) -> impl 'a + Iterator<Item = ValueId> {
        (0..self.instrs.len())
            .filter(|ndx| self.is_enabled[*ndx])
            .filter(|ndx| !self.is_loop_variant[*ndx])
            .map(|ndx| ValueId(ndx as u32))
    }
    fn iter_header(&self) -> impl Iterator<Item = (ValueId, &Instr)> {
        self.iter_add_instr(self.iter_header_vids())
    }

    /// Iterate over the trace's "loop body": the enabled loop-variant instructions
    fn iter_loop_body_vids<'a>(&'a self) -> impl 'a + Iterator<Item = ValueId> {
        (0..self.instrs.len())
            .filter(|ndx| self.is_enabled[*ndx])
            .filter(|ndx| self.is_loop_variant[*ndx])
            .map(|ndx| ValueId(ndx as u32))
    }
    fn iter_loop_body(&self) -> impl Iterator<Item = (ValueId, &Instr)> {
        self.iter_add_instr(self.iter_loop_body_vids())
    }

    fn iter_vids<'a>(&'a self) -> impl 'a + Iterator<Item = ValueId> {
        self.iter_header_vids().chain(self.iter_loop_body_vids())
    }
    fn iter_instrs<'a>(&'a self) -> impl 'a + Iterator<Item = (ValueId, &Instr)> {
        self.iter_add_instr(self.iter_vids())
    }

    fn iter_add_instr<'a, I>(&'a self, iter: I) -> impl 'a + Iterator<Item = (ValueId, &Instr)>
    where
        I: 'a + Iterator<Item = ValueId>,
    {
        iter.map(|vid| (vid, self.instrs.get(vid.0 as usize).unwrap()))
    }
}
