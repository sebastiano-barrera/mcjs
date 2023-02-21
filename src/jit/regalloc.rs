use crate::jit::builder::{ValueId, ValueType};

use super::builder::Instr;

type RegIndex = u16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HardReg {
    General(RegIndex),
    Numeric(RegIndex),
}

impl std::fmt::Debug for HardReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HardReg::General(ndx) => write!(f, "hg{ndx}"),
            HardReg::Numeric(ndx) => write!(f, "hn{ndx}"),
        }
    }
}

pub struct Allocation {
    hregs: Box<[Option<HardReg>]>,
    n_general: RegIndex,
    n_numeric: RegIndex,
}
impl Allocation {
    pub(crate) fn n_general(&self) -> RegIndex {
        self.n_general
    }
    pub(crate) fn n_instrs(&self) -> usize {
        self.hregs.len()
    }

    pub(crate) fn hreg_of_instr(&self, vid: ValueId) -> Option<HardReg> {
        self.hregs.get(vid.0 as usize).copied().flatten()
    }

    fn new(hregs: Box<[Option<HardReg>]>) -> Allocation {
        let mut n_general = 0;
        let mut n_numeric = 0;
        for hreg in hregs.iter() {
            match hreg {
                Some(HardReg::General(ndx)) => {
                    n_general = n_general.max(*ndx + 1);
                }
                Some(HardReg::Numeric(ndx)) => {
                    n_numeric = n_numeric.max(*ndx + 1);
                }
                None => {}
            }
        }

        Allocation {
            hregs,
            n_general,
            n_numeric,
        }
    }
}

pub(super) fn allocate_registers(
    code: &[Instr],
    num_general: RegIndex,
    num_numeric: RegIndex,
) -> Allocation {
    struct RegGroup {
        free_regs: Vec<RegIndex>,
        is_free: Vec<bool>,
    }

    impl RegGroup {
        fn new(count: RegIndex) -> Self {
            RegGroup {
                free_regs: (0..count).rev().collect(),
                is_free: vec![true; count as usize],
            }
        }

        fn set_free(&mut self, hreg_ndx: RegIndex) {
            eprintln!("regalloc: set_free {:?}", hreg_ndx);
            assert!(!self.is_free[hreg_ndx as usize]);
            self.free_regs.push(hreg_ndx);
            self.is_free[hreg_ndx as usize] = true;
        }

        fn pick_free(&mut self) -> RegIndex {
            let hreg_ndx = self.free_regs.pop().expect("out of hregs!");
            assert!(self.is_free[hreg_ndx as usize]);
            self.is_free[hreg_ndx as usize] = false;
            eprintln!("regalloc: pick_free -> {:?}", hreg_ndx);
            hreg_ndx
        }
    }

    let mut state_gen = RegGroup::new(num_general);
    let mut state_num = RegGroup::new(num_numeric);
    let mut allocation = vec![None; code.len()].into_boxed_slice();

    for (ndx, instr) in code.iter().enumerate().rev() {
        match allocation.get(ndx).unwrap() {
            Some(HardReg::General(regndx)) => state_gen.set_free(*regndx),
            Some(HardReg::Numeric(regndx)) => state_num.set_free(*regndx),
            _ => {}
        }

        for &ValueId(input_ndx) in instr.operands() {
            let alloc = allocation.get_mut(input_ndx as usize).unwrap();
            if alloc.is_none() {
                let src_instr = code.get(input_ndx as usize).unwrap_or_else(|| {
                    panic!("JIT bug: used value but no corresponding instr: {input_ndx}",)
                });
                let value_type = src_instr.result_type().unwrap_or_else(|| {
                    panic!(
                        "JIT bug: value used but instruction has no value: {:?}",
                        instr
                    )
                });
                if value_type == ValueType::Num {
                    let hreg_ndx = state_num.pick_free();
                    *alloc = Some(HardReg::Numeric(hreg_ndx));
                } else {
                    let hreg_ndx = state_gen.pick_free();
                    *alloc = Some(HardReg::General(hreg_ndx));
                }
            }
        }
    }

    Allocation::new(allocation)
}
