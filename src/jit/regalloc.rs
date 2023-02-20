use crate::jit::builder::ValueId;

use super::builder::Instr;

type RegIndex = u16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HardReg {
    General(RegIndex),
}

impl std::fmt::Debug for HardReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::General(ndx) => write!(f, "hg{ndx}"),
        }
    }
}

pub struct Allocation {
    hregs: Box<[Option<HardReg>]>,
    n_general: RegIndex,
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
        for hreg in hregs.iter() {
            match hreg {
                Some(HardReg::General(ndx)) => {
                    n_general = n_general.max(*ndx + 1);
                }
                None => {}
            }
        }

        Allocation { hregs, n_general }
    }
}

pub(super) fn allocate_registers(code: &[Instr], num_general: RegIndex) -> Allocation {
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

    let mut state = RegGroup::new(num_general);
    let mut allocation = vec![None; code.len()].into_boxed_slice();

    for (ndx, instr) in code.iter().enumerate().rev() {
        if let Some(HardReg::General(hreg)) = allocation.get(ndx).unwrap() {
            state.set_free(*hreg);
        }

        for &ValueId(input_ndx) in instr.operands() {
            let alloc = allocation.get_mut(input_ndx as usize).unwrap();
            if alloc.is_none() {
                let hreg_ndx = state.pick_free();
                *alloc = Some(HardReg::General(hreg_ndx));
            }
        }
    }

    Allocation::new(allocation)
}
