use crate::jit::builder::ValueId;

use super::builder::Instr;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HardReg(pub u32);

impl std::fmt::Debug for HardReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "h{}", self.0)
    }
}

pub struct Allocation {
    hregs: Box<[Option<HardReg>]>,
    n_hregs: u32,
}
impl Allocation {
    pub(crate) fn n_hregs(&self) -> u32 {
        self.n_hregs
    }
    pub(crate) fn n_instrs(&self) -> usize {
        self.hregs.len()
    }

    pub(crate) fn hreg_of_instr(&self, vid: ValueId) -> Option<HardReg> {
        self.hregs.get(vid.0 as usize).copied().flatten()
    }

    fn new(hregs: Box<[Option<HardReg>]>) -> Allocation {
        let n_hregs = hregs
            .iter()
            .filter_map(|opt| opt.map(|hreg| hreg.0))
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);
        Allocation { hregs, n_hregs }
    }
}

pub(super) fn allocate_registers(code: &[Instr], num_hregs: u32) -> Allocation {
    struct State {
        free_regs: Vec<HardReg>,
        is_free: Vec<bool>,
    }

    impl State {
        fn new(num_hregs: u32) -> Self {
            let free_regs = (0..num_hregs).map(|ndx| HardReg(ndx)).rev().collect();
            let is_free = vec![true; num_hregs as usize];

            State { free_regs, is_free }
        }

        fn set_free(&mut self, hreg: &HardReg) {
            eprintln!("regalloc: set_free {:?}", hreg);
            assert!(!self.is_free[hreg.0 as usize]);
            self.free_regs.push(*hreg);
            self.is_free[hreg.0 as usize] = true;
        }

        fn pick_free(&mut self) -> HardReg {
            let hreg = self.free_regs.pop().expect("out of hregs!");
            assert!(self.is_free[hreg.0 as usize]);
            self.is_free[hreg.0 as usize] = false;
            eprintln!("regalloc: pick_free -> {:?}", hreg);
            hreg
        }
    }

    let mut state = State::new(num_hregs);
    let mut allocation = vec![None; code.len()].into_boxed_slice();

    for (ndx, instr) in code.iter().enumerate().rev() {
        if let Some(hreg) = allocation.get(ndx).unwrap() {
            state.set_free(hreg);
        }

        for &ValueId(input_ndx) in instr.operands() {
            let alloc = allocation.get_mut(input_ndx as usize).unwrap();
            if alloc.is_none() {
                *alloc = Some(state.pick_free());
            }
        }
    }

    Allocation::new(allocation)
}
