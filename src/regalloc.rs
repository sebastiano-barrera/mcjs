#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HardReg(pub u32);

impl std::fmt::Debug for HardReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "h{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SoftReg(pub u32);

impl std::fmt::Debug for SoftReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "S{}", self.0)
    }
}

pub trait Instruction {
    fn inputs(&self) -> Vec<SoftReg>;
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

    pub(crate) fn hreg_of_instr(&self, instr_ndx: usize) -> Option<HardReg> {
        self.hregs.get(instr_ndx).copied().flatten()
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

    pub(super) fn reordered(&self, new_order: &[usize]) -> Allocation {
        todo!()
    }
}

pub fn allocate_registers<I: Instruction>(code: &[I], num_hregs: u32) -> Allocation {
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
        let sreg = SoftReg(ndx as u32);
        let inputs = instr.inputs();
        eprintln!("regalloc: {:2} {:?}", sreg.0, inputs);

        if let Some(hreg) = allocation.get(ndx).unwrap() {
            state.set_free(hreg);
        }

        for &SoftReg(input_ndx) in inputs.iter() {
            let alloc = allocation.get_mut(input_ndx as usize).unwrap();
            if alloc.is_none() {
                *alloc = Some(state.pick_free());
            }
        }
    }

    Allocation::new(allocation)
}

//    0 GetArg(0)
//    1 Unbox(Num, v0)
//    2 Cmp { ty: Num, op: GT, a: v1, b: Number(0.0) }
//    3 Not(v2)
//    4 Not(v3)
//    5 AssertTrue { cond: v4 }
//    6 Unbox(Num, v0)
//    7 Arith { op: Add, a: Number(0.0), b: v6 }
//    8 Unbox(Num, v0)
//    9 Arith { op: Sub, a: v8, b: Number(1.0) }
//   10 Cmp { ty: Num, op: GT, a: v9, b: Number(0.0) }
//   11 Not(v10)
//   12 Not(v11)
//   13 AssertTrue { cond: v12 }
//   14 Arith { op: Add, a: v7, b: v9 }
//   15 Arith { op: Sub, a: v9, b: Number(1.0) }
//   16 Cmp { ty: Num, op: GT, a: v15, b: Number(0.0) }
//   17 Not(v16)
//   18 AssertTrue { cond: v17 }
//   19 Box(v14)
//   20 Return(v19)
