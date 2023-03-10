use dynasmrt::x64::{Rq, Rx};

use crate::jit::builder::{ValueId, ValueType};

use super::builder::Instr;
use std::collections::HashMap;

type RegIndex = u8;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardReg {
    General(RegIndex),
    Numeric(RegIndex),
}

impl std::fmt::Debug for HardReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use dynasmrt::Register;
        match self {
            HardReg::General(regndx) => write!(f, "hg{}", regndx),
            HardReg::Numeric(regndx) => write!(f, "hn{}", regndx),
        }
    }
}

/// Expresses a register assignment: (pos, vid, hreg) means that IR
/// value ID `vid` is assigned to register `hreg` starting at position
/// `pos` (inclusive).
#[derive(Debug)]
pub struct RegAsmt {
    pos: ValueId,
    vid: ValueId,
    hreg: HardReg,
}

pub struct Allocation {
    asmts: Vec<RegAsmt>,
    n_general: RegIndex,
    n_numeric: RegIndex,
}

impl Allocation {
    pub(crate) fn n_general(&self) -> RegIndex {
        self.n_general
    }

    fn new(asmts: Vec<RegAsmt>) -> Allocation {
        let mut n_general = 0;
        let mut n_numeric = 0;
        for asmt in asmts.iter() {
            match asmt.hreg {
                HardReg::General(ndx) => {
                    n_general = n_general.max(ndx + 1);
                }
                HardReg::Numeric(ndx) => {
                    n_numeric = n_numeric.max(ndx + 1);
                }
            }
        }

        Allocation {
            asmts,
            n_general,
            n_numeric,
        }
    }

    pub(crate) fn hreg_of_instr(&self, vid: ValueId) -> Option<HardReg> {
        todo!("delete this method")
    }

    pub(crate) fn n_instrs(&self) -> usize {
        todo!("delete this method")
    }
}

/// A "constraint point" designates a register on which a constraint is
/// defined.  The constrained register is designated by its name (`vid`) and
/// the position at which the constraint becomes effective `pos`.  Both are
/// expressed as ValueIds, but the semantics are different.
///
/// Following this definition, if pos == vid, the expressed constraint is on
/// the result register; otherwise, on an operand.
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub(super) struct ConstraintPt {
    pos: ValueId,
    vid: ValueId,
}

struct RegGroup {
    next_free: RegIndex,
    vofh: Vec<Option<ValueId>>,
    #[cfg(test)]
    name: &'static str,
}

impl RegGroup {
    fn new(count: RegIndex) -> Self {
        assert!(count > 0);
        RegGroup {
            next_free: 0,
            vofh: vec![None; count as usize],
            #[cfg(test)]
            name: "???",
        }
    }
    #[cfg(test)]
    fn named(self, name: &'static str) -> Self {
        Self { name, ..self }
    }
    fn clear(&mut self, ndx: RegIndex) -> ValueId {
        #[cfg(test)]
        eprintln!("regalloc: {}: clear {:?}", self.name, ndx);
        self.vofh[ndx as usize].take().unwrap()
    }
    fn set(&mut self, hreg_ndx: RegIndex, vid: ValueId) {
        let slot = &mut self.vofh[hreg_ndx as usize];
        #[cfg(test)]
        eprintln!("regalloc: {}: alloc {:?} <- {:?}", self.name, hreg_ndx, vid);
        assert!(slot.is_none());
        *slot = Some(vid);
    }
    fn pick_free(&mut self) -> RegIndex {
        let n_regs = self.vofh.len() as u8;
        for _ in 0..n_regs {
            let slot = &self.vofh[self.next_free as usize];
            if slot.is_none() {
                return self.next_free;
            }
            self.next_free = (self.next_free + 1) % n_regs;
        }
        panic!("out of hregs!");
    }

    fn make_free(&mut self, hreg: RegIndex) -> Option<(ValueId, RegIndex)> {
        let ret = match self.vofh[hreg as usize].clone() {
            Some(incumbent_vid) => {
                let newreg = self.pick_free();
                self.set(newreg, incumbent_vid);
                self.clear(hreg);
                #[cfg(test)]
                eprintln!(
                    "regalloc: {}: move {:?} to {}",
                    self.name, incumbent_vid, newreg
                );
                Some((incumbent_vid, newreg))
            }
            None => None,
        };
        assert!(self.vofh[hreg as usize].is_none());
        ret
    }
}

struct RegState {
    gen: RegGroup,
    num: RegGroup,
    hofv: Box<[Option<HardReg>]>,
}

impl RegState {
    fn new(count_gen: RegIndex, count_num: RegIndex, count_vids: usize) -> Self {
        let gen = RegGroup::new(count_gen);
        let num = RegGroup::new(count_num);

        #[cfg(test)]
        let (gen, num) = (gen.named("gen"), num.named("num"));

        RegState {
            gen,
            num,
            hofv: vec![None; count_vids].into_boxed_slice(),
        }
    }
    fn clear(&mut self, hreg: HardReg) -> ValueId {
        let vid = match hreg {
            HardReg::General(regndx) => self.gen.clear(regndx),
            // CATCH this bug!
            HardReg::Numeric(regndx) => self.num.clear(regndx + 1),
        };
        self.hofv[vid.0 as usize] = None;
        vid
    }
    fn clear_vid(&mut self, vid: ValueId) -> Option<HardReg> {
        let hreg = self.hofv[vid.0 as usize]?;
        let alloced_vid = self.clear(hreg);
        assert_eq!(alloced_vid, vid);
        Some(hreg)
    }
    fn pick_free_gen(&mut self) -> HardReg {
        HardReg::General(self.gen.pick_free())
    }
    fn pick_free_num(&mut self) -> HardReg {
        HardReg::Numeric(self.num.pick_free())
    }

    fn set(&mut self, hreg: HardReg, vid: ValueId) {
        match hreg {
            // CATCH this bug!
            HardReg::General(regndx) => self.gen.set(regndx + 2, vid),
            HardReg::Numeric(regndx) => self.num.set(regndx, vid),
        };
        self.hofv[vid.0 as usize] = Some(hreg);
    }

    fn make_free(&mut self, hreg: HardReg) -> Option<(ValueId, HardReg)> {
        let transfer = match hreg {
            HardReg::General(regndx) => self
                .gen
                .make_free(regndx)
                // CATCH this bug!
                .map(|(pos, newreg)| (pos, HardReg::Numeric(newreg))),
            HardReg::Numeric(regndx) => self
                .num
                .make_free(regndx)
                .map(|(pos, newreg)| (pos, HardReg::Numeric(newreg))),
        };

        if let Some((moved_vid, _)) = transfer {
            self.hofv[moved_vid.0 as usize] = None;
        }

        transfer
    }
}

pub(super) fn allocate_registers<I: Instruction>(
    code: &[I],
    constraints: &HashMap<ConstraintPt, HardReg>,
    num_general: RegIndex,
    num_numeric: RegIndex,
) -> Allocation {
    let mut regs = RegState::new(num_general, num_numeric, code.len());
    let get_constraint = |pos, vid| constraints.get(&ConstraintPt { pos, vid });

    // Register assignments, in reverse order (last to first)
    let mut asmts: Vec<RegAsmt> = vec![];

    // Instructions are iterated backwards, last to first.  This way,
    // each vreg's last use is seen first, and its definition is seen
    // last (and clearly identified).
    for (pos_ndx, instr) in code.iter().enumerate().rev() {
        let pos = ValueId(pos_ndx as u32);
        let pos_next = ValueId(pos_ndx as u32 + 1);

        #[cfg(test)]
        eprintln!(". {:?}", pos);

        for oper_ndx in 0..instr.n_operands() {
            let operand = instr.get_operand(oper_ndx);
            #[cfg(test)]
            eprintln!("  ~ {:?}", operand);
            if let Some(&fixed_hreg) = get_constraint(pos, operand) {
                #[cfg(test)]
                eprintln!("    - constrained to {:?} ", fixed_hreg);

                let transfer = regs.make_free(fixed_hreg);
                if let Some((vid, newhreg)) = transfer {
                    // fixed_hreg was previously occupied by vid.  To make
                    // space for operand, we moved vid to newhreg.
                    asmts.push(RegAsmt {
                        pos: pos_next,
                        vid,
                        hreg: newhreg,
                    });
                }

                regs.set(fixed_hreg, operand);
            } else {
                let src_inst = &code[operand.0 as usize];
                let hreg = match src_inst.result_type().unwrap() {
                    ValueType::Num => regs.pick_free_num(),
                    _ => regs.pick_free_gen(),
                };
                regs.set(hreg, operand);
            }
        }

        #[cfg(test)]
        eprintln!("  <");
        if let Some(hreg) = regs.clear_vid(pos) {
            asmts.push(RegAsmt {
                pos,
                vid: pos,
                hreg,
            });
        }
    }

    Allocation::new(asmts)
}

pub(super) trait Instruction {
    fn n_operands(&self) -> usize;
    fn get_operand(&self, ndx: usize) -> ValueId;
    fn result_type(&self) -> Option<ValueType>;
}

#[cfg(test)]
mod tests {

    use crate::{interpreter::CmpOp, jit::builder::Cmp};

    use super::*;

    #[derive(Debug)]
    struct TestInstr {
        ty: Option<ValueType>,
        result_reg: ValueId,
        // TODO Make this more compact?
        operands: Vec<ValueId>,
    }

    impl Instruction for TestInstr {
        fn result_type(&self) -> Option<ValueType> {
            self.ty
        }

        fn n_operands(&self) -> usize {
            self.operands.len()
        }

        fn get_operand(&self, ndx: usize) -> ValueId {
            self.operands.get(ndx).copied().unwrap()
        }
    }

    struct InstSeqBuilder {
        seq: Vec<TestInstr>,
    }
    impl InstSeqBuilder {
        fn new() -> Self {
            Self { seq: Vec::new() }
        }
        fn inst(mut self, ty: Option<ValueType>, operands: &[ValueId]) -> Self {
            let ndx = self.seq.len();
            self.seq.push(TestInstr {
                ty,
                result_reg: ValueId(ndx as u32),
                operands: operands.to_owned(),
            });
            self
        }
        fn build(self) -> Vec<TestInstr> {
            self.seq
        }
    }

    fn constraint(pos: u32, vid: u32, hreg: HardReg) -> (ConstraintPt, HardReg) {
        let constraint_pt = ConstraintPt {
            pos: ValueId(pos),
            vid: ValueId(vid),
        };
        (constraint_pt, hreg)
    }

    fn check_allocation<I: Instruction>(instrs: &[I], alloc: &Allocation) {
        // Simulate a user of the register allocation, and check that it "would
        // work" for it (e.g. for the code generation)
        let mut asmts = alloc.asmts.iter().rev().peekable();

        let mut vofh = HashMap::new();

        for (pos_ndx, inst) in instrs.iter().enumerate() {
            let pos = ValueId(pos_ndx.try_into().unwrap());
            while let Some(asmt) = asmts.next_if(|asmt| asmt.pos == pos) {
                vofh.insert(asmt.hreg, asmt.vid);
            }

            for oper_ndx in 0..inst.n_operands() {
                let operand = inst.get_operand(oper_ndx);
                assert!(vofh.iter().any(|(_, v)| *v == operand));
            }
        }
    }

    #[test]
    fn test_constraint() {
        let instrs = InstSeqBuilder::new()
            .inst(Some(ValueType::Num), &[])
            .inst(Some(ValueType::Str), &[])
            .inst(Some(ValueType::Str), &[])
            .inst(Some(ValueType::Num), &[ValueId(0), ValueId(1)])
            .inst(Some(ValueType::Num), &[ValueId(3), ValueId(2)])
            .inst(Some(ValueType::Num), &[])
            .inst(Some(ValueType::Str), &[ValueId(3)])
            .build();
        let constraints: HashMap<_, _> = [
            constraint(3, 0, HardReg::Numeric(2)),
            constraint(3, 3, HardReg::Numeric(1)),
            constraint(4, 3, HardReg::Numeric(2)),
            constraint(3, 1, HardReg::General(1)),
        ]
        .into_iter()
        .collect();

        let num_general = 13;
        let num_numeric = 15;

        try_allocation(&instrs, &constraints, num_general, num_numeric);
    }

    fn try_allocation(
        instrs: &[TestInstr],
        constraints: &HashMap<ConstraintPt, HardReg>,
        num_general: u8,
        num_numeric: u8,
    ) {
        let allocation = allocate_registers(&instrs, &constraints, num_general, num_numeric);
        for asmt in &allocation.asmts {
            eprintln!(" - {:?}", asmt);
        }
        check_allocation(&instrs, &allocation);
    }
}
