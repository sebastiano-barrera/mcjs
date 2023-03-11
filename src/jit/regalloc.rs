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
    loc: Loc,
}

pub struct Allocation {
    asmts: Vec<RegAsmt>,
    n_general: RegIndex,
    n_numeric: RegIndex,
    stack_size: u16,
}

impl Allocation {
    pub(crate) fn n_general(&self) -> RegIndex {
        self.n_general
    }

    fn new(asmts: Vec<RegAsmt>) -> Allocation {
        let mut n_general = 0;
        let mut n_numeric = 0;
        let mut stack_size = 0u16;
        for asmt in asmts.iter() {
            match asmt.loc {
                Loc::HardReg(HardReg::General(ndx)) => n_general = n_general.max(ndx + 1),
                Loc::HardReg(HardReg::Numeric(ndx)) => n_numeric = n_numeric.max(ndx + 1),
                Loc::StackSlot(sslot) => stack_size = stack_size.max(sslot + 1),
            }
        }

        Allocation {
            asmts,
            n_general,
            n_numeric,
            stack_size,
        }
    }

    pub(crate) fn hreg_of_instr(&self, vid: ValueId) -> Option<HardReg> {
        todo!("delete this method")
    }

    pub(crate) fn n_instrs(&self) -> usize {
        todo!("delete this method")
    }
}

/// A "constraint point" designates an operand on which a constraint is
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
    fn pick_free(&mut self) -> Option<RegIndex> {
        let n_regs = self.vofh.len() as u8;
        for _ in 0..n_regs {
            let slot = &self.vofh[self.next_free as usize];
            if slot.is_none() {
                return Some(self.next_free);
            }
            self.next_free = (self.next_free + 1) % n_regs;
        }

        None
    }
}

type StackSlot = u16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Loc {
    HardReg(HardReg),
    StackSlot(StackSlot),
}

struct RegState {
    gen: RegGroup,
    num: RegGroup,

    // Free stack slots
    free_stacks: Vec<StackSlot>,

    loc_of_vid: Box<[Option<Loc>]>,
    // Register assignments, in reverse order (last to first instruction)
    asmts: Vec<RegAsmt>,
    cur_pos: ValueId,
}

impl RegState {
    fn new(count_gen: RegIndex, count_num: RegIndex, count_vids: u32) -> Self {
        let gen = RegGroup::new(count_gen);
        let num = RegGroup::new(count_num);

        #[cfg(test)]
        let (gen, num) = (gen.named("gen"), num.named("num"));

        RegState {
            gen,
            num,
            loc_of_vid: vec![None; count_vids as usize].into_boxed_slice(),
            asmts: Vec::new(),
            cur_pos: ValueId(count_vids - 1),
            free_stacks: Vec::new(),
        }
    }

    fn clear_vid(&mut self, vid: ValueId) {
        if let Some(loc) = self.loc_of_vid[vid.0 as usize].take() {
            let vid_chk = match loc {
                Loc::HardReg(HardReg::General(regndx)) => self.gen.clear(regndx),
                Loc::HardReg(HardReg::Numeric(regndx)) => self.num.clear(regndx),
                Loc::StackSlot(sslot) => {
                    self.free_stacks.push(sslot);
                    vid
                }
            };
            assert_eq!(vid_chk, vid);

            self.asmts.push(RegAsmt {
                pos: self.cur_pos,
                vid,
                loc,
            });
        }
    }

    // TODO Remove this from the "public" interface
    fn set_hreg(&mut self, hreg: HardReg, vid: ValueId) {
        match hreg {
            HardReg::General(regndx) => self.gen.set(regndx, vid),
            HardReg::Numeric(regndx) => self.num.set(regndx, vid),
        };
        self.loc_of_vid[vid.0 as usize] = Some(Loc::HardReg(hreg));
    }

    fn get_stack_slot(&mut self) -> StackSlot {
        todo!()
    }

    fn force_allocate(&mut self, fixed_hreg: HardReg, operand: ValueId) {
        let incumbent_vid = match fixed_hreg {
            HardReg::General(regndx) => self.gen.vofh[regndx as usize],
            HardReg::Numeric(regndx) => self.num.vofh[regndx as usize],
        };

        if let Some(incumbent_vid) = incumbent_vid {
            // fixed_hreg was previously occupied by incumbent_vid.  To make
            // space for operand, we move it.
            self.clear_vid(incumbent_vid);
        }

        self.set_hreg(fixed_hreg, operand);

        if let Some(incumbent_vid) = incumbent_vid {
            self.allocate_num(incumbent_vid);
        }
    }

    fn advance_to(&mut self, pos: ValueId) {
        assert!(pos.0 <= self.cur_pos.0);
        self.cur_pos = pos;
    }

    fn allocate_num(&mut self, vid: ValueId) {
        let regndx = match self.num.pick_free() {
            Some(regndx) => regndx,
            None => {
                // Evict something else to the stack
                let (evict_vid, freed_regndx) = self
                    .loc_of_vid
                    .iter()
                    .enumerate()
                    .find_map(|(vid, loc)| match loc {
                        Some(Loc::HardReg(HardReg::Numeric(regndx))) => {
                            let value_id = ValueId(u32::try_from(vid).unwrap());
                            Some((value_id, *regndx))
                        }
                        _ => None,
                    })
                    .expect("regalloc bug: no registers can be evicted to stack. (0 hregs?!)");
                let stack_slot = self.get_stack_slot();

                self.clear_vid(evict_vid);
                self.loc_of_vid[evict_vid.0 as usize] = Some(Loc::StackSlot(stack_slot));

                freed_regndx
            }
        };

        self.set_hreg(HardReg::Numeric(regndx), vid);
    }

    fn allocate_gen(&mut self, vid: ValueId) {
        let regndx = match self.num.pick_free() {
            Some(regndx) => regndx,
            None => {
                // Evict something else to the stack
                let (evict_vid, freed_regndx) = self
                    .loc_of_vid
                    .iter()
                    .enumerate()
                    .find_map(|(vid, loc)| match loc {
                        Some(Loc::HardReg(HardReg::General(regndx))) => {
                            let value_id = ValueId(u32::try_from(vid).unwrap());
                            Some((value_id, *regndx))
                        }
                        _ => None,
                    })
                    .expect("regalloc bug: no registers can be evicted to stack. (0 hregs?!)");

                self.clear_vid(evict_vid);

                let stack_slot = self.get_stack_slot();
                self.loc_of_vid[evict_vid.0 as usize] = Some(Loc::StackSlot(stack_slot));

                freed_regndx
            }
        };

        self.set_hreg(HardReg::General(regndx), vid);
    }

    fn where_is(&self, operand: ValueId) -> Option<Loc> {
        self.loc_of_vid[operand.0 as usize]
    }
}

pub(super) fn allocate_registers<I: Instruction>(
    code: &[I],
    constraints: &HashMap<ConstraintPt, HardReg>,
    num_general: RegIndex,
    num_numeric: RegIndex,
) -> Allocation {
    let get_constraint = |pos, vid| constraints.get(&ConstraintPt { pos, vid });

    let count_vids = u32::try_from(code.len()).unwrap();
    let mut regs = RegState::new(num_general, num_numeric, count_vids);

    // Instructions are iterated backwards, last to first.  This way,
    // each vreg's last use is seen first, and its definition is seen
    // last (and clearly identified).
    for (pos_ndx, instr) in code.iter().enumerate().rev() {
        let pos = ValueId(pos_ndx as u32);

        #[cfg(test)]
        eprintln!(". {:?}", pos);

        for oper_ndx in 0..instr.n_operands() {
            let operand = instr.get_operand(oper_ndx);

            #[cfg(test)]
            eprintln!("  ~ {:?}", operand);

            if let Some(&fixed_hreg) = get_constraint(pos, operand) {
                #[cfg(test)]
                eprintln!("    - constrained to {:?} ", fixed_hreg);

                regs.force_allocate(fixed_hreg, operand);
            } else {
                let allocate = |regs: &mut RegState, operand: ValueId| match code
                    [operand.0 as usize]
                    .result_type()
                    .unwrap()
                {
                    ValueType::Num => regs.allocate_num(operand),
                    _ => regs.allocate_gen(operand),
                };

                match regs.where_is(operand) {
                    None => allocate(&mut regs, operand),
                    // Nothing to do, we like operands in regs
                    Some(Loc::HardReg(_)) => {}
                    Some(Loc::StackSlot(_sslot)) => {
                        #[cfg(test)]
                        eprintln!("    - from stack #{} -> {:?}", _sslot, operand);
                        regs.clear_vid(operand);
                        allocate(&mut regs, operand);
                    }
                };
            }
        }

        regs.advance_to(pos);

        #[cfg(test)]
        eprintln!("  <");
        regs.clear_vid(pos);
    }

    Allocation::new(regs.asmts)
}

pub(super) trait Instruction {
    fn n_operands(&self) -> usize;
    fn get_operand(&self, ndx: usize) -> ValueId;
    fn result_type(&self) -> Option<ValueType>;
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{interpreter::CmpOp, jit::builder::Cmp};

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
                vofh.insert(asmt.loc, asmt.vid);
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

    proptest! {
        #[test]
        fn all_valueids_have_hreg(
            code_size in 0u8..100,
            read_frac in 0.0f32..=1.0,
            num_general in 1u8..100,
            num_numeric in 1u8..100,
        ) {
            xxx(code_size, read_frac, num_general, num_numeric);
        }
    }

    // TODO inline this function mack into all_valueids_have_hreg.  This is
    // useful to let rust-analyzer work with all its features in the function
    // body
    fn xxx(code_size: u8, read_frac: f32, num_general: u8, num_numeric: u8) {
        let mut rng = rand::thread_rng();

        let max_uses_per_instr = (read_frac * code_size as f32) as u8;

        let instrs = {
            let mut instrs = InstSeqBuilder::new();
            let mut uses = Vec::new();
            for _ in 0..code_size {
                let n_uses = rng.gen_range(0..=max_uses_per_instr);
                uses.resize(n_uses as usize, ValueId(0));
                for (ndx, x) in uses.iter_mut().enumerate().skip(1) {
                    *x = ValueId(rng.gen_range(0..ndx as u32));
                }

                instrs = instrs.inst(Some(ValueType::Num), &uses);
            }

            instrs.build()
        };

        // TODO
        let constraints = HashMap::new();

        try_allocation(&instrs, &constraints, num_general, num_numeric);
    }
}
