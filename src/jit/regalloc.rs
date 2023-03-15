use dynasmrt::x64::{Rq, Rx};

use crate::jit::builder::{ValueId, ValueType};

use super::builder::{Instr, OperandsSet};
use std::collections::HashMap;

type RegIndex = u8;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RegClass {
    General,
    Numeric,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HardReg {
    pub(super) class: RegClass,
    pub(super) index: RegIndex,
}

impl std::fmt::Debug for HardReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use dynasmrt::Register;
        let class_txt = match self.class {
            RegClass::General => "g",
            RegClass::Numeric => "n",
        };

        write!(f, "h{}{}", class_txt, self.index)
    }
}

/// Expresses a register assignment: (pos, vid, hreg) means that IR
/// value ID `vid` is assigned to register `hreg` starting at position
/// `pos` (inclusive).
#[derive(Debug, PartialEq, Eq)]
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
                Loc::HardReg(HardReg {
                    class: RegClass::General,
                    index: ndx,
                }) => n_general = n_general.max(ndx + 1),
                Loc::HardReg(HardReg {
                    class: RegClass::Numeric,
                    index: ndx,
                }) => n_numeric = n_numeric.max(ndx + 1),
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

    pub(crate) fn hreg_of_instr(&self, _vid: ValueId) -> Option<HardReg> {
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
    is_locked: Vec<bool>,
    vofh: Vec<Option<ValueId>>,
    #[cfg(test)]
    name: &'static str,
}

impl RegGroup {
    fn new(count: RegIndex, _debug_name: &'static str) -> Self {
        assert!(
            count >= 2,
            "too few registers ({count}; at least 2 requied)"
        );
        RegGroup {
            next_free: 0,
            is_locked: vec![false; count as usize],
            vofh: vec![None; count as usize],
            #[cfg(test)]
            name: _debug_name,
        }
    }

    fn unlock_all(&mut self) {
        self.is_locked.iter_mut().for_each(|x| *x = false)
    }

    fn take(&mut self, ndx: RegIndex) -> ValueId {
        #[cfg(test)]
        eprintln!("regalloc: {}: clear {:?}", self.name, ndx);

        self.vofh[ndx as usize].take().unwrap()
    }

    fn set(&mut self, hreg_ndx: RegIndex, vid: ValueId) {
        #[cfg(test)]
        eprintln!("regalloc: {}: alloc {:?} <- {:?}", self.name, hreg_ndx, vid);

        let slot = &mut self.vofh[hreg_ndx as usize];
        assert!(slot.is_none());
        *slot = Some(vid);
        self.is_locked[hreg_ndx as usize] = true;
    }

    fn pick_free(&mut self) -> Option<RegIndex> {
        for _ in 0..self.vofh.len() {
            let hreg_ndx = self.next_free as usize;
            if !self.is_locked[hreg_ndx] && self.vofh[hreg_ndx].is_none() {
                return Some(self.next_free);
            }
            self.next_free = (self.next_free + 1) % (self.vofh.len() as u8);
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
    stack_size: u16,

    loc_of_vid: Box<[Option<Loc>]>,
    // Register assignments, in reverse order (last to first instruction)
    asmts: Vec<RegAsmt>,
    cur_pos: ValueId,
}

impl RegState {
    fn new(count_gen: RegIndex, count_num: RegIndex, count_vids: u32) -> Self {
        let gen = RegGroup::new(count_gen, "gen");
        let num = RegGroup::new(count_num, "num");

        RegState {
            gen,
            num,
            free_stacks: Vec::new(),
            stack_size: 0,
            loc_of_vid: vec![None; count_vids as usize].into_boxed_slice(),
            asmts: Vec::new(),
            cur_pos: ValueId(count_vids - 1),
        }
    }

    fn group_for_mut(&mut self, reg_class: RegClass) -> &mut RegGroup {
        match reg_class {
            RegClass::General => &mut self.gen,
            RegClass::Numeric => &mut self.num,
        }
    }

    fn group_for(&self, reg_class: RegClass) -> &RegGroup {
        match reg_class {
            RegClass::General => &self.gen,
            RegClass::Numeric => &self.num,
        }
    }

    fn unlock_all(&mut self) {
        self.gen.unlock_all();
        self.num.unlock_all();
    }

    fn kill_vid(&mut self, vid: ValueId) {
        if let Some(loc) = self.loc_of_vid[vid.0 as usize].take() {
            match loc {
                Loc::HardReg(HardReg { index, class }) => {
                    let vid_chk = match class {
                        RegClass::General => self.gen.take(index),
                        RegClass::Numeric => self.num.take(index),
                    };
                    assert_eq!(vid_chk, vid);
                }
                Loc::StackSlot(sslot) => {
                    eprintln!("regalloc: to stack: v{} -> #{}", vid.0, sslot);
                    self.free_stacks.push(sslot);
                }
            };

            self.asmts.push(RegAsmt {
                pos: self.cur_pos,
                vid,
                loc,
            });
        }
    }

    // TODO Remove this from the "public" interface
    fn set_hreg(&mut self, hreg: HardReg, vid: ValueId) {
        self.group_for_mut(hreg.class).set(hreg.index, vid);
        self.loc_of_vid[vid.0 as usize] = Some(Loc::HardReg(hreg));
    }

    fn get_stack_slot(&mut self) -> StackSlot {
        if let Some(sslot) = self.free_stacks.pop() {
            return sslot;
        }

        let sslot = self.stack_size;
        self.stack_size += 1;
        sslot
    }

    fn force_allocate(&mut self, fixed_hreg: HardReg, vid: ValueId) {
        self.kill_vid(vid);

        let prev_inhabitant = self.group_for_mut(fixed_hreg.class).vofh[fixed_hreg.index as usize];
        if prev_inhabitant == Some(vid) {
            // Already assigned, lol
            return;
        }

        if let Some(incumbent_vid) = prev_inhabitant {
            // fixed_hreg was previously occupied by incumbent_vid.  To make
            // space for vid, we move it.
            self.kill_vid(incumbent_vid);
        }

        self.set_hreg(fixed_hreg, vid);

        if let Some(incumbent_vid) = prev_inhabitant {
            self.allocate(fixed_hreg.class, incumbent_vid);
        }
    }

    fn advance_to(&mut self, pos: ValueId) {
        assert!(pos.0 <= self.cur_pos.0);
        self.cur_pos = pos;
    }

    fn allocate(&mut self, class: RegClass, vid: ValueId) {
        let sought_class = class;

        let regndx = match self.group_for_mut(class).pick_free() {
            Some(regndx) => regndx,
            None => {
                #[cfg(test)]
                eprintln!("no free regs; we evict something to the stack");

                // Evict something else to the stack
                let (evict_vid, freed_regndx) = self
                    .loc_of_vid
                    .iter()
                    .enumerate()
                    .find_map(|(vid, loc)| match loc {
                        Some(Loc::HardReg(hreg))
                            if hreg.class == sought_class
                                && !self.group_for(class).is_locked[hreg.index as usize] =>
                        {
                            let value_id = ValueId(u32::try_from(vid).unwrap());
                            Some((value_id, hreg.index))
                        }
                        _ => None,
                    })
                    .expect("regalloc bug: no registers can be evicted to stack. (0 hregs?!)");
                let stack_slot = self.get_stack_slot();

                self.kill_vid(evict_vid);
                self.loc_of_vid[evict_vid.0 as usize] = Some(Loc::StackSlot(stack_slot));

                freed_regndx
            }
        };

        let hreg = HardReg {
            class,
            index: regndx,
        };
        self.set_hreg(hreg, vid);
    }

    fn where_is(&self, operand: ValueId) -> Option<Loc> {
        self.loc_of_vid[operand.0 as usize]
    }
}

pub(super) fn allocate_registers(
    reg_classes: &[Option<RegClass>],
    operands: &[OperandsSet],
    constraints: &HashMap<ConstraintPt, HardReg>,
    num_general: RegIndex,
    num_numeric: RegIndex,
) -> Allocation {
    if reg_classes.len() != operands.len() {
        panic!("bug: reg_class and operands must have same len()");
    }
    let n_instrs = reg_classes.len();

    if n_instrs == 0 {
        return Allocation::new(Vec::new());
    }

    let get_constraint = |pos, vid| constraints.get(&ConstraintPt { pos, vid });

    let count_vids = u32::try_from(n_instrs).unwrap();
    let mut regs = RegState::new(num_general, num_numeric, count_vids);

    // Instructions are iterated backwards, last to first.  This way,
    // each vreg's last use is seen first, and its definition is seen
    // last (and clearly identified).
    for (pos_ndx, operands) in operands.iter().enumerate().rev() {
        let pos = ValueId(pos_ndx as u32);

        #[cfg(test)]
        eprintln!(". {:?}", pos);

        regs.unlock_all();

        for &operand in operands.iter().flatten() {
            assert!((operand.0 as usize) < pos_ndx);

            let operand_class = reg_classes[operand.0 as usize]
                .expect("bug: malformed code: result used, but no reg class for it");

            #[cfg(test)]
            eprintln!("  ~ {:?} [{:?}]", operand, operand_class);

            if let Some(&fixed_hreg) = get_constraint(pos, operand) {
                #[cfg(test)]
                eprintln!("    - constrained to {:?} ", fixed_hreg);

                assert_eq!(
                    operand_class, fixed_hreg.class,
                    "invalid constraint: wrong register class!"
                );

                regs.force_allocate(fixed_hreg, operand);
            } else {
                match regs.where_is(operand) {
                    None => regs.allocate(operand_class, operand),
                    // Nothing to do, we like operands in regs
                    Some(Loc::HardReg(_)) => {}
                    Some(Loc::StackSlot(_sslot)) => {
                        #[cfg(test)]
                        eprintln!("    - from stack #{} -> {:?}", _sslot, operand);
                        regs.kill_vid(operand);
                        regs.allocate(operand_class, operand)
                    }
                };
            }
        }

        regs.advance_to(pos);

        #[cfg(test)]
        eprintln!("  <");
        regs.kill_vid(pos);
    }

    Allocation::new(regs.asmts)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::{
        interpreter::CmpOp,
        jit::builder::{Cmp, OperandsSet},
    };

    #[derive(Default)]
    struct InstSeq {
        reg_class: Vec<Option<RegClass>>,
        operands: Vec<OperandsSet>,
    }
    impl InstSeq {
        fn inst(&mut self, cls: Option<RegClass>, operands: &[ValueId]) {
            self.reg_class.push(cls);
            self.operands.push(OperandsSet::from(operands));
        }
    }

    fn check_allocation(instrs: &[OperandsSet], alloc: &Allocation) {
        // Simulate a user of the register allocation, and check that it "would
        // work" for it (e.g. for the code generation)
        let mut asmts = alloc.asmts.iter().rev().peekable();

        let mut vofh = HashMap::new();

        for (pos_ndx, operands) in instrs.iter().enumerate() {
            let pos = ValueId(pos_ndx.try_into().unwrap());
            while let Some(asmt) = asmts.next_if(|asmt| asmt.pos == pos) {
                vofh.insert(asmt.loc, asmt.vid);
            }

            for operand in operands.iter().flatten() {
                assert!(
                    vofh.iter().any(|(_, v)| v == operand),
                    "no hreg assigned to {:?} at pos {}",
                    operand,
                    pos_ndx
                );
            }
        }
    }

    fn hn(index: u8) -> HardReg {
        let class = RegClass::Numeric;
        HardReg { class, index }
    }

    fn hg(index: u8) -> HardReg {
        let class = RegClass::General;
        HardReg { class, index }
    }

    struct ConstrBuilder {
        constraints: HashMap<ConstraintPt, HardReg>,
    }

    impl ConstrBuilder {
        fn new() -> Self {
            let constraints = HashMap::new();
            ConstrBuilder { constraints }
        }

        fn constraint(mut self, pos: u32, vid: u32, hreg: HardReg) -> Self {
            let constraint_pt = ConstraintPt {
                pos: ValueId(pos),
                vid: ValueId(vid),
            };
            self.constraints.insert(constraint_pt, hreg);
            self
        }

        fn build(self) -> HashMap<ConstraintPt, HardReg> {
            self.constraints
        }
    }

    #[test]
    fn test_constraint() {
        let mut instrs = InstSeq::default();
        instrs.inst(Some(RegClass::Numeric), &[]);
        instrs.inst(Some(RegClass::General), &[]);
        instrs.inst(Some(RegClass::General), &[]);
        instrs.inst(Some(RegClass::Numeric), &[ValueId(0), ValueId(1)]);
        instrs.inst(None, &[ValueId(3), ValueId(2)]);
        instrs.inst(Some(RegClass::Numeric), &[]);
        instrs.inst(Some(RegClass::General), &[ValueId(3)]);

        let constraints = ConstrBuilder::new()
            .constraint(3, 0, hn(2))
            .constraint(3, 3, hn(1))
            .constraint(4, 3, hn(2))
            .constraint(3, 1, hg(1))
            .build();

        let num_general = 13;
        let num_numeric = 15;

        try_allocation(&instrs, &constraints, num_general, num_numeric);
    }

    fn try_allocation(
        instrs: &InstSeq,
        constraints: &HashMap<ConstraintPt, HardReg>,
        num_general: u8,
        num_numeric: u8,
    ) -> Allocation {
        let allocation = allocate_registers(
            &instrs.reg_class,
            &instrs.operands,
            &constraints,
            num_general,
            num_numeric,
        );
        for asmt in &allocation.asmts {
            eprintln!(" - {:?}", asmt);
        }
        check_allocation(&instrs.operands, &allocation);
        allocation
    }

    #[test]
    fn test_zero_case() {
        let allocation = allocate_registers(&[], &[], &HashMap::new(), 0, 0);
        assert!(allocation.asmts.is_empty());

        let constrs = ConstrBuilder::new()
            .constraint(12, 34, hn(56))
            .constraint(1, 2, hn(3))
            .build();
        let allocation = allocate_registers(&[], &[], &constrs, 0, 0);
        assert!(allocation.asmts.is_empty());
    }

    #[should_panic]
    #[test]
    fn test_one_instr_zero_hregs() {
        allocate_registers(
            &[Some(RegClass::General)],
            &[OperandsSet::none()],
            &HashMap::new(),
            0,
            0,
        );
    }

    #[should_panic]
    #[test]
    fn test_one_instr_one_hreg() {
        allocate_registers(
            &[Some(RegClass::General)],
            &[OperandsSet::none()],
            &HashMap::new(),
            1,
            1,
        );
    }

    #[test]
    fn test_one_instr() {
        for rcls in [RegClass::General, RegClass::Numeric] {
            let mut instrs = InstSeq::default();
            instrs.inst(Some(rcls), &[]);
            let allocation =
                allocate_registers(&instrs.reg_class, &instrs.operands, &HashMap::new(), 2, 2);
            // The instruction is unused
            assert!(allocation.asmts.is_empty());
        }
    }

    #[test]
    #[should_panic]
    fn test_two_instr_invalid_use() {
        for rcls in [RegClass::General, RegClass::Numeric] {
            let mut instrs = InstSeq::default();
            instrs.inst(Some(rcls), &[]);
            instrs.inst(Some(rcls), &[ValueId(1)]);
            allocate_registers(&instrs.reg_class, &instrs.operands, &HashMap::new(), 2, 2);
        }
    }

    #[test]
    fn test_two_instr() {
        for rcls in [RegClass::General, RegClass::Numeric] {
            let mut instrs = InstSeq::default();
            instrs.inst(Some(rcls), &[]);
            instrs.inst(Some(rcls), &[ValueId(0)]);
            let allocation =
                allocate_registers(&instrs.reg_class, &instrs.operands, &HashMap::new(), 2, 2);
            // The instruction is unused
            assert_eq!(1, allocation.asmts.len());
        }

        for (rcls, hreg) in [(RegClass::General, hg(2)), (RegClass::Numeric, hn(3))] {
            let mut instrs = InstSeq::default();
            instrs.inst(Some(rcls), &[]);
            instrs.inst(Some(rcls), &[ValueId(0)]);

            let constrs = ConstrBuilder::new().constraint(1, 0, hreg).build();

            let allocation =
                allocate_registers(&instrs.reg_class, &instrs.operands, &constrs, 5, 5);
            // The instruction is unused
            assert_eq!(1, allocation.asmts.len());
            assert_eq!(
                &RegAsmt {
                    pos: ValueId(0),
                    vid: ValueId(0),
                    loc: Loc::HardReg(hreg)
                },
                &allocation.asmts[0]
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_constraint_with_wrong_reg_class() {
        let mut instrs = InstSeq::default();
        instrs.inst(Some(RegClass::Numeric), &[]);
        instrs.inst(Some(RegClass::General), &[ValueId(0)]);
        instrs.inst(Some(RegClass::General), &[ValueId(1), ValueId(0)]);
        instrs.inst(Some(RegClass::General), &[ValueId(1)]);
        instrs.inst(Some(RegClass::Numeric), &[]);
        let constraints = ConstrBuilder::new().constraint(3, 1, hn(1)).build();

        try_allocation(&instrs, &constraints, 2, 2);
    }

    #[test]
    fn test_constraint_with_correct_reg_class() {
        let mut instrs = InstSeq::default();
        instrs.inst(Some(RegClass::Numeric), &[]);
        instrs.inst(Some(RegClass::General), &[ValueId(0)]);
        instrs.inst(Some(RegClass::General), &[ValueId(1), ValueId(0)]);
        instrs.inst(Some(RegClass::General), &[ValueId(0)]);
        instrs.inst(Some(RegClass::Numeric), &[]);
        let constraints = ConstrBuilder::new().constraint(3, 1, hg(1)).build();

        // just don't panic
        try_allocation(&instrs, &constraints, 2, 2);
    }

    #[test]
    fn test_program1() {
        let mut instrs = InstSeq::default();
        instrs.inst(Some(RegClass::Numeric), &[]);
        instrs.inst(Some(RegClass::General), &[ValueId(0)]);
        instrs.inst(Some(RegClass::General), &[ValueId(1), ValueId(0)]);
        instrs.inst(Some(RegClass::General), &[ValueId(1)]);
        instrs.inst(
            Some(RegClass::Numeric),
            &[ValueId(0), ValueId(2), ValueId(3)],
        );
        instrs.inst(Some(RegClass::Numeric), &[ValueId(1), ValueId(2)]);
        instrs.inst(Some(RegClass::Numeric), &[ValueId(1)]);
        instrs.inst(Some(RegClass::General), &[ValueId(0)]);
        instrs.inst(Some(RegClass::Numeric), &[ValueId(5)]);
        try_allocation(&instrs, &HashMap::new(), 2, 2);

        let constraints = ConstrBuilder::new()
            .constraint(1, 0, hn(0))
            .constraint(4, 4, hn(0))
            .constraint(2, 1, hg(0))
            .constraint(3, 1, hg(1))
            .build();

        for (ng, nn) in [(2, 2), (5, 5), (10, 10)] {
            let alloc = try_allocation(&instrs, &constraints, ng, nn);

            for asmt in alloc.asmts.iter() {
                let constr_pt = ConstraintPt {
                    pos: asmt.pos,
                    vid: asmt.vid,
                };
                if let Some(fixed_hreg) = constraints.get(&constr_pt) {
                    assert_eq!(Loc::HardReg(*fixed_hreg), asmt.loc);
                }
            }
        }
    }
}
