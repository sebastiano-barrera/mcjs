use std::collections::HashMap;

use crate::jit::builder::ValueId;

use super::{builder::Instr, Trace};

pub(super) fn find_loop_variant<F>(instrs: &[Instr], is_phi_target: F) -> Vec<bool>
where
    F: Fn(ValueId) -> bool,
{
    let mut is_loop_variant = vec![false; instrs.len()];

    // for each phi(old, new), old is NOT loop variant; all its dependents are! (and new is *supposed* to be)

    for (ndx, instr) in instrs.iter().enumerate() {
        if is_loop_variant[ndx] {
            continue;
        }

        is_loop_variant[ndx] = instr
            .operands()
            .any(|operand| is_loop_variant[operand.0 as usize] || is_phi_target(*operand));
    }

    eprint!("loop variant: ");
    for (ndx, is_lv) in is_loop_variant.iter().enumerate() {
        if *is_lv {
            eprint!("v{}, ", ndx);
        }
    }

    is_loop_variant
}
