use dynasm::dynasm;
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter};

use crate::{
    interpreter::CmpOp,
    jit::builder::{Cmp, Instr, ValueType},
};

use super::{BoxedValue, Trace};

pub struct NativeThunk {
    buf: dynasmrt::ExecutableBuffer,
    entry_offset: dynasmrt::AssemblyOffset,
    sink: Vec<BoxedValue>,
}

impl NativeThunk {
    pub(crate) fn run(&self) -> u64 {
        let ptr = self.buf.ptr(self.entry_offset);
        let thunk: extern "C" fn() -> u64 = unsafe { std::mem::transmute(ptr) };
        thunk()
    }
}

#[derive(PartialEq, Eq, EnumIter, EnumCount)]
enum ArchReg {
    RAX,
    RBX,
    RCX,
    RDX,
    RSI,
    RDI,
    RSP,
    RBP,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    // R14 is the scratch register.  Never allocated.
    // R15 is the type tags register.  Never allocated.
}

pub(super) fn to_native(trace: &Trace) -> NativeThunk {
    use dynasmrt::{DynasmApi, DynasmLabelApi};
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();

    dynasm!(asm
    ; .arch x64
    ; entry:
    );

    if trace.hreg_alloc.n_hregs() as usize > ArchReg::COUNT {
        panic!("too many hardregs used in reg allocation!");
    }

    assert_eq!(trace.instrs.len(), trace.hreg_alloc.n_instrs());

    for (indx, instr) in trace.instrs.iter().enumerate() {
        let hreg_ndx = |indx| {
            let hreg = trace.hreg_alloc.hreg_of_instr(indx);
            hreg.unwrap().0 as u8
        };

        match instr {
            Instr::TraceParam(ndx) => {
                //dynasm!(asm; );
            }
            Instr::GetArg(_) => todo!(),
            Instr::Const(value) => {
                let encoded_value = value.encode();
                dynasm!(asm; mov Rq (hreg_ndx(indx)), QWORD encoded_value);
            }
            Instr::Not(_) => todo!(),
            Instr::Arith { op, a, b } => todo!(),
            Instr::Cmp(Cmp {
                ty: ValueType::Num,
                op,
                a,
                b,
            }) => {
                let a = a.0 as usize;
                let b = b.0 as usize;
                match op {
                    CmpOp::GE => todo!(),
                    CmpOp::GT => todo!(),
                    CmpOp::LT => todo!(),
                    CmpOp::LE => {
                        // TODO Make sure the flag correctly goes to the instruction that needs it!
                        dynasm!(asm; cmp Rq (hreg_ndx(a)), Rq (hreg_ndx(b)));
                    }
                    CmpOp::EQ => todo!(),
                    CmpOp::NE => todo!(),
                }
            }
            Instr::Cmp(_) => {
                // Do nothing.  Rather, when an instruction *using* this result
                // occurs later, we'll do something about it. In particular,
                // we'll choose whether to do a cmp + jz/ja/je/etc. (using a
                // flag) or store the result in a dedicated boolean register.
            }
            Instr::BoolOp { op, a, b } => todo!(),
            Instr::Choose {
                ty,
                cond,
                if_true,
                if_false,
            } => todo!(),
            Instr::AssertTrue { cond } => {
                let cond_instr = trace.get_instr(*cond).unwrap();
                let cond_ty = cond_instr.result_type();
                assert_eq!(
                    cond_ty,
                    Some(ValueType::Bool),
                    "JIT bug: expected boolean, not {:?}",
                    cond_ty
                );

                if let Instr::Cmp(cmp) = cond_instr {
                    match cmp.ty {
                        ValueType::Boxed => {
                            todo!("Check type tag, then check register, then check the value as if it was unboxed.");
                        }
                        ValueType::Bool => todo!(),
                        ValueType::Num => todo!(),
                        ValueType::Str => todo!(),
                        ValueType::Obj => todo!(),
                        ValueType::Null => todo!(),
                        ValueType::Undefined => todo!(),
                        ValueType::Function => todo!(),
                    }
                } else {
                }
            }
            Instr::AssertEqConst { x, expected } => todo!(),
            Instr::Unbox(_, _) => todo!(),
            Instr::Box(_) => todo!(),
            Instr::Num2Str(_) => todo!(),
            Instr::ObjNew => todo!(),
            Instr::ObjSet { obj, key, value } => todo!(),
            Instr::ObjGet { obj, key } => todo!(),
            Instr::TypeOf(_) => todo!(),
            Instr::ClosureNew => todo!(),
            Instr::PushSink(_) => todo!(),
            Instr::Return(_) => todo!(),
        }
    }

    dynasm!(asm
    ; .arch x64
    ; ret
    );

    todo!("translate phis, and loopback IF this is a looping trace");

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        sink: Vec::new(),
    }
}

trait Encodable {
    fn encode(&self) -> i64;
}

impl Encodable for BoxedValue {
    fn encode(&self) -> i64 {
        match self {
            BoxedValue::Number(num) => unsafe { std::mem::transmute_copy(num) },
            BoxedValue::String(_) => todo!(),
            BoxedValue::Bool(_) => todo!(),
            BoxedValue::Object(_) => todo!(),
            BoxedValue::Null => todo!(),
            BoxedValue::Undefined => todo!(),
            BoxedValue::SelfFunction => todo!(),
            BoxedValue::NativeFunction(_) => todo!(),
            BoxedValue::Closure(_) => todo!(),
        }
    }
}

fn encode_tag(ty: ValueType) -> u8 {
    match ty {
        ValueType::Boxed => 0,
        ValueType::Bool => 1,
        ValueType::Num => 2,
        ValueType::Str => 3,
        ValueType::Obj => 4,
        ValueType::Null => 5,
        ValueType::Undefined => 6,
        ValueType::Function => 7,
    }
}

fn decode_tag(val: u8) -> Option<ValueType> {
    match val {
        0 => Some(ValueType::Boxed),
        1 => Some(ValueType::Bool),
        2 => Some(ValueType::Num),
        3 => Some(ValueType::Str),
        4 => Some(ValueType::Obj),
        5 => Some(ValueType::Null),
        6 => Some(ValueType::Undefined),
        7 => Some(ValueType::Function),
        _ => None,
    }
}

fn gen_tag_eq(asm: &mut dynasmrt::x64::Assembler, a_id: u8, b_id: u8) {
    use dynasmrt::DynasmApi;

    assert!(a_id < ArchReg::COUNT as u8);
    assert!(b_id < ArchReg::COUNT as u8);

    let (lo_id, hi_id) = min_max(a_id, b_id);
    let delta_id = hi_id - lo_id;

    dynasm!(asm
        ; .alias type_tags, r15
        ; .alias scratch, r14
        ; mov scratch, type_tags
        // align hi to lo, then compare them via xor
        ; shr scratch, 3 * delta_id as i8
        ; xor scratch, type_tags
        // move comparison result to LSB
        ; shr scratch, 3 * lo_id as i8
        ; and scratch, 0b111
    );
}

fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::builder::ValueType;

    #[test]
    fn test_tag_encoding() {
        use strum::IntoEnumIterator;

        for ty in ValueType::iter() {
            // Limit to 3 bits
            let encoded = encode_tag(ty) & 0b111;
            let readback_ty = decode_tag(encoded).unwrap();
            assert_eq!(ty, readback_ty);
        }
    }

    #[test]
    fn test_gen_tag_eq() {
        use strum::IntoEnumIterator;

        let pattern: u64 =
            0b0_001_010_011_100_101_110_111_000_001_010_011_100_101_110_111_000_001_010_011_100_101;

        let model = |a_id, b_id| {
            let a_tag = (pattern >> (a_id * 3)) & 0b111;
            let b_tag = (pattern >> (b_id * 3)) & 0b111;
            a_tag ^ b_tag
        };

        for (a_id, a) in ArchReg::iter().enumerate() {
            let a_id = a_id as u8;

            for (b_id, b) in ArchReg::iter().enumerate() {
                let b_id = b_id as u8;

                use dynasmrt::{DynasmApi, DynasmLabelApi};
                let mut asm = dynasmrt::x64::Assembler::new().unwrap();

                dynasm!(asm
                ; .arch x64
                ; entry:
                ; mov r15, QWORD pattern as _
                );

                gen_tag_eq(&mut asm, a_id, b_id);

                dynasm!(asm; mov rax, r14; ret);

                let entry_offset = asm.labels().resolve_local("entry").unwrap();
                let buf = asm.finalize().unwrap();
                let proc: extern "C" fn() -> u64 =
                    unsafe { std::mem::transmute(buf.ptr(entry_offset)) };

                let result = proc();
                let expected = model(a_id, b_id);
                assert_eq!(
                    result, expected,
                    "tag_eq({a_id}, {b_id}) = {result:03b} but {expected:03b} expected"
                );
            }
        }
    }
}
