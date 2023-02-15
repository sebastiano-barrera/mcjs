use dynasm::dynasm;
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter, FromRepr};

use crate::{
    interpreter::{ArithOp, CmpOp},
    jit::builder::{Cmp, Instr, ValueId, ValueType},
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

    #[cfg(test)]
    pub(crate) fn dump(&self) {
        use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction, NasmFormatter};

        let bytes = &self.buf[..];
        let bitness = 64;
        let start_rip = self.buf.as_ptr() as _;
        let mut decoder = Decoder::with_ip(bitness, bytes, start_rip, DecoderOptions::NONE);

        let mut formatter = NasmFormatter::new();

        // String implements FormatterOutput
        let mut line = String::new();

        for inst in decoder {
            // Format the instruction ("disassemble" it)
            line.clear();
            formatter.format(&inst, &mut line);

            // Eg. "00007FFAC46ACDB2    lea       rbp,[rsp-100h]"
            println!("{:016x} {}", inst.ip(), line);
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, EnumIter, EnumCount, FromRepr)]
#[repr(u8)]
enum ArchReg {
    // The order matters! If (a: ArchReg), then (a as u8) will be passed to dynasm to identify registers!
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

impl ArchReg {
    fn with_id(id: u8) -> ArchReg {
        ArchReg::from_repr(id).unwrap()
    }

    fn id(&self) -> u8 {
        *self as u8
    }
}

extern "C" fn ext_push_sink(value: u64) -> () {
    eprintln!("trace: push sink: {value} = {value:064b}");
}

pub(super) fn to_native(trace: &Trace) -> NativeThunk {
    use dynasmrt::{DynasmApi, DynasmLabelApi};
    use strum_macros::FromRepr;
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();

    dynasm!(asm
    ; .arch x64
    ; entry:
    );

    if trace.hreg_alloc.n_hregs() as usize > ArchReg::COUNT {
        panic!("too many hardregs used in reg allocation!");
    }

    assert_eq!(trace.instrs.len(), trace.hreg_alloc.n_instrs());

    let is_enabled = trace.enabled_mask();
    let hreg_of = |vid: ValueId| {
        let reg_id = trace
            .hreg_alloc
            .hreg_of_instr(vid.0 as usize)
            .unwrap_or_else(|| panic!("no hreg for {vid:?}"))
            .0 as u8;
        ArchReg::from_repr(reg_id).unwrap()
    };

    let mut processing_phis = false;

    for (indx, instr) in trace.instrs.iter().enumerate() {
        if !is_enabled[indx] {
            continue;
        }

        let vid = ValueId(indx as u32);

        if !processing_phis {
            match instr {
                Instr::TraceParam(ndx) => {
                    //dynasm!(asm; );
                }
                Instr::GetArg(_) => todo!(),
                Instr::Const(value) => {
                    let encoded_value = encode_value(&value);
                    dynasm!(asm; mov Rq (hreg_of(vid).id()), QWORD encoded_value);
                }
                Instr::Not(_) => todo!(),
                Instr::Arith { op, a, b } => {
                    let a = hreg_of(*a).id();
                    let b = hreg_of(*b).id();
                    let tgt = hreg_of(vid).id();
                    // TODO This would benefit from pin-pointing the target
                    // register to the first operand register
                    dynasm!(asm; mov Rq (tgt), Rq (a));
                    match op {
                        ArithOp::Add => dynasm!(asm; add Rq (tgt), Rq (b)),
                        ArithOp::Sub => dynasm!(asm; sub Rq (tgt), Rq (b)),
                        ArithOp::Mul => todo!("mul"),
                        ArithOp::Div => todo!("div"),
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

                    // TODO Refactor: move this somewhere else?
                    if let Instr::Cmp(cmp) = cond_instr {
                        let a = hreg_of(cmp.a);
                        let b = hreg_of(cmp.b);
                        gen_cmp(&mut asm, cmp.ty, cmp.op, a, b);
                    } else {
                        todo!("unsupported instruction for condition: {:?}", cond_instr);
                    }
                    dynasm!(asm; jnz >abort_trace);
                }
                Instr::AssertEqConst { x, expected } => todo!(),
                Instr::Unbox(ty, value) => {
                    let areg = hreg_of(*value);
                    gen_check_tag(&mut asm, areg, *ty);
                    dynasm!(asm; jnz >abort_trace);

                    let encoded_tag = encode_tag(*ty) as u64;
                    dynasm!(asm
                    ; mov r14, QWORD encoded_tag as _
                    ; shl r14, 3 * areg.id() as i8
                    ; or r15, r14
                    );
                }
                Instr::Box(_) => todo!(),
                Instr::Num2Str(_) => todo!(),
                Instr::ObjNew => todo!(),
                Instr::ObjSet { obj, key, value } => todo!(),
                Instr::ObjGet { obj, key } => todo!(),
                Instr::TypeOf(_) => todo!(),
                Instr::ClosureNew => todo!(),
                Instr::PushSink(vid) => {
                    let operand = hreg_of(*vid).id();
                    dynasm!(asm
                    ; push rsi
                    ; mov rsi, Rq (operand)
                    ; call ext_push_sink as _
                    ; pop rsi
                    );
                }
                Instr::Return(_) => todo!(),
                Instr::Phi(_, _) => {
                    processing_phis = true;
                }
            }
        }

        if processing_phis {
            match instr {
                Instr::Phi(old, new) => {
                    let old = hreg_of(*old).id();
                    let new = hreg_of(*new).id();
                    if old != new {
                        dynasm!(asm; mov Rq (old), Rq (new));
                    } else {
                        eprintln!(
                            "JIT codegen: phi old hreg = new hreg; no machine instr necessary"
                        );
                    }
                }
                _ => {
                    panic!("JIT bug: phi instructions must be located at the end of the trace")
                }
            }
        }
    }

    if trace.is_loop {
        dynasm!(asm; jmp <entry);
    } else {
        dynasm!(asm; ret);
    }

    dynasm!(asm
        ; abort_trace:
        ; hlt);

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        sink: Vec::new(),
    }
}

fn gen_check_tag(asm: &mut dynasmrt::x64::Assembler, reg: ArchReg, ty: ValueType) {
    use dynasmrt::DynasmApi;

    let encoded_tag = encode_tag(ty) as u32;

    dynasm!(asm
        ; .alias type_tags, r15
        ; .alias scratch, r14
        ; mov scratch, type_tags
        // align hi to lo, then compare them via xor
        ; shr scratch, 3 * (reg.id() as i8)
        ; and scratch, 0b111
        ; cmp scratch, DWORD encoded_tag as _
    );
}

fn gen_cmp(asm: &mut dynasmrt::x64::Assembler, ty: ValueType, op: CmpOp, a: ArchReg, b: ArchReg) {
    use dynasmrt::{DynasmApi, DynasmLabelApi};

    if ValueType::Boxed == ty {
        gen_tag_eq(asm, a, b);
    }

    dynasm!(asm
        ; jnz ->cmp_end
        ; cmp Rq (a as u8), Rq(b as u8)
        ; ->cmp_end:
    );
}

fn encode_value(value: &BoxedValue) -> i64 {
    match value {
        BoxedValue::Number(num) => unsafe { std::mem::transmute_copy(num) },
        BoxedValue::String(_) => todo!(),
        BoxedValue::Bool(_) => todo!(),
        BoxedValue::Object(_) => todo!(),
        BoxedValue::Null => 0 as i64,
        BoxedValue::Undefined => todo!(),
        BoxedValue::SelfFunction => todo!(),
        BoxedValue::NativeFunction(_) => todo!(),
        BoxedValue::Closure(_) => todo!(),
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

/// Generate code that compares the type tags for the two given registers.
///
/// The behavior of the generated code is ONLY defined IF the designated
/// registers hold values of type Boxed.
///
/// The generated code clobbers r14 (the scratch register) and sets ZF = true
/// iff the type tags are the same.
fn gen_tag_eq(asm: &mut dynasmrt::x64::Assembler, a: ArchReg, b: ArchReg) {
    use dynasmrt::DynasmApi;

    let a_id = a as u8;
    let b_id = b as u8;
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

        let model = |a: ArchReg, b: ArchReg| {
            let a_tag = (pattern >> (a.id() * 3)) & 0b111;
            let b_tag = (pattern >> (b.id() * 3)) & 0b111;
            a_tag ^ b_tag
        };

        for a in ArchReg::iter() {
            for b in ArchReg::iter() {
                use dynasmrt::{DynasmApi, DynasmLabelApi};
                let mut asm = dynasmrt::x64::Assembler::new().unwrap();

                dynasm!(asm
                ; .arch x64
                ; entry:
                ; mov r15, QWORD pattern as _
                );

                gen_tag_eq(&mut asm, a, b);

                dynasm!(asm; mov rax, r14; ret);

                let entry_offset = asm.labels().resolve_local("entry").unwrap();
                let buf = asm.finalize().unwrap();
                let proc: extern "C" fn() -> u64 =
                    unsafe { std::mem::transmute(buf.ptr(entry_offset)) };

                let result = proc();
                let expected = model(a, b);
                assert_eq!(
                    result,
                    expected,
                    "tag_eq({}, {}) = {:03b} but {:03b} expected",
                    a.id(),
                    b.id(),
                    result,
                    expected,
                );
            }
        }
    }
}
