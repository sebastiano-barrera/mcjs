use std::collections::HashSet;

use dynasm::dynasm;
use dynasmrt::x64::Rq;
use dynasmrt::Register;
use dynasmrt::{DynasmApi, DynasmLabelApi};
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter, FromRepr};

use crate::{
    interpreter::{self, ArithOp, CmpOp},
    jit::builder::{Cmp, Instr, ValueId, ValueType},
};

use super::{BoxedValue, Trace};

pub struct NativeThunk {
    buf: dynasmrt::ExecutableBuffer,
    snapshot_len: u16,
    entry_offset: dynasmrt::AssemblyOffset,
    sink: Vec<BoxedValue>,
}

fn encode_values(
    interpreter_values: &[interpreter::Value],
    rt_vals: &mut [i64],
    types: &mut [ValueType],
) {
    let len = interpreter_values.len();
    assert_eq!(len, rt_vals.len());
    assert_eq!(len, types.len());

    for i in 0..len {
        let (val, typ) = encode_value(&interpreter_values[i]);
        rt_vals[i] = val;
        types[i] = typ;
    }
}

fn encode_value(value: &interpreter::Value) -> (i64, ValueType) {
    let rt_val: i64 = match value {
        BoxedValue::Number(num) => unsafe { std::mem::transmute_copy(num) },
        BoxedValue::String(_) => todo!(),
        BoxedValue::Bool(bv) => {
            if *bv {
                1
            } else {
                0
            }
        }
        BoxedValue::Object(_) => todo!(),
        BoxedValue::Null => 0 as _,
        BoxedValue::Undefined => 0 as _,
        BoxedValue::SelfFunction => todo!(),
        BoxedValue::NativeFunction(_) => todo!(),
        BoxedValue::Closure(_) => todo!(),
    };

    (rt_val, ValueType::of(value))
}

fn decode_values(
    snap_rt_vals: &[i64],
    snap_types: &[ValueType],
    snapshot_values: &mut [interpreter::Value],
) {
    let len = snap_rt_vals.len();
    assert_eq!(len, snap_types.len());
    assert_eq!(len, snapshot_values.len());

    for (i, (val, typ)) in snap_rt_vals.iter().zip(snap_types.iter()).enumerate() {
        snapshot_values[i] = decode_value(*val, *typ);
    }
}

fn decode_value(rt_val: i64, typ: ValueType) -> interpreter::Value {
    match typ {
        ValueType::Bool => interpreter::Value::Bool(rt_val != 0),
        ValueType::Num => interpreter::Value::Number(unsafe { std::mem::transmute(rt_val) }),
        ValueType::Str => todo!(),
        ValueType::Obj => todo!(),
        ValueType::Null => interpreter::Value::Null,
        ValueType::Undefined => interpreter::Value::Undefined,
        ValueType::Function => todo!(),
        ValueType::Boxed => unreachable!(),
    }
}

impl NativeThunk {
    pub(crate) fn run(&self, interp_snapshot: &mut [interpreter::Value]) -> u64 {
        let snap_len = interp_snapshot.len();
        if snap_len != self.snapshot_len as usize {
            panic!(
                "wrong number of values for snapshot: {} instead of {}",
                interp_snapshot.len(),
                snap_len,
            );
        }

        let ptr = self.buf.ptr(self.entry_offset);

        let mut snap_rt_vals = vec![0; snap_len];
        let mut snap_types = vec![ValueType::Undefined; snap_len];
        encode_values(interp_snapshot, &mut snap_rt_vals, &mut snap_types);

        let thunk: extern "C" fn(
            snapshot_ptr: *mut i64,
            snapshot_types_ptr: *mut ValueType,
        ) -> u64 = unsafe { std::mem::transmute(ptr) };

        let ret = thunk(snap_rt_vals.as_mut_ptr(), snap_types.as_mut_ptr());

        decode_values(&snap_rt_vals, &snap_types, interp_snapshot);

        ret
    }

    #[cfg(test)]
    pub(crate) fn dump(&self) {
        use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction};

        let bytes = &self.buf[..];
        let bitness = 64;
        let start_rip = self.buf.as_ptr() as _;
        let decoder = Decoder::with_ip(bitness, bytes, start_rip, DecoderOptions::NONE);

        let mut formatter = iced_x86::NasmFormatter::new();
        {
            let opts = &mut formatter.options_mut();
            opts.set_show_branch_size(false);
        }

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

/// General purpose registers, available to be mapped to hregs.
///
/// The following registers are not part of the list, as they are reserved for other purposes:
///   - RDI (C func arg #1) points to the snapshot values (runtime encoded).
///   - RSI (C func arg #2) points to the array of value types.
///   - RSP is used as the stack pointer, in order to be compatible with the C calling convention.
///   - R15 is a scratch register
const GP_REGS: [Rq; 14] = [
    Rq::RAX,
    Rq::RBX,
    Rq::RCX,
    Rq::RDX,
    Rq::RSI,
    Rq::RDI,
    Rq::RBP,
    Rq::R8,
    Rq::R9,
    Rq::R10,
    Rq::R11,
    Rq::R12,
    Rq::R13,
    Rq::R14,
];

extern "C" fn ext_push_sink(value: u64) -> () {
    eprintln!("TODO: trace: push sink: {value} = {value:064b}");
}

pub(super) fn to_native(trace: &Trace) -> NativeThunk {
    use strum_macros::FromRepr;

    let snap_len = trace.snapshot_map.len();

    let mut asm = dynasmrt::x64::Assembler::new().unwrap();

    dynasm!(asm
    ; entry:
    );

    if trace.hreg_alloc.n_hregs() as usize > GP_REGS.len() {
        panic!("too many hardregs used in reg allocation!");
    }

    assert_eq!(trace.instrs.len(), trace.hreg_alloc.n_instrs());

    let is_enabled = trace.enabled_mask();
    let hreg_of_opt = |vid: ValueId| -> Option<Rq> {
        let reg_ndx = trace.hreg_alloc.hreg_of_instr(vid.0 as usize)?.0;
        GP_REGS.get(reg_ndx as usize).copied()
    };
    let hreg_of = |vid| hreg_of_opt(vid).unwrap_or_else(|| panic!("no hreg for {vid:?}"));

    let used_archregs: Vec<Rq> = {
        // establish an order. any order is fine, as long as we use it the
        // same in the trace exit
        let mut order = Vec::new();

        // the HashSet is used to dedup
        let mut used_archregs = HashSet::new();
        for indx in 0..trace.instrs.len() {
            if is_enabled[indx] {
                if let Some(areg) = hreg_of_opt(ValueId(indx as _)) {
                    used_archregs.insert(areg);
                }
            }
        }
        order.extend(used_archregs.into_iter());

        order.push(Rq::R15);
        order
    };

    for areg in used_archregs.iter() {
        dynasm!(asm; push Rq (areg.code()));
    }
    if trace.is_loop {
        dynasm!(asm
        ; loop_start:
        );
    }

    let assert_type = |trace: &Trace, vid: ValueId, expected_type: ValueType| {
        assert_eq!(
            trace.get_instr(vid).unwrap().result_type().unwrap(),
            expected_type
        );
    };

    let mut processing_phis = false;
    for (indx, instr) in trace.instrs.iter().enumerate() {
        if !is_enabled[indx] {
            continue;
        }

        let vid = ValueId(indx as u32);

        if !processing_phis {
            match instr {
                Instr::GetArg(_) => todo!(),
                Instr::Const(value) => {
                    // It is assumed that the IR already "knows" the type of
                    // this value, so we can discard it here
                    let (encoded_value, _) = encode_value(&value);
                    let areg = hreg_of(vid);
                    dynasm!(asm; mov Rq (areg.code()), QWORD encoded_value);
                }
                Instr::Not(_) => todo!(),
                Instr::Arith { op, a, b } => {
                    assert_type(trace, *a, ValueType::Num);
                    assert_type(trace, *b, ValueType::Num);

                    let a = hreg_of(*a).code();
                    let b = hreg_of(*b).code();
                    let tgt = hreg_of(vid).code();
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
                Instr::BoolOp { .. } => todo!(),
                Instr::Choose { .. } => todo!(),
                Instr::AssertTrue { cond } => {
                    assert_type(trace, *cond, ValueType::Bool);
                    let cond_instr = trace.get_instr(*cond).unwrap();

                    // TODO Refactor: move this somewhere else?
                    let cmp = if let Instr::Cmp(cmp) = cond_instr {
                        cmp
                    } else {
                        todo!("unsupported instruction for condition: {:?}", cond_instr);
                    };

                    let a = hreg_of(cmp.a);
                    let b = hreg_of(cmp.b);
                    trace_assert_cmp(&mut asm, cmp.ty, cmp.op, a, b);
                }
                Instr::AssertEqConst { .. } => todo!(),
                Instr::Unbox(ty, value) => {
                    let areg = hreg_of(*value);
                    gen_check_tag(&mut asm, snap_len, areg, *ty);
                }
                Instr::Box(_) => todo!(),
                Instr::Num2Str(_) => todo!(),
                Instr::ObjNew => todo!(),
                Instr::ObjSet { .. } => todo!(),
                Instr::ObjGet { .. } => todo!(),
                Instr::TypeOf(_) => todo!(),
                Instr::ClosureNew => todo!(),
                Instr::PushSink(vid) => {
                    let operand = hreg_of(*vid).code();
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

                // -- Snapshots
                // Variables are passed to and retrieved from the trace via an
                // array of interpreter::Value named "snapshot" (of length N).
                //
                // This array is allocated at runtime before starting the trace
                // proper. A pointer to the array is passed as register RBP.
                //
                // The interpeter initializes the snapshot with the values
                // indicated in the trace's snapshot_map (encoded in "native"
                // format, i.e. the format understood at runtime by the trace),
                // to allow the trace to read those values.
                //
                // When the trace exits, it updates the snapshot with the
                // values that have been mutated, in  order to allow the
                // interpreter to "apply" those changes to its own data
                // structures and continue execution.
                Instr::GetSnapshotItem { ndx: snap_ndx } => {
                    let snap_ndx = *snap_ndx as i32;
                    assert!(snap_ndx < snap_len as i32);

                    let tgt = hreg_of(vid).code();
                    let arr_ndx = (snap_len + tgt as usize) as _;

                    dynasm!(asm
                        ; mov Rq (tgt), [rdi + 8 * snap_ndx]
                        ; mov r15b, BYTE [rsi + snap_ndx]
                        ; mov BYTE [rsi + arr_ndx], r15b
                    );
                }
                Instr::SetSnapshotItem { ndx, value_id } => {
                    let argreg = hreg_of(*value_id).code();
                    dynasm!(asm
                    ; mov [rbp + 8 * (*ndx as i32)], Rq (argreg)
                    );
                }
            }
        }

        if processing_phis {
            match instr {
                Instr::Phi(old, new) => {
                    let old = hreg_of(*old).code();
                    let new = hreg_of(*new).code();
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
        dynasm!(asm; jmp <loop_start);
    } else {
        for areg in used_archregs.iter().rev() {
            dynasm!(asm; pop Rq (areg.code()));
        }
        dynasm!(asm; ret);
    }

    dynasm!(asm ; abort_trace:);
    // TODO Any chance of reusing this bit of code with the `!trace.is_loop` case?
    for areg in used_archregs.iter().rev() {
        dynasm!(asm; pop Rq (areg.code()));
    }
    // TODO This just won't work...
    dynasm!(asm; hlt);

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        sink: Vec::new(),
        snapshot_len: trace.snapshot_map.len() as u16,
    }
}

fn gen_check_tag(asm: &mut dynasmrt::x64::Assembler, snap_len: usize, reg: Rq, ty: ValueType) {
    let expected_tag = ty as u8;
    let type_arr_ndx = (snap_len + reg.code() as usize) as i32;

    dynasm!(asm
        ; cmp BYTE [rsi + type_arr_ndx], expected_tag as _
        ; jne >abort_trace);
}

fn trace_assert_cmp(asm: &mut dynasmrt::x64::Assembler, ty: ValueType, op: CmpOp, a: Rq, b: Rq) {
    if ValueType::Boxed == ty {
        trace_assert_typetag(asm, a, b);
    }

    dynasm!(asm; cmp Rq (a.code()), Rq(b.code()));
    match op {
        CmpOp::GE => dynasm!(asm; jl >abort_trace),
        CmpOp::GT => dynasm!(asm; jle >abort_trace),
        CmpOp::LT => dynasm!(asm; jge >abort_trace),
        CmpOp::LE => dynasm!(asm; jg >abort_trace),
        CmpOp::EQ => dynasm!(asm; jne >abort_trace),
        CmpOp::NE => dynasm!(asm; je >abort_trace),
    };
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
fn trace_assert_typetag(asm: &mut dynasmrt::x64::Assembler, a: Rq, b: Rq) {
    dynasm!(asm
        ; mov r15b, [rsi + a.code() as _]
        ; cmp r15b, [rsi + b.code() as _]
        ; jne >abort_trace
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
            let encoded = ty as u8 & 0b111;
            let readback_ty = decode_tag(encoded).unwrap();
            assert_eq!(ty, readback_ty);
        }
    }

    #[ignore]
    #[test]
    fn test_gen_tag_eq() {
        use strum::IntoEnumIterator;

        let pattern: u64 =
            0b0_001_010_011_100_101_110_111_000_001_010_011_100_101_110_111_000_001_010_011_100_101;

        let model = |a: Rq, b: Rq| {
            let a_tag = (pattern >> (a.code() * 3)) & 0b111;
            let b_tag = (pattern >> (b.code() * 3)) & 0b111;
            a_tag ^ b_tag
        };

        for a in GP_REGS.iter() {
            for b in GP_REGS.iter() {
                let mut asm = dynasmrt::x64::Assembler::new().unwrap();

                dynasm!(asm
                ; .arch x64
                ; entry:
                ; mov r15, QWORD pattern as _
                );

                trace_assert_typetag(&mut asm, *a, *b);

                dynasm!(asm; mov rax, r14; ret);

                let entry_offset = asm.labels().resolve_local("entry").unwrap();
                let buf = asm.finalize().unwrap();
                let proc: extern "C" fn() -> u64 =
                    unsafe { std::mem::transmute(buf.ptr(entry_offset)) };

                let result = proc();
                let expected = model(*a, *b);
                assert_eq!(
                    result,
                    expected,
                    "tag_eq({}, {}) = {:03b} but {:03b} expected",
                    a.code(),
                    b.code(),
                    result,
                    expected,
                );
            }
        }
    }
}
