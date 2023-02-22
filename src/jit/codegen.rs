use std::collections::{HashMap, HashSet};

use dynasm::dynasm;
use dynasmrt::x64::{Rq, Rx};
use dynasmrt::{DynamicLabel, Register};
use dynasmrt::{DynasmApi, DynasmLabelApi};
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter, FromRepr};

use crate::jit::regalloc::HardReg;
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
    code_size: usize,
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
    pub(crate) fn run(&self, interp_snapshot: &mut [interpreter::Value]) -> interpreter::Value {
        let snap_len = interp_snapshot.len();
        if snap_len != (self.snapshot_len as usize) {
            panic!(
                "wrong number of values for snapshot: {} instead of {}",
                snap_len, self.snapshot_len
            );
        }

        let ptr = self.buf.ptr(self.entry_offset);

        // The first snapshot slot is reserved for the return value, and is not mapped/considered
        // by `snapshot_map`, `snap_len` or anything else
        let mut snap_rt_vals = vec![0; snap_len + 1];
        let mut snap_types = vec![ValueType::Undefined; snap_len + 1];
        encode_values(
            interp_snapshot,
            &mut snap_rt_vals[1..],
            &mut snap_types[1..],
        );

        let thunk: extern "C" fn(snapshot_ptr: *mut i64, snapshot_types_ptr: *mut ValueType) =
            unsafe { std::mem::transmute(ptr) };

        thunk(snap_rt_vals.as_mut_ptr(), snap_types.as_mut_ptr());

        decode_values(&snap_rt_vals[1..], &snap_types[1..], interp_snapshot);
        let ret = decode_value(snap_rt_vals[0], snap_types[0]);

        ret
    }

    #[cfg(test)]
    pub(crate) fn dump(&self) {
        use iced_x86::{Decoder, DecoderOptions, Formatter, Instruction};

        let offset = self.entry_offset.0;
        let bytes = &self.buf[offset..];
        let bitness = 64;
        let start_rip = offset as u64;
        let decoder = Decoder::with_ip(bitness, bytes, start_rip, DecoderOptions::NONE);

        let mut formatter = iced_x86::NasmFormatter::new();
        {
            let opts = &mut formatter.options_mut();
            opts.set_show_branch_size(false);
            opts.set_branch_leading_zeros(false);
            opts.set_uppercase_hex(false);
            opts.set_signed_memory_displacements(true);
        }

        println!("-- Native ({} bytes)", self.code_size);
        // String implements FormatterOutput
        let mut line = String::new();

        for inst in decoder {
            // Format the instruction ("disassemble" it)
            line.clear();
            formatter.format(&inst, &mut line);

            // Eg. "00007FFAC46ACDB2    lea       rbp,[rsp-100h]"
            println!("{:4x}h  {}", inst.ip(), line);
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

const NUM_REGS: [Rx; 16] = [
    Rx::XMM0,
    Rx::XMM1,
    Rx::XMM2,
    Rx::XMM3,
    Rx::XMM4,
    Rx::XMM5,
    Rx::XMM6,
    Rx::XMM7,
    Rx::XMM8,
    Rx::XMM9,
    Rx::XMM10,
    Rx::XMM11,
    Rx::XMM12,
    Rx::XMM13,
    Rx::XMM14,
    Rx::XMM15,
];

extern "C" fn ext_push_sink(value: u64) -> () {
    eprintln!("TODO: trace: push sink: {value} = {value:064b}");
}

trait HardRegExt {
    fn expect_general(&self) -> Rq;
    fn expect_numeric(&self) -> Rx;
}

impl HardRegExt for HardReg {
    fn expect_general(&self) -> Rq {
        match self {
            HardReg::General(ndx) => GP_REGS[*ndx as usize],
            _ => panic!("expected a general reg"),
        }
    }

    fn expect_numeric(&self) -> Rx {
        match self {
            HardReg::Numeric(ndx) => NUM_REGS[*ndx as usize],
            _ => panic!("expected a numeric reg"),
        }
    }
}

pub(super) fn to_native(trace: &Trace) -> NativeThunk {
    if trace.hreg_alloc.n_general() as usize > GP_REGS.len() {
        panic!("too many hardregs used in reg allocation!");
    }
    assert_eq!(trace.instrs.len(), trace.hreg_alloc.n_instrs());

    let snap_len = trace.snapshot_map.len();
    let mut asm = dynasmrt::x64::Assembler::new().unwrap();

    let get_hreg = |vid| {
        trace
            .hreg_alloc
            .hreg_of_instr(vid)
            .unwrap_or_else(|| panic!("no hreg for {vid:?}"))
    };

    let assert_type = |trace: &Trace, vid: ValueId, expected_type: ValueType| {
        assert_eq!(
            trace.get_instr(vid).unwrap().result_type().unwrap(),
            expected_type
        );
    };

    // TODO Another good idea from LuaJIT: group all the constants at the start of the trace
    let num_consts = {
        let mut map = HashMap::new();
        for (vid, instr) in trace.iter_instrs() {
            if let Instr::Const(value) = instr {
                if let BoxedValue::Number(num) = value {
                    let lbl = asm.new_dynamic_label();
                    let num_raw = num.to_bits() as i64;
                    dynasm!(asm; =>lbl; .qword num_raw);
                    map.insert(vid, lbl);
                }
            }
        }

        map
    };

    dynasm!(asm ; entry: );

    let (used_gps, used_nums) = order_used_aregs(trace);
    for areg in used_gps.iter() {
        dynasm!(asm; push Rq (areg.code()));
    }

    let translate_instr = |asm: &mut dynasmrt::x64::Assembler, vid: ValueId, instr: &Instr| {
        match instr {
            Instr::Box(_) => todo!(),
            Instr::GetArg { .. } => todo!(),
            Instr::Const(value) => {
                // It is assumed that the IR already "knows" the type of
                // this value, so we can discard it here
                let (encoded_value, _) = encode_value(&value);

                match get_hreg(vid) {
                    HardReg::General(regndx) => {
                        let reg = GP_REGS[regndx as usize];
                        dynasm!(asm; mov Rq (reg.code()), QWORD encoded_value);
                    }
                    HardReg::Numeric(regndx) => {
                        let reg = NUM_REGS[regndx as usize];
                        let mem = *num_consts.get(&vid).unwrap();
                        dynasm!(asm; movsd Rx (reg.code()), [=>mem]);
                    }
                }
            }
            Instr::Not(_) => todo!(),
            Instr::Arith { op, a, b } => {
                assert_type(trace, *a, ValueType::Num);
                assert_type(trace, *b, ValueType::Num);

                let a = get_hreg(*a).expect_numeric();
                let b = get_hreg(*b).expect_numeric();
                let tgt = get_hreg(vid).expect_numeric();
                // TODO This would benefit from pin-pointing the target
                // register to the first operand register
                if tgt != a {
                    dynasm!(asm; movsd Rx (tgt.code()), Rx (a.code()));
                }
                // TODO This is completely wrong. The instructions for double-precision floating point numbers is MOVSD, ADDSD, VADDSD & co.
                match op {
                    ArithOp::Add => dynasm!(asm; addsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Sub => dynasm!(asm; subsd Rx (tgt.code()), Rx (b.code())),
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
            Instr::ExitUnless {
                cond: cmp,
                pre_snap_update,
            } => {
                write_snapshot(asm, &pre_snap_update, get_hreg);
                let a = get_hreg(cmp.a);
                let b = get_hreg(cmp.b);
                trace_assert_cmp(asm, cmp.ty, cmp.op, a, b);
            }
            Instr::Num2Str(_) => todo!(),
            Instr::ObjNew => todo!(),
            Instr::ObjSet { .. } => todo!(),
            Instr::ObjGet { .. } => todo!(),
            Instr::TypeOf(_) => todo!(),
            Instr::ClosureNew => todo!(),
            Instr::PushSink(vid) => {
                dynasm!(asm; push rsi);
                match get_hreg(*vid) {
                    HardReg::General(regndx) => dynasm!(asm; mov rsi, Rq (regndx as u8)),
                    HardReg::Numeric(regndx) => dynasm!(asm; movq rsi, Rx (regndx as u8)),
                }

                // NOTE We rely on ext_push_sink never touching any xmm*  register.
                // Otherwise, we would have to save and restore them, and that takes
                // quite a bit of code!q
                dynasm!(asm
                ; call ext_push_sink as _
                ; pop rsi
                );
            }
            Instr::Return(_) => todo!(),
            Instr::Phi(old, new) => {
                let old = get_hreg(*old);
                let new = get_hreg(*new);
                if old != new {
                    match (old, new) {
                        (HardReg::General(old), HardReg::General(new)) =>
                            dynasm!(asm; mov Rq (old as u8), Rq (new as u8)),

                        (HardReg::Numeric(old), HardReg::Numeric(new)) =>
                            dynasm!(asm; movsd Rx (old as u8), Rx (new as u8)),

                        _ => panic!(
                            "JIT bug: comparisons must be between either general or numeric registers, no mix"
                        ),
                    }
                } else {
                    eprintln!("JIT codegen: phi old hreg = new hreg; no machine instr necessary");
                }
            }

            // -- Snapshots
            // Variables are passed to and retrieved from the trace via the
            // "snapshot", which is a the 'juxtaposition' of two arrays:
            //   * values: [u64; N]        -- runtime-encoded values
            //   * types: [ValueType; N]   -- repr as u8
            //
            // These arrays are allocated at runtime before starting the
            // trace proper. A pointer to the array is passed as the first
            // argument to the C function that wraps the trace (reg RDI).
            //
            // `snapshot_map` tells which variable corresponds to each
            // index in the above values and types array.  The interpeter
            // uses this map to initialize the snapshot at trace start, and
            // to read values back from the snapshot at trace exit.
            Instr::GetSnapshotItem {
                ndx: snap_ndx,
                ty,
                post_snap_update,
            } => {
                let snap_ndx = *snap_ndx as i32;
                assert!(snap_ndx < snap_len as i32);

                dynasm!(asm
                 ; cmp BYTE [rsi + snap_ndx], *ty as i8
                 ; jne >abort_trace
                );

                match get_hreg(vid) {
                    HardReg::General(tgt) => {
                        dynasm!(asm; mov Rq (tgt as u8), QWORD [rdi + 8 * snap_ndx])
                    }
                    HardReg::Numeric(tgt) => {
                        dynasm!(asm; movsd Rx (tgt as u8), QWORD [rdi + 8 * snap_ndx])
                    }
                }
                write_snapshot(asm, &post_snap_update, get_hreg);
            }
        }
    };

    if trace.is_loop {
        dynasm!(asm; loop_start:);
    }
    for (vid, instr) in trace.iter_instrs() {
        translate_instr(&mut asm, vid, instr);
    }
    if trace.is_loop {
        dynasm!(asm; jmp <loop_start);
    }

    dynasm!(asm ; abort_trace:);
    // TODO Any chance of reusing this bit of code with the `!trace.is_loop` case?
    for areg in used_gps.iter().rev() {
        dynasm!(asm; pop Rq (areg.code()));
    }
    dynasm!(asm; ret);

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let code_size = asm.offset().0;
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        sink: Vec::new(),
        snapshot_len: trace.snapshot_map.len() as u16,
        code_size,
    }
}

fn write_snapshot<H>(asm: &mut dynasmrt::x64::Assembler, snap: &[Option<ValueId>], hreg_of: H)
where
    H: Fn(ValueId) -> HardReg,
{
    for (ndx, vid_opt) in snap.iter().enumerate() {
        if let Some(vid) = vid_opt {
            match hreg_of(*vid) {
                HardReg::General(regndx) => {
                    dynasm!(asm; mov [rbp + 8 * (ndx as i32)], Rq (regndx as u8))
                }
                HardReg::Numeric(regndx) => {
                    dynasm!(asm; movsd [rbp + 8 * (ndx as i32)], Rx (regndx as u8))
                }
            }
        }
    }
}

fn order_used_aregs(trace: &Trace) -> (Vec<Rq>, Vec<Rx>) {
    let mut used_gps = HashSet::new();
    let mut used_nums = HashSet::new();
    for vid in trace.iter_vids() {
        if let Some(hreg) = trace.hreg_alloc.hreg_of_instr(vid.clone().into()) {
            match hreg {
                HardReg::General(ndx) => {
                    used_gps.insert(GP_REGS.get(ndx as usize).copied().unwrap());
                }
                HardReg::Numeric(ndx) => {
                    used_nums.insert(NUM_REGS.get(ndx as usize).copied().unwrap());
                }
            }
        }
    }

    let mut order_gps: Vec<_> = used_gps.into_iter().collect();
    order_gps.push(Rq::R15);

    let order_nums = used_nums.into_iter().collect();
    (order_gps, order_nums)
}

fn trace_assert_type(asm: &mut dynasmrt::x64::Assembler, snap_len: usize, reg: Rq, ty: ValueType) {
    let expected_tag = ty as u8;
    let type_arr_ndx = (snap_len + reg.code() as usize) as i32;

    dynasm!(asm
        ; cmp BYTE [rsi + type_arr_ndx], expected_tag as _
        ; jne >abort_trace);
}

fn trace_assert_cmp(
    asm: &mut dynasmrt::x64::Assembler,
    ty: ValueType,
    op: CmpOp,
    a: HardReg,
    b: HardReg,
) {
    gen_cmp(ty, asm, a, b);
    match op {
        CmpOp::GE => dynasm!(asm; jl >abort_trace),
        CmpOp::GT => dynasm!(asm; jle >abort_trace),
        CmpOp::LT => dynasm!(asm; jge >abort_trace),
        CmpOp::LE => dynasm!(asm; jg >abort_trace),
        CmpOp::EQ => dynasm!(asm; jne >abort_trace),
        CmpOp::NE => dynasm!(asm; je >abort_trace),
    };
}

fn gen_cmp(ty: ValueType, asm: &mut dynasmrt::x64::Assembler, a: HardReg, b: HardReg) {
    if ValueType::Boxed == ty {
        let a = a.expect_general();
        let b = b.expect_general();
        trace_assert_typetag(asm, a, b);
    }

    match (a, b) {
        (HardReg::General(a), HardReg::General(b)) => dynasm!(asm; cmp Rq (a as u8), Rq (b as u8)),
        (HardReg::Numeric(a), HardReg::Numeric(b)) => {
            dynasm!(asm; ucomisd Rx (a as u8), Rx (b as u8))
        }
        _ => panic!(
            "JIT bug: comparisons must be between either general or numeric registers, no mix"
        ),
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
