use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;

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
    entry_offset: dynasmrt::AssemblyOffset,

    snapshot_len: u16,
    rt_data: RuntimeData,

    code_size: usize,
    #[cfg(test)]
    n_runs: AtomicUsize,
}

/// Data available at runtime, accessible from the running trace.
#[derive(Clone)]
struct RuntimeData {
    // TODO Any advantage in splitting const and mtu strs? They can be
    // distinguished, based on the context of the call to encode_value[s].
    strs: Vec<String>,

    sink: Vec<BoxedValue>,
}

type StringId = u64;

impl RuntimeData {
    fn new() -> Self {
        RuntimeData {
            strs: Vec::new(),
            sink: Vec::new(),
        }
    }

    fn add_str(&mut self, value: String) -> StringId {
        self.strs.push(value);
        (self.strs.len() - 1) as u64
    }
    fn get_str(&self, id: StringId) -> Option<&String> {
        self.strs.get(id as usize)
    }

    fn dump(&self) {
        println!("strs[{}]", self.strs.len());
        for (ndx, s) in self.strs.iter().enumerate() {
            println!("  [{}] = [{}]{:?}", ndx, s.len(), s);
        }
    }
}

fn encode_values(
    interpreter_values: &[interpreter::Value],
    rt_data: &mut RuntimeData,
    rt_vals: &mut [i64],
    types: &mut [ValueType],
) {
    let len = interpreter_values.len();
    assert_eq!(len, rt_vals.len());
    assert_eq!(len, types.len());

    for i in 0..len {
        let (val, typ) = encode_value(rt_data, &interpreter_values[i]);
        rt_vals[i] = val;
        types[i] = typ;
    }
}

fn encode_value(rt_data: &mut RuntimeData, value: &interpreter::Value) -> (i64, ValueType) {
    let rt_val: i64 = match value {
        BoxedValue::Number(num) => unsafe { std::mem::transmute_copy(num) },
        BoxedValue::String(str) => {
            // TODO Any safe way to avoid having to copy the string?
            let s = str.clone().into_owned();
            let sid = rt_data.add_str(s);
            sid as i64
        }
        BoxedValue::Bool(bv) => {
            if *bv {
                1
            } else {
                0
            }
        }
        BoxedValue::Object(_) => todo!("(big feat) encode object ref (remember the GC!)"),
        BoxedValue::Null => 0 as _,
        BoxedValue::Undefined => 0 as _,
        BoxedValue::SelfFunction => todo!("(small feat) SelfFunction"),
        BoxedValue::NativeFunction(_) => todo!("(big feat) NativeFunction"),
        BoxedValue::Closure(_) => todo!("(big feat) encode closure"),
    };

    (rt_val, ValueType::of(value))
}

fn decode_values(
    rt_data: &RuntimeData,
    snap_rt_vals: &[i64],
    snap_types: &[ValueType],
    snapshot_values: &mut [interpreter::Value],
) {
    let len = snap_rt_vals.len();
    assert_eq!(len, snap_types.len());
    assert_eq!(len, snapshot_values.len());

    for (i, (val, typ)) in snap_rt_vals.iter().zip(snap_types.iter()).enumerate() {
        snapshot_values[i] = decode_value(rt_data, *val, *typ);
    }
}

fn decode_value(rt_data: &RuntimeData, rt_val: i64, typ: ValueType) -> interpreter::Value {
    match typ {
        ValueType::Bool => interpreter::Value::Bool(rt_val != 0),
        ValueType::Num => interpreter::Value::Number(unsafe { std::mem::transmute(rt_val) }),
        ValueType::Str => {
            let sid = rt_val as u64;
            // TODO Any way to avoid this string copy?
            let s = rt_data.get_str(sid).unwrap().clone();
            interpreter::Value::String(s.into())
        }
        ValueType::Obj => todo!("(big feat) decode object ref (remember the GC!)"),
        ValueType::Null => interpreter::Value::Null,
        ValueType::Undefined => interpreter::Value::Undefined,
        ValueType::Function => todo!("(big feat) decode function ref"),
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

        // TODO Probably this copy could be saved, at least in part
        let mut rt_data = self.rt_data.clone();

        let ptr = self.buf.ptr(self.entry_offset);

        // The first snapshot slot is reserved for the return value, and is not mapped/considered
        // by `snapshot_map`, `snap_len` or anything else
        let mut snap_rt_vals = vec![0; snap_len + 1];
        let mut snap_types = vec![ValueType::Undefined; snap_len + 1];
        encode_values(
            interp_snapshot,
            &mut rt_data,
            &mut snap_rt_vals[1..],
            &mut snap_types[1..],
        );

        let thunk: extern "C" fn(
            snapshot_ptr: *mut i64,
            snapshot_types_ptr: *mut ValueType,
            rt_data: *mut RuntimeData,
        ) = unsafe { std::mem::transmute(ptr) };

        thunk(
            snap_rt_vals.as_mut_ptr(),
            snap_types.as_mut_ptr(),
            &mut rt_data,
        );

        decode_values(
            &rt_data,
            &snap_rt_vals[1..],
            &snap_types[1..],
            interp_snapshot,
        );
        let ret = decode_value(&rt_data, snap_rt_vals[0], snap_types[0]);

        #[cfg(test)]
        self.n_runs
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);

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
        println!("  init runtime data");
        self.rt_data.dump();
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
///   - RDX (C func arg #3) points to the RuntimeData.
///   - RSP is used as the stack pointer, in order to be compatible with the C calling convention.
///   - R15 is a scratch register
const GP_REGS: [Rq; 13] = [
    Rq::RAX,
    Rq::RBX,
    Rq::RCX,
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

pub extern "C" fn ext_push_sink(value: u64) -> () {
    // Figure out a better way of manipulating the `sink` vector
    println!("TODO(big feat): trace: push sink: {value:064b} = {value}");
}

// This function has the same prototype (as a C function) as the generated
// code. Since registers RDI, RSI and RDX are reserved (not used by the
// register allocation) and never touched in any other way, a naked `call`
// instruction "works" without further register shuffling.
pub extern "C" fn ext_str_cmp(value: u64) -> () {
    // Figure out a better way of manipulating the `sink` vector
    println!("TODO(big feat): trace: push sink: {value:064b} = {value}");
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
    let mut rt_data = RuntimeData::new();

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

    // TODO(big feat) Another good idea from LuaJIT: group all the constants at the start of the trace
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
    let saved_gps: Box<[_]> = used_gps
        .iter()
        .filter(|areg| is_callee_saved(**areg))
        .copied()
        .collect();

    // TODO(future rust): .next_multiple_of(16)
    let mut stack_codegen = StackCodegen::new();
    stack_codegen.start_chunk(&mut asm, &saved_gps, false);

    let mut translate_instr = |asm: &mut dynasmrt::x64::Assembler, vid: ValueId, instr: &Instr| {
        match instr {
            Instr::Box(_) => todo!("(big feat) Instr::Box"),
            Instr::GetArg { .. } => todo!("(big feat) Instr::GetArg"),
            Instr::Const(value) => {
                // It is assumed that the IR already "knows" the type of
                // this value, so we can discard it here
                let (encoded_value, _) = encode_value(&mut rt_data, &value);

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
            Instr::Not(_) => todo!("(small feat) Instr::Box"),
            Instr::Arith { op, a, b } => {
                assert_type(trace, *a, ValueType::Num);
                assert_type(trace, *b, ValueType::Num);

                let a = get_hreg(*a).expect_numeric();
                let b = get_hreg(*b).expect_numeric();
                let tgt = get_hreg(vid).expect_numeric();
                // TODO(idea) This would benefit from pin-pointing the target
                // register to the first operand register
                if tgt != a {
                    dynasm!(asm; movsd Rx (tgt.code()), Rx (a.code()));
                }

                match op {
                    ArithOp::Add => dynasm!(asm; addsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Sub => dynasm!(asm; subsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Mul => dynasm!(asm; mulsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Div => todo!("TODO(small feat) div"),
                }
            }
            Instr::Cmp(_) => {
                // Do nothing.  Rather, when an instruction *using* this result
                // occurs later, we'll do something about it. In particular,
                // we'll choose whether to do a cmp + jz/ja/je/etc. (using a
                // flag) or store the result in a dedicated boolean register.
            }
            Instr::BoolOp { .. } => todo!("TODO(small feat) boolop"),
            Instr::Choose { .. } => todo!("TODO(small feat) choose"),
            Instr::ExitUnless {
                cond: cmp,
                pre_snap_update,
            } => {
                write_snapshot(asm, &pre_snap_update, get_hreg);
                let a = get_hreg(cmp.a);
                let b = get_hreg(cmp.b);
                trace_assert_cmp(asm, cmp.ty, cmp.op, a, b);
            }
            Instr::Num2Str(_) => todo!("TODO(big feat) Num2Str"),
            Instr::ObjNew => todo!("TODO(big feat) objects in JIT"),
            Instr::ObjSet { .. } => todo!("TODO(big feat) objects in JIT"),
            Instr::ObjGet { .. } => todo!("TODO(big feat) objects in JIT"),
            Instr::TypeOf(_) => todo!("TODO(small feat) TypeOf"),
            Instr::ClosureNew => {
                // Nothing to do.  This instruction only serves to assign
                // the value ID to a new "virtual closure", which can only be
                // inlined later on. (It can't esacpe the trace, by design.)
                let tgt = get_hreg(vid).expect_general().code();
                dynasm!(asm; xor Rq (tgt), Rq (tgt));
            }
            Instr::PushSink(vid) => {
                let mut saved_regs = vec![Rq::RSI, Rq::RDI, Rq::RAX];
                for &areg in used_gps.iter() {
                    if areg != Rq::RSI
                        && areg != Rq::RDI
                        && areg != Rq::RAX
                        && !is_callee_saved(areg)
                    {
                        saved_regs.push(areg);
                    }
                }

                stack_codegen.start_chunk(asm, &saved_regs, true);

                match get_hreg(*vid) {
                    HardReg::General(regndx) => dynasm!(asm; mov rdi, Rq (regndx as u8)),
                    HardReg::Numeric(regndx) => dynasm!(asm; movq rdi, Rx (regndx as u8)),
                }

                let func = todo!();
                dynasm!(asm; call ext_push_sink as i32);

                stack_codegen.end_chunk(asm);
            }
            Instr::Return(_) => todo!("TODO return"),
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

                // `+ 1` because the first slot is reserved for the return value
                dynasm!(asm
                 ; cmp BYTE [rsi + snap_ndx + 1], *ty as i8
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

    stack_codegen.end_chunk(&mut asm);

    dynasm!(asm; ret);

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let code_size = asm.offset().0;
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        snapshot_len: trace.snapshot_map.len() as u16,
        code_size,
        rt_data,
        #[cfg(test)]
        n_runs: AtomicUsize::new(0),
    }
}

fn is_callee_saved(areg: Rq) -> bool {
    match areg {
        Rq::RBX | Rq::RSP | Rq::RBP | Rq::R12 | Rq::R13 | Rq::R14 | Rq::R15 => true,
        _ => false,
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

/// Helper for generating stack read/write.
///
/// Its main value is to make sure, in certain cases, that RSP is divisible
/// by 16 as required by the C ABI. If this isn't done, C functions called
/// directly or indirectly within this trace might, at some point, execute an
/// instruction such as this:
///
///    movaps %xmm2, 0xa0(%rsp)
///
/// which throws a hardware exception if the memory address is not aligned
/// to 16 bytes. (C compilers make sure that the offset (in this case 0xa0)
/// is divisible by 16, too, so %rsp is the only variable here.)
struct StackCodegen {
    regs: Vec<Rq>,
    chunks_size: Vec<(u8, u8)>,
}
impl StackCodegen {
    // TODO(big feat) Any way to make sure that start_chunk is coupled with an
    // end_chunk? Compile-time preferable, but runtime also useful. I had some
    // asserts in an impl Drop, but it made the program "panic while panicking",
    // which makes things harder.

    fn new() -> Self {
        StackCodegen {
            regs: Vec::new(),
            chunks_size: Vec::new(),
        }
    }

    // TODO(opt): encode set of registers as a bitmask
    fn start_chunk(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq], align16: bool) {
        let chunk_sz = regs.len() as u8;
        let padding = if align16 { 1 } else { 0 };

        self.chunks_size.push((chunk_sz, padding));
        self.regs.extend_from_slice(regs);

        let delta_rsp = 8 * (chunk_sz + padding) as i32;
        dynasm!(asm; sub rsp, delta_rsp);
        for (i, reg) in regs.iter().enumerate() {
            dynasm!(asm; mov [rsp + i as i32 * 8], Rq (reg.code()));
        }
    }

    fn end_chunk(&mut self, asm: &mut dynasmrt::x64::Assembler) {
        let (chunk_sz, padding) = self.chunks_size.pop().unwrap();
        let end = self.regs.len();
        let start = end - chunk_sz as usize;
        let regs = self.regs.drain(start..end);

        for (i, reg) in regs.enumerate() {
            dynasm!(asm; mov Rq (reg.code()), [rsp + i as i32 * 8]);
        }
        dynasm!(asm; add rsp, 8 * (chunk_sz + padding) as i32);
    }
}

fn trace_assert_type(asm: &mut dynasmrt::x64::Assembler, snap_len: usize, reg: Rq, ty: ValueType) {
    let expected_tag = ty as u8;
    // + 1 because the first slot is reserved for the return value
    let type_arr_ndx = (snap_len + 1 + reg.code() as usize) as i32;

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
    gen_cmp(asm, ty, a, b);
    match op {
        CmpOp::GE => dynasm!(asm; jl >abort_trace),
        CmpOp::GT => dynasm!(asm; jle >abort_trace),
        CmpOp::LT => dynasm!(asm; jge >abort_trace),
        CmpOp::LE => dynasm!(asm; jg >abort_trace),
        CmpOp::EQ => dynasm!(asm; jne >abort_trace),
        CmpOp::NE => dynasm!(asm; je >abort_trace),
    };
}

fn gen_cmp(asm: &mut dynasmrt::x64::Assembler, ty: ValueType, a: HardReg, b: HardReg) {
    match ty {
        ValueType::Boxed => {
            let a = a.expect_general();
            let b = b.expect_general();
            trace_assert_typetag(asm, a, b);
        }
        ValueType::Bool => todo!(),
        ValueType::Num => {
            let a = a.expect_numeric();
            let b = b.expect_numeric();
            dynasm!(asm; ucomisd Rx (a as u8), Rx (b as u8));
        }
        ValueType::Str => {
            let a = a.expect_general();
            let b = b.expect_general();
            dynasm!(asm
                ; cmp Rq (a.code()), Rq (b.code())
                ; je >strptreq
                // TODO TODO TODO
                ; call ext_str_cmp as _
                ; strptreq:
            );
        }
        ValueType::Obj => todo!(),
        ValueType::Null => todo!(),
        ValueType::Undefined => todo!(),
        ValueType::Function => todo!(),
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
        ; mov r15b, [rsi + (a.code() + 1) as _]
        ; cmp r15b, [rsi + (b.code() + 1) as _]
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

    struct Output {
        sink: Vec<interpreter::Value>,
        vm: interpreter::VM,
    }
    impl Output {
        fn get_trace(&self, trace_id: &str) -> Option<&Trace> {
            let (trace, _) = self.vm.get_trace(trace_id)?;
            Some(trace)
        }
    }

    fn quick_jit(code: &str) -> Output {
        let mut vm = interpreter::VM::new();

        let base_flags = Default::default();

        {
            let flags = interpreter::InterpreterFlags {
                jit_mode: interpreter::JitMode::Compile,
                ..base_flags
            };
            vm.run_script(code.to_string(), flags)
                .expect("first run (jit compilation) failed");
        }

        let (trace, _) = vm.get_trace("t").expect("no trace was produced!");
        trace.dump();
        assert!(vm.trace_ids().len() > 0);
        vm.take_sink(); // ... and discard it

        {
            let flags = interpreter::InterpreterFlags {
                jit_mode: interpreter::JitMode::UseTraces,
                ..base_flags
            };
            vm.run_script(code.to_string(), flags)
                .expect("second run (trace run) failed");
        }

        let (_, thunk) = vm.get_trace("t").unwrap();
        assert_eq!(thunk.n_runs.load(std::sync::atomic::Ordering::SeqCst), 1);

        Output {
            sink: vm.take_sink(),
            vm,
        }
    }

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

    #[test]
    fn test_create_closure_in_trace() {
        let output = quick_jit(
            "
            function foo() {
                __start_trace('t');
                let a = function() { return sink(30); };
                let b = function() { return sink(90); };
                let c = function() { return sink(120); };
                let d = function() { return sink(0); };


                d();
                c();
                a();
            }

            foo();
            ",
        );

        assert_eq!(
            &[
                BoxedValue::Number(0.0),
                BoxedValue::Number(120.0),
                BoxedValue::Number(30.0),
            ],
            &output.sink[..]
        );
    }

    #[test]
    fn test_unsupported_escaping_closures() {
        let code = "
            function foo() {

                let a = function() { return sink(999); };
                let b = function() { return sink(90); };
                let c = function() { return sink(120); };

                function inner() {
                    __start_trace('t');
                    a = function() { return sink(30); };
                }
                inner();

                let d = function() { return sink(0); };

                d();
                c();
                a();
            }

            foo();
        ";
        let mut vm = interpreter::VM::new();
        let flags = interpreter::InterpreterFlags {
            jit_mode: interpreter::JitMode::Compile,
            ..Default::default()
        };
        vm.run_script(code.to_string(), flags).unwrap();
        assert!(vm.get_trace("t").is_none());
    }
}
