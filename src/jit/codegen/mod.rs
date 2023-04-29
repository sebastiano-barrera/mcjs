use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

use dynasm::dynasm;
use dynasmrt::x64::{Rq, Rx};
use dynasmrt::{DynamicLabel, Register};
use dynasmrt::{DynasmApi, DynasmLabelApi};
use strum::EnumCount;
use strum_macros::{EnumCount, EnumIter, FromRepr};

use crate::jit::builder::reg_class_of_type;
use crate::jit::regalloc::{HardReg, RegClass};
use crate::{
    interpreter,
    bytecode::{IID, ArithOp, CmpOp},
    jit::builder::{Cmp, Instr, ValueId, ValueType},
};

use super::{regalloc, BoxedValue, Trace};

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
        ValueType::Bool => interpreter::Value::Bool(match rt_val {
            0 => false,
            1 => true,
            _ => panic!("decode_value: invalid bool"),
        }),
        ValueType::Num => interpreter::Value::Number(f64::from_bits(rt_val as u64)),
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
    pub(crate) fn n_runs(&self) -> usize {
        self.n_runs.load(std::sync::atomic::Ordering::Acquire)
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
/// The following registers are not part of the list, as they are reserved for other
/// purposes:
///   - RSP is used as the stack pointer, in order to be compatible with the C calling
///     convention.
///   - R15 is a scratch register
pub(crate) const GP_REGS: [Rq; 14] = [
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

pub(crate) const NUM_REGS: [Rx; 16] = [
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

pub(super) fn index_of_reg<R: dynasmrt::Register, O: TryFrom<usize>>(regs: &[R], needle: &R) -> O
where
    <O as TryFrom<usize>>::Error: std::fmt::Debug,
{
    regs.iter()
        .position(|r| r == needle)
        .unwrap()
        .try_into()
        .unwrap()
}

fn hreg_of_genreg(needle: Rq) -> HardReg {
    let index = index_of_reg(&GP_REGS, &needle);
    HardReg {
        class: RegClass::General,
        index,
    }
}

fn hreg_of_numreg(needle: Rx) -> HardReg {
    let index = index_of_reg(&NUM_REGS, &needle);
    HardReg {
        class: RegClass::Numeric,
        index,
    }
}

#[cfg(test)]
thread_local! {
    static TEST_SINK: RefCell<Vec<BoxedValue>> = RefCell::new(Vec::new());
}

#[cfg(test)]
pub fn take_test_sink() -> Vec<BoxedValue> {
    TEST_SINK.with(|test_sink| {
        let mut other = Vec::new();
        let mut test_sink = test_sink.borrow_mut();
        std::mem::swap(&mut other, &mut test_sink);
        other
    })
}

extern "C" fn ext_push_sink_gen(_value: i64, _ty: ValueType) {
    // Figure out a better way of manipulating the `sink` vector
    #[cfg(test)]
    TEST_SINK.with(|test_sink| {
        let addr = RefCell::as_ptr(test_sink) as usize;
        let mut test_sink = test_sink.borrow_mut();
        // TODO This is a kludge.  Remove the rtdata parameter from decode_value?
        let rtdata = unsafe { std::mem::transmute(std::ptr::null::<()>()) };
        let value = decode_value(rtdata, _value, _ty);
        eprintln!(
            "debug: push sink: value = {:?}/{} = {:?} (test_sink is at 0x{:08x}, thread is {:?})",
            _ty,
            _value,
            value,
            addr,
            std::thread::current().id()
        );
        test_sink.push(value);
    });
}

extern "C" fn ext_push_sink_num(_value: f64) {
    // Figure out a better way of manipulating the `sink` vector
    #[cfg(test)]
    TEST_SINK.with(|test_sink| {
        let addr = RefCell::as_ptr(test_sink) as usize;
        let mut test_sink = test_sink.borrow_mut();
        eprintln!(
            "debug: push sink numeric: value={} (test_sink is at 0x{:08x}, thread is {:?})",
            _value,
            addr as usize,
            std::thread::current().id()
        );
        test_sink.push(BoxedValue::Number(_value));
    });
}

extern "C" fn ext_str_cmp(sid_a: u64, sid_b: u64) {
    todo!("ext_str_cmp: {} <=> {}", sid_a, sid_b);
}

trait HardRegExt {
    fn expect_general(&self) -> Rq;
    fn expect_numeric(&self) -> Rx;
}

impl HardRegExt for HardReg {
    fn expect_general(&self) -> Rq {
        assert!(self.class == RegClass::General, "expected a general reg");
        GP_REGS[self.index as usize]
    }

    fn expect_numeric(&self) -> Rx {
        assert!(self.class == RegClass::Numeric, "expected a numeric reg");
        NUM_REGS[self.index as usize]
    }
}

// TODO better name?
struct LocMgr {
    stack_codegen: StackCodegen,
    cur_locs: HashMap<ValueId, regalloc::Loc>,
    used_gps: Vec<Rq>,
    used_nums: Vec<Rx>,
}

impl LocMgr {
    fn new() -> Self {
        LocMgr {
            stack_codegen: StackCodegen::new(),
            cur_locs: HashMap::new(),
            used_gps: Vec::new(),
            used_nums: Vec::new(),
        }
    }

    fn stack_push(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq]) {
        self.stack_codegen.start_chunk(asm, regs);
    }
    fn stack_pop(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq]) {
        self.stack_codegen.end_chunk(asm, regs);
    }

    fn pre_call(&mut self, asm: &mut dynasmrt::x64::Assembler) {
        let saved_genregs: Vec<_> = self
            .used_gps
            .iter()
            .filter(|areg| !is_callee_saved(**areg))
            .copied()
            .collect();

        self.stack_codegen.start_chunk(asm, &saved_genregs);
        self.stack_codegen.start_chunk_num(asm, &self.used_nums);
        self.stack_codegen.pre_call(asm);
    }

    fn post_call(&mut self, asm: &mut dynasmrt::x64::Assembler) {
        // TODO avoid having this vec at all
        let saved_genregs: Vec<_> = self
            .used_gps
            .iter()
            .filter(|areg| !is_callee_saved(**areg))
            .copied()
            .collect();

        self.stack_codegen.post_call(asm);
        self.stack_codegen.end_chunk_num(asm, &self.used_nums);
        self.stack_codegen.end_chunk(asm, &saved_genregs);
    }

    // TODO support locs, more general than HardRegs
    fn get_hreg(&self, vid: ValueId) -> HardReg {
        let loc = self
            .cur_locs
            .get(&vid)
            .unwrap_or_else(|| panic!("regalloc bug: no hreg or stack slot for {vid:?}"));
        match loc {
            regalloc::Loc::HardReg(hreg) => *hreg,
            regalloc::Loc::StackSlot(_) => todo!("stack slots"),
        }
    }

    fn set_loc(
        &mut self,
        asmt: &regalloc::RegAsmt,
        asm: &mut dynasmrt::Assembler<dynasmrt::x64::X64Relocation>,
    ) {
        // TODO this "algorithm" overestimates the set of used registers, as they remain marked as
        // used, and are never cleared
        match asmt.loc {
            regalloc::Loc::HardReg(HardReg { class, index }) => match class {
                // TODO Bad data structure
                RegClass::General => {
                    let rq = GP_REGS[index as usize];
                    if !self.used_gps.contains(&rq) {
                        self.used_gps.push(rq);
                    }
                }
                RegClass::Numeric => {
                    let rx = NUM_REGS[index as usize];
                    if !self.used_nums.contains(&rx) {
                        self.used_nums.push(rx);
                    }
                }
            },
            _ => {}
        }

        let prev = self.cur_locs.insert(asmt.vid, asmt.loc);
        if let Some(prev) = prev {
            emit_mov(asm, prev, asmt.loc);
        }
    }
}

pub(super) fn to_native(trace: &Trace) -> NativeThunk {
    assert!(!trace.is_loop, "loops still unsupported");

    if trace.hreg_alloc.n_general() as usize > GP_REGS.len() {
        panic!("too many hardregs used in reg allocation!");
    }

    let snap_len = trace.snapshot_map.len();

    let mut asm = dynasmrt::x64::Assembler::new().unwrap();
    let mut rt_data = RuntimeData::new();

    let get_ty = |vid: ValueId| trace.get_instr(vid).unwrap().result_type().unwrap();
    let assert_type = |vid: ValueId, expected_type: ValueType| {
        assert_eq!(get_ty(vid), expected_type);
    };

    // TODO(big feat) Another good idea from LuaJIT: group all the constants at the start of
    // the trace
    let num_consts = {
        let mut map = HashMap::new();
        for (vid, instr) in trace.iter_instrs() {
            if let Instr::Const(BoxedValue::Number(num)) = instr {
                let lbl = asm.new_dynamic_label();
                let num_raw = num.to_bits() as i64;
                dynasm!(asm; =>lbl; .qword num_raw);
                map.insert(vid, lbl);
            }
        }

        map
    };

    dynasm!(asm ; entry: );

    let (used_gps, used_nums) = order_used_aregs(&trace.hreg_alloc);
    let saved_gps: Box<[_]> = used_gps
        .iter()
        .filter(|areg| is_callee_saved(**areg))
        .copied()
        .collect();

    let mut locmgr = LocMgr::new();
    locmgr.stack_push(&mut asm, &saved_gps);

    let mut translate_instr = |asm: &mut dynasmrt::x64::Assembler,
                               locmgr: &mut LocMgr,
                               vid: ValueId,
                               instr: &Instr| {
        let write_snap_update =
            |asm: &mut dynasmrt::x64::Assembler, update: &super::builder::SnapshotUpdate| {
                assert!(
                    update.len() <= snap_len as usize,
                    "snapshot update has wrong len ({}, must be at most {})",
                    update.len(),
                    snap_len
                );
                for (ndx, slot_vid) in update.iter().enumerate() {
                    if let Some(slot_vid) = slot_vid {
                        let hreg = locmgr.get_hreg(*slot_vid);
                        let ty = get_ty(*slot_vid);
                        write_snapshot_item(asm, ndx, ty, hreg);
                    }
                }
            };

        match instr {
            Instr::Box(_) => todo!("(big feat) Instr::Box"),
            Instr::GetArg { .. } => todo!("(big feat) Instr::GetArg"),
            Instr::Const(value) => {
                // It is assumed that the IR already "knows" the type of
                // this value, so we can discard it here
                let (encoded_value, _) = encode_value(&mut rt_data, value);

                let HardReg { class, index } = locmgr.get_hreg(vid);
                assert_eq!(
                    class,
                    reg_class_of_type(ValueType::of(value)),
                    "wrong register class: {class:?}"
                );

                match class {
                    RegClass::General => {
                        let reg = GP_REGS[index as usize];
                        dynasm!(asm; mov Rq (reg.code()), QWORD encoded_value);
                    }
                    RegClass::Numeric => {
                        let reg = NUM_REGS[index as usize];
                        let mem = *num_consts.get(&vid).unwrap();
                        dynasm!(asm; movsd Rx (reg.code()), [=>mem]);
                    }
                }
            }
            Instr::Not(_) => todo!("(small feat) Instr::Box"),
            Instr::Arith { op, a, b } => {
                assert_type(*a, ValueType::Num);
                assert_type(*b, ValueType::Num);

                let a = locmgr.get_hreg(*a).expect_numeric();
                let b = locmgr.get_hreg(*b).expect_numeric();
                let tgt = locmgr.get_hreg(vid).expect_numeric();
                // TODO(idea) This would benefit from pin-pointing the target
                // register to the first operand register
                if tgt != a {
                    dynasm!(asm; movsd Rx (tgt.code()), Rx (a.code()));
                }

                match op {
                    ArithOp::Add => dynasm!(asm; addsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Sub => dynasm!(asm; subsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Mul => dynasm!(asm; mulsd Rx (tgt.code()), Rx (b.code())),
                    ArithOp::Div => dynasm!(asm; divsd Rx (tgt.code()), Rx (b.code())),
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
                write_snap_update(asm, &pre_snap_update);
                let a = locmgr.get_hreg(cmp.a);
                let b = locmgr.get_hreg(cmp.b);
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
                let tgt = locmgr.get_hreg(vid).expect_general().code();
                dynasm!(asm; xor Rq (tgt), Rq (tgt));
            }
            Instr::PushSink(arg) => {
                let hreg = locmgr.get_hreg(*arg);
                let ty = get_ty(*arg);

                locmgr.pre_call(asm);
                match reg_class_of_type(ty) {
                    RegClass::General => {
                        assert_eq!(hreg.class, RegClass::General);
                        assert_eq!(GP_REGS[hreg.index as usize], Rq::RDI);

                        let func_ptr = ext_push_sink_gen as *const ();
                        dynasm!(asm
                        ; mov rax, QWORD func_ptr as i64
                        ; mov rsi, QWORD ty as _
                        ; call rax);
                    }
                    RegClass::Numeric => {
                        assert_eq!(hreg.class, RegClass::Numeric);
                        assert_eq!(NUM_REGS[hreg.index as usize], Rx::XMM0);

                        let func_ptr = ext_push_sink_num as *const ();
                        dynasm!(asm
                        ; mov rax, QWORD func_ptr as i64
                        ; call rax);
                    }
                }
                locmgr.post_call(asm);
            }
            Instr::Return(vid) => {
                let ty = get_ty(*vid);
                let HardReg { class, index } = locmgr.get_hreg(*vid);

                dynasm!(asm; mov BYTE [rsi], ty as _);
                match class {
                    RegClass::General => {
                        let rq = GP_REGS[index as usize];
                        dynasm!(asm; mov [rdi], Rq (rq.code()))
                    }
                    RegClass::Numeric => {
                        let rx = NUM_REGS[index as usize];
                        dynasm!(asm; movsd [rdi], Rx (rx.code()))
                    }
                }
            }
            Instr::Phi(old, new) => {
                let old = locmgr.get_hreg(*old);
                let new = locmgr.get_hreg(*new);
                if old != new {
                    match (old.class, new.class) {
                        (RegClass::General, RegClass::General) =>
                            dynasm!(asm; mov Rq (old.index), Rq (new.index)),

                        (RegClass::Numeric, RegClass::Numeric) =>
                            dynasm!(asm; movsd Rx (old.index), Rx (new.index)),

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
            //   * values: u64; N        -- runtime-encoded values
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
                assert!(*snap_ndx < snap_len);
                // `+ 1` because the first slot is reserved for the return value
                let slot_ndx = 1 + *snap_ndx as i32;

                dynasm!(asm
                 ; cmp BYTE [rsi + slot_ndx], *ty as i8
                 ; jne >abort_trace
                );

                let HardReg { class, index: tgt } = locmgr.get_hreg(vid);
                assert_eq!(class, reg_class_of_type(*ty));
                match class {
                    RegClass::General => {
                        let tgt = GP_REGS[tgt as usize];
                        dynasm!(asm; mov Rq (tgt.code()), QWORD [rdi + 8 * slot_ndx])
                    }
                    RegClass::Numeric => {
                        let tgt = NUM_REGS[tgt as usize];
                        dynasm!(asm; movsd Rx (tgt.code()), QWORD [rdi + 8 * slot_ndx])
                    }
                }

                write_snap_update(asm, &post_snap_update);
            }

            #[cfg(test)]
            Instr::ClobberCallerSaved => {
                locmgr.pre_call(asm);

                // A value that is pretty evident in a debugger
                let mut val: i64 = 0x12345678abcdef01;
                for reg in &GP_REGS {
                    if !is_callee_saved(*reg) {
                        dynasm!(asm; mov Rq (reg.code()), QWORD val);
                        val = val.rotate_left(1);
                    }
                }

                dynasm!(asm; sub rsp, 8);
                for reg in &NUM_REGS {
                    dynasm!(asm; mov DWORD [rsp], val as i32);
                    dynasm!(asm; mov DWORD [rsp + 4], (val >> 32) as i32);
                    dynasm!(asm; movsd Rx (reg.code()), QWORD [rsp]);
                    val = val.rotate_left(1);
                }
                dynasm!(asm; add rsp, 8);

                locmgr.post_call(asm);
            }
        }
    };

    let mut reg_asmts = trace.hreg_alloc.asmts.iter().rev().peekable();

    if trace.is_loop {
        dynasm!(asm; loop_start:);
    }
    for (vid, instr) in trace.iter_instrs() {
        while let Some(asmt) = reg_asmts.next_if(|asmt| asmt.pos.0 <= vid.0) {
            eprintln!("  #{:2} reg: {:?} -> {:?}", asmt.pos.0, asmt.vid, asmt.loc);
            locmgr.set_loc(asmt, &mut asm);
        }
        eprintln!("{:4?} {:?}", vid, instr);
        translate_instr(&mut asm, &mut locmgr, vid, instr);
    }
    if trace.is_loop {
        dynasm!(asm; jmp <loop_start);
    }

    dynasm!(asm ; abort_trace:);

    locmgr.stack_pop(&mut asm, &saved_gps);

    dynasm!(asm; ret);

    let entry_offset = asm.labels().resolve_local("entry").unwrap();
    let code_size = asm.offset().0;
    let buf = asm.finalize().unwrap();
    NativeThunk {
        buf,
        entry_offset,
        snapshot_len: trace.snapshot_map.len(),
        code_size,
        rt_data,
        #[cfg(test)]
        n_runs: AtomicUsize::new(0),
    }
}

fn emit_mov(asm: &mut dynasmrt::x64::Assembler, prev: regalloc::Loc, dest: regalloc::Loc) {
    use regalloc::Loc;
    eprintln!("  (emitting move from {:?} to {:?})", prev, dest);
    if let (Loc::HardReg(hfrom), Loc::HardReg(hto)) = (prev, dest) {
        assert_eq!(
            hfrom.class, hto.class,
            "regalloc bug: can't move from a {:?} register to a {:?} one",
            hfrom.class, hto.class
        );

        match hfrom.class {
            RegClass::General => {
                let from_ndx = GP_REGS[hfrom.index as usize].code();
                let to_ndx = GP_REGS[hto.index as usize].code();
                dynasm!(asm; mov Rq (to_ndx), Rq (from_ndx));
            }
            RegClass::Numeric => {
                let from_ndx = NUM_REGS[hfrom.index as usize].code();
                let to_ndx = NUM_REGS[hto.index as usize].code();
                dynasm!(asm; movsd Rx (to_ndx), Rx (from_ndx));
            }
        }
    } else {
        todo!("emit mov from {:?} to {:?}", prev, dest);
    }
}

fn is_callee_saved(areg: Rq) -> bool {
    // TODO(performance) use an array or a bitmask for this
    matches!(
        areg,
        Rq::RBX | Rq::RSP | Rq::RBP | Rq::R12 | Rq::R13 | Rq::R14 | Rq::R15
    )
}

fn write_snapshot_item(
    asm: &mut dynasmrt::x64::Assembler,
    ndx: usize,
    ty: ValueType,
    hreg: HardReg,
) {
    dynasm!(asm; mov BYTE [rsi + 1 + ndx as i32], ty as _);
    match hreg.class {
        // Mind the `1 + ndx`: the first slot is reserved for the return value
        // (in case the trace spans a whole function)
        RegClass::General => {
            let rq = GP_REGS[hreg.index as usize];
            dynasm!(asm; mov [rdi + 8 * (1 + ndx as i32)], Rq (rq.code()))
        }
        RegClass::Numeric => {
            let rx = NUM_REGS[hreg.index as usize];
            dynasm!(asm; movsd [rdi + 8 * (1 + ndx as i32)], Rx (rx.code()))
        }
    }
}

fn order_used_aregs(reg_alloc: &regalloc::Allocation) -> (Vec<Rq>, Vec<Rx>) {
    use regalloc::Loc;

    let mut used_gps = HashSet::new();
    let mut used_nums = HashSet::new();
    for asmt in reg_alloc.asmts.iter() {
        if let Loc::HardReg(hreg) = asmt.loc {
            match hreg.class {
                RegClass::General => {
                    used_gps.insert(GP_REGS.get(hreg.index as usize).copied().unwrap());
                }
                RegClass::Numeric => {
                    used_nums.insert(NUM_REGS.get(hreg.index as usize).copied().unwrap());
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
    chunks_size: Vec<u8>,
    // Expressed in number of slots
    cur_height: u16,
    in_call: bool,
}
struct StackCodegenInCall(StackCodegen);
impl StackCodegen {
    // TODO(big feat) Any way to make sure that start_chunk is coupled with an
    // end_chunk? Compile-time preferable, but runtime also useful. I had some
    // asserts in an impl Drop, but it made the program "panic while panicking",
    // which makes things harder.

    fn new() -> Self {
        StackCodegen {
            chunks_size: Vec::new(),
            cur_height: 0,
            in_call: false,
        }
    }

    fn check_invariant(&self) {
        assert_eq!(
            self.cur_height,
            self.chunks_size.iter().map(|x| *x as u16).sum()
        );
    }

    // TODO(opt): encode set of registers as a bitmask
    fn start_chunk(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq]) {
        let chunk_sz = regs.len() as u8;
        self.push_chunk(asm, chunk_sz);
        self.write_regs(asm, regs);

        self.check_invariant();
    }
    fn end_chunk(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq]) {
        self.read_regs(asm, regs);
        self.pop_chunk(asm);

        self.check_invariant();
    }
    fn start_chunk_num(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rx]) {
        let chunk_sz = regs.len() as u8;
        self.push_chunk(asm, chunk_sz);
        self.write_regs_num(asm, regs);

        self.check_invariant();
    }
    fn end_chunk_num(&mut self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rx]) {
        self.read_regs_num(asm, regs);
        self.pop_chunk(asm);

        self.check_invariant();
    }

    fn pre_call(&mut self, asm: &mut dynasmrt::x64::Assembler) {
        assert!(!self.in_call);
        // cur_height is expressed in number 8-byte slots
        match self.cur_height % 2 {
            0 => dynasm!(asm; sub rsp, 8),
            1 => { /* already OK for a C call */ }
            other => {
                panic!(
                    "codegen bug: stack height is {}, but it should be either 16- or 8-aligned!",
                    other * 8
                );
            }
        }
        self.in_call = true;
    }
    fn post_call(&mut self, asm: &mut dynasmrt::x64::Assembler) {
        assert!(self.in_call);
        // cur_height is expressed in number 8-byte slots
        match self.cur_height % 2 {
            0 => dynasm!(asm; add rsp, 8),
            1 => { /* we were already OK for a C call */ }
            other => {
                panic!(
                    "codegen bug: stack height is {}, but it should be either 16- or 8-aligned!",
                    other * 8
                );
            }
        }
        self.in_call = false;
    }

    fn push_chunk(&mut self, asm: &mut dynasmrt::x64::Assembler, chunk_sz: u8) {
        debug_assert!(!self.in_call);
        self.chunks_size.push(chunk_sz);
        if chunk_sz > 0 {
            self.cur_height += chunk_sz as u16;
            let delta_rsp = 8 * chunk_sz as i32;
            dynasm!(asm; sub rsp, delta_rsp);
        }
        self.check_invariant();
    }

    fn write_regs(&self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq]) {
        debug_assert!(!self.in_call);
        for (i, reg) in regs.iter().enumerate() {
            dynasm!(asm; mov [rsp + i as i32 * 8], Rq (reg.code()));
        }
        self.check_invariant();
    }

    fn read_regs(&self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rq]) {
        debug_assert!(!self.in_call);
        for (i, reg) in regs.iter().enumerate() {
            dynasm!(asm; mov Rq (reg.code()), [rsp + i as i32 * 8]);
        }
        self.check_invariant();
    }

    fn write_regs_num(&self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rx]) {
        debug_assert!(!self.in_call);
        for (i, reg) in regs.iter().enumerate() {
            dynasm!(asm; movsd [rsp + i as i32 * 8], Rx (reg.code()));
        }
        self.check_invariant();
    }

    fn read_regs_num(&self, asm: &mut dynasmrt::x64::Assembler, regs: &[Rx]) {
        debug_assert!(!self.in_call);
        for (i, reg) in regs.iter().enumerate() {
            dynasm!(asm; movsd Rx (reg.code()), [rsp + i as i32 * 8]);
        }
        self.check_invariant();
    }

    fn pop_chunk(&mut self, asm: &mut dynasmrt::x64::Assembler) {
        debug_assert!(!self.in_call);
        let chunk_sz = self.chunks_size.pop().unwrap();
        if chunk_sz > 0 {
            self.cur_height -= chunk_sz as u16;
            let delta_rsp = 8 * chunk_sz as i32;
            dynasm!(asm; add rsp, delta_rsp);
        }
        self.check_invariant();
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
            todo!("I need a big refactoring in order to always generate correct code for arbitrary C funcs");
            assert_eq!(a, Rq::RDI);
            assert_eq!(b, Rq::RSI);
            dynasm!(asm
            ; call ext_str_cmp as _
            ; cmp rax, 0
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::{
        builder::{SnapshotMap, SnapshotMapItem, SnapshotUpdate, ValueType},
        regalloc::{Allocation, RegAsmt},
    };

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

    fn simple_trace(instrs: Vec<Instr>, hreg_alloc: Allocation) -> Trace {
        let is_enabled = instrs.iter().map(|_| true).collect();
        Trace {
            hreg_alloc,
            snapshot_map: SnapshotMap::new(),
            snapshot_final_update: Vec::new(),
            instrs,
            is_loop: false,
            is_enabled,
        }
    }

    #[test]
    fn test_const_general() {
        let trace = simple_trace(
            [
                Instr::Const(BoxedValue::Bool(true)),
                Instr::PushSink(ValueId(0u32)),
            ]
            .into(),
            Allocation::new(
                [
                    RegAsmt::hreg(1, 0, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(0, 0, super::hreg_of_genreg(Rq::RCX)),
                ]
                .into(),
            ),
        );

        let thunk = super::to_native(&trace);
        thunk.dump();
        let ret = thunk.run(&mut []);

        assert_eq!(ret, BoxedValue::Undefined);
        assert_eq!(&super::take_test_sink(), &[BoxedValue::Bool(true)]);
    }

    #[test]
    fn test_caller_saved_reg() {
        let trace = simple_trace(
            [
                Instr::Const(BoxedValue::Bool(true)),
                Instr::Const(BoxedValue::Bool(true)),
                Instr::Const(BoxedValue::Bool(false)),
                Instr::Const(BoxedValue::Bool(true)),
                Instr::Const(BoxedValue::Bool(true)),
                Instr::Const(BoxedValue::Bool(false)),
                Instr::Const(BoxedValue::Bool(false)),
                Instr::Const(BoxedValue::Bool(false)),
                Instr::Const(BoxedValue::Bool(true)),
                Instr::Const(BoxedValue::Number(4.1234)),
                Instr::Const(BoxedValue::Number(1.2344)),
                Instr::Const(BoxedValue::Number(2.3441)),
                Instr::Const(BoxedValue::Number(2.3441)),
                Instr::Const(BoxedValue::Number(3.4123)),
                Instr::Const(BoxedValue::Number(4.1234)),
                Instr::Const(BoxedValue::Number(1.2345)),
                Instr::Const(BoxedValue::Number(5.3214)),
                Instr::ClobberCallerSaved,
                Instr::PushSink(ValueId(0)),
                Instr::PushSink(ValueId(1)),
                Instr::PushSink(ValueId(2)),
                Instr::PushSink(ValueId(3)),
                Instr::PushSink(ValueId(4)),
                Instr::PushSink(ValueId(5)),
                Instr::PushSink(ValueId(6)),
                Instr::PushSink(ValueId(7)),
                Instr::PushSink(ValueId(8)),
                Instr::PushSink(ValueId(9)),
                Instr::PushSink(ValueId(10)),
                Instr::PushSink(ValueId(11)),
                Instr::PushSink(ValueId(12)),
                Instr::PushSink(ValueId(13)),
                Instr::PushSink(ValueId(14)),
                Instr::PushSink(ValueId(15)),
                Instr::PushSink(ValueId(16)),
            ]
            .into(),
            Allocation::new(
                [
                    RegAsmt::hreg(35, 17, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(34, 16, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(33, 15, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(32, 14, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(31, 13, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(30, 12, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(29, 11, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(28, 10, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(27, 9, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(26, 8, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(25, 7, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(24, 6, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(23, 5, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(22, 4, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(21, 3, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(20, 2, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(19, 1, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(18, 0, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(16, 16, super::hreg_of_numreg(Rx::XMM7)),
                    RegAsmt::hreg(15, 15, super::hreg_of_numreg(Rx::XMM6)),
                    RegAsmt::hreg(14, 14, super::hreg_of_numreg(Rx::XMM5)),
                    RegAsmt::hreg(13, 13, super::hreg_of_numreg(Rx::XMM4)),
                    RegAsmt::hreg(12, 12, super::hreg_of_numreg(Rx::XMM3)),
                    RegAsmt::hreg(11, 11, super::hreg_of_numreg(Rx::XMM2)),
                    RegAsmt::hreg(10, 10, super::hreg_of_numreg(Rx::XMM1)),
                    RegAsmt::hreg(9, 9, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(8, 8, super::hreg_of_genreg(Rq::RAX)),
                    RegAsmt::hreg(7, 7, super::hreg_of_genreg(Rq::RCX)),
                    RegAsmt::hreg(6, 6, super::hreg_of_genreg(Rq::RDX)),
                    RegAsmt::hreg(5, 5, super::hreg_of_genreg(Rq::RSI)),
                    RegAsmt::hreg(4, 4, super::hreg_of_genreg(Rq::RDI)),
                    RegAsmt::hreg(3, 3, super::hreg_of_genreg(Rq::R8)),
                    RegAsmt::hreg(2, 2, super::hreg_of_genreg(Rq::R9)),
                    RegAsmt::hreg(1, 1, super::hreg_of_genreg(Rq::R10)),
                    RegAsmt::hreg(0, 0, super::hreg_of_genreg(Rq::R11)),
                ]
                .into(),
            ),
        );

        let thunk = super::to_native(&trace);
        thunk.dump();
        let ret = thunk.run(&mut []);

        assert_eq!(ret, BoxedValue::Undefined);
        assert_eq!(
            &super::take_test_sink(),
            &[
                BoxedValue::Bool(true),
                BoxedValue::Bool(true),
                BoxedValue::Bool(false),
                BoxedValue::Bool(true),
                BoxedValue::Bool(true),
                BoxedValue::Bool(false),
                BoxedValue::Bool(false),
                BoxedValue::Bool(false),
                BoxedValue::Bool(true),
                BoxedValue::Number(4.1234),
                BoxedValue::Number(1.2344),
                BoxedValue::Number(2.3441),
                BoxedValue::Number(2.3441),
                BoxedValue::Number(3.4123),
                BoxedValue::Number(4.1234),
                BoxedValue::Number(1.2345),
                BoxedValue::Number(5.3214),
            ]
        );
    }

    #[test]
    #[should_panic]
    fn test_pushsink_wrong_reg() {
        let trace = simple_trace(
            [
                Instr::Const(BoxedValue::Number(1032.312)),
                Instr::PushSink(ValueId(0u32)),
            ]
            .into(),
            Allocation::new(
                [
                    RegAsmt::hreg(1, 0, super::hreg_of_numreg(Rx::XMM3)),
                    RegAsmt::hreg(0, 0, super::hreg_of_numreg(Rx::XMM0)),
                ]
                .into(),
            ),
        );

        super::to_native(&trace);
    }

    #[test]
    fn test_const_numeric() {
        let num_val = BoxedValue::Number(1032.312);
        let trace = simple_trace(
            [
                Instr::Const(num_val.clone()),
                Instr::PushSink(ValueId(0u32)),
            ]
            .into(),
            Allocation::new(
                [
                    RegAsmt::hreg(1, 0, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(0, 0, super::hreg_of_numreg(Rx::XMM3)),
                ]
                .into(),
            ),
        );

        let thunk = super::to_native(&trace);
        thunk.dump();
        let ret = thunk.run(&mut []);

        assert_eq!(ret, BoxedValue::Undefined);
        assert_eq!(&super::take_test_sink(), &[num_val]);
    }

    #[test]
    fn test_arith() {
        let a = 1032.312;
        let b = 99.2321;
        let trace = simple_trace(
            [
                Instr::Const(BoxedValue::Number(a)),
                Instr::Const(BoxedValue::Number(b)),
                Instr::Arith {
                    op: ArithOp::Add,
                    a: ValueId(0u32),
                    b: ValueId(1u32),
                },
                Instr::Arith {
                    op: ArithOp::Sub,
                    a: ValueId(0u32),
                    b: ValueId(1u32),
                },
                Instr::Arith {
                    op: ArithOp::Mul,
                    a: ValueId(0u32),
                    b: ValueId(1u32),
                },
                Instr::Arith {
                    op: ArithOp::Div,
                    a: ValueId(0u32),
                    b: ValueId(1u32),
                },
                Instr::PushSink(ValueId(2u32)),
                Instr::PushSink(ValueId(3u32)),
                Instr::PushSink(ValueId(4u32)),
                Instr::PushSink(ValueId(5u32)),
            ]
            .into(),
            Allocation::new(
                [
                    RegAsmt::hreg(9, 5, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(8, 4, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(7, 3, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(6, 2, super::hreg_of_numreg(Rx::XMM0)),
                    RegAsmt::hreg(5, 5, super::hreg_of_numreg(Rx::XMM5)),
                    RegAsmt::hreg(4, 4, super::hreg_of_numreg(Rx::XMM4)),
                    RegAsmt::hreg(3, 3, super::hreg_of_numreg(Rx::XMM3)),
                    RegAsmt::hreg(2, 2, super::hreg_of_numreg(Rx::XMM2)),
                    RegAsmt::hreg(1, 1, super::hreg_of_numreg(Rx::XMM1)),
                    RegAsmt::hreg(0, 0, super::hreg_of_numreg(Rx::XMM0)),
                ]
                .into(),
            ),
        );

        let thunk = super::to_native(&trace);
        thunk.dump();
        let ret = thunk.run(&mut []);

        assert_eq!(ret, BoxedValue::Undefined);
        assert_eq!(
            &super::take_test_sink(),
            &[
                BoxedValue::Number(a + b),
                BoxedValue::Number(a - b),
                BoxedValue::Number(a * b),
                BoxedValue::Number(a / b),
            ]
        );
    }

    #[test]
    fn test_exit_snapshot() {
        let trace = {
            let instrs: Vec<_> = [
                Instr::Const(BoxedValue::Number(122.3)),
                Instr::Const(BoxedValue::Number(9.001)),
                Instr::Const(BoxedValue::Number(75.30)),
                Instr::ExitUnless {
                    cond: Cmp {
                        ty: ValueType::Num,
                        op: CmpOp::EQ,
                        a: ValueId(0),
                        b: ValueId(1),
                    },
                    pre_snap_update: [Some(ValueId(2)), None, Some(ValueId(0))].into(),
                },
            ]
            .into();
            let hreg_alloc = Allocation::new(
                [
                    RegAsmt::hreg(2, 2, super::hreg_of_numreg(Rx::XMM2)),
                    RegAsmt::hreg(1, 1, super::hreg_of_numreg(Rx::XMM1)),
                    RegAsmt::hreg(0, 0, super::hreg_of_numreg(Rx::XMM0)),
                ]
                .into(),
            );
            let is_enabled = instrs.iter().map(|_| true).collect();
            // just dummies
            Trace {
                hreg_alloc,
                snapshot_map: dummy_snapshot_map(3),
                snapshot_final_update: Vec::new(),
                instrs,
                is_loop: false,
                is_enabled,
            }
        };

        let thunk = super::to_native(&trace);
        thunk.dump();
        let mut snap = vec![
            BoxedValue::Number(2222.2),
            BoxedValue::Bool(true),
            BoxedValue::Number(11.111),
        ];
        let ret = thunk.run(&mut snap);

        assert_eq!(ret, BoxedValue::Undefined);
        assert_eq!(
            &snap,
            &[
                BoxedValue::Number(75.30),
                BoxedValue::Bool(true),
                BoxedValue::Number(122.3)
            ]
        );
    }

    // NOTE: In this test module, the register assignments are done "manually",
    // without using any register allocation algorithm.  It has to be correct
    // or the trace will have a wrong result/behavior so be careful!

    fn check_getsnapshotitem_wrong_regclass(ty: ValueType, hreg: HardReg) {
        let instr = Instr::GetSnapshotItem {
            ndx: 0,
            ty,
            post_snap_update: vec![],
        };
        let instrs = vec![instr];
        let hreg_asmts = vec![RegAsmt::hreg(0, 0, hreg)];
        let is_enabled = vec![true];
        let trace = Trace {
            hreg_alloc: Allocation::new(hreg_asmts),
            snapshot_map: dummy_snapshot_map(3),
            snapshot_final_update: Vec::new(),
            instrs,
            is_loop: false,
            is_enabled,
        };

        super::to_native(&trace);
    }

    #[test]
    #[should_panic]
    fn test_getsnapshotitem_wrong_regclass_1() {
        check_getsnapshotitem_wrong_regclass(ValueType::Num, super::hreg_of_genreg(Rq::RBX));
    }

    #[test]
    #[should_panic]
    fn test_getsnapshotitem_wrong_regclass_2() {
        check_getsnapshotitem_wrong_regclass(ValueType::Str, super::hreg_of_numreg(Rx::XMM2));
    }

    #[test]
    fn test_snap_inout() {
        let trace = {
            let instrs: Vec<_> = [
                Instr::GetSnapshotItem {
                    ndx: 1,
                    ty: ValueType::Num,
                    post_snap_update: vec![],
                },
                Instr::GetSnapshotItem {
                    ndx: 2,
                    ty: ValueType::Str,
                    post_snap_update: vec![],
                },
                Instr::Const(BoxedValue::Number(5.0)),
                Instr::Arith {
                    op: ArithOp::Mul,
                    a: ValueId(0),
                    b: ValueId(2),
                },
                Instr::Const(BoxedValue::String("lol123".into())),
                Instr::ExitUnless {
                    cond: Cmp {
                        ty: ValueType::Num,
                        op: CmpOp::NE,
                        a: ValueId(0),
                        b: ValueId(0),
                    },
                    pre_snap_update: [Some(ValueId(3)), None, Some(ValueId(4))].into(),
                },
            ]
            .into();
            let hreg_alloc = Allocation::new(
                [
                    RegAsmt::hreg(4, 4, super::hreg_of_genreg(Rq::RBX)),
                    RegAsmt::hreg(3, 3, super::hreg_of_numreg(Rx::XMM3)),
                    RegAsmt::hreg(2, 2, super::hreg_of_numreg(Rx::XMM2)),
                    RegAsmt::hreg(1, 1, super::hreg_of_genreg(Rq::RCX)),
                    RegAsmt::hreg(0, 0, super::hreg_of_numreg(Rx::XMM0)),
                ]
                .into(),
            );
            let is_enabled = instrs.iter().map(|_| true).collect();
            // just dummies
            Trace {
                hreg_alloc,
                snapshot_map: dummy_snapshot_map(3),
                snapshot_final_update: Vec::new(),
                instrs,
                is_loop: false,
                is_enabled,
            }
        };

        let thunk = super::to_native(&trace);
        thunk.dump();

        let mut snap: [_; 3] = [
            BoxedValue::Bool(true),
            BoxedValue::Number(11.111),
            BoxedValue::String("asdlol".into()),
        ];
        let ret = thunk.run(&mut snap[..]);

        assert_eq!(ret, BoxedValue::Undefined);
        let (n0, n1, s) = match &snap[..] {
            [BoxedValue::Number(n0), BoxedValue::Number(n1), BoxedValue::String(s)] => {
                (*n0, *n1, s.clone())
            }
            _ => panic!("wrong post-trace snapshot content: {:?}", snap),
        };

        // Re-doing the multiplication explicitly is necessary, as the
        // result will be approximate.  The multiplication performed by
        // the trace and here in Rust will have "the same error".
        assert_eq!(n0, 11.111 * 5.0);
        assert!(n1 == 11.111);

        assert_eq!(s.as_ref(), "lol123");
    }

    #[should_panic]
    #[test]
    fn test_emit_mov_wrong_regclass() {
        use crate::jit::regalloc::Loc;

        let mut asm = dynasmrt::x64::Assembler::new().unwrap();
        let a = super::HardReg {
            class: RegClass::General,
            index: 1,
        };
        let b = super::HardReg {
            class: RegClass::Numeric,
            index: 1,
        };
        super::emit_mov(&mut asm, Loc::HardReg(a), Loc::HardReg(b));
    }

    // There is NO use filling in a SnapshotMap properly in the codegen module
    // TODO Can this be excluded/moved elsewhere
    fn dummy_snapshot_map(count: usize) -> SnapshotMap {
        let items: Vec<_> = (0..count)
            .map(|i| SnapshotMapItem {
                write_on_entry: false,
                write_on_exit: true,
                operand: IID(i as u32),
            })
            .collect();

        SnapshotMap::from(items)
    }

    #[test]
    fn test_getsnapshotitem_wrong_ty() {}

    #[should_panic]
    #[test]
    fn test_getsnapshotitem_wrong_snap_len() {
        let trace = {
            let instrs: Vec<_> = [
                Instr::GetSnapshotItem {
                    ndx: 0,
                    ty: ValueType::Str,
                    post_snap_update: [None, None].into(),
                },
                Instr::Return(ValueId(0)),
            ]
            .into();
            let hreg_alloc =
                Allocation::new(vec![RegAsmt::hreg(0, 0, super::hreg_of_genreg(Rq::RDI))]);
            let is_enabled = instrs.iter().map(|_| true).collect();
            // just dummies
            Trace {
                hreg_alloc,
                // 1 is smaller than the post_snap_update
                snapshot_map: dummy_snapshot_map(1),
                snapshot_final_update: Vec::new(),
                instrs,
                is_loop: false,
                is_enabled,
            }
        };

        super::to_native(&trace);
    }

    #[ignore]
    #[test]
    fn test_str_cmp() {
        let trace = {
            let instrs: Vec<_> = [
                Instr::GetSnapshotItem {
                    ndx: 0,
                    ty: ValueType::Str,
                    post_snap_update: [None, None].into(),
                },
                Instr::GetSnapshotItem {
                    ndx: 1,
                    ty: ValueType::Str,
                    post_snap_update: [None, None].into(),
                },
                Instr::ExitUnless {
                    cond: Cmp {
                        ty: ValueType::Str,
                        op: CmpOp::NE,
                        a: ValueId(0),
                        b: ValueId(1),
                    },
                    pre_snap_update: [None, None].into(),
                },
                Instr::Const(BoxedValue::Bool(true)),
                Instr::Return(ValueId(3)),
            ]
            .into();
            let hreg_alloc = Allocation::new(
                [
                    RegAsmt::hreg(3, 3, super::hreg_of_genreg(Rq::RAX)),
                    RegAsmt::hreg(1, 1, super::hreg_of_genreg(Rq::RSI)),
                    RegAsmt::hreg(0, 0, super::hreg_of_genreg(Rq::RDI)),
                ]
                .into(),
            );
            let is_enabled = instrs.iter().map(|_| true).collect();
            // just dummies
            Trace {
                hreg_alloc,
                snapshot_map: dummy_snapshot_map(2),
                snapshot_final_update: Vec::new(),
                instrs,
                is_loop: false,
                is_enabled,
            }
        };

        let thunk = super::to_native(&trace);
        thunk.dump();

        let mut snap: [_; 2] = [
            BoxedValue::String("asd".into()),
            BoxedValue::String("rofl".into()),
        ];
        let ret = thunk.run(&mut snap[..]);

        assert_eq!(ret, BoxedValue::Bool(true));
    }
}
