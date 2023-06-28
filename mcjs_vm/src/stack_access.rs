use crate::bytecode::{self, FnId, VReg, IID};
use crate::interpreter::{UpvalueId, Value};
use std::cell::Cell;
use std::env::args;
use std::mem::size_of;
use std::{marker::PhantomData, ops::Range};

// TODO put the real type here instead of u64
type ResultSlot = Slot;
type ArgSlot = Slot;
type CaptureSlot = UpvalueId;

#[derive(Clone, Copy)]
pub(crate) enum Slot {
    Inline(Value),
    Upvalue(UpvalueId),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FrameHeader {
    #[cfg(test)]
    pub(crate) magic: [u8; 8],
    pub(crate) n_instrs: u32,
    pub(crate) n_args: u8,
    pub(crate) n_captures: u16,
    pub(crate) this: Value,
    pub(crate) return_value_vreg: Option<VReg>,
    pub(crate) return_to_iid: Option<IID>,
    pub(crate) fn_id: FnId,
}

#[cfg(test)]
impl FrameHeader {
    pub(crate) const MAGIC: [u8; 8] = *b"THEMAGIC";
}

pub(crate) struct Stack {
    top_offset: usize,
    store: Box<[u8]>,
}

impl Stack {
    pub(crate) fn new(store: Box<[u8]>) -> Self {
        let size = store.len();
        Stack {
            store,
            top_offset: size,
        }
    }

    fn capacity(&self) -> usize {
        self.store.len()
    }

    pub(crate) fn push_frame(&mut self, header: FrameHeader) {
        let layout = FrameLayout::measure(&header);
        assert!(self.top_offset >= layout.size);
        self.top_offset -= layout.size;

        let frame_raw = &mut self.store[self.top_offset..self.top_offset + layout.size];
        layout.header.write(frame_raw, header);
        for (i, var_slot) in layout.vars.get_mut(frame_raw).iter_mut().enumerate() {
            *var_slot = Slot::Inline(Value::Number(i as f64));
        }
        for (i, arg_slot) in layout.args.get_mut(frame_raw).iter_mut().enumerate() {
            *arg_slot = Slot::Inline(Value::Number(i as f64));
        }
        for (i, cap_slot) in layout.captures.get_mut(frame_raw).iter_mut().enumerate() {
            *cap_slot = Slot::Inline(Value::Number(i as f64));
        }
    }

    pub(crate) fn pop_frame(&mut self) {
        let layout = self.cur_layout();
        assert!(self.top_offset + layout.size <= self.store.len());
        self.top_offset += layout.size;
    }

    fn cur_layout(&self) -> FrameLayout {
        // TODO TODO cache this layout and return a & instead
        FrameLayout::measure(self.top_header())
    }

    fn stack(&self) -> &[u8] {
        &self.store[self.top_offset..]
    }
    fn stack_mut(&mut self) -> &mut [u8] {
        &mut self.store[self.top_offset..]
    }

    pub(crate) fn top_header(&self) -> &FrameHeader {
        let header_field = Field::just();
        header_field.get(&self.store[self.top_offset..])
    }

    pub(crate) fn args(&self) -> &[Slot] {
        self.cur_layout().args.get(self.stack())
    }
    pub(crate) fn args_mut(&mut self) -> &mut [Slot] {
        self.cur_layout().args.get_mut(self.stack_mut())
    }

    pub(crate) fn vars(&self) -> &[Slot] {
        self.cur_layout().vars.get(self.stack())
    }
    pub(crate) fn vars_mut(&mut self) -> &mut [Slot] {
        self.cur_layout().vars.get_mut(self.stack_mut())
    }

    pub(crate) fn captures(&self) -> &[Slot] {
        self.cur_layout().captures.get(self.stack())
    }
    pub(crate) fn captures_mut(&mut self) -> &mut [Slot] {
        self.cur_layout().captures.get_mut(self.stack_mut())
    }
}

struct FrameLayout {
    header: Field<FrameHeader>,
    vars: SliceField<Slot>,
    args: SliceField<Slot>,
    captures: SliceField<Slot>,
    size: usize,
}
impl FrameLayout {
    fn measure(header_value: &FrameHeader) -> Self {
        let mut lb = LayoutBuilder::new();
        let header = lb.field();
        let vars = lb.field_slice(header_value.n_instrs as usize);
        let args = lb.field_slice(header_value.n_args as usize);
        let captures = lb.field_slice(header_value.n_captures as usize);

        FrameLayout {
            header,
            vars,
            args,
            captures,
            size: lb.offset,
        }
    }
}

struct LayoutBuilder {
    offset: usize,
}
impl LayoutBuilder {
    fn new() -> Self {
        LayoutBuilder { offset: 0 }
    }

    fn field<T>(&mut self) -> Field<T> {
        let size = size_of::<T>();
        let field = Field {
            range: (self.offset..self.offset + size),
            ph: PhantomData,
        };
        self.offset += size;
        field
    }

    fn field_slice<T>(&mut self, count: usize) -> SliceField<T> {
        let size = size_of::<T>() * count;
        let field = SliceField {
            range: (self.offset..self.offset + size),
            ph: PhantomData,
        };
        self.offset += size;
        field
    }
}

struct Field<T> {
    range: Range<usize>,
    ph: PhantomData<T>,
}
impl<T> Field<T> {
    fn just() -> Self {
        let size = size_of::<T>();
        Field {
            range: 0..size,
            ph: PhantomData,
        }
    }

    fn get<'b>(&self, buf: &'b [u8]) -> &'b T {
        let bytes = &buf[self.range.clone()];
        assert_eq!(bytes.len(), size_of::<T>());
        unsafe { &*(bytes as *const [u8] as *const T) }
    }

    fn get_mut<'b>(&self, buf: &'b mut [u8]) -> &'b mut T {
        let bytes = &mut buf[self.range.clone()];
        assert_eq!(bytes.len(), size_of::<T>());
        unsafe { &mut *(bytes as *mut [u8] as *mut T) }
    }

    fn write<'b>(&self, buf: &'b mut [u8], value: T) {
        let bytes = &mut buf[self.range.clone()];
        assert_eq!(bytes.len(), size_of::<T>());
        let bytes = bytes as *mut [u8] as *mut T;

        // write_unaligned?
        unsafe { std::ptr::write(bytes, value) }
    }
}
struct SliceField<T> {
    range: Range<usize>,
    ph: PhantomData<T>,
}
impl<T> SliceField<T> {
    fn get<'b>(&self, buf: &'b [u8]) -> &'b [T] {
        if self.range.len() == 0 {
            return &[];
        }
        assert_eq!(self.range.len() % size_of::<T>(), 0);
        assert!(buf.len() >= self.range.end);
        let slice = std::ptr::slice_from_raw_parts(
            &buf[self.range.start] as *const u8 as *const T,
            self.range.len() / size_of::<T>(),
        );
        unsafe { &*slice }
    }

    fn get_mut<'b>(&self, buf: &'b mut [u8]) -> &'b mut [T] {
        if self.range.len() == 0 {
            return &mut [];
        }
        assert_eq!(self.range.len() % size_of::<T>(), 0);
        assert!(buf.len() >= self.range.end);
        let slice = std::ptr::slice_from_raw_parts_mut(
            &mut buf[self.range.start] as *mut u8 as *mut T,
            self.range.len() / size_of::<T>(),
        );
        unsafe { &mut *slice }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_header() -> FrameHeader {
        FrameHeader {
            magic: FrameHeader::MAGIC,
            n_instrs: rand::random::<u32>() % 100,
            n_captures: rand::random::<u16>() % 100,
            n_args: rand::random::<u8>() % 100,
            this: Value::Undefined,
            return_value_vreg: Some(VReg(rand::random())),
            return_to_iid: Some(IID(rand::random())),
            fn_id: FnId(rand::random::<u32>()),
        }
    }

    #[cfg(none)]
    #[test]
    fn test_multiple_frames_slots_do_not_overlap() {
        // ordered bottom to top
        let headers: Vec<_> = (0..10).map(|_| random_header()).collect();
        let buf_size = headers.iter().map(|hdr| hdr.expected_frame_size()).sum();
        let buf = vec![0u8; buf_size].into_boxed_slice();
        let mut stack = Stack::new(buf);

        for header in headers.iter().rev() {
            stack.push_frame(*header);
        }

        for header in headers.iter() {
            let frame = stack.top();
            let check_header = frame.header.get();
            assert_eq!(&check_header, header);
            stack.pop_frame();
        }
    }
}
