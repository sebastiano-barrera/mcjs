//
//  Frame layout:
//      Header =
//          n_instrs: u32
//          n_args: u32
//      Results buffer = [n_instrs; Result]
//
//  Result
//      = enum { Stack(Value), Heap(UpvalueId), }
//      = Number(f64),
//      | String(Cow<'static, str>),
//      | Bool(bool),
//      | Object(Object),
//      | Null,
//      | Undefined,
//      | // TODO(cleanup) Delete, Closure supersedes this
//      | SelfFunction,
//      | // TODO(cleanup) Delete, Closure supersedes this
//      | NativeFunction(u32),
//      | Closure(Closure),
//  repr Result =
//    << STACK : 1, padding,  >>
//
//  const STACK = 0
//  const HEAP = 1
//
//

use crate::bytecode::{FnId, VReg, IID};
use crate::interpreter::{UpvalueId, Value};
use std::mem::size_of;
use std::{marker::PhantomData, ops::Range};

pub(crate) struct Offset<T: 'static> {
    offset: usize,
    ph: PhantomData<T>,
}

impl<T: 'static> Offset<T> {
    pub(crate) const fn at(offset: usize) -> Self {
        Offset {
            offset,
            ph: PhantomData,
        }
    }

    pub(crate) fn offset(&self) -> usize {
        self.offset
    }
    pub(crate) fn size(&self) -> usize {
        std::mem::size_of::<T>()
    }
    pub(crate) fn range(&self) -> Range<usize> {
        self.offset()..(self.offset() + self.size())
    }

    pub(crate) fn get<'a>(&self, buf: &'a [u8]) -> &'a T {
        let ofs = self.offset();
        unsafe { std::mem::transmute(&buf[ofs]) }
    }

    pub(crate) fn get_mut<'a>(&self, buf: &'a mut [u8]) -> &'a mut T {
        let ofs = self.offset();
        unsafe { std::mem::transmute(&mut buf[ofs]) }
    }
}

impl<T: 'static> std::fmt::Debug for Offset<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_name = std::any::type_name::<T>();
        write!(f, "{:?}@{}", type_name, self.offset)
    }
}

impl<T: 'static> std::ops::Add<usize> for Offset<T> {
    type Output = Offset<T>;

    fn add(self, rhs: usize) -> Self::Output {
        Offset {
            offset: rhs + self.offset,
            ph: PhantomData,
        }
    }
}

pub(crate) struct FrameMetrics {
    pub(crate) top: usize,
}

// TODO put the real type here instead of u64
type ResultSlot = Slot;
type ArgSlot = Slot;
type CaptureSlot = UpvalueId;

pub(crate) enum Slot {
    Inline(Value),
    Upvalue(UpvalueId),
}

#[derive(Debug, Clone)]
#[repr(packed)]
pub(crate) struct FrameHeader {
    pub(crate) n_instrs: u32,
    pub(crate) n_args: u8,
    pub(crate) n_captures: u16,
    pub(crate) return_value_vreg: Option<VReg>,
    pub(crate) return_to_iid: Option<IID>,
    pub(crate) fn_id: FnId,
}

impl FrameHeader {
    pub(crate) fn expected_frame_size(&self) -> usize {
        size_of::<Self>()
            + size_of::<CaptureSlot>() * self.n_captures as usize
            + size_of::<ResultSlot>() * self.n_instrs as usize
            + size_of::<ArgSlot>() * self.n_args as usize
    }
}

impl FrameMetrics {
    pub(crate) const fn header(&self) -> Offset<FrameHeader> {
        Offset::at(self.top)
    }
    const HEADER_SIZE: usize = size_of::<FrameHeader>();

    pub(crate) fn frame_size(&self, buf: &[u8]) -> usize {
        let hdr = self.header().get(buf);
        hdr.expected_frame_size()
    }

    pub(crate) fn capture_slot(&self, capture_ndx: usize, buf: &[u8]) -> Offset<CaptureSlot> {
        let hdr = self.header().get(buf);
        assert!(capture_ndx < hdr.n_captures as usize);
        Offset::at(self.top + Self::HEADER_SIZE + size_of::<CaptureSlot>() * capture_ndx)
    }

    // TODO Fix in-memory repr of Value
    //
    // These two functions are, of course, supreme bullshit.
    //
    // Why?  Because as soon as you get a Offset<interpreter::Value>, you're going to do get()
    // or get_mut() on it, which means you're going to reinterpret a slice of bytes from
    // your buffer as a Value.
    //
    // But Value has a bunch of semantics currently attached to it. For example, Value::Object
    // holds a shared reference to an object via RefCell<Rc<...>>.  That means that *somebody*
    // has to increase/decrease that reference count!  Even worse, that someone has to
    // *remember* to do it correctly every time.   Value::String holds (via a Cow, so not
    // always) an actual whole-ass String object. So one problem is that the current
    // scheme VERY PROBABLY LEAKS MEMORY.
    //
    // This is all going to be solved by:
    //  - having Value only hold simple, dumb, Copy handles to objects or strings
    //  - using those handles to access those objects/strings via the GC-managed heap
    //  - having an actual GC to scan the stack/heap/whatever
    //
    // That's not a problem I'm going to solve tonight though, so, I'll keep the unsafe
    // situation going for a little while more.

    pub(crate) fn result_slot(&self, result_ndx: usize, buf: &[u8]) -> Option<Offset<ResultSlot>> {
        let hdr = self.header().get(buf);
        if result_ndx < hdr.n_instrs as usize {
            Some(Offset::at(
                self.top
                    + Self::HEADER_SIZE
                    + size_of::<CaptureSlot>() * hdr.n_captures as usize
                    + size_of::<ResultSlot>() * result_ndx,
            ))
        } else {
            None
        }
    }

    pub(crate) fn arg_slot(&self, arg_ndx: usize, buf: &[u8]) -> Option<Offset<ArgSlot>> {
        let hdr = self.header().get(buf);
        if arg_ndx < hdr.n_args as usize {
            Some(Offset::at(
                self.top
                    + Self::HEADER_SIZE
                    + size_of::<CaptureSlot>() * hdr.n_captures as usize
                    + size_of::<ResultSlot>() * hdr.n_instrs as usize
                    + size_of::<ArgSlot>() * arg_ndx,
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn for_each_slot<F>(metrics: &FrameMetrics, buf: &[u8], mut handle_range: F)
    where
        F: FnMut(Range<usize>, &str),
    {
        handle_range(metrics.header().range(), "header");

        let hdr = metrics.header().get(buf);

        for i in 0..hdr.n_captures {
            let comment = format!("capture #{i}");
            handle_range(metrics.capture_slot(i as usize, buf).range(), &comment);
        }
        for i in 0..hdr.n_instrs {
            let comment = format!("result #{i}");
            handle_range(metrics.result_slot(i as usize, buf).unwrap().range(), &comment);
        }
        for i in 0..hdr.n_args {
            let comment = format!("arg #{i}");
            handle_range(metrics.arg_slot(i as usize, buf).unwrap().range(), &comment);
        }
    }

    fn all_slots(metrics: &FrameMetrics, buf: &[u8]) -> Vec<Range<usize>> {
        let mut slots = Vec::new();
        for_each_slot(metrics, buf, |range, _| slots.push(range));
        slots
    }

    fn ranges_intersect(a: &Range<usize>, b: &Range<usize>) -> bool {
        !(a.end <= b.start || a.start >= b.end)
    }

    fn random_header() -> FrameHeader {
        FrameHeader {
            n_instrs: rand::random::<u32>() % 100,
            n_captures: rand::random::<u16>() % 100,
            n_args: rand::random::<u8>() % 100,
            return_value_vreg: Some(VReg(rand::random())),
            return_to_iid: Some(IID(rand::random())),
            fn_id: FnId(rand::random::<u32>()),
        }
    }

    #[test]
    fn test_slots_do_not_overlap() {
        let hdr = random_header();
        let buf_size = hdr.expected_frame_size();
        let metrics = FrameMetrics { top: 0 };
        let mut buf = vec![0u8; buf_size];
        *metrics.header().get_mut(&mut buf) = hdr;

        let slots = all_slots(&metrics, &buf);
        for i in 0..slots.len() {
            for j in 0..i {
                let range_a = &slots[i];
                let range_b = &slots[j];

                assert!(
                    !ranges_intersect(range_a, range_b),
                    "ranges unexpectedly intersect: #{} ({:?}) and #{} ({:?})",
                    i,
                    range_a,
                    j,
                    range_b
                );
            }
        }
    }

    #[test]
    fn test_multiple_frames_slots_do_not_overlap() {
        // ordered bottom to top
        let headers: Vec<_> = (0..10).map(|_| random_header()).collect();
        let buf_size = headers.iter().map(|hdr| hdr.expected_frame_size()).sum();
        let mut buf = vec![0u8; buf_size];

        let mut metrics = FrameMetrics { top: buf_size };
        for (ndx, header) in headers.iter().enumerate() {
            let frame_sz = header.expected_frame_size();
            metrics.top -= frame_sz;
            eprintln!("frame #{}: {} bytes", ndx, frame_sz);

            *metrics.header().get_mut(&mut buf) = header.clone();

            for_each_slot(&metrics, &buf, |range, comment| {
                assert!(
                    range.start >= metrics.top,
                    "assertion failed: range.start ({}) >= metrics.top ({}) ({})",
                    range.start,
                    metrics.top,
                    comment,
                );
                assert!(
                    range.end <= metrics.top + frame_sz,
                    "assertion failed: range.end ({}) < metrics.top ({}) + frame_sz ({}) = {} ({})",
                    range.start,
                    metrics.top,
                    frame_sz,
                    metrics.top + frame_sz,
                    comment
                );
            });
        }
    }

    #[test]
    fn test_frame_slots_within_frame() {
        let padding = 221;

        // ordered bottom to top
        let header = random_header();
        let frame_sz = header.expected_frame_size();
        eprintln!("frame is {} bytes", frame_sz);

        let buf_size = padding + frame_sz;
        let mut buf = vec![0u8; buf_size];
        eprintln!("buffer is {} bytes ({} is padding)", buf_size, padding);

        let metrics = FrameMetrics { top: padding };
        *metrics.header().get_mut(&mut buf) = header.clone();

        for_each_slot(&metrics, &buf, |range, comment| {
            assert!(
                range.start >= padding,
                "assertion failed: range.start ({}) >= padding ({}) ({})",
                range.start,
                padding,
                comment,
            );
            assert!(
                range.end <= buf_size,
                "assertion failed: range.end ({}) < buf_size ({}) ({})",
                range.end,
                buf_size,
                comment
            );
        });
    }

    #[test]
    fn test_whole_frame_covered() {
        let padding = 200;
        let hdr = random_header();
        let buf_size = padding + hdr.expected_frame_size();
        let metrics = FrameMetrics { top: padding };
        let mut buf = vec![0u8; buf_size];
        *metrics.header().get_mut(&mut buf) = hdr;
        let mut covered = vec![false; buf_size];

        eprintln!("frame size = {}", buf_size);

        let slots = all_slots(&metrics, &buf);
        for range in slots {
            for ndx in range {
                assert!(covered[ndx] == false);
                covered[ndx] = true;
            }
        }

        eprintln!("Frame map: ");
        for (ndx, cell) in covered.iter().enumerate() {
            if ndx % 32 == 0 {
                eprintln!();
                eprint!("{:8}: ", ndx);
            } else if ndx % 8 == 0 {
                eprint!(" ");
            }

            eprint!("{}", if *cell { '#' } else { '_' });
        }
        eprintln!();
        assert!(covered.iter().skip(padding).all(|x| *x == true));
    }

    #[test]
    fn test_decoding() {
        let bytes = [9_u8, 22, 54, 212, 73, 5, 99, 11, 90];
        let ofs = Offset::at(2);
        let value: &i32 = ofs.get(&bytes);

        assert_eq!(*value, 88724534);
    }

    #[test]
    fn test_encoding() {
        let mut bytes = [9_u8, 22, 0, 0, 0, 0, 99, 11, 90];
        let ofs = Offset::at(2);
        let value: &mut i32 = ofs.get_mut(&mut bytes);
        *value = 88724534;

        assert_eq!(&[9_u8, 22, 54, 212, 73, 5, 99, 11, 90], &bytes);
    }

    #[test]
    fn test_encoding_sanity() {
        let mut bytes = [0u8; 90];
        check_encoding_sanity(&mut bytes, 7, 85834561234589145_u64);
        check_encoding_sanity(&mut bytes, 8, 99.23445f64);
        check_encoding_sanity(&mut bytes, 9, '#');
    }
    fn check_encoding_sanity<T>(buf: &mut [u8], offset: usize, value: T)
    where
        T: 'static + PartialEq + Clone + Copy + std::fmt::Debug,
    {
        let offset = Offset::at(offset);
        {
            let buf_field = offset.get_mut(buf);
            *buf_field = value.clone();
        }
        {
            let buf_field = offset.get(buf);
            assert_eq!(*buf_field, value);
        }
    }
}
