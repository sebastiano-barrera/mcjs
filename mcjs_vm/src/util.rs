#[derive(Clone)]
pub struct Mask(Vec<bool>);

impl Mask {
    pub fn new() -> Self {
        Mask(Vec::new())
    }

    pub fn set(&mut self, ndx: usize, value: bool) {
        while self.0.len() <= ndx {
            self.0.push(false)
        }
        self.0[ndx] = value;
    }

    pub fn get(&self, ndx: usize) -> bool {
        self.0.get(ndx).copied().unwrap_or(false)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &bool> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|x| !(*x))
    }
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (ndx, value) in self.iter().enumerate() {
            if *value {
                write!(f, "{}", ndx)?;
            }
        }
        write!(f, "}}")
    }
}

pub struct LimVec<const N: usize, T: Clone + Copy>([Option<T>; N]);

impl<const N: usize, T: Clone + Copy> Default for LimVec<N, T> {
    fn default() -> Self {
        LimVec([None; N])
    }
}

impl<const N: usize, T> LimVec<N, T>
where
    T: Clone + Copy,
{
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter().filter_map(|slot| slot.as_ref())
    }

    pub fn from_iter<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        let mut arr = [None; N];
        for (i, item) in iter.enumerate() {
            arr[i] = Some(item);
        }
        LimVec(arr)
    }
}

pub fn shorten_by<T>(xs: &mut Vec<T>, n: usize) {
    xs.truncate(xs.len() - n);
}

pub fn pop_while<T>(vec: &mut Vec<T>, pred: impl Fn(&T) -> bool) {
    while let Some(item) = vec.last() {
        if pred(item) {
            vec.pop();
        } else {
            break;
        }
    }
}

pub fn write_comma_sep<W, T>(wrt: &mut W, values: impl Iterator<Item = T>) -> std::fmt::Result
where
    W: std::fmt::Write,
    T: ToString,
{
    for (ndx, value) in values.enumerate() {
        if ndx > 0 {
            write!(wrt, ", ")?;
        }
        write!(wrt, "{}", value.to_string())?;
    }

    Ok(())
}

pub use formatter::{Dump, DumpExt, Formatter};

mod formatter {
    pub struct Formatter<W: std::fmt::Write> {
        indent_level: usize,
        deferred_indent: bool,
        writer: W,
    }

    pub trait Dump {
        fn dump<W: std::fmt::Write>(&self, f: &mut Formatter<W>) -> std::fmt::Result;
    }

    #[macro_export]
    macro_rules! impl_debug_via_dump {
        ($type:ty) => {
            impl std::fmt::Debug for $type {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    use $crate::util::Dump;
                    let mut dump_fmter = $crate::util::Formatter::new(f);
                    self.dump(&mut dump_fmter)
                }
            }
        };
    }

    impl<W: std::fmt::Write> Formatter<W> {
        pub fn new(writer: W) -> Self {
            Formatter {
                indent_level: 0,
                deferred_indent: false,
                writer,
            }
        }

        pub fn indent(&mut self) {
            self.indent_level += 1;
        }
        pub fn dedent(&mut self) {
            self.indent_level -= 1;
        }

        pub fn finish(self) -> W {
            self.writer
        }
    }

    impl<W: std::fmt::Write> std::fmt::Write for Formatter<W> {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            fn print_indent<W: std::fmt::Write>(
                wrt: &mut W,
                indent_level: usize,
            ) -> std::fmt::Result {
                wrt.write_str("\n")?;
                for _ in 0..indent_level {
                    wrt.write_str("  ")?;
                }
                Ok(())
            }

            if self.deferred_indent {
                print_indent(&mut self.writer, self.indent_level)?;
                self.deferred_indent = false;
            }

            for (ndx, line) in s.split('\n').enumerate() {
                if ndx > 0 {
                    if !line.is_empty() {
                        print_indent(&mut self.writer, self.indent_level)?;
                    } else {
                        self.deferred_indent = true;
                    }
                }
                self.writer.write_str(line)?;
            }

            Ok(())
        }
    }

    pub trait DumpExt {
        fn dump_to_string(&self) -> String;
    }

    impl<T: Dump> DumpExt for T {
        fn dump_to_string(&self) -> String {
            let mut fmter = Formatter::new(String::new());
            self.dump(&mut fmter).unwrap();
            fmter.finish()
        }
    }
}
