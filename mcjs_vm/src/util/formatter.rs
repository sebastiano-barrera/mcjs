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

