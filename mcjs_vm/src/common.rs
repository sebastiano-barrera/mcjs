use std::rc::Rc;

use swc_common::{SourceMap, Span};

struct Item {
    span: Option<Span>,
    vm_filename: String,
    vm_lineno: u32,
    message: String,
}

impl std::fmt::Debug for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(span) = self.span {
            write!(
                f,
                "{:?}: {} [VM code at {}:{}]",
                span, self.message, self.vm_filename, self.vm_lineno
            )
        } else {
            write!(
                f,
                "{} [VM code at {}:{}]",
                self.message, self.vm_filename, self.vm_lineno
            )
        }
    }
}

impl Item {
    fn message(&self, buf: &mut String, source_map: Option<&SourceMap>, indent_level: usize) {
        use std::fmt::Write;

        for _ in 0..indent_level {
            write!(buf, "  ").unwrap();
        }

        match (source_map, self.span) {
            (Some(source_map), Some(span)) => {
                let lo = source_map.lookup_char_pos(span.lo);
                let hi = source_map.lookup_char_pos(span.hi);
                write!(
                    buf,
                    "{}: {},{} - {},{}: {} [VM code at {}:{}]",
                    lo.file.name,
                    lo.line,
                    lo.col_display,
                    hi.line,
                    hi.col_display,
                    self.message,
                    self.vm_filename,
                    self.vm_lineno
                )
                .unwrap();
            }

            (None, Some(span)) => {
                write!(
                    buf,
                    "byte {} - byte {}: {} [VM code at {}:{}]",
                    span.lo.0, span.hi.0, self.message, self.vm_filename, self.vm_lineno
                )
                .unwrap();
            }

            (_, None) => buf.push_str(self.message.as_str()),
        }
    }
}

pub struct Error {
    head: Item,
    chain: Vec<Item>,
    source_map: Option<Rc<SourceMap>>,
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())?;
        let mut buf = String::new();
        for ctx_item in self.chain.iter() {
            ctx_item.message(&mut buf, self.source_map.as_deref(), 0);
            write!(f, "\n  {}", buf)?;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! error {
    ($($args:expr),+) => {{
        let message = format!($($args),*);
        $crate::common::Error::new(message, file!().to_owned(), line!())
    }}
}

impl From<std::io::Error> for Error {
    fn from(io_err: std::io::Error) -> Error {
        error!("{}", io_err)
    }
}

impl Error {
    pub(crate) fn new(message: String, src_filename: String, src_lineno: u32) -> Self {
        Error {
            head: Item {
                span: None,
                vm_filename: src_filename,
                vm_lineno: src_lineno,
                message,
            },
            chain: Vec::new(),
            source_map: None,
        }
    }

    pub(crate) fn with_span(mut self, span: Span) -> Self {
        self.head.span = Some(span);
        self
    }

    pub(crate) fn with_source_map(mut self, source_map: Rc<SourceMap>) -> Self {
        self.source_map = Some(source_map);
        self
    }

    pub fn message(&self) -> String {
        self.message_ex(0)
    }

    pub fn message_ex(&self, indent_level: usize) -> String {
        let mut buf = String::new();
        let sm = self.source_map.as_ref().map(|rc| rc.as_ref());
        self.head.message(&mut buf, sm, indent_level);
        for ctx_item in self.chain.iter() {
            buf.push('\n');
            ctx_item.message(&mut buf, sm, indent_level + 1);
        }

        buf
    }

    pub fn messages(&self) -> impl '_ + Iterator<Item = String> {
        std::iter::once(&self.head)
            .chain(self.chain.iter())
            .map(|item| {
                let mut buf = String::new();
                item.message(&mut buf, self.source_map.as_deref(), 0);
                buf
            })
    }
}

pub trait Context<Err> {
    fn with_context(self, other: Err) -> Self;
}

impl Context<Error> for Error {
    fn with_context(mut self, mut other: Self) -> Self {
        self.chain.push(other.head);
        self.chain.extend(other.chain);
        if self.source_map.is_none() {
            self.source_map = other.source_map.take();
        }
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl<T> Context<Error> for Result<T> {
    fn with_context(self, other: Error) -> Self {
        self.map_err(|err| err.with_context(other))
    }
}

// TODO: It would be cool to allow make a 'with_context_fn' that takes a closure and
// allows makign the Error object only in case of a Result::Err
