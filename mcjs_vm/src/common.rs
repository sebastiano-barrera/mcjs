use std::{fmt::Write, rc::Rc};

use swc_common::{SourceMap, Span};

use crate::{bytecode, Loader};

pub struct ErrorItem {
    src_span: Option<Span>,
    src_giid: Option<bytecode::GlobalIID>,
    vm_filename: String,
    vm_lineno: u32,
    message: String,
}

impl ErrorItem {
    pub(crate) fn new(message: String, vm_filename: String, vm_lineno: u32) -> Self {
        ErrorItem {
            src_span: None,
            src_giid: None,
            vm_filename,
            vm_lineno,
            message,
        }
    }

    pub(crate) fn set_giid(&mut self, giid: bytecode::GlobalIID) {
        self.src_giid = Some(giid);
    }
    pub(crate) fn with_giid(mut self, giid: bytecode::GlobalIID) -> Self {
        self.set_giid(giid);
        self
    }

    fn write_to<W: Write>(&self, out: &mut W, loader: Option<&Loader>) {
        if let Some(bytecode::GlobalIID(fnid, iid)) = &self.src_giid {
            write!(out, "f{}/i{}: ", fnid.0, iid.0).unwrap();
        }

        let src_span = self.src_span.or_else(|| {
            // No span? Try getting one out of the giid, if we have it
            let loader = loader?;
            let giid = self.src_giid?;
            let bytecode::GlobalIID(fnid, iid) = giid;
            let breakranges = loader.function_breakranges(fnid)?;
            let breakrange = breakranges
                .map(|(_, br)| br)
                .filter(|br| br.iid_start <= iid && iid < br.iid_end)
                .min_by_key(|br| br.hi - br.lo)?;

            Some(swc_common::Span::new(
                breakrange.lo,
                breakrange.hi,
                swc_common::SyntaxContext::default(),
            ))
        });

        if let Some(src_span) = &src_span {
            let source_map = loader.map(|l| l.get_source_map());
            if let Some(source_map) = source_map {
                let lo = source_map.lookup_char_pos(src_span.lo);
                let hi = source_map.lookup_char_pos(src_span.hi);
                write!(
                    out,
                    "{}:{}:{} to {}:{}: ",
                    lo.file.name, lo.line, lo.col_display, hi.line, hi.col_display,
                )
                .unwrap();
            } else {
                write!(
                    out,
                    "byte {} - byte {}: {} [VM code at {}:{}]",
                    src_span.lo.0, src_span.hi.0, self.message, self.vm_filename, self.vm_lineno
                )
                .unwrap();
            }
        }

        write!(out, "{}", self.message).unwrap();
    }
}

pub struct Error {
    head: ErrorItem,
    chain: Vec<ErrorItem>,
    source_map: Option<Rc<SourceMap>>,
}

impl Error {
    pub(crate) fn new(head: ErrorItem) -> Self {
        Error {
            head,
            chain: Vec::new(),
            source_map: None,
        }
    }

    pub(crate) fn from_err<E: std::error::Error>(value: E) -> Self {
        let message = value.to_string();
        Error::new(ErrorItem::new(message, String::new(), 0))
    }

    pub(crate) fn push_context(&mut self, context: ErrorItem) {
        self.chain.push(context);
    }

    pub fn write_to<W: Write>(&self, out: &mut W, loader: Option<&Loader>) -> std::fmt::Result {
        self.head.write_to(out, loader);
        writeln!(out)?;
        for ctx_item in self.chain.iter() {
            write!(out, "    ")?;
            ctx_item.write_to(out, loader);
            writeln!(out, "")?;
        }
        Ok(())
    }
}
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_to(f, None)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
impl std::error::Error for Error {}

#[macro_export]
macro_rules! error_item {
    ($($args:expr),+) => {{
        let message = format!($($args),*);
        $crate::common::ErrorItem::new(
            message,
            file!().to_owned(),
            line!(),
        )
    }}
}

#[macro_export]
macro_rules! error {
    ($($args:expr),+) => {{
        use crate::error_item;
        $crate::common::Error::new(error_item!($($args),*))
    }}
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait Context<E> {
    fn with_context(self, other: E) -> Self;
    fn with_source_map(self, source_map: Rc<SourceMap>) -> Self;
}

impl Context<ErrorItem> for Error {
    fn with_context(mut self, context: ErrorItem) -> Self {
        self.push_context(context);
        self
    }

    fn with_source_map(mut self, source_map: Rc<SourceMap>) -> Self {
        self.source_map = Some(source_map);
        self
    }
}

impl<T> Context<ErrorItem> for std::result::Result<T, Error> {
    fn with_context(self, other: ErrorItem) -> Self {
        self.map_err(|err| err.with_context(other))
    }

    fn with_source_map(self, source_map: Rc<SourceMap>) -> Self {
        self.map_err(|err| err.with_source_map(source_map))
    }
}

pub type MultiErrResult<T> = std::result::Result<T, MultiError>;

pub struct MultiError(pub Vec<Error>);

impl MultiError {
    pub fn into_single(self) -> Error {
        assert!(!self.0.is_empty());
        if self.0.len() == 1 {
            self.0.into_iter().next().unwrap()
        } else {
            let mut str_buf = String::new();
            write!(str_buf, "{:?}", self).unwrap();
            error!("{}", str_buf)
        }
    }
}

impl std::fmt::Debug for MultiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} errors:", self.0.len())?;
        for (ndx, err) in self.0.iter().enumerate() {
            writeln!(f, "error #{}: {}", ndx + 1, err)?;
        }
        Ok(())
    }
}

pub struct ErrorWithLoader<'a> {
    err: Error,
    loader: &'a Loader,
}
impl Error {
    pub(crate) fn with_loader<'a>(self, loader: &'a Loader) -> ErrorWithLoader<'a> {
        ErrorWithLoader { err: self, loader }
    }
}
impl<'a> std::fmt::Debug for ErrorWithLoader<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.err.write_to(f, Some(self.loader))
    }
}

// TODO: It would be cool to allow make a 'with_context_fn' that takes a closure and
// allows makign the Error object only in case of a Result::Err
