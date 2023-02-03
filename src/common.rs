pub type Result<T> = std::result::Result<T, Error>;
use swc_common::{SourceMap, Span};

// TODO Use `thiserror`
#[derive(Debug)]
pub enum Error {
    FileNotFound,
    Io(std::io::Error),
    ParseError,
    UnsupportedItem { span: Span, details: &'static str },
    UnboundVariable { span: Span, ident: String },
    IllegalAssignment { span: Span, target: String },

    NativeInvalidArgs,
    NativeNoSuchFunction(u32),

    NoSuchModule(String),
}

impl From<std::io::Error> for Error {
    fn from(io_err: std::io::Error) -> Self {
        Error::Io(io_err)
    }
}

impl Error {
    pub fn span(&self) -> Option<&Span> {
        match self {
            Error::UnsupportedItem { span, .. } => Some(span),
            Error::UnboundVariable { span, .. } => Some(span),
            Error::IllegalAssignment { span, .. } => Some(span),
            _ => None,
        }
    }

    pub fn message(&self, source_map: &SourceMap) -> String {
        match self {
            Error::FileNotFound => "file not found".to_string(),
            Error::Io(io_err) => io_err.to_string(),
            Error::ParseError => "parse error".to_string(),
            Error::UnsupportedItem { span, details } => {
                let span_str = source_map.span_to_string(*span);
                format!("{span_str}: compiler limitation: {details}")
            }
            Error::UnboundVariable { span, ident } => {
                let span_str = source_map.span_to_string(*span);
                format!("{span_str}: unbound variable `{ident}`")
            }
            Error::IllegalAssignment { span, target } => {
                let span_str = source_map.span_to_string(*span);
                format!("{span_str}: illegal assignment to `{target}` (const?)")
            }
            Error::NativeInvalidArgs => "native function: invalid args".to_string(),
            Error::NativeNoSuchFunction(nfid) => format!("no such native function with ID {nfid}"),
            Error::NoSuchModule(name) => format!("no such module `{name}`"),
        }
    }
}

