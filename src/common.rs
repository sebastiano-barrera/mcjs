use swc_common::{SourceMap, Span};

struct Item {
    span: Option<Span>,
    message: String,
}

impl std::fmt::Debug for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(span) = self.span {
            write!(f, "{:?}: {}", span, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl Item {
    fn message(&self, buf: &mut String, source_map: &SourceMap, indent_level: usize) {
        use std::fmt::Write;

        for _ in 0..indent_level {
            write!(buf, "  ").unwrap();
        }

        match self.span {
            Some(span) => {
                let span_str = source_map.span_to_string(span);
                write!(buf, "{span_str}: {}", self.message).unwrap();
            }
            None => buf.push_str(self.message.as_str()),
        }
    }
}

pub struct Error {
    head: Item,
    chain: Vec<Item>,
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.head)?;
        for ctx_item in self.chain.iter() {
            write!(f, "\n  {:?}", ctx_item)?;
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! error {
    ($($args:expr),+) => {{
        let message = format!($($args),*);
        Error::new(message)
    }}
}

impl From<std::io::Error> for Error {
    fn from(io_err: std::io::Error) -> Error {
        error!("{}", io_err)
    }
}

impl Error {
    pub(crate) fn new(message: String) -> Self {
        Error {
            head: Item {
                span: None,
                message,
            },
            chain: Vec::new(),
        }
    }

    pub(crate) fn with_span(mut self, span: Span) -> Self {
        self.head.span = Some(span);
        self
    }

    pub fn message(&self, source_map: &SourceMap) -> String {
        self.message_ex(source_map, 0)
    }

    pub fn message_ex(&self, source_map: &SourceMap, indent_level: usize) -> String {
        let mut buf = String::new();
        self.head.message(&mut buf, source_map, indent_level);
        for ctx_item in self.chain.iter() {
            buf.push('\n');
            ctx_item.message(&mut buf, source_map, indent_level + 1);
        }

        buf
    }
}

pub trait Context<Err> {
    fn with_context(self, other: Err) -> Self;
}

impl Context<Error> for Error {
    fn with_context(mut self, other: Self) -> Self {
        self.chain.push(other.head);
        self.chain.extend(other.chain);
        self
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl<T> Context<Error> for Result<T> {
    fn with_context(self, other: Error) -> Self {
        self.map_err(|err| err.with_context(other))
    }
}
