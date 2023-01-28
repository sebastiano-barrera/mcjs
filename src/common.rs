
pub type Result<T> = std::result::Result<T, Error>;

// TODO Use `thiserror`
#[derive(Debug)]
pub enum Error {
    FileNotFound,
    Io(std::io::Error),
    ParseError,
    UnsupportedItem,
    UnboundVariable(String),
}

impl From<std::io::Error> for Error {
    fn from(io_err: std::io::Error) -> Self {
        Error::Io(io_err)
    }
}

