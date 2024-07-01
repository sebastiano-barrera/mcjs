mod formatter;

#[macro_export]
macro_rules! define_flag {
    ($vis:vis $type_name:ident) => {
        #[derive(PartialEq, Eq, Clone, Copy, Debug)]
        $vis enum $type_name {
            Yes,
            No,
        }

        impl From<$type_name> for bool {
            fn from(value: $type_name) -> bool {
                value == $type_name::Yes
            }
        }
    };
}

#[cfg(test)]
pub use formatter::DumpExt;
pub use formatter::{Dump, Formatter};

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
