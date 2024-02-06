mod formatter;

pub use formatter::{Dump, DumpExt, Formatter};

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
