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
