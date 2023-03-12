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
