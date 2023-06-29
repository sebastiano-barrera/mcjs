use std::path::PathBuf;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum Root {
    ModuleImport(String),
    InlineScript(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Case {
    pub include_paths: Vec<PathBuf>,
    pub root: Root,
}

