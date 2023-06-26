use std::path::PathBuf;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub enum Root {
    ModuleImport(String),
    InlineScript(String),
}

#[derive(Serialize, Deserialize)]
pub struct Case {
    pub include_paths: Vec<PathBuf>,
    pub root: Root,
}

