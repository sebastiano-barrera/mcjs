use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

pub fn export_inspector_case(include_paths: Vec<PathBuf>, root: Root) -> PathBuf {
    let case_file_path = {
        let mut counter = 0;
        loop {
            let path = PathBuf::from(format!("/tmp/mcjs-inspector-{}.case", counter));
            if !path.exists() {
                break path;
            }
            counter += 1;
        }
    };

    let case = Case {
        include_paths,
        root,
    };

    let mut f = std::fs::File::create(&case_file_path).expect("could not create case file");
    let mut serializer = rmp_serde::Serializer::new(&mut f).with_binary();
    case.serialize(&mut serializer)
        .expect("could not serialize");
    eprintln!();
    eprintln!("exported inspector case file: {}", case_file_path.display());
    case_file_path
}
