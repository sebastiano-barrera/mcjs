use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::bytecode;
use crate::bytecode_compiler::Loader;

pub struct FileLoader {
    module_paths: Vec<PathBuf>,
    module_ids: HashMap<PathBuf, bytecode::ModuleId>,
    path_of_module: HashMap<bytecode::ModuleId, PathBuf>,
    next_module_id: u16,
}

impl FileLoader {
    pub fn new(module_paths: Vec<PathBuf>) -> Self {
        let module_paths = module_paths
            .into_iter()
            .map(|p| p.canonicalize().expect("could not canonicalize path"))
            .collect();
        FileLoader {
            module_paths,
            module_ids: HashMap::new(),
            path_of_module: HashMap::new(),
            next_module_id: 0,
        }
    }
}

impl Loader for FileLoader {
    fn get_module_id(&mut self, relative_path: &str) -> bytecode::ModuleId {
        let relative_path = Path::new(relative_path);

        let file_path = self
            .module_paths
            .iter()
            .find_map(|mp| {
                let candidate_file_path = mp.join(&relative_path);
                match candidate_file_path.is_file() {
                    true => Some(candidate_file_path),
                    false => None,
                }
            })
            .unwrap_or_else(|| panic!("module not found: '{}'", relative_path.to_string_lossy()));

        if let Some(mid) = self.module_ids.get(&file_path) {
            *mid
        } else {
            let mid = bytecode::ModuleId(self.next_module_id);
            self.module_ids.insert(file_path.clone(), mid);
            self.path_of_module.insert(mid, file_path);
            self.next_module_id += 1;
            mid
        }
    }

    fn read_source(&self, module_id: crate::bytecode::ModuleId) -> String {
        let path = self.path_of_module.get(&module_id).unwrap();
        std::fs::read_to_string(path).expect("read error")
    }
}
