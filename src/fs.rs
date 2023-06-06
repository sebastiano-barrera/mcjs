use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::bytecode::{self, ModuleId};
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
    fn get_module_id(&mut self, relative_path: &str) -> Option<bytecode::ModuleId> {
        let relative_path = Path::new(relative_path);

        let file_path = self.module_paths.iter().find_map(|mp| {
            let candidate_file_path = mp.join(&relative_path);
            match candidate_file_path.is_file() {
                true => Some(candidate_file_path),
                false => None,
            }
        })?;

        let mid = if let Some(mid) = self.module_ids.get(&file_path) {
            *mid
        } else {
            let mid = bytecode::ModuleId(self.next_module_id);
            self.module_ids.insert(file_path.clone(), mid);
            self.path_of_module.insert(mid, file_path);
            self.next_module_id += 1;
            mid
        };

        Some(mid)
    }

    fn read_source(&self, module_id: crate::bytecode::ModuleId) -> String {
        let path = self.path_of_module.get(&module_id).unwrap();
        std::fs::read_to_string(path).expect("read error")
    }
}

pub struct MockLoader {
    mid_of_file: HashMap<String, bytecode::ModuleId>,
    code_of_module: HashMap<bytecode::ModuleId, String>,
}

impl MockLoader {
    pub fn new() -> Self {
        MockLoader {
            mid_of_file: HashMap::new(),
            code_of_module: HashMap::new(),
        }
    }
    pub fn add_module(&mut self, filename: String, module_id: bytecode::ModuleId, code: String) {
        self.mid_of_file.insert(filename, module_id);
        self.code_of_module.insert(module_id, code);
    }
}

impl Loader for MockLoader {
    fn get_module_id(&mut self, filename: &str) -> Option<bytecode::ModuleId> {
        self.mid_of_file.get(filename).copied()
    }

    fn read_source(&self, module_id: bytecode::ModuleId) -> String {
        self.code_of_module.get(&module_id).cloned().unwrap()
    }
}

pub struct CombinedLoader {
    inner: Vec<Box<dyn Loader>>,
    module_ids: HashMap<bytecode::ModuleId, (usize, bytecode::ModuleId)>,
    last_module_id: u16,
}

impl CombinedLoader {
    pub fn new(inner: Vec<Box<dyn Loader>>) -> Self {
        CombinedLoader {
            inner,
            module_ids: HashMap::new(),
            last_module_id: 0,
        }
    }

    fn next_module_id(&mut self) -> ModuleId {
        let new_module_id = bytecode::ModuleId(self.last_module_id);
        self.last_module_id += 1;
        new_module_id
    }
}

impl Loader for CombinedLoader {
    fn get_module_id(&mut self, filename: &str) -> Option<bytecode::ModuleId> {
        for (ndx, loader) in self.inner.iter_mut().enumerate() {
            if let Some(inner_module_id) = loader.get_module_id(filename) {
                let new_module_id = self.next_module_id();
                self.module_ids
                    .insert(new_module_id, (ndx, inner_module_id));
                return Some(new_module_id);
            }
        }

        None
    }

    fn read_source(&self, module_id: bytecode::ModuleId) -> String {
        let (loader_ndx, inner_mod_id) = *self.module_ids.get(&module_id).unwrap();
        let loader = &self.inner[loader_ndx];
        loader.read_source(inner_mod_id)
    }
}
