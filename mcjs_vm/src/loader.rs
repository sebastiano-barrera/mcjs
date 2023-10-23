use std::path::Path;
use std::rc::Rc;
use std::sync::Mutex;
use std::{collections::HashMap, path::PathBuf};

use crate::bytecode;
use crate::bytecode_compiler;
use crate::common::{Context, Error, Result};
use crate::error;

/// A `Loader` manages the loading, compilation and storage of all parts of a
/// JavaScript codebase.
///
/// It is the main point of entry to the AST->bytecode compilation system (impl:
/// although it does not include the bytecode compiler itself).
///
/// It stores modules, associating them to unique IDs.
pub struct Loader {
    root_pkg_path: Option<PathBuf>,

    next_module_id: u16,
    next_package_id: u16,

    // Every loader always contains a single script, which can only ever be *accreted* by compiling
    // many chunks of code. (See Loader::load_script).
    script: Script,

    modules: HashMap<bytecode::ModuleId, Module>,
    // Key is the absolute filename of the (previously loaded) module
    mod_of_path: HashMap<PathBuf, bytecode::ModuleId>,
    packages: HashMap<PackageId, Package>,
    pkg_of_name: HashMap<String, PackageId>,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct PackageId(u16);

const ROOT_PKG_ID: PackageId = PackageId(0);

struct Module {
    compiled: bytecode_compiler::CompiledChunk,
    /// ID of the package this module belongs to.
    package_id: PackageId,
    /// This module's path, *relative* to the package's root path.
    path: String,
}

struct Package {
    /// Absolute path of the package's root directory.  Modules belonging to
    /// this package can be designated by a relative path from this directory.
    root_path: PathBuf,
    /// Relative path to the main module. (Relative to `root_path`)
    main_filename: PathBuf,
}
struct PkgJson {
    main: String,
}

struct Script {
    functions: HashMap<bytecode::LocalFnId, bytecode::Function>,
    max_fnid: u16,
}

impl Script {
    fn add_functions(&mut self, new_funcs: HashMap<bytecode::LocalFnId, bytecode::Function>) {
        if new_funcs.is_empty() {
            return;
        }

        let min_lfnid = new_funcs.keys().min().unwrap().0;
        let max_lfnid = new_funcs.keys().max().unwrap().0;

        assert!(min_lfnid > self.max_fnid);
        self.functions.extend(new_funcs.into_iter());

        self.max_fnid = max_lfnid;
    }
}

/// Represent where an import statement comes from (i.e., from a specific module or a
/// script). Determines the resolution of a module path to a specific file in
/// `Loader::load_module`.
pub enum ImportSite<'a> {
    /// Import comes from a specific module, which has this ID.
    Module(bytecode::ModuleId),
    /// Import comes from a script.
    Script(&'a Path),
}

impl Loader {
    pub fn new(root_pkg_path: Option<PathBuf>) -> Self {
        Loader {
            root_pkg_path,
            next_module_id: 1,
            next_package_id: 1,
            modules: HashMap::new(),
            mod_of_path: HashMap::new(),
            packages: HashMap::new(),
            pkg_of_name: HashMap::new(),
            script: Script {
                functions: HashMap::new(),
                max_fnid: 0,
            },
        }
    }

    pub fn get_module_root_fn(&self, module_id: bytecode::ModuleId) -> Option<bytecode::LocalFnId> {
        match module_id {
            bytecode::SCRIPT_MODULE_ID => None,
            module_id => Some(self.modules.get(&module_id)?.compiled.root_fnid),
        }
    }

    pub(crate) fn loaded_modules_paths(&self) -> &HashMap<PathBuf, bytecode::ModuleId> {
        &self.mod_of_path
    }

    pub fn get_function(&self, fnid: bytecode::FnId) -> Option<&bytecode::Function> {
        let bytecode::FnId(mod_id, lfnid) = fnid;

        match mod_id {
            bytecode::SCRIPT_MODULE_ID => self.script.functions.get(&lfnid),
            mod_id => self.modules.get(&mod_id)?.compiled.functions.get(&lfnid),
        }
    }

    /// Load a module (only if necessary) from an import statement.
    ///
    /// The arguments reflect the "coordinates" of the import statement:
    ///
    ///  - `import_path`: The string that identifies the module or package to import.
    ///    Supported forms:
    ///
    ///    - *relative paths* (e.g. `./asd/lol.js`) are for importing another module in
    ///      the same package as the importing one.
    ///
    ///    - *bare specifiers* (e.g. `react`) are for importing another package. The other
    ///      package's main module is loaded.
    ///
    ///  - `import_site`: the ID of the module where the import is taking place. This is
    ///    used to resolve relative paths, among other things.
    pub fn load_import(
        &mut self,
        import_path: String,
        import_site: bytecode::ModuleId,
    ) -> Result<bytecode::FnId> {
        if import_path.starts_with("./") {
            // Relative path

            // The starting point is the module's parent package
            let mod_id = match import_site {
                bytecode::SCRIPT_MODULE_ID => return Err(error!("invalid import: relative path not allowed if import statement comes from a script")),
                m => m,
            };

            let module = self.modules.get(&mod_id).ok_or_else(|| {
                error!("invalid import site: no such module with ID {:?}", mod_id)
            })?;
            // TODO Cleanup the choice of data types, so as to avoid so many unwrap()'s?
            let resolved_module_path = PathBuf::from(module.path.clone())
                .parent()
                .unwrap()
                .join(import_path)
                .into_os_string()
                .into_string()
                .unwrap();

            self.load_module(module.package_id, &resolved_module_path)
        } else if is_valid_package_name(&import_path) {
            // Package name
            if !self.pkg_of_name.contains_key(&import_path) {
                let root_pkg_path = self.root_pkg_path.as_ref().ok_or_else(|| {
                    error!("can't load packages from this loader: no root path has been configured")
                })?;
                let pkg_path = root_pkg_path.join("node_modules").join(&import_path);

                let package = parse_package(&pkg_path).map_err(|err| {
                    Error::from(err)
                        .with_context(error!("while loading imported package '{}'", import_path))
                })?;

                let pkg_id = self.gen_pkg_id();
                self.packages.insert(pkg_id, package);
                self.pkg_of_name.insert(import_path.clone(), pkg_id);
            }

            let pkg_id = *self.pkg_of_name.get(&import_path).unwrap();
            let pkg = self.packages.get(&pkg_id).unwrap();
            let module_path = pkg.main_filename.as_os_str().to_str().unwrap().to_string();

            self.load_module(pkg_id, &module_path)
        } else {
            Err(error!(
                "invalid module path: `{}` (only allowed: './relative/paths', and 'package-name')",
                import_path
            ))
        }
    }

    fn load_module(&mut self, package_id: PackageId, module_path: &str) -> Result<bytecode::FnId> {
        let package = self
            .packages
            .get(&package_id)
            .ok_or_else(|| error!("no package with ID {}", package_id.0))?;

        assert!(package.root_path.is_absolute());
        let filename = package.root_path.join(module_path);
        assert!(filename.is_absolute());

        if !filename.starts_with(&package.root_path) {
            return Err(error!(
                "invalid module path (resolves to a path outside the package)"
            ));
        }

        if let Some(mod_id) = self.mod_of_path.get(&filename) {
            // TODO Make the root FnId fixed (the same) across all modules (then
            // we can avoid this lookup)
            let module = self
                .modules
                .get(mod_id)
                .expect("loader bug: inconsistent `module` and `module_id`");
            return Ok(bytecode::FnId(*mod_id, module.compiled.root_fnid));
        }

        let content = std::fs::read_to_string(&filename).map_err(|err| {
            Error::from(err)
                .with_context(error!(
                    "while reading file `{}`",
                    filename.to_string_lossy()
                ))
                .with_context(error!("while loading module `{module_path}`"))
        })?;

        // TODO Remove this impedance mismatch between PathBufs and Strings
        let flags = bytecode_compiler::CompileFlags::default();
        let compiled = bytecode_compiler::compile_file(
            filename.to_string_lossy().into_owned(),
            content,
            &flags,
        )?;

        let root_fnid = compiled.root_fnid;
        let module = Module {
            compiled,
            package_id,
            path: module_path.to_string(),
        };

        let module_id = self.gen_mod_id();
        self.modules.insert(module_id, module);
        self.mod_of_path.insert(filename, module_id);
        Ok(bytecode::FnId(module_id, root_fnid))
    }

    /// Compile a new chunk of script code.
    ///
    /// This differs from module code in the following ways:
    ///
    ///  - each module is assigned to a separate ModuleId, whereas functions in the script
    ///    context always have module ID == bytecode::SCRIPT_MODULE_ID,
    ///
    ///  - each Loader always stores exactly one script.  No script can be created,
    ///    removed or replaced for a single Loader.
    ///
    ///  - each call to `load_module` generates a new module, completely independent from
    ///    previous compilations, whereas each call to load_script only ever *adds* more
    ///    functions to this Loader's script.
    ///
    ///    - nevertheless, compiling new script chunks never invalidates previously
    ///      compiled function IDs.
    pub fn load_script(
        &mut self,
        filename: Option<String>,
        content: String,
    ) -> Result<bytecode::FnId> {
        let filename = filename.unwrap_or_else(|| "<input>".to_string());
        let flags = bytecode_compiler::CompileFlags {
            min_fnid: Some(self.script.max_fnid + 1),
        };
        let compiled = bytecode_compiler::compile_file(filename, content, &flags)?;

        let chunk_fnid = compiled.root_fnid;
        self.script.add_functions(compiled.functions);
        // TODO Do something with the chunk's source map?

        Ok(bytecode::FnId(bytecode::SCRIPT_MODULE_ID, chunk_fnid))
    }

    pub fn resolve_loc(&self, filename: &str, line_number: usize) -> Result<Vec<GlobalIID>> {
        // Resolve the (potentially partial) filename to a full path
        let (filename, module_id) = {
            let all_paths = self.loaded_modules_paths();
            let mut matching_paths = all_paths
                .iter()
                .filter(|(path, _)| path.ends_with(filename));
            let resolved = matching_paths.next().ok_or(error!("no module with path `{}` (either full or suffix)", filename )?;

            // TODO Could it be useful to return all the matching paths? To help the user
            // narrow it down?
            if matching_paths.next().is_some() {
                return Err(ProbeError::AmbiguousFilename);
            }

            resolved
        };

        let (byte_lo, byte_hi) = {
            let source_map = self.get_source_map(*module_id).unwrap();
            let source_file = source_map
                .get_source_file(&swc_common::FileName::Real(filename.clone()))
                .ok_or(ProbeError::NoSourceMap)?;

            source_file.line_bounds(line_number)
        };
    }

    fn gen_mod_id(&mut self) -> bytecode::ModuleId {
        let mod_id = self.next_module_id;
        self.next_module_id += 1;
        bytecode::ModuleId::from(mod_id)
    }

    fn gen_pkg_id(&mut self) -> PackageId {
        let pkg_id = self.next_package_id;
        self.next_package_id += 1;
        assert_ne!(pkg_id, ROOT_PKG_ID.0);
        PackageId(pkg_id)
    }

    pub(crate) fn get_source_map(
        &self,
        module_id: bytecode::ModuleId,
    ) -> Option<&swc_common::SourceMap> {
        let module = self.modules.get(&module_id)?;
        Some(Rc::as_ref(&module.compiled.source_map))
    }

    // TODO(performance) Does it make sense to mmap the input file?
    // TODO Indirect filesystem access? (Check if needed)
    // TODO Virtual module (for testing/debugging)? (Check if needed)
}

fn is_valid_package_name(import_path: &str) -> bool {
    import_path
        .chars()
        .all(|ch| ch.is_alphabetic() || ch.is_digit(10) || ch == '-' || ch == '_')
}

// Private: packages are only loaded as part of the  'import' implementation (load_import)
fn parse_package(pkg_path: &Path) -> Result<Package> {
    let pkg_json_path = pkg_path.join("package.json");
    let pkg_json_file = std::fs::File::open(&pkg_json_path)
        .map_err(Error::from)
        .with_context(error!(
            "while opening package.json for {}",
            pkg_path.to_string_lossy()
        ))?;

    let pkg_spec_raw = std::io::read_to_string(&pkg_json_file)
        .map_err(Error::from)
        .with_context(error!(
            "while reading package spec [{}]",
            pkg_json_path.to_string_lossy()
        ))?;

    let json_value = json::parse(&pkg_spec_raw)
        .map_err(|err| error!("JSON parse error: {}", err))
        .with_context(error!(
            "while parsing the JSON in package.json file [{}]",
            pkg_json_path.to_string_lossy()
        ))?;

    let pkg_spec = match json_value {
        json::JsonValue::Object(obj) => obj,
        _ => {
            return Err(error!(
                "invalid package.json: not an object [{}]",
                pkg_json_path.to_string_lossy()
            ))
        }
    };

    let main_filename_orig = pkg_spec
        .get("module")
        .and_then(|val| val.as_str())
        .unwrap_or("index.js");
    let main_filename = pkg_path.join(main_filename_orig).canonicalize().unwrap();
    if !main_filename.starts_with(pkg_path.canonicalize().unwrap()) {
        return Err(error!(
                "invalid 'main' in package.json: path '{}' resolves to '{}', which is outside the package's root directory",
                 main_filename_orig, main_filename.to_string_lossy()
            ));
    }

    let root_path = pkg_path.canonicalize().unwrap();

    Ok(Package {
        root_path,
        main_filename,
    })
}
