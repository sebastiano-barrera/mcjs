use std::cell::OnceCell;
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
    base_path: Option<PathBuf>,

    next_module_id: u16,
    next_package_id: u16,

    // Every loader always contains at most a single script.
    // Other than its address (See Loader::load_script).
    script: Script,

    functions: HashMap<bytecode::FnId, bytecode::Function>,
    modules: HashMap<bytecode::ModuleId, Module>,
    // This is just a cache of previously parsed package.json.  Defining the PackageId is also
    // useful to identify packages without further checks.
    packages: HashMap<PackageId, Package>,

    // Key is the absolute filename of the (previously loaded) module
    mod_of_path: HashMap<PathBuf, bytecode::ModuleId>,
    pkg_of_path: HashMap<PathBuf, PackageId>,

    boot_script_fnid: Option<bytecode::FnId>,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct PackageId(u16);

struct Module {
    root_fnid: bytecode::LocalFnId,
    source_map: Rc<swc_common::SourceMap>,
    breakable_ranges: Vec<bytecode::BreakRange>,
    abs_path: PathBuf,
    // The package that the module belongs to could be cached from abs_path.
    // Worth the increase in complexity/trickiness?
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct BreakRangeID(bytecode::ModuleId, usize);

impl std::fmt::Display for BreakRangeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{},{}", self.0 .0, self.1)
    }
}

impl BreakRangeID {
    pub fn module_id(&self) -> bytecode::ModuleId {
        self.0
    }

    pub fn parse_string(s: &str) -> Option<Self> {
        let (mod_id_s, ndx_s) = s.split_once(',')?;
        let mod_id = mod_id_s.parse().ok()?;
        let ndx = ndx_s.parse().ok()?;
        Some(BreakRangeID(bytecode::ModuleId(mod_id), ndx))
    }
}

struct Package {
    /// Absolute path of the package's root directory.  Modules belonging to
    /// this package can be designated by a relative path from this directory.
    root_path: PathBuf,
    /// Relative path to the main module. (Relative to `root_path`)
    main_filename: PathBuf,
}

struct Script {
    max_fnid: u16,
    // In order to share a single set of breakable ranges among all script chunks, we need to
    // have a single source map (and a single module ID, but we have bytecode::SCRIPT_MODULE_ID
    // already).
    source_map: Rc<swc_common::SourceMap>,
    breakable_ranges: Vec<bytecode::BreakRange>,
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
    /// Create a new empty loader.
    ///
    /// `base_path` is a directory (this function will panic otherwise). To a first
    /// approximation, it's the directory where the main file or the REPL is running.
    /// More concretely, it is considered as the initial search path for relative
    /// imports (e.g. when `import * as x from './asd/lol'` appears in a script/REPL
    /// chunk).
    pub fn new(base_path: Option<PathBuf>) -> Self {
        let mut loader = Loader {
            base_path,
            next_module_id: 1,
            next_package_id: 1,
            script: Script {
                max_fnid: 0,
                source_map: Rc::new(swc_common::SourceMap::default()),
                breakable_ranges: Vec::new(),
            },
            functions: HashMap::new(),
            modules: HashMap::new(),
            packages: HashMap::new(),
            mod_of_path: HashMap::new(),
            pkg_of_path: HashMap::new(),
            boot_script_fnid: None,
        };

        let fnid = loader
            .load_script(
                None,
                r#"
                // TODO TODO TODO This needs to be updated to support more than 8 args
                Function.prototype.call = function (new_this, arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
                    // `this` is the function to call
                    const bound = this.bind(new_this);
                    return bound(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
                }

                Function.prototype.apply = function (new_this, args) {
                    // TODO change once spread syntax (e.g. `f(...args)`) is implemented
                    // `this` is the function to call
                    return this.bind(new_this)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
                }
                "#
                .into(),
            )
            .unwrap();
        loader.boot_script_fnid = Some(fnid);

        loader
    }

    pub fn boot_script_fnid(&self) -> bytecode::FnId {
        self.boot_script_fnid.unwrap()
    }

    pub fn get_module_root_fn(&self, module_id: bytecode::ModuleId) -> Option<bytecode::LocalFnId> {
        assert_ne!(
            module_id,
            bytecode::SCRIPT_MODULE_ID,
            "module_id == SCRIPT_MODULE_ID, but only modules have a root function"
        );

        Some(self.modules.get(&module_id)?.root_fnid)
    }

    pub fn get_function(&self, fnid: bytecode::FnId) -> Option<&bytecode::Function> {
        self.functions.get(&fnid)
    }

    pub fn get_abs_path(&self, module_id: bytecode::ModuleId) -> Option<&Path> {
        match module_id {
            bytecode::SCRIPT_MODULE_ID => None,
            module_id => Some(self.modules.get(&module_id)?.abs_path.as_path()),
        }
    }

    fn get_module(&self, mod_id: bytecode::ModuleId) -> Result<&Module> {
        assert_ne!(mod_id, bytecode::SCRIPT_MODULE_ID);
        self.modules
            .get(&mod_id)
            .ok_or_else(|| error!("no such module with ID {:?}", mod_id))
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
        import_path: &str,
        import_site: bytecode::ModuleId,
    ) -> Result<bytecode::FnId> {
        // The starting point is the module's parent package
        let base_path = match import_site {
            bytecode::SCRIPT_MODULE_ID =>
                self.base_path.as_deref()
                    .ok_or_else(|| error!("imports not allowed in script context, because no base path configured for this Loader"))?,

            import_site => self.get_module(import_site)?
                .abs_path
                .parent()
                .expect("loader bug: module path has no parent"),
        };
        let base_path = base_path.to_owned();

        let resolved_module_path = if import_path.starts_with("./") {
            let import_path = Path::new(&import_path);
            debug_assert!(import_path.is_relative());
            base_path.join(import_path)
        } else if is_valid_package_name(import_path) {
            // Package name ("bare import specifier")
            let pkg_id = self.resolve_package_import(&base_path, import_path)?;
            let pkg = self.packages.get(&pkg_id).unwrap();
            let mod_path = pkg.root_path.join(&pkg.main_filename);

            // Otherwise the file's path would resolve to outside the package...
            // Can't allow that
            debug_assert!(mod_path.canonicalize().unwrap().starts_with(&pkg.root_path));

            mod_path
        } else {
            return Err(error!(
                "invalid module path: `{}` (only allowed: './relative/paths', and 'package-name')",
                import_path
            ));
        };

        self.load_module(&resolved_module_path)
    }

    fn resolve_package_import(
        &mut self,
        importing_mod_dir: &Path,
        pkg_name: &str,
    ) -> Result<PackageId> {
        assert!(is_valid_package_name(pkg_name));
        // There are a bunch of corner cases around '.' and '..' that we can avoid
        assert!(!pkg_name.contains('.'));

        let mut cur_dir = importing_mod_dir;
        assert!(cur_dir.is_absolute());
        debug_assert_eq!(cur_dir.canonicalize().unwrap(), cur_dir);

        eprintln!("resolving package import {}", pkg_name);
        let pkg_id = 'cycle: loop {
            let pkg_dir = cur_dir.join("node_modules").join(pkg_name);
            eprintln!(" - checking in {}", pkg_dir.to_string_lossy());
            if let Some(pkg_id) = self.pkg_of_path.get(&pkg_dir) {
                break 'cycle *pkg_id;
            } else if pkg_dir.join("package.json").is_file() {
                let package = parse_package(&pkg_dir).map_err(|err| {
                    err.with_context(error!("while loading imported package '{}'", pkg_name))
                })?;

                let pkg_id = self.gen_pkg_id();
                self.packages.insert(pkg_id, package);
                self.pkg_of_path.insert(pkg_dir, pkg_id);
                break 'cycle pkg_id;
            }

            cur_dir = match cur_dir.parent() {
                Some(parent) => parent,
                None => return Err(error!("no such package: {}", pkg_name)),
            };
        };

        Ok(pkg_id)
    }

    fn load_module(&mut self, module_path: &Path) -> Result<bytecode::FnId> {
        assert!(module_path.is_absolute());

        if let Some(mod_id) = self.mod_of_path.get(module_path) {
            // TODO Make the root FnId fixed (the same) across all modules (then
            // we can avoid this lookup)
            let module = self
                .modules
                .get(mod_id)
                .expect("loader bug: inconsistent `module` and `module_id`");
            return Ok(bytecode::FnId(*mod_id, module.root_fnid));
        }

        let content = std::fs::read_to_string(module_path).map_err(|err| {
            Error::from(err).with_context(error!(
                "while loading module `{}`",
                module_path.to_string_lossy()
            ))
        })?;

        let source_map = Rc::new(Default::default());
        let flags = bytecode_compiler::CompileFlags {
            min_fnid: 1,
            source_type: bytecode_compiler::SourceType::Module,
        };
        let bytecode_compiler::CompiledChunk {
            root_fnid,
            functions,
            source_map,
            breakable_ranges,
        } = bytecode_compiler::compile_file(
            module_path.to_string_lossy().into_owned(),
            content,
            source_map,
            flags,
        )?;

        let module = Module {
            root_fnid,
            source_map,
            breakable_ranges,
            abs_path: module_path.to_owned(),
        };
        let module_id = self.gen_mod_id();

        self.add_functions(functions.into_iter(), module_id);

        self.modules.insert(module_id, module);
        self.mod_of_path.insert(module_path.to_owned(), module_id);
        Ok(bytecode::FnId(module_id, root_fnid))
    }

    fn add_functions(
        &mut self,
        functions: impl Iterator<Item = (bytecode::LocalFnId, bytecode::Function)>,
        module_id: crate::ModuleId,
    ) {
        for (lfnid, func) in functions {
            let prev = self
                .functions
                .insert(bytecode::FnId(module_id, lfnid), func);
            assert!(prev.is_none(), "loader bug: duplicate fnid");
        }
    }

    /// Compile a new chunk of script code.
    ///
    /// Since we're not in a browser, the only script code that should
    /// realistically pass through here is a  REPL chunk.
    ///
    /// The main characteristic of script code (at compile time) is that functions and
    /// variables defined in script context always have module ID ==
    /// `bytecode::SCRIPT_MODULE_ID`.  In other words, they appear as if coming from a
    /// single file, even though they go through the compiler in chunks.
    pub fn load_script(
        &mut self,
        filename: Option<String>,
        content: String,
    ) -> Result<bytecode::FnId> {
        let filename = filename.unwrap_or_else(|| "<input>".to_string());
        let min_fnid = self.script.max_fnid + 1;

        let source_map = Rc::clone(&self.script.source_map);
        let flags = bytecode_compiler::CompileFlags {
            min_fnid,
            source_type: bytecode_compiler::SourceType::Script,
        };
        let compiled = bytecode_compiler::compile_file(filename, content, source_map, flags)?;

        assert!(compiled.functions.keys().all(|lfnid| lfnid.0 >= min_fnid));
        if let Some(cur_max_fnid) = compiled.functions.keys().max() {
            self.script.max_fnid = cur_max_fnid.0;
        } else {
            // No change if no function was compiled
            debug_assert!(compiled.breakable_ranges.is_empty());
        }

        // Note that we discard everything other than the functions (e.g.
        // source_map, breakable_ranges)
        self.add_functions(compiled.functions.into_iter(), bytecode::SCRIPT_MODULE_ID);
        self.script
            .breakable_ranges
            .extend(compiled.breakable_ranges);

        Ok(bytecode::FnId(
            bytecode::SCRIPT_MODULE_ID,
            compiled.root_fnid,
        ))
    }

    pub fn resolve_break_loc(
        &self,
        module_id: bytecode::ModuleId,
        byte_pos: swc_common::BytePos,
    ) -> Result<Vec<(BreakRangeID, &bytecode::BreakRange)>> {
        // TODO Scanning the whole set every time resolve_loc is called may not be ideal. Interval
        // tree?
        Ok(self
            .module_breakranges(module_id)?
            .iter()
            .enumerate()
            .map(|(ndx, brange)| (BreakRangeID(module_id, ndx), brange))
            .filter(|(_, brange)| brange.lo <= byte_pos && byte_pos <= brange.hi)
            .collect())
    }

    pub fn breakrange_at_giid(
        &self,
        giid: bytecode::GlobalIID,
    ) -> Option<(BreakRangeID, &bytecode::BreakRange)> {
        let bytecode::GlobalIID(fnid, iid) = giid;

        // TODO We expect there to be at most 1 range matching the filter.  Any better mechanism?
        self.function_breakranges(fnid)?
            .find(|(_, brange)| brange.iid_start <= iid && iid < brange.iid_end)
    }

    pub fn function_breakranges(
        &self,
        fnid: bytecode::FnId,
    ) -> Option<impl Iterator<Item = (BreakRangeID, &bytecode::BreakRange)>> {
        let bytecode::FnId(mod_id, lfnid) = fnid;

        let branges = self
            .module_breakranges(mod_id)
            .ok()?
            .iter()
            .enumerate()
            .filter(move |(_, brange)| brange.local_fnid == lfnid)
            .map(move |(ndx, brange)| (BreakRangeID(mod_id, ndx), brange));
        Some(branges)
    }

    fn module_breakranges(&self, mod_id: crate::ModuleId) -> Result<&Vec<bytecode::BreakRange>> {
        if mod_id == bytecode::SCRIPT_MODULE_ID {
            Ok(&self.script.breakable_ranges)
        } else {
            let module = self.get_module(mod_id)?;
            // TODO We scan the whole vector every time.  This can be solved by segregating
            // breakable_ranges by function, or with an index data structure.  But for now, I can't be
            // bothered
            Ok(&module.breakable_ranges)
        }
    }

    pub fn get_break_range(&self, brange_id: BreakRangeID) -> Option<&bytecode::BreakRange> {
        let BreakRangeID(mod_id, ndx) = brange_id;
        if mod_id == bytecode::SCRIPT_MODULE_ID {
            self.script.breakable_ranges.get(ndx)
        } else {
            let module = self.get_module(mod_id).ok()?;
            module.breakable_ranges.get(ndx)
        }
    }

    /// Resolve the given path to a module ID.
    ///
    /// Note that the `filename` argument may be abbreviated to only a suffix of the
    /// desired file's path.  As long as there is only one loaded file whose filename
    /// matches that suffix, it will be automatically resolved to its full path. If
    /// more than one file matches, an error will be returned.
    pub fn find_module(&self, filename: &str) -> Result<&bytecode::ModuleId> {
        let all_paths = self.mod_of_path.keys().map(|pb| pb.as_path());
        let filename = resolve_path_by_suffix(all_paths, filename)?;
        let module_id = self.mod_of_path.get(filename).unwrap();
        Ok(module_id)
    }

    fn gen_mod_id(&mut self) -> bytecode::ModuleId {
        let mod_id = self.next_module_id;
        self.next_module_id += 1;
        bytecode::ModuleId::from(mod_id)
    }

    fn gen_pkg_id(&mut self) -> PackageId {
        let pkg_id = self.next_package_id;
        self.next_package_id += 1;
        PackageId(pkg_id)
    }

    pub fn get_source_map(&self, module_id: bytecode::ModuleId) -> Option<&swc_common::SourceMap> {
        let source_map_rc = if module_id == bytecode::SCRIPT_MODULE_ID {
            &self.script.source_map
        } else {
            &self.modules.get(&module_id)?.source_map
        };
        Some(Rc::as_ref(source_map_rc))
    }

    // TODO(performance) Does it make sense to mmap the input file?
    // TODO Indirect filesystem access? (Check if needed)
}

fn resolve_path_by_suffix<'a>(
    all_paths: impl Iterator<Item = &'a Path>,
    filename: &str,
) -> Result<&'a Path> {
    let mut matching_paths = all_paths.filter(|path| path.ends_with(filename));
    let resolved = matching_paths.next().ok_or(error!(
        "no module with path `{}` (either full or suffix)",
        filename
    ))?;

    // TODO Could it be useful to return all the matching paths? To help the user
    // narrow it down?
    if matching_paths.next().is_some() {
        return Err(error!(
            "ambiguous filename: multiple loaded files match suffix `{}`",
            filename
        ));
    }

    Ok(resolved)
}

fn is_valid_package_name(import_path: &str) -> bool {
    import_path
        .chars()
        .all(|ch| ch.is_alphabetic() || ch.is_ascii_digit() || ch == '-' || ch == '_')
}

// Private: packages are only loaded as part of the  'import' implementation (load_import)
//
// `pkg_path` must be the Path to the package's *directory*.
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
