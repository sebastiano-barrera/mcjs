use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
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
    /// This is the "current directory" used the import resolution algoritm for
    /// import statements that appear in REPL fragments.
    base_path: PathBuf,

    next_anon_id: usize,

    max_fnid: u16,
    functions: HashMap<bytecode::FnId, bytecode::Function>,
    func_extra: HashMap<bytecode::FnId, FuncInfo>,

    // This is just a cache of previously parsed package.json.
    // The key is the package's root path (Package::root_path).
    packages: HashMap<PathBuf, Rc<Package>>,

    // Key is the absolute filename of the (previously loaded) module
    files: HashMap<PathBuf, Rc<FileInfo>>,
    repl_frags: HashMap<usize, Rc<FileInfo>>,

    // A single source map for literally all of the code that is ever
    // loaded through this Loader.
    source_map: Rc<swc_common::SourceMap>,

    boot_script_fnid: Option<bytecode::FnId>,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct PackageId(u16);

struct FuncInfo {
    file: Rc<FileInfo>,
    breakranges: Vec<bytecode::BreakRange>,
}

/// Unique identifer for a "chunk" of JavaScript code that we've loaded.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum FileID {
    /// REPL fragment
    Anon(usize),
    /// An actual file.  The path is always absolute.
    File(PathBuf),
}

/// Represents a "file" containing executable JavaScript code.
///
/// This is a unified representation for 3 types of files:
///  - module files
///  - script files
///  - REPL fragments (these are scripts, too)
///
/// All 3 can be 'run' by accessing the root function (via `root_fnid`).
struct FileInfo {
    root_fnid: bytecode::FnId,

    /// Used in the lookups for import statements. Always absolute.
    directory: PathBuf,
}

struct Package {
    /// Absolute path to the package's root directory.  Modules belonging to
    /// this package can be designated by a relative path from this directory.
    root_path: PathBuf,
    /// Path to the main file (the one imported with a naked import specifier).  Relative
    /// to `root_path`.
    main_filename: PathBuf,
}

struct Module {
    root_fnid: bytecode::LocalFnId,
    source_map: Rc<swc_common::SourceMap>,
    breakable_ranges: Vec<bytecode::BreakRange>,
    abs_path: PathBuf,
    // The package that the module belongs to could be cached from abs_path.
    // Worth the increase in complexity/trickiness?
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct BreakRangeID(bytecode::FnId, usize);

impl std::fmt::Display for BreakRangeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "br{},{}", self.0 .0 .0, self.1)
    }
}

impl BreakRangeID {
    pub fn parse_string(s: &str) -> Option<Self> {
        let s = s.strip_prefix("br")?;
        let (fnid_s, ndx_s) = s.split_once(',')?;
        let fnid = fnid_s.parse().ok()?;
        let ndx = ndx_s.parse().ok()?;
        Some(BreakRangeID(bytecode::FnId(bytecode::LocalFnId(fnid)), ndx))
    }
}

impl Loader {
    pub fn new_cwd() -> Self {
        Self::new(std::env::current_dir().unwrap())
    }

    /// Create a new empty loader.
    ///
    /// `base_path` is a directory (this function will panic otherwise). To a first
    /// approximation, it's the directory where the main file or the REPL is running.
    /// More concretely, it is considered as the initial search path for relative
    /// imports (e.g. when `import * as x from './asd/lol'` appears in a script/REPL
    /// chunk).
    pub fn new(base_path: PathBuf) -> Self {
        let mut loader = Loader {
            base_path,
            next_anon_id: 0,
            max_fnid: 0,
            functions: HashMap::new(),
            func_extra: HashMap::new(),
            packages: HashMap::new(),
            files: HashMap::new(),
            repl_frags: HashMap::new(),
            source_map: Rc::new(swc_common::SourceMap::default()),
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
                    args = args || [];
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

    pub fn get_function(&self, fnid: bytecode::FnId) -> Option<&bytecode::Function> {
        self.functions.get(&fnid)
    }

    pub fn functions(&self) -> impl Iterator<Item = &bytecode::FnId> {
        self.functions.keys()
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
    ///  - `import_site`: the ID of the function where the import statement is located.
    ///    Because import statements can only be present at the top-level of a module,
    ///    this *must* to be a module's root function. An error will be returned
    ///    otherwise.
    pub fn load_import(
        &mut self,
        import_path: &str,
        import_site: bytecode::FnId,
    ) -> Result<bytecode::FnId> {
        // The starting point is the module's parent package
        let importing_file = Rc::clone(
            &self
                .func_extra
                .get(&import_site)
                .expect("no such function")
                .file,
        );

        if import_path.starts_with("./") {
            // relative paths (.e.g './asd/lol.js')
            assert!(importing_file.directory.is_absolute());
            let mod_path = importing_file.directory.join(import_path);
            self.load_module(mod_path)
        } else if is_valid_package_name(import_path) {
            // Package name ("bare import specifier")
            let pkg_name = import_path;
            let package = self.resolve_package_import(&importing_file.directory, pkg_name)?;
            let mod_path = package.root_path.join(&package.main_filename);
            self.load_module(mod_path)
        } else {
            Err(error!(
                "invalid module path: `{}` (only allowed: './relative/paths', and 'package-name')",
                import_path
            ))
        }
    }

    /// Find the package with the given name.
    ///
    /// The search starts from the package identified by `start_pkg_id`, and proceeds
    /// by walking "upwards", from the package's root directory towards its
    /// parents. At each step, we look into `node_modules`.
    fn resolve_package_import(&mut self, start_path: &Path, pkg_name: &str) -> Result<Rc<Package>> {
        assert!(is_valid_package_name(pkg_name));
        // There are a bunch of corner cases around '.' and '..' that we can avoid
        // Packages can't legally start by '.' anyway.
        assert!(!pkg_name.contains('.'));

        let mut cur_dir = start_path;
        assert!(cur_dir.is_absolute());
        debug_assert_eq!(cur_dir.canonicalize().unwrap(), cur_dir);

        eprintln!("resolving package import {}", pkg_name);
        loop {
            let pkg_dir = cur_dir.join("node_modules").join(pkg_name);
            eprintln!(" - checking in {}", pkg_dir.to_string_lossy());

            if let Some(package) = self.packages.get(&pkg_dir) {
                // Cached; return immediately
                return Ok(Rc::clone(&package));
            }

            if pkg_dir.join("package.json").is_file() {
                let package = parse_package(&pkg_dir).map_err(|err| {
                    err.with_context(error!("while loading imported package '{}'", pkg_name))
                })?;
                let package = Rc::new(package);

                self.packages
                    .insert(package.root_path.clone(), Rc::clone(&package));
                return Ok(package);
            }

            cur_dir = match cur_dir.parent() {
                Some(parent) => parent,
                None => return Err(error!("no such package: {}", pkg_name)),
            };
        }
    }

    fn load_module(&mut self, module_path: PathBuf) -> Result<bytecode::FnId> {
        assert!(module_path.is_absolute());

        if let Some(file_info) = self.files.get(&module_path) {
            return Ok(file_info.root_fnid);
        }

        let content = std::fs::read_to_string(&module_path).map_err(|err| {
            Error::from(err).with_context(error!(
                "while loading module `{}`",
                module_path.to_string_lossy()
            ))
        })?;

        self.load_code(
            FileID::File(module_path),
            content,
            bytecode_compiler::SourceType::Module,
        )
    }

    fn load_code(
        &mut self,
        file_id: FileID,
        content: String,
        source_type: bytecode_compiler::SourceType,
    ) -> Result<bytecode::FnId> {
        let directory = match &file_id {
            FileID::Anon(_) => self.base_path.to_owned(),
            FileID::File(path) => path.parent().expect("invalid file ID!").to_owned(),
        };

        if let FileID::Anon(_) = &file_id {
            assert_eq!(source_type, bytecode_compiler::SourceType::Script);
        }

        let min_lfnid = self.max_fnid + 1;
        let flags = bytecode_compiler::CompileFlags {
            min_lfnid,
            source_type,
        };

        // TODO Modify compile_file so that we don't have to copy the Rc
        let source_map = Rc::clone(&self.source_map);
        let bytecode_compiler::CompiledChunk {
            root_fnid,
            functions,
            source_map: _,
            breakable_ranges,
        } = bytecode_compiler::compile_file(&file_id, content, source_map, flags)?;
        assert!(functions.keys().all(|lfnid| lfnid.0 >= min_lfnid));
        if let Some(cur_max_fnid) = functions.keys().max() {
            assert!(self.max_fnid <= cur_max_fnid.0);
            self.max_fnid = cur_max_fnid.0;
        } else {
            // No change if no function was compiled
            debug_assert!(breakable_ranges.is_empty());
        }

        assert!(directory.is_absolute());
        let root_fnid = bytecode::FnId(root_fnid);
        let file = Rc::new(FileInfo {
            root_fnid,
            directory,
        });

        let functions = functions
            .into_iter()
            .map(|(lfnid, func)| (bytecode::FnId(lfnid), func));
        self.add_functions(functions, &file);

        for br in breakable_ranges {
            self.func_extra
                .get_mut(&br.fnid)
                .unwrap()
                .breakranges
                .push(br);
        }

        match file_id {
            FileID::Anon(id) => self.repl_frags.insert(id, file),
            FileID::File(path) => self.files.insert(path, file),
        };

        Ok(root_fnid)
    }

    fn add_functions(
        &mut self,
        functions: impl Iterator<Item = (bytecode::FnId, bytecode::Function)>,
        file: &Rc<FileInfo>,
    ) {
        for (fnid, func) in functions {
            let prev = self.functions.insert(fnid, func);
            assert!(prev.is_none(), "loader bug: duplicate fnid");

            let func_extra = FuncInfo {
                breakranges: Vec::new(),
                file: Rc::clone(file),
            };
            let prev = self.func_extra.insert(fnid, func_extra);
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
        filename: Option<PathBuf>,
        content: String,
    ) -> Result<bytecode::FnId> {
        let file_id = match filename {
            Some(path) => FileID::File(path),
            None => {
                self.next_anon_id += 1;
                FileID::Anon(self.next_anon_id)
            }
        };
        self.load_code(file_id, content, bytecode_compiler::SourceType::Script)
    }

    pub fn resolve_break_loc(
        &self,
        fnid: bytecode::FnId,
        byte_pos: swc_common::BytePos,
    ) -> Result<Vec<(BreakRangeID, &bytecode::BreakRange)>> {
        // TODO Scanning the whole set every time resolve_loc is called may not be ideal. Interval
        // tree?
        Ok(self
            .func_extra
            .get(&fnid)
            .ok_or_else(|| error!("no such function"))?
            .breakranges
            .iter()
            .enumerate()
            .filter(|(_, brange)| brange.lo <= byte_pos && byte_pos <= brange.hi)
            .map(|(ndx, brange)| (BreakRangeID(fnid, ndx), brange))
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
        let brs = &self.func_extra.get(&fnid)?.breakranges;
        Some(
            brs.iter()
                .enumerate()
                .map(move |(ndx, br)| (BreakRangeID(fnid, ndx), br)),
        )
    }

    pub fn get_break_range(&self, brange_id: BreakRangeID) -> Option<&bytecode::BreakRange> {
        let BreakRangeID(fnid, ndx) = brange_id;
        let fnxtra = self.func_extra.get(&fnid).unwrap();
        fnxtra.breakranges.get(ndx)
    }

    // TODO Remove the _fnid parameter (we now have a single source map for everything)
    pub fn get_source_map(&self, _fnid: bytecode::FnId) -> Option<&Rc<swc_common::SourceMap>> {
        Some(&self.source_map)
    }

    pub fn lookup_function(&self, fnid: bytecode::FnId) -> Option<FunctionLookup> {
        let func = self.get_function(fnid)?;
        let span = *func.span();
        let swc_common::SourceFileAndBytePos {
            sf: source_file,
            pos: local_lo,
        } = self.source_map.lookup_byte_offset(span.lo);
        let local_hi = local_lo + (span.hi - span.lo);

        Some(FunctionLookup {
            source_map: Rc::clone(&self.source_map),
            source_file,
            span,
            local_range: local_lo..local_hi,
        })
    }

    // TODO(performance) Does it make sense to mmap the input file?
    // TODO Indirect filesystem access? (Check if needed)
}

pub struct FunctionLookup {
    pub source_map: Rc<swc_common::SourceMap>,
    pub source_file: Rc<swc_common::SourceFile>,
    pub span: swc_common::Span,
    /// Byte pos range for the function's text, local to source_file.src
    pub local_range: Range<swc_common::BytePos>,
}
impl FunctionLookup {
    pub fn text(&self) -> &str {
        let local_lo = self.local_range.start.0 as usize;
        let local_hi = self.local_range.end.0 as usize;
        &self.full_text()[local_lo..local_hi]
    }
    pub fn full_text(&self) -> &str {
        &self.source_file.src
    }
    pub fn local_range_usize(&self) -> Range<usize> {
        let start = self.local_range.start.0 as usize;
        let end = self.local_range.end.0 as usize;
        start..end
    }
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
