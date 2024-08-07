use std::ffi::OsStr;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::{collections::HashMap, path::PathBuf};

use crate::bytecode_compiler;
use crate::common::{Context, Error, Result};
use crate::error;
use crate::{bytecode, error_item};

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
    ///
    /// Always absolute.
    base_path: PathBuf,

    next_anon_id: usize,

    max_fnid: u32,
    functions: HashMap<bytecode::FnID, bytecode::Function>,
    func_extra: HashMap<bytecode::FnID, FuncInfo>,

    // This is just a cache of previously parsed package.json.
    // The key is the package's root path (Package::root_path).
    packages: HashMap<PathBuf, Rc<Package>>,

    // Key is the absolute filename of the (previously loaded) module
    files: HashMap<PathBuf, Rc<FileInfo>>,

    // A single source map for literally all of the code that is ever
    // loaded through this Loader.
    source_map: Rc<swc_common::SourceMap>,

    boot_script_fnid: Option<bytecode::FnID>,
}

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

#[derive(Clone, Copy)]
pub enum FileIDRef<'a> {
    #[allow(dead_code)]
    Anon(usize),
    File(&'a Path),
}

impl FileID {
    // AsRef and Borrow are not quite right here
    fn as_fileidref(&self) -> FileIDRef {
        match self {
            FileID::Anon(ndx) => FileIDRef::Anon(*ndx),
            FileID::File(path) => FileIDRef::File(path),
        }
    }
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
    root_fnid: bytecode::FnID,

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

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct BreakRangeID(bytecode::FnID, usize);

impl std::fmt::Display for BreakRangeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let BreakRangeID(fnid, ndx) = self;
        write!(f, "br{},{}", fnid.0, ndx)
    }
}

impl BreakRangeID {
    pub fn parse_string(s: &str) -> Option<Self> {
        let s = s.strip_prefix("br")?;
        let (fnid_s, ndx_s) = s.split_once(',')?;
        let fnid = fnid_s.parse().ok()?;
        let ndx = ndx_s.parse().ok()?;
        Some(BreakRangeID(bytecode::FnID(fnid), ndx))
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
        let base_path = base_path.canonicalize().unwrap();
        let mut loader = Loader {
            base_path,
            next_anon_id: 0,
            max_fnid: 0,
            functions: HashMap::new(),
            func_extra: HashMap::new(),
            packages: HashMap::new(),
            files: HashMap::new(),
            source_map: Rc::new(swc_common::SourceMap::default()),
            boot_script_fnid: None,
        };

        const BOOTSTRAP_SCRIPT: &str = include_str!("bootstrap.js");
        let file_id = loader.gen_file_id();
        let fnid = loader
            .load_code(
                file_id,
                BOOTSTRAP_SCRIPT.into(),
                bytecode_compiler::SourceType::Script,
                bytecode_compiler::AllowDirectForms::Yes,
            )
            .unwrap();
        loader.boot_script_fnid = Some(fnid);

        loader
    }

    pub fn boot_script_fnid(&self) -> bytecode::FnID {
        self.boot_script_fnid.unwrap()
    }

    pub fn get_function(&self, fnid: bytecode::FnID) -> Option<&bytecode::Function> {
        self.functions.get(&fnid)
    }

    pub fn functions(&self) -> impl Iterator<Item = &bytecode::FnID> {
        self.functions.keys()
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
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
    ///  - `import_site`: the ID of the file where the import statement is located.
    pub fn load_import_from_fn(
        &mut self,
        import_path: &str,
        importing_fnid: bytecode::FnID,
    ) -> Result<bytecode::FnID> {
        let directory = &self
            .func_extra
            .get(&importing_fnid)
            .ok_or_else(|| error!("no such file for this ID"))?
            .file
            .directory
            .clone();

        self.load_import_from_dir(import_path, directory)
    }

    pub fn load_import_from_dir(
        &mut self,
        import_path: &str,
        directory: &Path,
    ) -> Result<bytecode::FnID> {
        if import_path.starts_with("./") {
            // relative paths (.e.g './asd/lol.js')
            assert!(directory.is_absolute());
            let mod_path = directory.join(import_path);
            self.load_module(&mod_path)
        } else if is_valid_package_name(import_path) {
            // Package name ("bare import specifier")
            let pkg_name = import_path;
            let package = self.resolve_package_import(directory, pkg_name)?;
            let mod_path = package.root_path.join(&package.main_filename);
            self.load_module(&mod_path)
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
                return Ok(Rc::clone(package));
            }

            if pkg_dir.join("package.json").is_file() {
                let package = parse_package(&pkg_dir).map_err(|err| {
                    err.with_context(error_item!("while loading imported package '{}'", pkg_name))
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

    fn load_module(&mut self, module_path: &Path) -> Result<bytecode::FnID> {
        assert!(module_path.is_absolute());

        if let Some(file_info) = self.files.get(module_path) {
            return Ok(file_info.root_fnid);
        }

        let content = std::fs::read_to_string(module_path).map_err(|err| {
            Error::from_err(err).with_context(error_item!(
                "while loading module `{}`",
                module_path.to_string_lossy()
            ))
        })?;

        self.load_code(
            FileID::File(module_path.to_owned()),
            content,
            bytecode_compiler::SourceType::Module,
            bytecode_compiler::AllowDirectForms::No,
        )
    }

    #[cfg(test)]
    pub(crate) fn load_code_forced(
        &mut self,
        file_id: FileID,
        content: String,
        source_type: bytecode_compiler::SourceType,
    ) -> Result<bytecode::FnID> {
        self.load_code(
            file_id,
            content,
            source_type,
            bytecode_compiler::AllowDirectForms::No,
        )
    }

    fn load_code(
        &mut self,
        file_id: FileID,
        content: String,
        source_type: bytecode_compiler::SourceType,
        allow_direct_forms: bytecode_compiler::AllowDirectForms,
    ) -> Result<bytecode::FnID> {
        let directory = match &file_id {
            FileID::Anon(_) => self.base_path.to_owned(),
            FileID::File(path) => path.parent().expect("invalid file ID!").to_owned(),
        };

        if let FileID::Anon(_) = &file_id {
            assert_eq!(source_type, bytecode_compiler::SourceType::Script);
        }

        let min_fnid = self.max_fnid + 1;
        let flags = bytecode_compiler::CompileFlags {
            min_fnid,
            source_type,
            allow_direct_forms,
        };

        // TODO Modify compile_file so that we don't have to copy the Rc
        let source_map = Rc::clone(&self.source_map);
        let bytecode_compiler::CompiledChunk {
            root_fnid,
            functions,
            breakable_ranges,
        } = bytecode_compiler::compile_file(file_id.as_fileidref(), content, source_map, flags)?;
        assert!(functions.keys().all(|lfnid| lfnid.0 >= min_fnid));
        if let Some(cur_max_fnid) = functions.keys().max() {
            assert!(self.max_fnid <= cur_max_fnid.0);
            self.max_fnid = cur_max_fnid.0;
        } else {
            // No change if no function was compiled
            debug_assert!(breakable_ranges.is_empty());
        }

        assert!(directory.is_absolute());
        let file = Rc::new(FileInfo {
            root_fnid,
            directory,
        });

        self.add_functions(functions.into_iter(), &file);

        for br in breakable_ranges {
            self.func_extra
                .get_mut(&br.fnid)
                .unwrap()
                .breakranges
                .push(br);
        }

        if let FileID::File(path) = file_id {
            self.files.insert(path, file);
        }

        Ok(root_fnid)
    }

    fn add_functions(
        &mut self,
        functions: impl Iterator<Item = (bytecode::FnID, bytecode::Function)>,
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

    /// Compile a new chunk of anonymous script code.
    ///
    /// "Anonymous" here means that the code is not linked to any file. This is the case,
    /// for example, for a piece of code coming from a REPL or similar tool.
    pub fn load_script_anon(&mut self, content: String) -> Result<bytecode::FnID> {
        let file_id = self.gen_file_id();
        self.load_code(
            file_id,
            content,
            bytecode_compiler::SourceType::Script,
            bytecode_compiler::AllowDirectForms::No,
        )
    }

    fn gen_file_id(&mut self) -> FileID {
        self.next_anon_id += 1;
        FileID::Anon(self.next_anon_id)
    }

    pub fn load_script_file(&mut self, filename: &Path) -> Result<bytecode::FnID> {
        let content = std::fs::read_to_string(filename).map_err(|err| {
            Error::from_err(err).with_context(error_item!(
                "while loading script `{}`",
                filename.to_string_lossy()
            ))
        })?;

        // Now that the file has been read successfully, we can reasonably
        // assume that "canonicalize" will also succeed.
        let filename = filename.canonicalize().unwrap();

        let source_type = if filename.extension() == Some(OsStr::new("mjs")) {
            bytecode_compiler::SourceType::Module
        } else {
            bytecode_compiler::SourceType::Script
        };

        self.load_code(
            FileID::File(filename),
            content,
            source_type,
            bytecode_compiler::AllowDirectForms::No,
        )
    }

    pub fn resolve_break_loc(
        &self,
        fnid: bytecode::FnID,
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
        fnid: bytecode::FnID,
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
    pub fn get_source_map(&self) -> &swc_common::SourceMap {
        Rc::as_ref(&self.source_map)
    }

    pub fn lookup_function(&self, fnid: bytecode::FnID) -> Option<FunctionLookup> {
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
        .map_err(Error::from_err)
        .with_context(error_item!(
            "while opening package.json for {}",
            pkg_path.to_string_lossy()
        ))?;

    let pkg_spec_raw = std::io::read_to_string(&pkg_json_file)
        .map_err(Error::from_err)
        .with_context(error_item!(
            "while reading package spec [{}]",
            pkg_json_path.to_string_lossy()
        ))?;

    let json_value = json::parse(&pkg_spec_raw)
        .map_err(|err| error!("JSON parse error: {}", err))
        .with_context(error_item!(
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
        .get("main")
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
