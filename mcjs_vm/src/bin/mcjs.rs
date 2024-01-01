use std::path::PathBuf;

use mcjs_vm::{Interpreter, Loader, Realm};

fn main() {
    let main_path = match std::env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("Usage: mcjs <filename>   (loader base path is cwd)");
            return;
        }
    };
    let main_path = PathBuf::from(main_path).canonicalize().unwrap();

    let base_path = main_path.parent().unwrap();
    let filename = main_path
        .file_name()
        .unwrap()
        .to_str()
        .expect("can't convert main filename to UTF-8");
    let import_path = format!("./{}", filename);

    let mut loader = Loader::new(Some(base_path.to_owned()));
    let mut realm = Realm::new(&mut loader);

    let main_fnid = loader
        .load_import(&import_path, mcjs_vm::SCRIPT_MODULE_ID)
        .expect("compile error");

    use mcjs_vm::interpreter::Exit;

    println!();
    println!("running...");

    let mut intrp = Interpreter::new(&mut realm, &mut loader, main_fnid);

    loop {
        match intrp.run().expect("runtime error") {
            Exit::Finished(_) => break,
            Exit::Suspended(next_intrp) => {
                println!("(suspended; resuming immediately)");
                intrp = next_intrp;
            }
        }
    }

    println!("interpreter finished");
}
