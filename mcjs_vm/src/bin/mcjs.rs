use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

use mcjs_vm::{Interpreter, Loader, Realm};

fn main() {
    let config = parse_args();

    let cwd = std::env::current_dir().unwrap();
    let mut loader = Loader::new(Some(cwd));
    let mut realm = Realm::new(&mut loader);

    for path in config.paths {
        let path = PathBuf::from(&path).canonicalize().unwrap();
        let content = std::fs::read_to_string(&path).expect("could not read file");

        let filename = path
            .file_name()
            .unwrap()
            .to_str()
            .expect("can't convert main filename to UTF-8");
        let main_fnid = loader
            .load_script(Some(filename.to_owned()), content)
            .expect("compile error");

        println!();
        println!("running...");

        let intrp = Interpreter::new(&mut realm, &mut loader, main_fnid);
        run_to_completion(intrp);
    }

    if config.start_shell {
        println!("Welcome to mcjs.  It's got no version number yet.");
        println!("Type some JavaScript, see what you can get away with.");

        let mut stdin = std::io::stdin().lock();
        let mut stdout = std::io::stdout().lock();
        let mut line = String::new();
        loop {
            print!("> ");
            stdout.flush().unwrap();

            line.clear();
            let n_read = stdin.read_line(&mut line).unwrap();
            if n_read == 0 {
                // we reached EOF
                break;
            }

            let res = loader.load_script(None, line.clone());

            match res {
                Ok(fnid) => {
                    let intrp = Interpreter::new(&mut realm, &mut loader, fnid);
                    run_to_completion(intrp);
                }
                Err(err) => {
                    println!(" (!!) compile error: {}", err.message());
                }
            }
        }
    }
}

fn run_to_completion(mut intrp: Interpreter<'_>) {
    use mcjs_vm::interpreter::Exit;

    loop {
        match intrp.run().expect("runtime error") {
            Exit::Finished(_) => break,
            Exit::Suspended(next_intrp) => {
                intrp = next_intrp;
            }
        }
    }
}

struct Config {
    paths: Vec<String>,
    start_shell: bool,
}

fn parse_args() -> Config {
    let mut paths = Vec::new();
    let mut start_shell = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--shell" || arg == "-s" {
            start_shell = true;
        } else {
            paths.push(arg);
        }
    }

    if paths.is_empty() {
        start_shell = true;
    }

    Config { paths, start_shell }
}
