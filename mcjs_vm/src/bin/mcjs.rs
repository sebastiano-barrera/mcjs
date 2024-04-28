use std::{
    io::{BufRead, Write},
    path::PathBuf,
};

use mcjs_vm::{FnId, Interpreter, Loader, Realm};

fn main() {
    let mut stdout = std::io::stdout().lock();
    let config = parse_args();

    let mut loader = Loader::new_cwd();
    let mut realm = Realm::new(&mut loader);

    for path in config.paths {
        let path = PathBuf::from(&path).canonicalize().unwrap();
        let main_fnid = loader.load_script_file(&path).expect("compile error");

        if config.dump_bytecode {
            for fnid in loader.functions() {
                println!(
                    "fn {:?} {}",
                    fnid,
                    if fnid == &main_fnid { "[root]" } else { "" }
                );
                loader
                    .get_function(*fnid)
                    .unwrap()
                    .dump(&mut stdout)
                    .unwrap();
            }
        }

        println!();
        println!("running...");

        run_to_completion(&mut realm, &mut loader, main_fnid);
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

            let res = loader.load_script_anon(line.clone());

            match res {
                Ok(fnid) => {
                    run_to_completion(&mut realm, &mut loader, fnid);
                }
                Err(err) => {
                    println!(" (!!) compile error: {}", err);
                }
            }
        }
    }
}

fn run_to_completion(realm: &mut Realm, loader: &mut Loader, fnid: FnId) {
    let intrp = Interpreter::new(realm, loader, fnid);
    intrp.run().expect("runtime error").expect_finished();
}

struct Config {
    paths: Vec<String>,
    start_shell: bool,
    dump_bytecode: bool,
}

fn parse_args() -> Config {
    let mut paths = Vec::new();
    let mut start_shell = false;
    let mut dump_bytecode = false;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--shell" | "-s" => {
                start_shell = true;
            }
            "--dump-bytecode" => {
                dump_bytecode = true;
            }
            _ => {
                paths.push(arg);
            }
        }
    }

    if paths.is_empty() {
        start_shell = true;
    }

    Config {
        paths,
        start_shell,
        dump_bytecode,
    }
}
