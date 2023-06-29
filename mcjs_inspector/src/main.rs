use std::path::PathBuf;

use mcjs_vm::{inspector_case, FnId, GlobalIID, InspectorAction, IID};

fn main() {
    let case_file_path = {
        let mut args = std::env::args().skip(1);
        PathBuf::from(args.next().expect("usage: mcjs_inspector <case file>"))
    };

    let mut f = std::fs::File::open(&case_file_path).expect("could not open case file");
    let case: inspector_case::Case =
        rmp_serde::from_read(&mut f).expect("could not decode case file!");
    eprintln!("Case file content = {:#?}", case);

    let mut builder = mcjs_vm::BuilderParams {
        loader: Box::new(mcjs_vm::FileLoader::new(case.include_paths)),
    }
    .to_builder();
    let test_mod_id = match case.root {
        inspector_case::Root::ModuleImport(path) => builder
            .compile_file(path)
            .unwrap_or_else(|err| panic!("compile error: {:?}", err)),
        inspector_case::Root::InlineScript(_) => {
            todo!("sorry, Root::InlineScript is not supported yet")
        }
    };
    let codebase = builder.build();

    struct Breakpoint(GlobalIID);

    let breakpoint = Breakpoint(GlobalIID(FnId(70), IID(6)));
    let mut code_path = Vec::new();
    let mut on_step = |step: &mcjs_vm::InspectorStep| {
        code_path.push(step.giid);
        if step.giid == breakpoint.0 {
            InspectorAction::Fail
        } else {
            InspectorAction::Continue
        }
    };

    let vm = mcjs_vm::Interpreter::new(&codebase).with_step_handler(&mut on_step);
    let result = vm.run_module(test_mod_id);

    match result {
        Ok(output) => {
            println!("interpreter finished OK.");
            println!("sink = {:?}", output.sink);
        }
        Err(error) => {
            eprintln!("code path:");
            for giid in code_path {
                eprintln!(" - {:?}", giid);
            }
            eprintln!("interpreter error: {:?}", error.error);

            if let Some(core_dump) = error.core_dump {
                let header = core_dump.data.header();
                eprintln!("top frame header: {:?}", header);

                eprintln!("results ({}):", header.n_instrs);
                for i in 0u32..header.n_instrs {
                    let vreg = mcjs_vm::bytecode::VReg(i.try_into().unwrap());
                    eprintln!("  [{}] = {:?}", i, core_dump.data.get_result(vreg));
                }

                eprintln!("args ({}):", header.n_args);
                for i in 0..header.n_args {
                    let argndx = mcjs_vm::bytecode::ArgIndex(i);
                    eprintln!("  [{}] = {:?}", i, core_dump.data.get_arg(argndx));
                }

                eprintln!("captures ({}):", header.n_captures);
                for i in 0..header.n_captures {
                    let capndx = mcjs_vm::bytecode::CaptureIndex(i);
                    eprintln!("  [{}] = {:?}", i, core_dump.data.get_capture(capndx));
                }

            } else {
                eprintln!("-- no interpreter core dumped");
            }
        }
    }
}
