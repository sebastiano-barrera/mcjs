use slint_build::CompilerConfiguration;


fn main() {
    slint_build::compile_with_config(
        "ui/hello.slint", 
        CompilerConfiguration::new().with_style("fluent".to_owned()))
    .unwrap();
}

