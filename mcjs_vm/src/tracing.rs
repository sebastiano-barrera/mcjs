#![cfg_attr(not(test), allow(unused_variables))]

pub fn section(header: &str) -> Section {
    internal::section(header)
}
pub type Section = internal::Section;

#[cfg(not(test))]
mod internal {
    use std::fmt::Debug;

    pub fn section(_header: &str) -> Section {
        Section
    }
    pub struct Section;

    impl Section {
        pub fn log_value<T: Debug>(&self, _tag: &str, _value: &T) {}
        pub fn log(&self, _tag: &str, _value: &str) {}
    }
}

#[cfg(test)]
mod internal {
    use lazy_static::lazy_static;

    use std::borrow::BorrowMut;
    use std::cell::RefCell;
    use std::fmt::Debug;
    use std::fs::File;
    use std::io::{Result, Write};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::thread_local;

    #[must_use]
    pub fn section(header: &str) -> Section {
        #[cfg(test)]
        THREAD_LOGGER.with(|logger| {
            let mut logger = logger.borrow_mut();
            logger.start_section(header);
        });

        Section
    }

    pub struct Section;
    impl Section {
        pub fn log_value<T: Debug>(&self, tag: &str, value: &T) {
            #[cfg(test)]
            THREAD_LOGGER.with(|logger| {
                let mut logger = logger.borrow_mut();
                logger.log_value(tag, value);
            });
        }

        pub fn log(&self, tag: &str, value: &str) {
            #[cfg(test)]
            THREAD_LOGGER.with(|logger| {
                let mut logger = logger.borrow_mut();
                logger.log(tag, value);
            })
        }
    }

    impl Drop for Section {
        fn drop(&mut self) {
            THREAD_LOGGER.with(|logger| {
                let mut logger = logger.borrow_mut();
                logger.end_section();
            })
        }
    }

    struct SinkConfig {
        sink: SinkType,
    }
    impl Default for SinkConfig {
        fn default() -> Self {
            SinkConfig {
                sink: SinkType::Stdout,
            }
        }
    }
    impl SinkConfig {
        fn from_env() -> Self {
            let sink = match std::env::var("MCJS_TRACING_SINK") {
                Ok(filename) => SinkType::File(PathBuf::from(filename)),
                _ => SinkType::Stdout,
            };
            SinkConfig { sink }
        }
    }

    enum SinkType {
        File(PathBuf),
        Stdout,
        Buffer,
    }

    lazy_static! {
        static ref GLOBAL_SINK: Arc<Mutex<Sink>> =
            Arc::new(Mutex::new(Sink::from_config(SinkConfig::from_env())));
    }

    fn get_sink() -> MutexGuard<'static, Sink> {
        GLOBAL_SINK.lock().expect("could not lock global logger!")
    }
    fn set_sink(new_sink: Sink) {
        let mut sink = get_sink();
        *sink = new_sink;
    }

    enum Sink {
        File(File),
        Stdout,
        Buffer(String),
    }
    impl Sink {
        fn from_config(config: SinkConfig) -> Self {
            match config.sink {
                SinkType::File(filename) => {
                    let file = File::create(filename).expect("could not open logger sink file");
                    Sink::File(file)
                }
                SinkType::Stdout => Sink::Stdout,
                SinkType::Buffer => Sink::Buffer(String::new()),
            }
        }

        fn write(&mut self, data: &str) {
            match self {
                Sink::File(f) => {
                    f.write(data.as_bytes()).unwrap();
                }
                Sink::Stdout => {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write(data.as_bytes()).unwrap();
                }
                Sink::Buffer(buf) => {
                    buf.push_str(data);
                }
            }
        }

        #[cfg(test)]
        fn take_buffer(&mut self) -> String {
            let buf = match self {
                Sink::Buffer(buf) => buf,
                _ => panic!("take_buffer, but sink is not Sink::Buffer(_)"),
            };

            std::mem::replace(buf, String::new())
        }
    }

    thread_local! {
        static THREAD_LOGGER: RefCell<Logger> = RefCell::new(Logger::new());
    }

    struct Logger {
        buf: String,
        sections_depth: usize,
    }

    impl Default for Logger {
        fn default() -> Self {
            Logger {
                buf: String::new(),
                sections_depth: 0,
            }
        }
    }

    impl Logger {
        fn new() -> Self {
            Logger::default()
        }

        fn start_section(&mut self, header: &str) {
            self.write(&format!("<{}>", header));
            self.sections_depth += 1;
        }

        fn end_section(&mut self) {
            assert!(self.sections_depth > 0);
            self.sections_depth -= 1;
            self.write("</>");

            if self.sections_depth == 0 {
                let mut sink = get_sink();
                sink.write(&self.buf);
                self.buf.clear();
            }
        }

        fn write(&mut self, text: &str) {
            use std::fmt::Write;

            for line in text.split('\n') {
                for _ in 0..self.sections_depth {
                    write!(self.buf, "    ").unwrap();
                }
                writeln!(self.buf, "{}", line).unwrap();
            }
        }

        fn log_value<T: Debug>(&mut self, tag: &str, value: &T) {
            self.write(&format!("{}: {:#?}\n", tag, value));
        }
        fn log(&mut self, tag: &str, value: &str) {
            self.write(&format!("{}: {}\n", tag, value));
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_section() {
            set_sink(Sink::from_config(SinkConfig {
                sink: SinkType::Buffer,
            }));

            {
                let _s = section("section name");
                _s.log_value("the_value", &(1, 2, 3));
            }

            let output = get_sink().take_buffer();
            insta::assert_snapshot!(output);
        }
    }
}
