#![cfg_attr(not(test), allow(unused_variables))]

pub fn section(header: &str) -> Section {
    internal::section(header)
}
pub type Section = internal::Section;

#[cfg(all(not(test), not(feature = "tracing")))]
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

#[cfg(any(test, feature = "tracing"))]
mod internal {
    use lazy_static::lazy_static;

    use std::cell::RefCell;
    use std::fmt::Debug;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::thread_local;

    //
    // Public API
    // (does not require a handle, simply forwards into *the* thread-local handle)
    //

    #[must_use]
    pub fn section(header: &str) -> Section {
        THREAD_LOGGER.with(|logger| {
            let mut logger = logger.borrow_mut();
            logger.start_section(header);
        });

        Section
    }

    pub struct Section;
    impl Section {
        pub fn log_value<T: Debug>(&self, tag: &str, value: &T) {
            THREAD_LOGGER.with(|logger| {
                let mut logger = logger.borrow_mut();
                logger.log_value(tag, value);
            });
        }

        pub fn log(&self, tag: &str, value: &str) {
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
        sink_filename: PathBuf,
    }
    impl Default for SinkConfig {
        fn default() -> Self {
            SinkConfig {
                sink_filename: PathBuf::from("/dev/null"),
            }
        }
    }
    impl SinkConfig {
        fn from_env() -> Self {
            let sink_filename: PathBuf = std::env::var("MCJS_TRACING_SINK")
                .unwrap_or_else(|_| "/dev/null".to_string())
                .into();
            SinkConfig { sink_filename }
        }
    }

    //
    // Thread-local part
    //

    thread_local! {
        static THREAD_LOGGER: RefCell<Logger> = RefCell::new(Logger::new());
    }

    struct Logger {
        thread_prefix: String,
        buf: String,
        sections_depth: usize,
    }

    impl Default for Logger {
        fn default() -> Self {
            let thread_id = format!("{:?}: ", std::thread::current().id());
            Logger {
                thread_prefix: thread_id,
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
            self.writeln(&format!("<{}>", header));
            self.sections_depth += 1;
        }

        fn log_value<T: Debug>(&mut self, tag: &str, value: &T) {
            self.writeln(&format!("{}: {:#?}", tag, value));
        }
        fn log(&mut self, tag: &str, value: &str) {
            self.writeln(&format!("{}: {}", tag, value));
        }

        fn end_section(&mut self) {
            assert!(self.sections_depth > 0);
            self.sections_depth -= 1;
            self.writeln("</>");
        }

        fn writeln(&mut self, text: &str) {
            use std::fmt::Write;

            self.buf.clear();
            for line in text.split('\n') {
                self.buf.push_str(&self.thread_prefix);
                for _ in 0..self.sections_depth {
                    write!(self.buf, "    ").unwrap();
                }

                write!(self.buf, "{}\n", line).unwrap();
            }

            // flush global sink file. inefficient, but tracing is often employed to debug VM
            // crashes. we might be crashing right after returning, so let's work to make it more
            // likely that the data is not lost.
            let mut sink = get_global_sink();
            sink.write(&self.buf);
        }
    }

    //
    // Global part (process-wide, thread-nonlocal)
    //
    lazy_static! {
        static ref GLOBAL_SINK: Arc<Mutex<Sink>> =
            Arc::new(Mutex::new(Sink::from_config(SinkConfig::from_env())));
    }

    fn get_global_sink() -> MutexGuard<'static, Sink> {
        GLOBAL_SINK.lock().expect("could not lock global logger!")
    }

    #[cfg(test)]
    fn set_global_sink_memory() {
        let mut gsink = GLOBAL_SINK.lock().expect("could not lock global logger!");
        *gsink = Sink::Memory(String::new());
    }

    #[cfg(test)]
    fn read_memory_buffer_this_thread() -> String {
        let mut gsink = GLOBAL_SINK.lock().expect("could not lock global logger!");
        let buf = match &mut *gsink {
            Sink::Memory(buf) => buf,
            _ => panic!("global sink not initialized as memory buffer"),
        };

        let prefix = format!("{:?}: ", std::thread::current().id());
        let mut tl_buf = String::new();
        for line in buf.lines() {
            if let Some(tl_line) = line.strip_prefix(&prefix) {
                tl_buf.push_str(tl_line);
                tl_buf.push('\n');
            }
        }

        tl_buf
    }

    enum Sink {
        File(File),
        Memory(String),
    }
    impl Sink {
        fn from_config(config: SinkConfig) -> Self {
            let file = File::create(config.sink_filename).expect("could not open logger sink file");
            Sink::File(file)
        }

        fn write(&mut self, data: &str) {
            match self {
                Sink::File(f) => {
                    f.write(data.as_bytes()).unwrap();

                    // inefficient, but tracing is often employed when the program is crashing. we might be
                    // crashing right after returning from this call, so let's flush immediately.
                    f.flush().unwrap();
                }
                Sink::Memory(buf) => {
                    // no flushing necessary
                    buf.push_str(data);
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_section() {
            set_global_sink_memory();

            {
                let _s = section("section name");
                _s.log_value("the_value", &(1, 2, 3));
            }

            let output = read_memory_buffer_this_thread();
            // Remove thread ID
            insta::assert_snapshot!(output);
        }
    }
}
