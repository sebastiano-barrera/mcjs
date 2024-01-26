use lazy_static::lazy_static;

use std::fmt::Debug;
use std::fs::File;
use std::io::{Result, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

pub struct Config {
    pub sink: SinkType,
}

pub enum SinkType {
    File(PathBuf),
    Stdout,
    Buffer,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sink: SinkType::Stdout,
        }
    }
}

pub fn init(config: Config) {
    let mut logger = get();
    *logger = Logger::new(config);
}

#[must_use]
pub fn section(header: &str) -> Section {
    let mut logger = get();
    logger.start_section(header);
    Section
}

pub struct Section;
impl Section {
    pub fn log_value<T: Debug>(&self, tag: &str, value: &T) {
        let mut logger = get();
        logger.log_value(tag, value);
    }

    pub fn log(&self, tag: &str, value: &str) {
        let mut logger = get();
        logger.log(tag, value);
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        let mut logger = get();
        logger.end_section();
    }
}

lazy_static! {
    static ref GLOBAL_LOGGER: Arc<Mutex<Logger>> = Arc::new(Mutex::new(Logger::default()));
}

fn get() -> MutexGuard<'static, Logger> {
    GLOBAL_LOGGER.lock().expect("could not lock global logger!")
}

struct Logger {
    sink: Sink,
    sections_depth: usize,
}
enum Sink {
    File(File),
    Stdout,
    Buffer(String),
}

impl Default for Logger {
    fn default() -> Self {
        Logger::new(Config::default())
    }
}

impl Logger {
    fn new(config: Config) -> Self {
        let sink = match config.sink {
            SinkType::File(filename) => {
                let file = File::open(filename).expect("could not open tracing sink file");
                Sink::File(file)
            }
            SinkType::Stdout => Sink::Stdout,
            SinkType::Buffer => Sink::Buffer(String::new()),
        };

        Logger {
            sink,
            sections_depth: 0,
        }
    }

    fn start_section(&mut self, header: &str) {
        self.write(&format!("<{}>", header));
        self.sections_depth += 1;
    }

    fn end_section(&mut self) {
        assert!(self.sections_depth > 0);
        self.sections_depth -= 1;
        self.write("</>");
    }

    fn write(&mut self, text: &str) {
        use std::fmt::Write;

        let mut line_buf = String::new();
        for line in text.split('\n') {
            for _ in 0..self.sections_depth {
                write!(line_buf, "    ").unwrap();
            }
            writeln!(line_buf, "{}", line).unwrap();
        }

        match &mut self.sink {
            Sink::File(f) => {
                f.write(line_buf.as_bytes()).unwrap();
            }
            Sink::Stdout => {
                let mut stdout = std::io::stdout().lock();
                stdout.write(line_buf.as_bytes()).unwrap();
            }
            Sink::Buffer(buf) => {
                buf.push_str(&line_buf);
            }
        }
    }

    fn log_value<T: Debug>(&mut self, tag: &str, value: &T) {
        self.write(&format!("{}: {:#?}\n", tag, value));
    }
    fn log(&mut self, tag: &str, value: &str) {
        self.write(&format!("{}: {}\n", tag, value));
    }

    fn take_buffer(&mut self) -> String {
        let buf = match &mut self.sink {
            Sink::Buffer(buf) => buf,
            _ => panic!("take_buffer, but sink is not Sink::Buffer(_)"),
        };

        std::mem::replace(buf, String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section() {
        init(Config {
            sink: SinkType::Buffer,
        });

        {
            let _s = section("section name");
            _s.log_value("the_value", &(1, 2, 3));
        }

        let output = get().take_buffer();
        insta::assert_snapshot!(output);
    }
}
