mod config;
mod copy;

use els::Server;
use erg_common::config::ErgMode;
use erg_common::spawn::exec_new_thread;
use pylyzer_core::{PythonAnalyzer, SimplePythonParser};

use crate::config::files_to_be_checked;
use crate::copy::copy_dot_erg;

fn run() {
    copy_dot_erg();
    let cfg = config::parse_args();
    if cfg.mode == ErgMode::LanguageServer {
        let lang_server = Server::<PythonAnalyzer, SimplePythonParser>::new(cfg, None);
        lang_server.run();
    } else {
        let mut code = 0;
        let files = files_to_be_checked();
        if files.is_empty() {
            let mut analyzer = PythonAnalyzer::new(cfg);
            code = analyzer.run();
        } else {
            for path in files {
                let cfg = cfg.inherit(path);
                let mut analyzer = PythonAnalyzer::new(cfg);
                let c = analyzer.run();
                if c != 0 {
                    code = 1;
                }
            }
        }
        std::process::exit(code);
    }
}

fn main() {
    exec_new_thread(run, "pylyzer");
}
