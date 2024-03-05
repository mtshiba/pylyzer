mod analyze;
mod config;
mod handle_err;
mod copy;

use analyze::{PythonAnalyzer, SimplePythonParser};
use els::Server;
use erg_common::config::ErgMode;
use erg_common::spawn::exec_new_thread;

use crate::copy::copy_dot_erg;

fn run() {
    copy_dot_erg();
    let cfg = config::parse_args();
    if cfg.mode == ErgMode::LanguageServer {
        let lang_server = Server::<PythonAnalyzer, SimplePythonParser>::new(cfg, None);
        lang_server.run();
    } else {
        let mut analyzer = PythonAnalyzer::new(cfg);
        analyzer.run();
    }
}

fn main() {
    exec_new_thread(run, "pylyzer");
}
