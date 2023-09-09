mod analyze;
mod config;
mod download;
mod handle_err;

use analyze::{PythonAnalyzer, SimplePythonParser};
use download::download_dependencies;
use els::Server;
use erg_common::config::ErgMode;
use erg_common::spawn::exec_new_thread;

fn run() {
    download_dependencies().unwrap();
    let cfg = config::parse_args();
    if cfg.mode == ErgMode::LanguageServer {
        let mut lang_server = Server::<PythonAnalyzer, SimplePythonParser>::new(cfg, None);
        lang_server.run().unwrap_or_else(|_| {
            std::process::exit(1);
        });
    } else {
        let mut analyzer = PythonAnalyzer::new(cfg);
        analyzer.run();
    }
}

fn main() {
    exec_new_thread(run, "pylyzer");
}
