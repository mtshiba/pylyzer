mod analyze;
mod handle_err;

use std::env;
use std::path::PathBuf;
use std::str::FromStr;

use analyze::PythonAnalyzer;
use els::Server;
use erg_common::config::{ErgConfig, Input};
use erg_common::spawn::exec_new_thread;

pub fn parse_args() -> ErgConfig {
    let mut args = env::args();
    args.next(); // "pylyzer"
    let mut cfg = ErgConfig::default();
    while let Some(arg) = args.next() {
        match &arg[..] {
            "--" => {
                for arg in args {
                    cfg.runtime_args.push(Box::leak(arg.into_boxed_str()));
                }
                break;
            }
            "-c" | "--code" => {
                cfg.input = Input::Str(args.next().expect("the value of `-c` is not passed"));
            }
            "--server" => {
                cfg.mode = "server";
                cfg.quiet_repl = true;
            }
            "--dump-decl" => {
                cfg.output_dir = Some("");
            }
            "--verbose" => {
                cfg.verbose = args
                    .next()
                    .expect("the value of `--verbose` is not passed")
                    .parse::<u8>()
                    .expect("the value of `--verbose` is not a number");
            }
            "-V" | "--version" => {
                println!("Erg {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            other if other.starts_with('-') => {
                println!(
                    "\
invalid option: {other}

USAGE:
pylyzer [OPTIONS] [SUBCOMMAND] [ARGS]...

For more information try `pylyzer --help`"
                );
                std::process::exit(2);
            }
            _ => {
                cfg.input = Input::File(
                    PathBuf::from_str(&arg[..])
                        .unwrap_or_else(|_| panic!("invalid file path: {}", arg)),
                );
                if let Some("--") = args.next().as_ref().map(|s| &s[..]) {
                    for arg in args {
                        cfg.runtime_args.push(Box::leak(arg.into_boxed_str()));
                    }
                }
                break;
            }
        }
    }
    cfg
}

fn run() {
    let cfg = parse_args();
    if cfg.mode == "server" {
        let mut lang_server = Server::<PythonAnalyzer>::new(cfg);
        lang_server.run().unwrap_or_else(|_| {
            std::process::exit(1);
        });
    } else {
        let mut analyzer = PythonAnalyzer::new(cfg);
        analyzer.run();
    }
}

fn main() {
    exec_new_thread(run);
}
