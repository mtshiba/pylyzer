use std::env;
use std::path::PathBuf;
use std::str::FromStr;

use erg_common::config::{ErgConfig, ErgMode};
use erg_common::io::Input;
use erg_common::switch_lang;

fn command_message() -> &'static str {
    switch_lang!(
        "japanese" =>
        "\
USAGE:
    pylyzer [OPTIONS] [ARGS]...

ARGS:
    <script> スクリプトファイルからプログラムを読み込む

OPTIONS
    --help/-?/-h                         このhelpを表示
    --version/-V                         バージョンを表示
    --verbose 0|1|2                      冗長性レベルを指定
    --server                             Language Serverを起動
    --code/-c cmd                        文字列をプログラムに渡す",

    "simplified_chinese" =>
    "\
USAGE:
    pylyzer [OPTIONS] [ARGS]...

ARGS:
    <script> 从脚本文件读取程序

OPTIONS
    --help/-?/-h                         显示帮助
    --version/-V                         显示版本
    --verbose 0|1|2                      指定细致程度
    --server                             启动 Language Server
    --code/-c cmd                        作为字符串传入程序",

    "traditional_chinese" =>
        "\
USAGE:
    pylyzer [OPTIONS] [ARGS]...

ARGS:
    <script> 從腳本檔案讀取程式

OPTIONS
    --help/-?/-h                         顯示幫助
    --version/-V                         顯示版本
    --verbose 0|1|2                      指定細緻程度
    --server                             啟動 Language Server
    --code/-c cmd                        作為字串傳入程式",

    "english" =>
        "\
USAGE:
    pylyzer [OPTIONS] [ARGS]...

ARGS:
    <script> program read from script file

OPTIONS
    --help/-?/-h                         show this help
    --version/-V                         show version
    --verbose 0|1|2                      verbosity level
    --server                             start the Language Server
    --code/-c cmd                        program passed in as string",
    )
}

#[allow(unused)]
pub(crate) fn parse_args() -> ErgConfig {
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
                cfg.input = Input::str(args.next().expect("the value of `-c` is not passed"));
            }
            "-?" | "-h" | "--help" => {
                println!("{}", command_message());
                std::process::exit(0);
            }
            "--server" => {
                cfg.mode = ErgMode::LanguageServer;
                cfg.quiet_repl = true;
            }
            "--dump-decl" => {
                cfg.dist_dir = Some("");
            }
            "--verbose" => {
                cfg.verbose = args
                    .next()
                    .expect("the value of `--verbose` is not passed")
                    .parse::<u8>()
                    .expect("the value of `--verbose` is not a number");
            }
            "-V" | "--version" => {
                println!("pylyzer {}", env!("CARGO_PKG_VERSION"));
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
                cfg.input = Input::file(
                    PathBuf::from_str(&arg[..])
                        .unwrap_or_else(|_| panic!("invalid file path: {arg}")),
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
