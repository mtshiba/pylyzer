use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use erg_common::config::{ErgConfig, ErgMode};
use erg_common::io::Input;
use erg_common::pathutil::project_entry_file_of;
use erg_common::switch_lang;

use crate::copy::clear_cache;

fn entry_file() -> Option<PathBuf> {
    project_entry_file_of(&env::current_dir().ok()?).or_else(|| {
        let mut opt_path = None;
        for ent in Path::new(".").read_dir().ok()? {
            let ent = ent.ok()?;
            if ent.file_type().ok()?.is_file() {
                let path = ent.path();
                if path.file_name().is_some_and(|name| name == "__init__.py") {
                    return Some(path);
                } else if path.extension().is_some_and(|ext| ext == "py") {
                    if opt_path.is_some() {
                        return None;
                    } else {
                        opt_path = Some(path);
                    }
                }
            }
        }
        opt_path
    })
}

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
    --clear-cache                        キャッシュをクリア
    --code/-c cmd                        文字列をプログラムに渡す
    --dump-decl                          型宣言ファイルを出力
    --disable                            指定した機能を無効化",

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
    --clear-cache                        清除缓存
    --code/-c cmd                        作为字符串传入程序
    --dump-decl                          输出类型声明文件
    --disable                            禁用指定功能",

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
    --clear-cache                        清除快取
    --code/-c cmd                        作為字串傳入程式
    --dump-decl                          輸出類型宣告檔案
    --disable                            禁用指定功能",

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
    --clear-cache                        clear cache
    --code/-c cmd                        program passed in as string
    --dump-decl                          output type declaration file
    --disable                            disable specified features",
    )
}

#[allow(unused)]
pub(crate) fn parse_args() -> ErgConfig {
    let mut args = env::args();
    args.next(); // "pylyzer"
    let mut cfg = ErgConfig {
        effect_check: false,
        ownership_check: false,
        respect_pyi: true,
        ..ErgConfig::default()
    };
    let mut runtime_args: Vec<&'static str> = Vec::new();
    while let Some(arg) = args.next() {
        match &arg[..] {
            "--" => {
                for arg in args {
                    runtime_args.push(Box::leak(arg.into_boxed_str()));
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
            "--disable" => {
                let arg = args.next().expect("the value of `--disable` is not passed");
                runtime_args.push(Box::leak(arg.into_boxed_str()));
            }
            "-V" | "--version" => {
                println!("pylyzer {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--clear-cache" => {
                clear_cache();
                std::process::exit(0);
            }
            "--no-infer-fn-type" => {
                cfg.no_infer_fn_type = true;
            }
            "--fast-error-report" => {
                cfg.fast_error_report = true;
            }
            "--hurry" => {
                cfg.no_infer_fn_type = true;
                cfg.fast_error_report = true;
            }
            "--do-not-show-ext-errors" => {
                cfg.do_not_show_ext_errors = true;
            }
            "--do-not-respect-pyi" => {
                cfg.respect_pyi = false;
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
                        runtime_args.push(Box::leak(arg.into_boxed_str()));
                    }
                }
                break;
            }
        }
    }
    if !cfg.mode.is_language_server() && cfg.input.is_repl() {
        if let Some(entry) = entry_file() {
            cfg.input = Input::file(entry);
        } else {
            eprintln!("No entry file found in the current project");
            std::process::exit(1);
        }
    }
    cfg.runtime_args = runtime_args.into();
    cfg
}
