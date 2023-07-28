use std::path::PathBuf;

use erg_common::config::ErgConfig;
use erg_common::io::Input;
use erg_common::spawn::exec_new_thread;
use erg_common::traits::Stream;
use erg_compiler::artifact::{CompleteArtifact, IncompleteArtifact};
use pylyzer::PythonAnalyzer;

pub fn exec_analyzer(file_path: &'static str) -> Result<CompleteArtifact, IncompleteArtifact> {
    let cfg = ErgConfig {
        input: Input::file(PathBuf::from(file_path)),
        ..Default::default()
    };
    let mut analyzer = PythonAnalyzer::new(cfg);
    let py_code = analyzer.cfg.input.read();
    analyzer.analyze(py_code, "exec")
}

fn _expect(file_path: &'static str, warns: usize, errors: usize) {
    println!("Testing {file_path} ...");
    match exec_analyzer(file_path) {
        Ok(artifact) => {
            assert_eq!(artifact.warns.len(), warns);
            assert_eq!(errors, 0);
        }
        Err(artifact) => {
            assert_eq!(artifact.warns.len(), warns);
            assert_eq!(artifact.errors.len(), errors);
        }
    }
}

pub fn expect(file_path: &'static str, warns: usize, errors: usize) {
    exec_new_thread(
        move || {
            _expect(file_path, warns, errors);
        },
        file_path,
    );
}

#[test]
fn exec_test() {
    expect("tests/test.py", 0, 11);
}

#[test]
fn exec_import() {
    expect("tests/import.py", 1, 2);
}

#[test]
fn exec_export() {
    expect("tests/export.py", 0, 0);
}

#[test]
fn exec_func() {
    expect("tests/func.py", 0, 1);
}

#[test]
fn exec_class() {
    expect("tests/class.py", 0, 4);
}

#[test]
fn exec_errors() {
    expect("tests/errors.py", 0, 3);
}

#[test]
fn exec_warns() {
    expect("tests/warns.py", 2, 0);
}

#[test]
fn exec_typespec() {
    expect("tests/typespec.py", 0, 7);
}

#[test]
fn exec_projection() {
    expect("tests/projection.py", 0, 4);
}

#[test]
fn exec_narrowing() {
    expect("tests/narrowing.py", 0, 1);
}

#[test]
fn exec_casting() {
    expect("tests/casting.py", 1, 1);
}

#[test]
fn exec_collections() {
    expect("tests/collections.py", 0, 4);
}

#[test]
fn exec_call() {
    expect("tests/call.py", 0, 3);
}

#[test]
fn exec_shadowing() {
    expect("tests/shadowing.py", 0, 3);
}
