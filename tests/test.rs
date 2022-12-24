use std::path::PathBuf;

use erg_common::traits::Stream;
use erg_common::config::{ErgConfig, Input};
use erg_compiler::artifact::{IncompleteArtifact, CompleteArtifact};
use pylyzer::PythonAnalyzer;

pub fn exec_analyzer(file_path: &'static str) -> Result<CompleteArtifact, IncompleteArtifact> {
    let cfg = ErgConfig { python_compatible_mode: true, input: Input::File(PathBuf::from(file_path)), ..Default::default() };
    let mut analyzer = PythonAnalyzer::new(cfg);
    let py_code = analyzer.cfg.input.read();
    analyzer.analyze(py_code, "exec")
}

pub fn expect(file_path: &'static str, warns: usize, errors: usize) {
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

#[test]
fn exec_test() {
    expect("tests/test.py", 0, 8);
}

#[test]
fn exec_import() {
    expect("tests/import.py", 0, 2);
}

#[test]
fn exec_export() {
    expect("tests/export.py", 0, 0);
}

#[test]
fn exec_class() {
    expect("tests/class.py", 0, 1);
}

#[test]
fn exec_errors() {
    expect("tests/errors.py", 0, 3);
}

#[test]
fn exec_warns() {
    expect("tests/warns.py", 2, 0);
}
