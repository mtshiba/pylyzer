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

fn _expect(file_path: &'static str, warns: usize, errors: usize) -> Result<(), String> {
    println!("Testing {file_path} ...");
    match exec_analyzer(file_path) {
        Ok(artifact) => {
            if artifact.warns.len() != warns {
                return Err(format!(
                    "Expected {warns} warnings, found {}",
                    artifact.warns.len()
                ));
            }
            if errors != 0 {
                return Err(format!("Expected {errors} errors, found 0"));
            }
            Ok(())
        }
        Err(artifact) => {
            if artifact.warns.len() != warns {
                return Err(format!(
                    "Expected {warns} warnings, found {}",
                    artifact.warns.len()
                ));
            }
            if artifact.errors.len() != errors {
                return Err(format!(
                    "Expected {errors} errors, found {}",
                    artifact.errors.len()
                ));
            }
            Ok(())
        }
    }
}

pub fn expect(file_path: &'static str, warns: usize, errors: usize) -> Result<(), String> {
    exec_new_thread(move || _expect(file_path, warns, errors), file_path)
}

#[test]
fn exec_test() -> Result<(), String> {
    expect("tests/test.py", 0, 15)
}

#[test]
fn exec_import() -> Result<(), String> {
    expect("tests/import.py", 1, 2)
}

#[test]
fn exec_export() -> Result<(), String> {
    expect("tests/export.py", 0, 0)
}

#[test]
fn exec_func() -> Result<(), String> {
    expect("tests/func.py", 0, 1)
}

#[test]
fn exec_class() -> Result<(), String> {
    expect("tests/class.py", 0, 4)
}

#[test]
fn exec_errors() -> Result<(), String> {
    expect("tests/errors.py", 0, 3)
}

#[test]
fn exec_warns() -> Result<(), String> {
    expect("tests/warns.py", 2, 0)
}

#[test]
fn exec_typespec() -> Result<(), String> {
    expect("tests/typespec.py", 0, 7)
}

#[test]
fn exec_projection() -> Result<(), String> {
    expect("tests/projection.py", 0, 4)
}

#[test]
fn exec_narrowing() -> Result<(), String> {
    expect("tests/narrowing.py", 0, 1)
}

#[test]
fn exec_casting() -> Result<(), String> {
    expect("tests/casting.py", 1, 1)
}

#[test]
fn exec_collections() -> Result<(), String> {
    expect("tests/collections.py", 0, 4)
}

#[test]
fn exec_call() -> Result<(), String> {
    expect("tests/call.py", 0, 3)
}

#[test]
fn exec_shadowing() -> Result<(), String> {
    expect("tests/shadowing.py", 0, 3)
}

#[test]
fn exec_widening() -> Result<(), String> {
    expect("tests/widening.py", 0, 1)
}
