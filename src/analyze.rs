use erg_common::traits::{Runnable, Stream};
use erg_common::config::{ErgConfig};
use erg_common::error::MultiErrorDisplay;
use erg_compiler::artifact::{BuildRunnable, CompleteArtifact, IncompleteArtifact, Buildable};
use erg_compiler::context::Context;
use erg_compiler::erg_parser::ast::AST;
use erg_compiler::error::{CompileErrors, CompileError};
use erg_compiler::lower::ASTLowerer;
use py2erg::dump_decl_er;
use rustpython_parser::parser;

use crate::handle_err;

#[derive(Debug, Default)]
pub struct PythonAnalyzer {
    pub cfg: ErgConfig,
    checker: ASTLowerer,
}

impl Runnable for PythonAnalyzer {
    type Err = CompileError;
    type Errs = CompileErrors;
    const NAME: &'static str =  "Python Analyzer";
    fn new(cfg: ErgConfig) -> Self {
        let checker = ASTLowerer::new(cfg.clone());
        Self {
            checker,
            cfg,
        }
    }
    fn cfg(&self) -> &ErgConfig {
        &self.cfg
    }
    fn finish(&mut self) {
        self.checker.finish();
    }
    fn initialize(&mut self) {
        self.checker.initialize();
    }
    fn clear(&mut self) {
        self.checker.clear();
    }
    fn eval(&mut self, src: String) -> Result<String, Self::Errs> {
        self.checker.eval(src)
    }
    fn exec(&mut self) -> Result<i32, Self::Errs> {
        self.checker.exec()
    }
}

impl Buildable for PythonAnalyzer {
    fn build(&mut self, code: String, mode: &str) -> Result<CompleteArtifact, IncompleteArtifact> {
        self.analyze(code, mode)
    }
    fn pop_context(&mut self) -> Option<Context> {
        Some(self.checker.pop_mod_ctx())
    }
    fn get_context(&self) -> Option<&Context> {
        Some(self.checker.get_mod_ctx())
    }
}

impl BuildRunnable for PythonAnalyzer {}

impl PythonAnalyzer {
    pub fn analyze(&mut self, py_code: String, mode: &str) -> Result<CompleteArtifact, IncompleteArtifact> {
        let filename = self.cfg.input.filename();
        let py_program = parser::parse_program(&py_code).unwrap();
        let erg_module = py2erg::convert_program(py_program);
        let erg_ast = AST::new(erg_common::Str::rc(filename), erg_module);
        erg_common::log!("AST: {erg_ast}");
        self.checker.lower(erg_ast, mode).map_err(|iart| {
            let filtered = handle_err::filter_errors(self.checker.get_mod_ctx(), iart.errors);
            IncompleteArtifact::new(iart.object, filtered, iart.warns)
        })
    }

    pub fn run(&mut self) {
        let filename = self.cfg.input.filename();
        let py_code = self.cfg.input.read();
        println!("Start checking: {filename}");
        match self.analyze(py_code, "exec") {
            Ok(artifact) => {
                if !artifact.warns.is_empty() {
                    println!("Found warnings: {}", artifact.warns.len());
                    artifact.warns.fmt_all_stderr();
                }
                println!("All checks OK.");
                if self.cfg.output_dir.is_some() {
                    dump_decl_er(artifact.object);
                    println!("A declaration file has been generated to __pycache__ directory.");
                }
                std::process::exit(0);
            }
            Err(artifact) => {
                if !artifact.warns.is_empty() {
                    println!("Found warnings: {}", artifact.warns.len());
                    artifact.warns.fmt_all_stderr();
                }
                if artifact.errors.is_empty() {
                    println!("All checks OK.");
                    std::process::exit(0);
                } else {
                    println!("Found errors: {}", artifact.errors.len());
                    artifact.errors.fmt_all_stderr();
                    std::process::exit(1);
                }
            }
        }
    }
}
