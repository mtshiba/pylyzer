use erg_common::traits::{Runnable, Stream};
use erg_common::config::{ErgConfig};
use erg_common::error::{MultiErrorDisplay, ErrorCore, ErrorKind};
use erg_compiler::artifact::{BuildRunnable, CompleteArtifact, IncompleteArtifact, Buildable};
use erg_compiler::context::Context;
use erg_compiler::erg_parser::ast::AST;
use erg_compiler::error::{CompileErrors, CompileError};
use erg_compiler::lower::ASTLowerer;
use py2erg::ShadowingMode;
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
    #[inline]
    fn cfg(&self) -> &ErgConfig {
        &self.cfg
    }
    #[inline]
    fn cfg_mut(&mut self) -> &mut ErgConfig {
        &mut self.cfg
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
        self.checker.pop_mod_ctx()
    }
    fn get_context(&self) -> Option<&Context> {
        Some(self.checker.get_mod_ctx())
    }
}

impl BuildRunnable for PythonAnalyzer {}

impl PythonAnalyzer {
    pub fn new(cfg: ErgConfig) -> Self {
        Runnable::new(cfg)
    }

    pub fn analyze(&mut self, py_code: String, mode: &str) -> Result<CompleteArtifact, IncompleteArtifact> {
        let filename = self.cfg.input.filename();
        let py_program = parser::parse_program(&py_code).map_err(|err| {
            let core = ErrorCore::new(
                vec![],
                err.to_string(),
                0,
                ErrorKind::SyntaxError,
                erg_common::error::Location::Line(err.location.row())
            );
            let err = CompileError::new(core, self.cfg.input.clone(),  "".into());
            IncompleteArtifact::new(None, CompileErrors::from(err), CompileErrors::empty())
        })?;
        let converter = py2erg::ASTConverter::new(self.cfg.copy(), ShadowingMode::Invisible);
        let CompleteArtifact{ object: erg_module, mut warns } = converter.convert_program(py_program);
        let erg_ast = AST::new(erg_common::Str::rc(filename), erg_module);
        erg_common::log!("AST: {erg_ast}");
        match self.checker.lower(erg_ast, mode) {
            Ok(mut artifact) => {
                artifact.warns.extend(warns);
                Ok(artifact)
            }
            Err(iart) => {
                let errors = handle_err::filter_errors(self.checker.get_mod_ctx(), iart.errors);
                let ws = handle_err::filter_errors(self.checker.get_mod_ctx(), iart.warns);
                warns.extend(ws);
                Err(IncompleteArtifact::new(iart.object, errors, warns))
            }
        }
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
                    dump_decl_er(self.cfg.input.clone(), artifact.object);
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
