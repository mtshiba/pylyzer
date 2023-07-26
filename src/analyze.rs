use erg_common::config::ErgConfig;
use erg_common::error::{ErrorCore, ErrorKind, MultiErrorDisplay};
use erg_common::style::colors::{BLUE, GREEN, RED, YELLOW};
use erg_common::style::RESET;
use erg_common::traits::{ExitStatus, Runnable, Stream};
use erg_common::Str;
use erg_compiler::artifact::{BuildRunnable, Buildable, CompleteArtifact, IncompleteArtifact};
use erg_compiler::context::register::CheckStatus;
use erg_compiler::context::ModuleContext;
use erg_compiler::erg_parser::ast::AST;
use erg_compiler::erg_parser::error::{
    CompleteArtifact as PCompleteArtifact, IncompleteArtifact as PIncompleteArtifact, ParseErrors,
};
use erg_compiler::erg_parser::parse::Parsable;
use erg_compiler::error::{CompileError, CompileErrors};
use erg_compiler::lower::ASTLowerer;
use erg_compiler::module::SharedCompilerResource;
use py2erg::{dump_decl_er, reserve_decl_er, ShadowingMode};
use rustpython_parser::parser;

use crate::handle_err;

pub struct SimplePythonParser {}

impl Parsable for SimplePythonParser {
    fn parse(code: String) -> Result<PCompleteArtifact, PIncompleteArtifact> {
        let py_program = parser::parse_program(&code).map_err(|_err| ParseErrors::empty())?;
        let shadowing = if cfg!(feature = "debug") {
            ShadowingMode::Visible
        } else {
            ShadowingMode::Invisible
        };
        let converter = py2erg::ASTConverter::new(ErgConfig::default(), shadowing);
        let art = converter.convert_program(py_program);
        if art.errors.is_empty() {
            Ok(PCompleteArtifact::new(
                art.object.unwrap(),
                art.warns.into(),
            ))
        } else {
            Err(PIncompleteArtifact::new(
                art.object,
                art.errors.into(),
                art.warns.into(),
            ))
        }
    }
}

#[derive(Debug, Default)]
pub struct PythonAnalyzer {
    pub cfg: ErgConfig,
    checker: ASTLowerer,
}

impl Runnable for PythonAnalyzer {
    type Err = CompileError;
    type Errs = CompileErrors;
    const NAME: &'static str = "Python Analyzer";
    fn new(cfg: ErgConfig) -> Self {
        let checker = ASTLowerer::new(cfg.clone());
        Self { checker, cfg }
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
    fn exec(&mut self) -> Result<ExitStatus, Self::Errs> {
        self.checker.exec()
    }
}

impl Buildable for PythonAnalyzer {
    fn inherit(cfg: ErgConfig, shared: SharedCompilerResource) -> Self {
        let mod_name = Str::rc(&cfg.input.file_stem());
        Self {
            cfg: cfg.copy(),
            checker: ASTLowerer::new_with_cache(cfg, mod_name, shared),
        }
    }
    fn build(&mut self, code: String, mode: &str) -> Result<CompleteArtifact, IncompleteArtifact> {
        self.analyze(code, mode)
    }
    fn pop_context(&mut self) -> Option<ModuleContext> {
        self.checker.pop_mod_ctx()
    }
    fn get_context(&self) -> Option<&ModuleContext> {
        Some(self.checker.get_mod_ctx())
    }
}

impl BuildRunnable for PythonAnalyzer {}

impl PythonAnalyzer {
    pub fn new(cfg: ErgConfig) -> Self {
        Runnable::new(cfg)
    }

    pub fn analyze(
        &mut self,
        py_code: String,
        mode: &str,
    ) -> Result<CompleteArtifact, IncompleteArtifact> {
        let filename = self.cfg.input.filename();
        let py_program = parser::parse_program(&py_code).map_err(|err| {
            let core = ErrorCore::new(
                vec![],
                err.to_string(),
                0,
                ErrorKind::SyntaxError,
                erg_common::error::Location::Line(err.location.row() as u32),
            );
            let err = CompileError::new(core, self.cfg.input.clone(), "".into());
            IncompleteArtifact::new(None, CompileErrors::from(err), CompileErrors::empty())
        })?;
        let shadowing = if cfg!(feature = "debug") {
            ShadowingMode::Visible
        } else {
            ShadowingMode::Invisible
        };
        let converter = py2erg::ASTConverter::new(self.cfg.copy(), shadowing);
        let IncompleteArtifact{ object: Some(erg_module), mut errors, mut warns } = converter.convert_program(py_program) else { unreachable!() };
        let erg_ast = AST::new(erg_common::Str::rc(&filename), erg_module);
        erg_common::log!("AST:\n{erg_ast}");
        match self.checker.lower(erg_ast, mode) {
            Ok(mut artifact) => {
                artifact.warns.extend(warns);
                artifact.warns =
                    handle_err::filter_errors(self.checker.get_mod_ctx(), artifact.warns);
                if errors.is_empty() {
                    Ok(artifact)
                } else {
                    Err(IncompleteArtifact::new(
                        Some(artifact.object),
                        errors,
                        artifact.warns,
                    ))
                }
            }
            Err(iart) => {
                errors.extend(iart.errors);
                let errors = handle_err::filter_errors(self.checker.get_mod_ctx(), errors);
                warns.extend(iart.warns);
                let warns = handle_err::filter_errors(self.checker.get_mod_ctx(), warns);
                Err(IncompleteArtifact::new(iart.object, errors, warns))
            }
        }
    }

    pub fn run(&mut self) {
        if self.cfg.dist_dir.is_some() {
            reserve_decl_er(self.cfg.input.clone());
        }
        let py_code = self.cfg.input.read();
        let filename = self.cfg.input.filename();
        println!("{BLUE}Start checking{RESET}: {filename}");
        match self.analyze(py_code, "exec") {
            Ok(artifact) => {
                if !artifact.warns.is_empty() {
                    println!(
                        "{YELLOW}Found {} warnings{RESET}: {}",
                        artifact.warns.len(),
                        self.cfg.input.filename()
                    );
                    artifact.warns.write_all_stderr();
                }
                println!("{GREEN}All checks OK{RESET}: {}", self.cfg.input.filename());
                if self.cfg.dist_dir.is_some() {
                    dump_decl_er(
                        self.cfg.input.clone(),
                        artifact.object,
                        CheckStatus::Succeed,
                    );
                    println!("A declaration file has been generated to __pycache__ directory.");
                }
                std::process::exit(0);
            }
            Err(artifact) => {
                if !artifact.warns.is_empty() {
                    println!(
                        "{YELLOW}Found {} warnings{RESET}: {}",
                        artifact.warns.len(),
                        self.cfg.input.filename()
                    );
                    artifact.warns.write_all_stderr();
                }
                let code = if artifact.errors.is_empty() {
                    println!("{GREEN}All checks OK{RESET}: {}", self.cfg.input.filename());
                    0
                } else {
                    println!(
                        "{RED}Found {} errors{RESET}: {}",
                        artifact.errors.len(),
                        self.cfg.input.filename()
                    );
                    artifact.errors.write_all_stderr();
                    1
                };
                // Even if type checking fails, some APIs are still valid, so generate a file
                if self.cfg.dist_dir.is_some() {
                    dump_decl_er(
                        self.cfg.input.clone(),
                        artifact.object.unwrap(),
                        CheckStatus::Failed,
                    );
                    println!("A declaration file has been generated to __pycache__ directory.");
                }
                std::process::exit(code);
            }
        }
    }
}
