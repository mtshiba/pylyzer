use erg_common::config::ErgConfig;
use erg_common::error::{ErrorCore, ErrorKind, MultiErrorDisplay};
use erg_common::style::colors::{BLUE, GREEN, RED, YELLOW};
use erg_common::style::RESET;
use erg_common::traits::{New, ExitStatus, Runnable, Stream};
use erg_common::Str;
use erg_compiler::GenericHIRBuilder;
use erg_compiler::artifact::{BuildRunnable, Buildable, CompleteArtifact, IncompleteArtifact};
use erg_compiler::build_package::{GenericPackageBuilder, CheckStatus};
use erg_compiler::context::ModuleContext;
use erg_compiler::erg_parser::ast::{Module, AST};
use erg_compiler::erg_parser::build_ast::ASTBuildable;
use erg_compiler::erg_parser::error::{
    CompleteArtifact as ParseArtifact, IncompleteArtifact as IncompleteParseArtifact, ParseErrors, ParserRunnerErrors,
};
use erg_compiler::erg_parser::parse::Parsable;
use erg_compiler::error::{CompileError, CompileErrors};
use erg_compiler::module::SharedCompilerResource;
use py2erg::{dump_decl_er, reserve_decl_er, ShadowingMode};
use rustpython_ast::source_code::{RandomLocator, SourceRange};
use rustpython_ast::{Fold, ModModule};
use rustpython_parser::{Parse, ParseErrorType};

use crate::handle_err;

#[derive(Debug, Default)]
pub struct SimplePythonParser {
    cfg: ErgConfig,
}

impl Parsable for SimplePythonParser {
    fn parse(code: String) -> Result<ParseArtifact, IncompleteParseArtifact<Module, ParseErrors>> {
        let mut slf = Self::new(ErgConfig::string(code.clone()));
        slf.build_ast(code)
            .map(|art| {
                ParseArtifact::new(art.ast.module, art.warns.into())
            })
            .map_err(|iart| {
                IncompleteParseArtifact::new(
                    iart.ast.map(|art| art.module),
                    iart.errors.into(),
                    iart.warns.into(),
                )
            })
    }
}

impl New for SimplePythonParser {
    fn new(cfg: ErgConfig) -> Self {
        Self { cfg }
    }
}

impl ASTBuildable for SimplePythonParser {
    fn build_ast(
        &mut self,
        code: String,
    ) -> Result<ParseArtifact<AST, ParserRunnerErrors>, IncompleteParseArtifact<AST, ParserRunnerErrors>> {
        let filename = self.cfg.input.filename();
        let py_program = self.parse_py_code(code)?;
        let shadowing = if cfg!(feature = "debug") {
            ShadowingMode::Visible
        } else {
            ShadowingMode::Invisible
        };
        let converter = py2erg::ASTConverter::new(ErgConfig::default(), shadowing);
        let IncompleteArtifact{ object: Some(erg_module), errors, warns } = converter.convert_program(py_program) else { unreachable!() };
        let erg_ast = AST::new(erg_common::Str::rc(&filename), erg_module);
        if errors.is_empty() {
            Ok(ParseArtifact::new(erg_ast, warns.into()))
        } else {
            Err(IncompleteParseArtifact::new(
                Some(erg_ast),
                errors.into(),
                warns.into(),
            ))
        }
    }
}

impl SimplePythonParser {
    pub fn parse_py_code(&self, code: String) -> Result<ModModule<SourceRange>, IncompleteParseArtifact<AST, ParserRunnerErrors>>{
        let py_program = ModModule::parse(&code, "<stdin>").map_err(|err| {
            let mut locator = RandomLocator::new(&code);
            // let mut locator = LinearLocator::new(&py_code);
            let err = locator.locate_error::<_, ParseErrorType>(err);
            let msg = err.to_string();
            let loc = err.location.unwrap();
            let core = ErrorCore::new(
                vec![],
                msg,
                0,
                ErrorKind::SyntaxError,
                erg_common::error::Location::range(
                    loc.row.get(),
                    loc.column.to_zero_indexed(),
                    loc.row.get(),
                    loc.column.to_zero_indexed(),
                ),
            );
            let err = CompileError::new(core, self.cfg.input.clone(), "".into());
            IncompleteParseArtifact::new(None, ParserRunnerErrors::from(err), ParserRunnerErrors::empty())
        })?;
        let mut locator = RandomLocator::new(&code);
        // let mut locator = LinearLocator::new(&code);
        let module = locator
            .fold(py_program)
            .map_err(|_err| ParserRunnerErrors::empty())?;
        Ok(module)
    }
}

#[derive(Debug, Default)]
pub struct PythonAnalyzer {
    pub cfg: ErgConfig,
    checker: GenericPackageBuilder<SimplePythonParser, GenericHIRBuilder<SimplePythonParser>>,
}

impl New for PythonAnalyzer {
    fn new(cfg: ErgConfig) -> Self {
        let checker = GenericPackageBuilder::new(cfg.clone(), SharedCompilerResource::new(cfg.clone()));
        Self { checker, cfg }
    }
}

impl Runnable for PythonAnalyzer {
    type Err = CompileError;
    type Errs = CompileErrors;
    const NAME: &'static str = "Python Analyzer";
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
            checker: GenericPackageBuilder::new_with_cache(cfg, mod_name, shared),
        }
    }
    fn inherit_with_name(cfg: ErgConfig, mod_name: Str, shared: SharedCompilerResource) -> Self {
        Self {
            cfg: cfg.copy(),
            checker: GenericPackageBuilder::new_with_cache(cfg, mod_name, shared),
        }
    }
    fn build(&mut self, code: String, mode: &str) -> Result<CompleteArtifact, IncompleteArtifact> {
        self.analyze(code, mode)
    }
    fn build_from_ast(
        &mut self,
        ast: AST,
        mode: &str,
    ) -> Result<CompleteArtifact<erg_compiler::hir::HIR>, IncompleteArtifact<erg_compiler::hir::HIR>> {
        self.check(ast, CompileErrors::empty(), CompileErrors::empty(), mode)
    }
    fn pop_context(&mut self) -> Option<ModuleContext> {
        self.checker.pop_context()
    }
    fn get_context(&self) -> Option<&ModuleContext> {
        self.checker.get_context()
    }
}

impl BuildRunnable for PythonAnalyzer {}

impl PythonAnalyzer {
    pub fn new(cfg: ErgConfig) -> Self {
        New::new(cfg)
    }

    fn check(&mut self, erg_ast: AST, mut errors: CompileErrors, mut warns: CompileErrors, mode: &str) -> Result<CompleteArtifact, IncompleteArtifact> {
        match self.checker.build_from_ast(erg_ast, mode) {
            Ok(mut artifact) => {
                artifact.warns.extend(warns);
                artifact.warns =
                    handle_err::filter_errors(self.get_context().unwrap(), artifact.warns);
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
                let errors = handle_err::filter_errors(self.get_context().unwrap(), errors);
                warns.extend(iart.warns);
                let warns = handle_err::filter_errors(self.get_context().unwrap(), warns);
                Err(IncompleteArtifact::new(iart.object, errors, warns))
            }
        }
    }

    pub fn analyze(
        &mut self,
        py_code: String,
        mode: &str,
    ) -> Result<CompleteArtifact, IncompleteArtifact> {
        let filename = self.cfg.input.filename();
        let parser = SimplePythonParser::new(self.cfg.copy());
        let py_program = parser.parse_py_code(py_code)
            .map_err(|iart| {
                IncompleteArtifact::new(
                    None,
                    iart.errors.into(),
                    iart.warns.into(),
                )
            })?;
        let shadowing = if cfg!(feature = "debug") {
            ShadowingMode::Visible
        } else {
            ShadowingMode::Invisible
        };
        let converter = py2erg::ASTConverter::new(self.cfg.copy(), shadowing);
        let IncompleteArtifact{ object: Some(erg_module), errors, warns } = converter.convert_program(py_program) else { unreachable!() };
        let erg_ast = AST::new(erg_common::Str::rc(&filename), erg_module);
        erg_common::log!("AST:\n{erg_ast}");
        self.check(erg_ast, errors, warns, mode)
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
