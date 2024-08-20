use wasm_bindgen::prelude::*;

use erg_common::error::ErrorCore;
use erg_common::error::Location as Loc;
use erg_common::traits::{Runnable, Stream};
use erg_compiler::context::ContextProvider;
use erg_compiler::erg_parser::ast::VarName;
use erg_compiler::error::CompileError;
use erg_compiler::ty::Type as Ty;
use erg_compiler::varinfo::VarInfo;
use pylyzer_core::PythonAnalyzer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[wasm_bindgen]
pub enum CompItemKind {
    Method = 0,
    Function = 1,
    Constructor = 2,
    Field = 3,
    Variable = 4,
    Class = 5,
    Struct = 6,
    Interface = 7,
    Module = 8,
    Property = 9,
    Event = 10,
    Operator = 11,
    Unit = 12,
    Value = 13,
    Constant = 14,
    Enum = 15,
    EnumMember = 16,
    Keyword = 17,
    Text = 18,
    Color = 19,
    File = 20,
    Reference = 21,
    Customcolor = 22,
    Folder = 23,
    TypeParameter = 24,
    User = 25,
    Issue = 26,
    Snippet = 27,
}

#[derive(Debug, Clone)]
#[wasm_bindgen]
pub struct Location(Loc);

impl From<Loc> for Location {
    fn from(loc: Loc) -> Self {
        Self(loc)
    }
}

impl Location {
    pub const UNKNOWN: Location = Location(Loc::Unknown);
}

#[derive(Debug, Clone)]
#[wasm_bindgen]
#[allow(dead_code)]
pub struct Type(Ty);

#[derive(Debug, Clone)]
#[wasm_bindgen]
pub struct VarEntry {
    name: VarName,
    vi: VarInfo,
}

impl VarEntry {
    pub fn new(name: VarName, vi: VarInfo) -> Self {
        Self { name, vi }
    }
}

#[wasm_bindgen]
impl VarEntry {
    pub fn name(&self) -> String {
        self.name.to_string()
    }
    pub fn item_kind(&self) -> CompItemKind {
        match &self.vi.t {
            Ty::Callable { .. } => CompItemKind::Function,
            Ty::Subr(subr) => {
                if subr.self_t().is_some() {
                    CompItemKind::Method
                } else {
                    CompItemKind::Function
                }
            }
            Ty::Quantified(quant) => match quant.as_ref() {
                Ty::Callable { .. } => CompItemKind::Function,
                Ty::Subr(subr) => {
                    if subr.self_t().is_some() {
                        CompItemKind::Method
                    } else {
                        CompItemKind::Function
                    }
                }
                _ => unreachable!(),
            },
            Ty::ClassType => CompItemKind::Class,
            Ty::TraitType => CompItemKind::Interface,
            Ty::Poly { name, .. } if &name[..] == "Module" => CompItemKind::Module,
            _ if self.vi.muty.is_const() => CompItemKind::Constant,
            _ => CompItemKind::Variable,
        }
    }
    pub fn typ(&self) -> String {
        self.vi.t.to_string()
    }
}

#[wasm_bindgen]
impl Location {
    pub fn ln_begin(&self) -> Option<u32> {
        self.0.ln_begin()
    }

    pub fn ln_end(&self) -> Option<u32> {
        self.0.ln_end()
    }

    pub fn col_begin(&self) -> Option<u32> {
        self.0.col_begin()
    }

    pub fn col_end(&self) -> Option<u32> {
        self.0.col_end()
    }
}

#[derive(Debug, Clone)]
#[wasm_bindgen(getter_with_clone)]
pub struct Error {
    pub errno: usize,
    pub is_warning: bool,
    // pub kind: ErrorKind,
    pub loc: Location,
    pub desc: String,
    pub hint: Option<String>,
}

fn find_fallback_loc(err: &ErrorCore) -> Loc {
    if err.loc == Loc::Unknown {
        for sub in &err.sub_messages {
            if sub.loc != Loc::Unknown {
                return sub.loc;
            }
        }
        Loc::Unknown
    } else {
        err.loc
    }
}

impl From<CompileError> for Error {
    fn from(err: CompileError) -> Self {
        let loc = Location(find_fallback_loc(&err.core));
        let sub_msg = err
            .core
            .sub_messages
            .first()
            .map(|sub| {
                sub.msg
                    .iter()
                    .fold("\n".to_string(), |acc, s| acc + s + "\n")
            })
            .unwrap_or_default();
        let desc = err.core.main_message + &sub_msg;
        Self {
            errno: err.core.errno,
            is_warning: err.core.kind.is_warning(),
            // kind: err.kind(),
            loc,
            desc,
            hint: err
                .core
                .sub_messages
                .first()
                .and_then(|sub| sub.hint.clone()),
        }
    }
}

impl Error {
    pub const fn new(
        errno: usize,
        is_warning: bool,
        loc: Location,
        desc: String,
        hint: Option<String>,
    ) -> Self {
        Self {
            errno,
            is_warning,
            loc,
            desc,
            hint,
        }
    }
}

#[wasm_bindgen]
// #[derive()]
pub struct Analyzer {
    analyzer: PythonAnalyzer,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Analyzer {
    pub fn new() -> Self {
        Analyzer {
            analyzer: PythonAnalyzer::default(),
        }
    }

    pub fn clear(&mut self) {
        self.analyzer.clear();
    }

    pub fn start_message(&self) -> String {
        self.analyzer.start_message()
    }

    pub fn dir(&mut self) -> Box<[VarEntry]> {
        self.analyzer
            .dir()
            .into_iter()
            .map(|(n, vi)| VarEntry::new(n.clone(), vi.clone()))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    pub fn check(&mut self, input: &str) -> Box<[Error]> {
        match self.analyzer.analyze(input.to_string(), "exec") {
            Ok(artifact) => artifact
                .warns
                .into_iter()
                .map(Error::from)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            Err(mut err_artifact) => {
                err_artifact.errors.extend(err_artifact.warns);
                let errs = err_artifact
                    .errors
                    .into_iter()
                    .map(Error::from)
                    .collect::<Vec<_>>();
                errs.into_boxed_slice()
            }
        }
    }
}
