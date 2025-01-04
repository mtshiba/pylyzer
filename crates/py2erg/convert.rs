use core::fmt;
use std::path::Path;

use erg_common::config::ErgConfig;
use erg_common::dict::Dict as HashMap;
use erg_common::error::Location as ErgLocation;
use erg_common::fresh::FRESH_GEN;
use erg_common::set::Set as HashSet;
use erg_common::traits::{Locational, Stream};
use erg_common::{fmt_vec, log, set};
use erg_compiler::artifact::IncompleteArtifact;
use erg_compiler::erg_parser::ast::{
    Accessor, Args, BinOp, Block, ClassAttr, ClassAttrs, ClassDef, Compound, ConstAccessor,
    ConstApp, ConstArgs, ConstAttribute, ConstBinOp, ConstBlock, ConstDict, ConstExpr,
    ConstKeyValue, ConstLambda, ConstList, ConstListWithLength, ConstNormalList, ConstNormalSet,
    ConstPosArg, ConstSet, Decorator, Def, DefBody, DefId, DefaultParamSignature, Dict,
    DictComprehension, Dummy, Expr, Identifier, KeyValue, KwArg, Lambda, LambdaSignature, List,
    ListComprehension, Literal, Methods, Module, NonDefaultParamSignature, NormalDict, NormalList,
    NormalRecord, NormalSet, NormalTuple, ParamPattern, ParamTySpec, Params, PosArg,
    PreDeclTypeSpec, ReDef, Record, RecordAttrs, Set, SetComprehension, Signature, SubrSignature,
    SubrTypeSpec, Tuple, TupleTypeSpec, TypeAppArgs, TypeAppArgsKind, TypeAscription,
    TypeBoundSpec, TypeBoundSpecs, TypeSpec, TypeSpecWithOp, UnaryOp, VarName, VarPattern,
    VarRecordAttr, VarRecordAttrs, VarRecordPattern, VarSignature, VisModifierSpec,
};
use erg_compiler::erg_parser::desugar::Desugarer;
use erg_compiler::erg_parser::token::{Token, TokenKind, AS, COLON, DOT, EQUAL};
use erg_compiler::erg_parser::Parser;
use erg_compiler::error::{CompileError, CompileErrors};
use rustpython_ast::located::LocatedMut;
use rustpython_ast::source_code::RandomLocator;
use rustpython_parser::ast::located::{
    self as py_ast, Alias, Arg, Arguments, BoolOp, CmpOp, ExprConstant, Keyword, Located,
    ModModule, Operator, Stmt, String, Suite, TypeParam, UnaryOp as UnOp,
};
use rustpython_parser::ast::Fold;
use rustpython_parser::source_code::{
    OneIndexed, SourceLocation as PyLocation, SourceRange as PySourceRange,
};
use rustpython_parser::Parse;

use crate::ast_util::accessor_name;
use crate::error::*;

macro_rules! global_unary_collections {
    () => {
        "Collection" | "Container" | "Generator" | "Iterable" | "Iterator" | "Sequence" | "Set"
    };
}

macro_rules! global_mutable_unary_collections {
    () => {
        "MutableSequence" | "MutableSet" | "MutableMapping"
    };
}

macro_rules! global_binary_collections {
    () => {
        "Mapping"
    };
}

pub const ARROW: Token = Token::dummy(TokenKind::FuncArrow, "->");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenameKind {
    Let,
    Phi,
    Redef,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NameKind {
    Variable,
    Class,
    Function,
}

impl NameKind {
    pub const fn is_variable(&self) -> bool {
        matches!(self, Self::Variable)
    }
    pub const fn is_class(&self) -> bool {
        matches!(self, Self::Class)
    }
    pub const fn is_function(&self) -> bool {
        matches!(self, Self::Function)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
    If,
    /// else, except, finally
    Else {
        if_block_id: usize,
    },
    For,
    While,
    Try,
    With,
    Function,
    AsyncFunction,
    Class,
    Module,
}

impl BlockKind {
    pub const fn is_if(&self) -> bool {
        matches!(self, Self::If)
    }
    pub const fn is_function(&self) -> bool {
        matches!(self, Self::Function | Self::AsyncFunction)
    }
    pub const fn makes_scope(&self) -> bool {
        matches!(self, Self::Function | Self::AsyncFunction | Self::Class)
    }
    pub const fn is_else(&self) -> bool {
        matches!(self, Self::Else { .. })
    }
    pub const fn if_block_id(&self) -> Option<usize> {
        match self {
            Self::Else { if_block_id } => Some(*if_block_id),
            _ => None,
        }
    }
}

/// Variables are automatically rewritten with `py_compat`,
/// but types are rewritten here because they are complex components used inseparably in the Erg system.
fn escape_name(name: String) -> String {
    match &name[..] {
        "object" => "Obj".into(),
        "int" => "Int".into(),
        "float" => "Float".into(),
        "complex" => "Complex".into(),
        "str" => "Str".into(),
        "bool" => "Bool".into(),
        "list" => "GenericList".into(),
        "bytes" => "Bytes".into(),
        "bytearray" => "ByteArray!".into(),
        // "range" => "GenericRange".into(),
        "dict" => "GenericDict".into(),
        "set" => "GenericSet".into(),
        "tuple" => "GenericTuple".into(),
        "type" => "Type".into(),
        "ModuleType" => "GeneticModule".into(),
        "MutableSequence" => "Sequence!".into(),
        "MutableMapping" => "Mapping!".into(),
        _ => name,
    }
}

fn quoted_symbol(sym: &str, lineno: u32, col_begin: u32) -> Token {
    let col_end = col_begin + sym.chars().count() as u32;
    Token {
        kind: TokenKind::StrLit,
        content: format!("\"{sym}\"").into(),
        lineno,
        col_begin,
        col_end,
    }
}

fn op_to_token(op: Operator) -> Token {
    let (kind, cont) = match op {
        Operator::Add => (TokenKind::Plus, "+"),
        Operator::Sub => (TokenKind::Minus, "-"),
        Operator::Mult => (TokenKind::Star, "*"),
        Operator::Div => (TokenKind::Slash, "/"),
        Operator::Mod => (TokenKind::Mod, "%"),
        Operator::Pow => (TokenKind::Pow, "**"),
        Operator::LShift => (TokenKind::Shl, "<<"),
        Operator::RShift => (TokenKind::Shr, ">>"),
        Operator::BitOr => (TokenKind::BitOr, "||"),
        Operator::BitXor => (TokenKind::BitXor, "^^"),
        Operator::BitAnd => (TokenKind::BitAnd, "&&"),
        Operator::FloorDiv => (TokenKind::FloorDiv, "//"),
        Operator::MatMult => (TokenKind::AtSign, "@"),
    };
    Token::from_str(kind, cont)
}

pub fn pyloc_to_ergloc(range: PySourceRange) -> erg_common::error::Location {
    erg_common::error::Location::range(
        range.start.row.get(),
        range.start.column.to_zero_indexed(),
        range.end.unwrap().row.get(),
        range.end.unwrap().column.to_zero_indexed(),
    )
}

pub fn ergloc_to_pyloc(loc: erg_common::error::Location) -> PySourceRange {
    PySourceRange::new(
        PyLocation {
            row: OneIndexed::from_zero_indexed(loc.ln_begin().unwrap_or(0)),
            column: OneIndexed::from_zero_indexed(loc.col_begin().unwrap_or(0)),
        },
        PyLocation {
            row: OneIndexed::from_zero_indexed(loc.ln_end().unwrap_or(0)),
            column: OneIndexed::from_zero_indexed(loc.col_end().unwrap_or(0)),
        },
    )
}

fn attr_name_loc(value: &Expr) -> PyLocation {
    PyLocation {
        row: OneIndexed::from_zero_indexed(value.ln_end().unwrap_or(0)).saturating_sub(1),
        column: OneIndexed::from_zero_indexed(value.col_end().unwrap_or(0)).saturating_add(1),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DefinedPlace {
    Known(String),
    Unknown,
}

impl PartialEq<str> for DefinedPlace {
    fn eq(&self, other: &str) -> bool {
        match self {
            Self::Known(s) => s == other,
            Self::Unknown => false,
        }
    }
}

impl PartialEq<String> for DefinedPlace {
    fn eq(&self, other: &String) -> bool {
        match self {
            Self::Known(s) => s == other,
            Self::Unknown => false,
        }
    }
}

impl DefinedPlace {
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NameInfo {
    rename: Option<String>,
    defined_in: DefinedPlace,
    defined_block_id: usize,
    defined_times: usize,
    referenced: HashSet<String>,
}

impl NameInfo {
    pub fn new(
        rename: Option<String>,
        defined_in: DefinedPlace,
        defined_block_id: usize,
        defined_times: usize,
    ) -> Self {
        Self {
            rename,
            defined_in,
            defined_block_id,
            defined_times,
            referenced: HashSet::new(),
        }
    }

    // TODO: referrer can be usize
    pub fn add_referrer(&mut self, referrer: String) {
        self.referenced.insert(referrer);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowingMode {
    Invisible,
    Visible,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeVarInfo {
    name: String,
    constraints: Vec<Expr>,
    bound: Option<Expr>,
}

impl fmt::Display for TypeVarInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(bound) = &self.bound {
            write!(
                f,
                "TypeVarInfo({}, [{}], bound={})",
                self.name,
                fmt_vec(&self.constraints),
                bound
            )
        } else {
            write!(
                f,
                "TypeVarInfo({}, [{}])",
                self.name,
                fmt_vec(&self.constraints)
            )
        }
    }
}

impl TypeVarInfo {
    pub const fn new(name: String, constraints: Vec<Expr>, bound: Option<Expr>) -> Self {
        Self {
            name,
            constraints,
            bound,
        }
    }
}

#[derive(Debug)]
pub struct BlockInfo {
    pub id: usize,
    pub kind: BlockKind,
}

#[derive(Debug)]
pub enum ReturnKind {
    None,
    Return,
    Yield,
}

impl ReturnKind {
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

#[derive(Debug)]
pub struct LocalContext {
    pub name: String,
    pub kind: BlockKind,
    /// Erg does not allow variables to be defined multiple times, so rename them using this
    names: HashMap<String, NameInfo>,
    type_vars: HashMap<String, TypeVarInfo>,
    // e.g. def id(x: T) -> T: ... => appeared_types = {T}
    appeared_type_names: HashSet<String>,
    return_kind: ReturnKind,
}

impl LocalContext {
    pub fn new(name: String, kind: BlockKind) -> Self {
        Self {
            name,
            kind,
            names: HashMap::new(),
            type_vars: HashMap::new(),
            appeared_type_names: HashSet::new(),
            return_kind: ReturnKind::None,
        }
    }
}

#[derive(Debug, Default)]
pub struct CommentStorage {
    comments: HashMap<u32, (String, Option<py_ast::Expr>)>,
}

impl fmt::Display for CommentStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (comment, expr)) in &self.comments {
            writeln!(f, "line {i}: \"{comment}\" (expr: {})", expr.is_some())?;
        }
        Ok(())
    }
}

impl CommentStorage {
    pub fn new() -> Self {
        Self {
            comments: HashMap::new(),
        }
    }

    pub fn read(&mut self, code: &str) {
        // NOTE: This locater is meaningless.
        let mut locater = RandomLocator::new(code);
        for (i, line) in code.lines().enumerate() {
            let mut split = line.split('#');
            let _code = split.next().unwrap();
            if let Some(comment) = split.next() {
                let comment = comment.to_string();
                let trimmed = comment.trim_start();
                let expr = if trimmed.starts_with("type:") {
                    let typ = trimmed.trim_start_matches("type:").trim();
                    let typ = if typ == "ignore" { "Any" } else { typ };
                    rustpython_ast::Expr::parse(typ, "<module>")
                        .ok()
                        .and_then(|expr| locater.fold(expr).ok())
                } else {
                    None
                };
                self.comments.insert(i as u32, (comment, expr));
            }
        }
    }

    /// line: 0-origin
    pub fn get_code(&self, line: u32) -> Option<&String> {
        self.comments.get(&line).map(|(code, _)| code)
    }

    /// line: 0-origin
    pub fn get_type(&self, line: u32) -> Option<&py_ast::Expr> {
        self.comments.get(&line).and_then(|(_, ty)| ty.as_ref())
    }
}

#[derive(Debug, Clone)]
pub struct PyFuncTypeSpec {
    type_params: Vec<py_ast::TypeParam>,
    args: py_ast::Arguments,
    returns: Option<py_ast::Expr>,
}

#[derive(Debug, Clone)]
pub enum PyTypeSpec {
    Var(py_ast::Expr),
    Func(PyFuncTypeSpec),
}

#[derive(Debug, Default)]
pub struct PyiTypeStorage {
    decls: HashMap<String, PyTypeSpec>,
    classes: HashMap<String, HashMap<String, PyTypeSpec>>,
}

impl fmt::Display for PyiTypeStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (name, t_spec) in &self.decls {
            writeln!(f, "{name}: {t_spec:?}")?;
        }
        for (class, methods) in &self.classes {
            writeln!(f, "class {class}:")?;
            for (name, t_spec) in methods {
                writeln!(f, "    {name}: {t_spec:?}")?;
            }
        }
        Ok(())
    }
}

impl PyiTypeStorage {
    pub fn new() -> Self {
        Self {
            decls: HashMap::new(),
            classes: HashMap::new(),
        }
    }

    pub fn parse(&mut self, filename: &str) {
        let Ok(code) = std::fs::read_to_string(filename) else {
            return;
        };
        let Ok(py_program) = rustpython_ast::ModModule::parse(&code, filename) else {
            return;
        };
        let mut locator = RandomLocator::new(&code);
        let Ok(py_program) = locator.fold(py_program);
        for stmt in py_program.body {
            match stmt {
                py_ast::Stmt::AnnAssign(assign) => {
                    let py_ast::Expr::Name(name) = *assign.target else {
                        continue;
                    };
                    self.decls
                        .insert(name.id.to_string(), PyTypeSpec::Var(*assign.annotation));
                }
                py_ast::Stmt::FunctionDef(def) => {
                    let returns = def.returns.map(|anot| *anot);
                    self.decls.insert(
                        def.name.to_string(),
                        PyTypeSpec::Func(PyFuncTypeSpec {
                            type_params: def.type_params,
                            args: *def.args,
                            returns,
                        }),
                    );
                }
                py_ast::Stmt::AsyncFunctionDef(def) => {
                    let returns = def.returns.map(|anot| *anot);
                    self.decls.insert(
                        def.name.to_string(),
                        PyTypeSpec::Func(PyFuncTypeSpec {
                            type_params: def.type_params,
                            args: *def.args,
                            returns,
                        }),
                    );
                }
                py_ast::Stmt::ClassDef(class) => {
                    let mut methods = HashMap::new();
                    for stmt in class.body {
                        match stmt {
                            py_ast::Stmt::AnnAssign(assign) => {
                                let py_ast::Expr::Name(name) = *assign.target else {
                                    continue;
                                };
                                methods.insert(
                                    name.id.to_string(),
                                    PyTypeSpec::Var(*assign.annotation),
                                );
                            }
                            py_ast::Stmt::FunctionDef(def) => {
                                let returns = def.returns.map(|anot| *anot);
                                methods.insert(
                                    def.name.to_string(),
                                    PyTypeSpec::Func(PyFuncTypeSpec {
                                        type_params: def.type_params,
                                        args: *def.args,
                                        returns,
                                    }),
                                );
                            }
                            py_ast::Stmt::AsyncFunctionDef(def) => {
                                let returns = def.returns.map(|anot| *anot);
                                methods.insert(
                                    def.name.to_string(),
                                    PyTypeSpec::Func(PyFuncTypeSpec {
                                        type_params: def.type_params,
                                        args: *def.args,
                                        returns,
                                    }),
                                );
                            }
                            _ => {}
                        }
                    }
                    self.classes.insert(class.name.to_string(), methods);
                }
                _ => {}
            }
        }
    }

    pub fn get_type(&self, name: &str) -> Option<&PyTypeSpec> {
        self.decls.get(name)
    }

    pub fn get_class_member_type(&self, class: &str, name: &str) -> Option<&PyTypeSpec> {
        self.classes
            .get(class)
            .and_then(|methods| methods.get(name))
    }
}

/// AST must be converted in the following order:
///
/// Params -> Block -> Signature
///
/// If you convert it from left to right, for example
///
/// ```python
/// i = 1
/// i = i + 1
/// ```
///
/// would be converted as follows. This is a mistake.
///
/// ```python
/// i = 1
/// i = i2 + 1
/// ```
///
/// The correct conversion is as follows.
///
/// ```python
/// i = 1
/// i2 = i + 1
/// ```
#[derive(Debug)]
pub struct ASTConverter {
    cfg: ErgConfig,
    shadowing: ShadowingMode,
    comments: CommentStorage,
    pyi_types: PyiTypeStorage,
    block_id_counter: usize,
    /// block != scope (if block doesn't make a scope, for example)
    blocks: Vec<BlockInfo>,
    contexts: Vec<LocalContext>,
    warns: CompileErrors,
    errs: CompileErrors,
}

impl ASTConverter {
    pub fn new(cfg: ErgConfig, shadowing: ShadowingMode, comments: CommentStorage) -> Self {
        let mut pyi_types = PyiTypeStorage::new();
        pyi_types.parse(&cfg.input.path().with_extension("pyi").to_string_lossy());
        Self {
            shadowing,
            pyi_types,
            cfg,
            comments,
            block_id_counter: 0,
            blocks: vec![BlockInfo {
                id: 0,
                kind: BlockKind::Module,
            }],
            contexts: vec![LocalContext::new("<module>".into(), BlockKind::Module)],
            warns: CompileErrors::empty(),
            errs: CompileErrors::empty(),
        }
    }

    fn get_name(&self, name: &str) -> Option<&NameInfo> {
        for ctx in self.contexts.iter().rev() {
            if let Some(ni) = ctx.names.get(name) {
                return Some(ni);
            }
        }
        None
    }

    fn get_mut_name(&mut self, name: &str) -> Option<&mut NameInfo> {
        for ctx in self.contexts.iter_mut().rev() {
            if let Some(ni) = ctx.names.get_mut(name) {
                return Some(ni);
            }
        }
        None
    }

    fn get_type_var(&self, name: &str) -> Option<&TypeVarInfo> {
        for ctx in self.contexts.iter().rev() {
            if let Some(tv) = ctx.type_vars.get(name) {
                return Some(tv);
            }
        }
        None
    }

    fn define_name(&mut self, name: String, info: NameInfo) {
        self.contexts.last_mut().unwrap().names.insert(name, info);
    }

    fn declare_name(&mut self, name: String, info: NameInfo) {
        self.contexts.first_mut().unwrap().names.insert(name, info);
    }

    fn define_type_var(&mut self, name: String, info: TypeVarInfo) {
        self.contexts
            .last_mut()
            .unwrap()
            .type_vars
            .insert(name, info);
    }

    fn grow(&mut self, namespace: String, kind: BlockKind) {
        self.contexts.push(LocalContext::new(namespace, kind));
    }

    fn pop(&mut self) {
        self.contexts.pop();
    }

    fn cur_block_kind(&self) -> BlockKind {
        self.blocks.last().unwrap().kind
    }

    fn cur_block_id(&self) -> usize {
        self.blocks.last().unwrap().id
    }

    /// foo.bar.baz
    fn cur_namespace(&self) -> String {
        self.contexts
            .iter()
            .map(|ctx| &ctx.name[..])
            .collect::<Vec<_>>()
            .join(".")
    }

    // baz
    fn cur_name(&self) -> &str {
        &self.contexts.last().unwrap().name
    }

    fn cur_context(&self) -> &LocalContext {
        self.contexts.last().unwrap()
    }

    fn cur_context_mut(&mut self) -> &mut LocalContext {
        self.contexts.last_mut().unwrap()
    }

    fn parent_name(&self) -> &str {
        &self.contexts[self.contexts.len().saturating_sub(2)].name
    }

    fn cur_appeared_type_names(&self) -> &HashSet<String> {
        &self.contexts.last().unwrap().appeared_type_names
    }

    fn register_name_info(&mut self, name: &str, kind: NameKind) -> RenameKind {
        let cur_namespace = self.cur_namespace();
        let cur_block_id = self.cur_block_id();
        let cur_block_kind = self.cur_block_kind();
        if let Some(name_info) = self.get_mut_name(name) {
            if name_info.defined_in == cur_namespace && name_info.defined_block_id == cur_block_id {
                name_info.defined_times += 1;
            }
            if name_info.defined_in.is_unknown() {
                name_info.defined_in = DefinedPlace::Known(cur_namespace);
                name_info.defined_times += 1; // 0 -> 1
            }
            if cur_block_kind
                .if_block_id()
                .is_some_and(|id| id == name_info.defined_block_id)
            {
                RenameKind::Phi
            } else if cur_block_kind.makes_scope()
                || name_info.defined_block_id == cur_block_id
                || name_info.defined_times == 0
            {
                RenameKind::Let
            } else {
                RenameKind::Redef
            }
        } else {
            // In Erg, classes can only be defined in uppercase
            // So if not, prefix it with `Type_`
            let rename = if kind.is_class() && !name.starts_with(char::is_uppercase) {
                Some(format!("Type_{name}"))
            } else {
                None
            };
            let defined_in = DefinedPlace::Known(self.cur_namespace());
            let info = NameInfo::new(rename, defined_in, self.cur_block_id(), 1);
            self.define_name(String::from(name), info);
            RenameKind::Let
        }
    }

    fn convert_ident(&mut self, name: String, loc: PyLocation) -> Identifier {
        let shadowing = self.shadowing;
        let name = escape_name(name);
        let cur_namespace = self.cur_namespace();
        let cur_block_id = self.cur_block_id();
        let cont = if let Some(name_info) = self.get_mut_name(&name) {
            let different_namespace = name_info.defined_in != cur_namespace;
            if cur_block_id != 0 {
                // current is <module>?
                name_info.add_referrer(cur_namespace);
            }
            if different_namespace {
                name
            } else {
                let name = name_info
                    .rename
                    .as_ref()
                    .map_or_else(|| &name, |renamed| renamed);
                if name_info.defined_times > 1 {
                    if shadowing == ShadowingMode::Invisible {
                        // HACK: add zero-width characters as postfix
                        format!("{name}{}", "\0".repeat(name_info.defined_times))
                    } else {
                        format!("{name}__{}", name_info.defined_times)
                    }
                } else {
                    name.clone()
                }
            }
        } else {
            let defined_in = if self.cur_namespace() == "<module>" {
                DefinedPlace::Known(self.cur_namespace())
            } else {
                DefinedPlace::Unknown
            };
            let mut info = NameInfo::new(None, defined_in, cur_block_id, 0);
            info.add_referrer(cur_namespace);
            self.declare_name(name.clone(), info);
            name
        };
        let token = Token::new(
            TokenKind::Symbol,
            cont,
            loc.row.get(),
            loc.column.to_zero_indexed(),
        );
        let name = VarName::new(token);
        let dot = Token::new(
            TokenKind::Dot,
            ".",
            loc.row.get(),
            loc.column.to_zero_indexed(),
        );
        Identifier::new(VisModifierSpec::Public(dot.loc()), name)
    }

    // TODO: module member mangling
    fn convert_attr_ident(&mut self, name: String, loc: PyLocation) -> Identifier {
        let token = Token::new(
            TokenKind::Symbol,
            name,
            loc.row.get(),
            loc.column.to_zero_indexed(),
        );
        let name = VarName::new(token);
        let dot = Token::new(
            TokenKind::Dot,
            ".",
            loc.row.get(),
            loc.column.to_zero_indexed(),
        );
        Identifier::new(VisModifierSpec::Public(dot.loc()), name)
    }

    // Duplicate param names will result in an error at the parser. So we don't need to check it here.
    fn convert_param_pattern(&mut self, arg: String, loc: PyLocation) -> ParamPattern {
        self.register_name_info(&arg, NameKind::Variable);
        let ident = self.convert_ident(arg, loc);
        ParamPattern::VarName(ident.name)
    }

    fn get_cur_scope_t_spec(&self) -> Option<&PyTypeSpec> {
        if self.contexts.len() == 2 {
            let func_name = self.cur_name();
            self.pyi_types.get_type(func_name)
        } else {
            let class = self.parent_name();
            let func_name = self.cur_name();
            self.pyi_types.get_class_member_type(class, func_name)
        }
    }

    fn convert_nd_param(&mut self, param: Arg) -> NonDefaultParamSignature {
        let pat = self.convert_param_pattern(param.arg.to_string(), param.location());
        let t_spec = param
            .annotation
            .or_else(|| {
                let PyTypeSpec::Func(func) = self.get_cur_scope_t_spec()? else {
                    return None;
                };
                func.args
                    .args
                    .iter()
                    .chain(&func.args.kwonlyargs)
                    .find(|arg| arg.def.arg == param.arg)
                    .and_then(|arg| arg.def.annotation.clone())
            })
            .map(|anot| {
                (
                    self.convert_type_spec(*anot.clone()),
                    self.convert_expr(*anot),
                )
            })
            .map(|(t_spec, expr)| {
                let as_op = Token::new(
                    TokenKind::As,
                    "as",
                    t_spec.ln_begin().unwrap_or(0),
                    t_spec.col_begin().unwrap_or(0),
                );
                TypeSpecWithOp::new(as_op, t_spec, expr)
            });
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn convert_default_param(&mut self, kw: Arg, default: py_ast::Expr) -> DefaultParamSignature {
        let sig = self.convert_nd_param(kw);
        let default = self.convert_expr(default);
        DefaultParamSignature::new(sig, default)
    }

    fn convert_params(&mut self, params: Arguments) -> Params {
        #[allow(clippy::type_complexity)]
        fn split_args(
            params: Arguments,
        ) -> (Vec<Arg>, Option<Arg>, Vec<(Arg, py_ast::Expr)>, Option<Arg>) {
            let mut args = Vec::new();
            let mut with_defaults = Vec::new();
            let var_args = params.vararg.map(|x| *x);
            let kw_args = params.kwarg.map(|x| *x);
            for arg in params
                .posonlyargs
                .into_iter()
                .chain(params.args.into_iter())
                .chain(params.kwonlyargs.into_iter())
            {
                if let Some(default) = arg.default {
                    with_defaults.push((arg.def, *default));
                } else {
                    args.push(arg.def);
                }
            }
            (args, var_args, with_defaults, kw_args)
        }
        let (non_defaults, var_args, defaults, kw_args) = split_args(params);
        let non_defaults = non_defaults
            .into_iter()
            .map(|p| self.convert_nd_param(p))
            .collect();
        let var_params = var_args.map(|p| self.convert_nd_param(p));
        let defaults = defaults
            .into_iter()
            .map(|(kw, default)| self.convert_default_param(kw, default))
            .collect();
        let kw_var_params = kw_args.map(|p| self.convert_nd_param(p));
        Params::new(non_defaults, var_params, defaults, kw_var_params, None)
    }

    fn convert_for_param(&mut self, name: String, loc: PyLocation) -> NonDefaultParamSignature {
        let pat = self.convert_param_pattern(name, loc);
        let t_spec = None;
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn param_pattern_to_var(pat: ParamPattern) -> VarPattern {
        match pat {
            ParamPattern::VarName(name) => VarPattern::Ident(Identifier::new(
                VisModifierSpec::Public(ErgLocation::Unknown),
                name,
            )),
            ParamPattern::Discard(token) => VarPattern::Discard(token),
            other => todo!("{other}"),
        }
    }

    /// (i, j) => $1 (i = $1[0]; j = $1[1])
    fn convert_opt_expr_to_param(
        &mut self,
        expr: Option<py_ast::Expr>,
    ) -> (NonDefaultParamSignature, Vec<Expr>) {
        if let Some(expr) = expr {
            self.convert_expr_to_param(expr)
        } else {
            let token = Token::new(TokenKind::UBar, "_", 0, 0);
            (
                NonDefaultParamSignature::new(ParamPattern::Discard(token), None),
                vec![],
            )
        }
    }

    fn convert_expr_to_param(
        &mut self,
        expr: py_ast::Expr,
    ) -> (NonDefaultParamSignature, Vec<Expr>) {
        match expr {
            py_ast::Expr::Name(expr) => (
                self.convert_for_param(expr.id.to_string(), expr.location()),
                vec![],
            ),
            py_ast::Expr::Tuple(expr) => {
                let loc = expr.location();
                let tmp = FRESH_GEN.fresh_varname();
                let tmp_name = VarName::from_str_and_line(tmp, expr.location().row.get());
                let tmp_expr = Expr::Accessor(Accessor::Ident(Identifier::new(
                    VisModifierSpec::Public(ErgLocation::Unknown),
                    tmp_name.clone(),
                )));
                let mut block = vec![];
                for (i, elem) in expr.elts.into_iter().enumerate() {
                    let index = Literal::new(Token::new(
                        TokenKind::NatLit,
                        i.to_string(),
                        elem.location().row.get(),
                        elem.location().column.to_zero_indexed(),
                    ));
                    let (param, mut blocks) = self.convert_expr_to_param(elem);
                    let sig = Signature::Var(VarSignature::new(
                        Self::param_pattern_to_var(param.pat),
                        param.t_spec,
                    ));
                    let method = tmp_expr
                        .clone()
                        .attr_expr(self.convert_ident("__getitem__".to_string(), loc));
                    let tuple_acc = method.call1(Expr::Literal(index));
                    let body = DefBody::new(EQUAL, Block::new(vec![tuple_acc]), DefId(0));
                    let def = Expr::Def(Def::new(sig, body));
                    block.push(def);
                    block.append(&mut blocks);
                }
                let pat = ParamPattern::VarName(tmp_name);
                (NonDefaultParamSignature::new(pat, None), block)
            }
            other => {
                let token = Token::new(
                    TokenKind::UBar,
                    "_",
                    other.location().row.get(),
                    other.location().column.to_zero_indexed(),
                );
                (
                    NonDefaultParamSignature::new(ParamPattern::Discard(token), None),
                    vec![],
                )
            }
        }
    }

    fn convert_for_body(&mut self, lhs: Option<py_ast::Expr>, body: Suite) -> Lambda {
        let (param, block) = self.convert_opt_expr_to_param(lhs);
        let params = Params::new(vec![param], None, vec![], None, None);
        self.block_id_counter += 1;
        self.blocks.push(BlockInfo {
            id: self.block_id_counter,
            kind: BlockKind::For,
        });
        let body = body
            .into_iter()
            .map(|stmt| self.convert_statement(stmt, true))
            .collect::<Vec<_>>();
        self.blocks.pop();
        let body = block.into_iter().chain(body).collect();
        let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
        let op = Token::from_str(TokenKind::FuncArrow, "->");
        Lambda::new(sig, op, Block::new(body), DefId(0))
    }

    fn convert_ident_type_spec(&mut self, name: String, range: PySourceRange) -> TypeSpec {
        let loc = pyloc_to_ergloc(range);
        let global = ConstExpr::Accessor(ConstAccessor::Local(Identifier::private_with_loc(
            "global".into(),
            loc,
        )));
        let obj = ConstExpr::Accessor(ConstAccessor::Local(Identifier::private_with_loc(
            "Obj".into(),
            loc,
        )));
        match &name[..] {
            "dict" => {
                let kv = ConstKeyValue::new(obj.clone(), obj.clone());
                let (l, r) = Self::gen_enclosure_tokens(TokenKind::LSqBr, range);
                let dict = ConstDict::new(l, r, vec![kv]);
                TypeSpec::poly(
                    global.attr(Identifier::private_with_loc("Dict!".into(), loc)),
                    ConstArgs::single(ConstExpr::Dict(dict)),
                )
            }
            "set" => TypeSpec::poly(
                global.attr(Identifier::private_with_loc("Set!".into(), loc)),
                ConstArgs::single(obj),
            ),
            "tuple" => TypeSpec::poly(
                global.attr(Identifier::private_with_loc("HomogenousTuple".into(), loc)),
                ConstArgs::single(obj),
            ),
            // Iterable[T] => Iterable(T), Iterable => Iterable(Obj)
            global_unary_collections!() => TypeSpec::poly(
                global.attr(Identifier::private_with_loc(name.into(), loc)),
                ConstArgs::single(obj),
            ),
            // MutableSequence[T] => Sequence!(T), MutableSequence => Sequence!(Obj)
            global_mutable_unary_collections!() => TypeSpec::poly(
                global.attr(Identifier::private_with_loc(
                    format!("{}!", name.trim_start_matches("Mutable")).into(),
                    loc,
                )),
                ConstArgs::single(obj),
            ),
            // Mapping => Mapping(Obj, Obj)
            global_binary_collections!() => TypeSpec::poly(
                global.attr(Identifier::private_with_loc(name.into(), loc)),
                ConstArgs::pos_only(
                    vec![ConstPosArg::new(obj.clone()), ConstPosArg::new(obj)],
                    None,
                ),
            ),
            _ => TypeSpec::mono(self.convert_ident(name, range.start)),
        }
    }

    fn gen_dummy_type_spec(loc: PyLocation) -> TypeSpec {
        TypeSpec::Infer(Token::new(
            TokenKind::UBar,
            "_",
            loc.row.get(),
            loc.column.to_zero_indexed(),
        ))
    }

    fn convert_const_expr(&mut self, expr: ConstExpr) -> ConstExpr {
        match expr {
            ConstExpr::UnaryOp(un) if un.op.is(TokenKind::Mutate) => *un.expr,
            ConstExpr::App(app)
                if app
                    .attr_name
                    .as_ref()
                    .is_some_and(|n| n.inspect() == "__getitem__") =>
            {
                let obj = self.convert_const_expr(*app.obj);
                let mut args = app.args.map(&mut |arg| self.convert_const_expr(arg));
                if args.pos_args.is_empty() {
                    return ConstExpr::App(ConstApp::new(obj, app.attr_name, args));
                }
                let mut args = match args.pos_args.remove(0).expr {
                    ConstExpr::Tuple(tuple) => tuple.elems,
                    other => {
                        args.pos_args.insert(0, ConstPosArg::new(other));
                        args
                    }
                };
                match obj.local_name() {
                    Some("Union") => {
                        if args.pos_args.len() >= 2 {
                            let first = args.pos_args.remove(0).expr;
                            let or_op = Token::dummy(TokenKind::OrOp, "or");
                            args.pos_args.into_iter().fold(first, |acc, expr| {
                                ConstExpr::BinOp(ConstBinOp::new(or_op.clone(), acc, expr.expr))
                            })
                        } else if args.pos_args.len() == 1 {
                            args.pos_args.remove(0).expr
                        } else {
                            ConstExpr::App(ConstApp::new(obj, app.attr_name, args))
                        }
                    }
                    Some("GenericDict" | "Dict") => {
                        if args.pos_args.len() == 2 {
                            let key = args.pos_args.remove(0).expr;
                            let value = args.pos_args.remove(0).expr;
                            let key_value = ConstKeyValue::new(key, value);
                            ConstExpr::Dict(ConstDict::new(
                                Token::DUMMY,
                                Token::DUMMY,
                                vec![key_value],
                            ))
                        } else {
                            ConstExpr::App(ConstApp::new(obj, app.attr_name, args))
                        }
                    }
                    Some("GenericList" | "List") => {
                        if args.pos_args.len() == 2 {
                            let elem = args.pos_args.remove(0).expr;
                            let len = args.pos_args.remove(0).expr;
                            let l_brace = Token::dummy(TokenKind::LSqBr, "[");
                            let r_brace = Token::dummy(TokenKind::RSqBr, "]");
                            ConstExpr::List(ConstList::WithLength(ConstListWithLength::new(
                                l_brace, r_brace, elem, len,
                            )))
                        } else {
                            let obj = ConstExpr::Accessor(ConstAccessor::Local(
                                Identifier::private("List".into()),
                            ));
                            ConstExpr::App(ConstApp::new(obj, None, args))
                        }
                    }
                    Some("GenericTuple" | "Tuple") => {
                        if args.pos_args.get(1).is_some_and(|arg| matches!(&arg.expr, ConstExpr::Lit(l) if l.is(TokenKind::EllipsisLit))) {
                            let ty = args.pos_args.remove(0).expr;
                            let obj = ConstExpr::Accessor(ConstAccessor::Local(
                                Identifier::private("HomogenousTuple".into()),
                            ));
                            let args = ConstArgs::single(ty);
                            ConstExpr::App(ConstApp::new(obj, None, args))
                        } else {
                            let obj = ConstExpr::Accessor(ConstAccessor::Local(
                                Identifier::private("Tuple".into()),
                            ));
                            let range = ergloc_to_pyloc(args.loc());
                            let (l, r) = Self::gen_enclosure_tokens(TokenKind::LSqBr, range);
                            let list = ConstList::Normal(ConstNormalList::new(l, r, args, None));
                            let args = ConstArgs::single(ConstExpr::List(list));
                            ConstExpr::App(ConstApp::new(obj, None, args))
                        }
                    }
                    Some("Optional") => {
                        let arg = args.pos_args.remove(0).expr;
                        let none = ConstExpr::Accessor(ConstAccessor::Local(Identifier::private(
                            "NoneType".into(),
                        )));
                        let or_op = Token::dummy(TokenKind::OrOp, "or");
                        ConstExpr::BinOp(ConstBinOp::new(or_op, arg, none))
                    }
                    Some("Literal") => {
                        let set = ConstNormalSet::new(Token::DUMMY, Token::DUMMY, args);
                        ConstExpr::Set(ConstSet::Normal(set))
                    }
                    Some("Callable") => {
                        let params = if args.pos_args.is_empty() {
                            self.errs.push(CompileError::syntax_error(
                                self.cfg.input.clone(),
                                line!() as usize,
                                args.loc(),
                                self.cur_namespace(),
                                "`Callable` takes an input type list and a return type".into(),
                                None,
                            ));
                            ConstArgs::empty()
                        } else {
                            match args.pos_args.remove(0).expr {
                                ConstExpr::List(ConstList::Normal(list)) => list.elems,
                                other => {
                                    args.pos_args.insert(0, ConstPosArg::new(other));
                                    args.clone()
                                }
                            }
                        };
                        let non_defaults = params
                            .pos_args
                            .into_iter()
                            .map(|param| {
                                let expr = match param.expr.downgrade() {
                                    Expr::Literal(lit) if lit.is(TokenKind::NoneLit) => {
                                        Expr::Accessor(Accessor::Ident(Identifier::private(
                                            "NoneType".into(),
                                        )))
                                    }
                                    other => other,
                                };
                                let ty = Parser::expr_to_type_spec(expr.clone())
                                    .unwrap_or(TypeSpec::mono(Identifier::private("Any".into())));
                                let discard = Token::dummy(TokenKind::UBar, "_");
                                let t_spec = TypeSpecWithOp::new(
                                    Token::dummy(TokenKind::Colon, ":"),
                                    ty,
                                    expr,
                                );
                                NonDefaultParamSignature::new(
                                    ParamPattern::Discard(discard),
                                    Some(t_spec),
                                )
                            })
                            .collect();
                        let params = Params::new(non_defaults, None, vec![], None, None);
                        let ret = if args.pos_args.is_empty() {
                            self.errs.push(CompileError::syntax_error(
                                self.cfg.input.clone(),
                                line!() as usize,
                                args.loc(),
                                self.cur_namespace(),
                                "Expected a return type".into(),
                                None,
                            ));
                            ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Any".into())))
                        } else {
                            match args.pos_args.remove(0).expr {
                                ConstExpr::Lit(lit) if lit.is(TokenKind::NoneLit) => {
                                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private(
                                        "NoneType".into(),
                                    )))
                                }
                                other => other,
                            }
                        };
                        let op = Token::dummy(TokenKind::ProcArrow, "=>");
                        let body = ConstBlock::new(vec![ret]);
                        let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                        ConstExpr::Lambda(ConstLambda::new(sig, op, body, DefId(0)))
                    }
                    _ => ConstExpr::App(ConstApp::new(obj, app.attr_name, args)),
                }
            }
            _ => expr.map(&mut |expr| self.convert_const_expr(expr)),
        }
    }

    fn convert_expr_to_const(&mut self, expr: py_ast::Expr) -> Option<ConstExpr> {
        let expr = self.convert_expr(expr);
        match Parser::validate_const_expr(expr) {
            Ok(expr) => Some(self.convert_const_expr(expr)),
            Err(err) => {
                let err =
                    CompileError::new(err.into(), self.cfg.input.clone(), self.cur_namespace());
                self.errs.push(err);
                None
            }
        }
    }

    // TODO:
    fn convert_compound_type_spec(&mut self, name: String, args: py_ast::Expr) -> TypeSpec {
        match &name[..] {
            "Union" => {
                let py_ast::Expr::Tuple(mut tuple) = args else {
                    let err = CompileError::syntax_error(
                        self.cfg.input.clone(),
                        line!() as usize,
                        pyloc_to_ergloc(args.range()),
                        self.cur_namespace(),
                        "`Union` takes at least 2 types".into(),
                        None,
                    );
                    self.errs.push(err);
                    return Self::gen_dummy_type_spec(args.location());
                };
                let lhs = self.convert_type_spec(tuple.elts.remove(0));
                if tuple.elts.is_empty() {
                    return lhs;
                }
                let rhs = self.convert_type_spec(tuple.elts.remove(0));
                let mut union = TypeSpec::or(lhs, rhs);
                for elem in tuple.elts {
                    let t = self.convert_type_spec(elem);
                    union = TypeSpec::or(union, t);
                }
                union
            }
            "Optional" => {
                let loc = args.location();
                let t = self.convert_type_spec(args);
                let ident = Identifier::private_with_line("NoneType".into(), loc.row.get());
                let none = TypeSpec::mono(ident);
                TypeSpec::or(t, none)
            }
            "Literal" => {
                let py_ast::Expr::Tuple(tuple) = args else {
                    return Self::gen_dummy_type_spec(args.location());
                };
                let mut elems = vec![];
                for elem in tuple.elts {
                    if let Some(expr) = self.convert_expr_to_const(elem) {
                        elems.push(ConstPosArg::new(expr));
                    }
                }
                let elems = ConstArgs::new(elems, None, vec![], None, None);
                TypeSpec::Enum(elems)
            }
            // TODO: distinguish from collections.abc.Callable
            "Callable" => {
                let mut tuple = match args {
                    py_ast::Expr::Tuple(tuple) if tuple.elts.len() == 2 => tuple,
                    _ => {
                        let err = CompileError::syntax_error(
                            self.cfg.input.clone(),
                            line!() as usize,
                            pyloc_to_ergloc(args.range()),
                            self.cur_namespace(),
                            "`Callable` takes an input type list and a return type".into(),
                            None,
                        );
                        self.errs.push(err);
                        return Self::gen_dummy_type_spec(args.location());
                    }
                };
                let params = tuple.elts.remove(0);
                let mut non_defaults = vec![];
                match params {
                    py_ast::Expr::List(list) => {
                        for param in list.elts.into_iter() {
                            let t_spec = self.convert_type_spec(param);
                            non_defaults.push(ParamTySpec::anonymous(t_spec));
                        }
                    }
                    other => {
                        let err = CompileError::syntax_error(
                            self.cfg.input.clone(),
                            line!() as usize,
                            pyloc_to_ergloc(other.range()),
                            self.cur_namespace(),
                            "Expected a list of parameters".into(),
                            None,
                        );
                        self.errs.push(err);
                    }
                }
                let ret = self.convert_type_spec(tuple.elts.remove(0));
                TypeSpec::Subr(SubrTypeSpec::new(
                    TypeBoundSpecs::empty(),
                    None,
                    non_defaults,
                    None,
                    vec![],
                    None,
                    ARROW,
                    ret,
                ))
            }
            "Iterable" | "Iterator" | "Collection" | "Container" | "Sequence"
            | "MutableSequence" => {
                let elem_t = match self.convert_expr_to_const(args) {
                    Some(elem_t) => elem_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let elem_t = ConstPosArg::new(elem_t);
                let global =
                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("global".into())));
                let acc = ConstAccessor::Attr(ConstAttribute::new(
                    global,
                    Identifier::private(escape_name(name).into()),
                ));
                TypeSpec::poly(acc, ConstArgs::pos_only(vec![elem_t], None))
            }
            "Mapping" | "MutableMapping" => {
                let mut tuple = match args {
                    py_ast::Expr::Tuple(tuple) if tuple.elts.len() == 2 => tuple,
                    _ => {
                        let err = CompileError::syntax_error(
                            self.cfg.input.clone(),
                            line!() as usize,
                            pyloc_to_ergloc(args.range()),
                            self.cur_namespace(),
                            "`Mapping` takes 2 types".into(),
                            None,
                        );
                        self.errs.push(err);
                        return Self::gen_dummy_type_spec(args.location());
                    }
                };
                let key_t = match self.convert_expr_to_const(tuple.elts.remove(0)) {
                    Some(key_t) => key_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let key_t = ConstPosArg::new(key_t);
                let value_t = match self.convert_expr_to_const(tuple.elts.remove(0)) {
                    Some(value_t) => value_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let value_t = ConstPosArg::new(value_t);
                let global =
                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("global".into())));
                let acc = ConstAccessor::Attr(ConstAttribute::new(
                    global,
                    Identifier::private(escape_name(name).into()),
                ));
                TypeSpec::poly(acc, ConstArgs::pos_only(vec![key_t, value_t], None))
            }
            "List" | "list" => {
                let len = ConstExpr::Accessor(ConstAccessor::Local(Identifier::private_with_loc(
                    "_".into(),
                    pyloc_to_ergloc(args.range()),
                )));
                let elem_t = match self.convert_expr_to_const(args) {
                    Some(elem_t) => elem_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let elem_t = ConstPosArg::new(elem_t);
                let len = ConstPosArg::new(len);
                let global =
                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("global".into())));
                let acc = ConstAccessor::Attr(ConstAttribute::new(
                    global,
                    Identifier::private("List!".into()),
                ));
                TypeSpec::poly(
                    acc,
                    ConstArgs::new(vec![elem_t, len], None, vec![], None, None),
                )
            }
            "Dict" | "dict" => {
                let mut tuple = match args {
                    py_ast::Expr::Tuple(tuple) if tuple.elts.len() == 2 => tuple,
                    _ => {
                        let err = CompileError::syntax_error(
                            self.cfg.input.clone(),
                            line!() as usize,
                            pyloc_to_ergloc(args.range()),
                            self.cur_namespace(),
                            "`dict` takes 2 types".into(),
                            None,
                        );
                        self.errs.push(err);
                        return Self::gen_dummy_type_spec(args.location());
                    }
                };
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, tuple.range);
                let key_t = match self.convert_expr_to_const(tuple.elts.remove(0)) {
                    Some(key_t) => key_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let val_t = match self.convert_expr_to_const(tuple.elts.remove(0)) {
                    Some(val_t) => val_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let dict = ConstPosArg::new(ConstExpr::Dict(ConstDict::new(
                    l_brace,
                    r_brace,
                    vec![ConstKeyValue::new(key_t, val_t)],
                )));
                let global =
                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("global".into())));
                let acc = ConstAccessor::Attr(ConstAttribute::new(
                    global,
                    Identifier::private("Dict!".into()),
                ));
                TypeSpec::poly(acc, ConstArgs::new(vec![dict], None, vec![], None, None))
            }
            "Tuple" | "tuple" => {
                let py_ast::Expr::Tuple(mut tuple) = args else {
                    return Self::gen_dummy_type_spec(args.location());
                };
                // tuple[T, ...] == HomogenousTuple T
                if tuple.elts.get(1).is_some_and(|ex| matches!(ex, py_ast::Expr::Constant(c) if matches!(c.value, py_ast::Constant::Ellipsis))) {
                    let acc = ConstAccessor::local(Token::symbol("HomogenousTuple"));
                    let ty = tuple.elts.remove(0);
                    let args = ConstArgs::single(self.convert_expr_to_const(ty).unwrap());
                    return TypeSpec::poly(acc, args);
                }
                let tys = tuple
                    .elts
                    .into_iter()
                    .map(|elem| self.convert_type_spec(elem))
                    .collect();
                let (l, r) = Self::gen_enclosure_tokens(TokenKind::LParen, tuple.range);
                let tuple = TupleTypeSpec::new(Some((l.loc(), r.loc())), tys);
                TypeSpec::Tuple(tuple)
            }
            "Set" | "set" => {
                let elem_t = match self.convert_expr_to_const(args) {
                    Some(elem_t) => elem_t,
                    None => {
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let elem_t = ConstPosArg::new(elem_t);
                let global =
                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("global".into())));
                let acc = ConstAccessor::Attr(ConstAttribute::new(
                    global,
                    Identifier::private("Set!".into()),
                ));
                TypeSpec::poly(acc, ConstArgs::pos_only(vec![elem_t], None))
            }
            _ => Self::gen_dummy_type_spec(args.location()),
        }
    }

    fn convert_type_spec(&mut self, expr: py_ast::Expr) -> TypeSpec {
        #[allow(clippy::collapsible_match)]
        match expr {
            py_ast::Expr::Name(name) => {
                self.contexts
                    .last_mut()
                    .unwrap()
                    .appeared_type_names
                    .insert(name.id.to_string());
                self.convert_ident_type_spec(name.id.to_string(), name.range)
            }
            py_ast::Expr::Constant(cons) => {
                if cons.value.is_none() {
                    self.convert_ident_type_spec("NoneType".into(), cons.range)
                } else if let Some(name) = cons.value.as_str() {
                    self.convert_ident_type_spec(name.into(), cons.range)
                } else {
                    let err = CompileError::syntax_error(
                        self.cfg.input.clone(),
                        line!() as usize,
                        pyloc_to_ergloc(cons.range()),
                        self.cur_namespace(),
                        format!("{:?} is not a type", cons.value),
                        None,
                    );
                    self.errs.push(err);
                    Self::gen_dummy_type_spec(cons.location())
                }
            }
            py_ast::Expr::Attribute(attr) => {
                let namespace = Box::new(self.convert_expr(*attr.value));
                let t = self.convert_ident(attr.attr.to_string(), attr_name_loc(&namespace));
                if namespace
                    .full_name()
                    .is_some_and(|n| n == "typing" || n == "collections.abc")
                {
                    match &t.inspect()[..] {
                        global_unary_collections!()
                        | global_mutable_unary_collections!()
                        | global_binary_collections!() => {
                            return self.convert_ident_type_spec(attr.attr.to_string(), attr.range)
                        }
                        "Any" => return TypeSpec::PreDeclTy(PreDeclTypeSpec::Mono(t)),
                        _ => {}
                    }
                }
                let predecl = PreDeclTypeSpec::Attr { namespace, t };
                TypeSpec::PreDeclTy(predecl)
            }
            // value[slice]
            py_ast::Expr::Subscript(subs) => match *subs.value {
                py_ast::Expr::Name(name) => {
                    self.convert_compound_type_spec(name.id.to_string(), *subs.slice)
                }
                py_ast::Expr::Attribute(attr) => {
                    let loc = attr.location();
                    match accessor_name(*attr.value).as_ref().map(|s| &s[..]) {
                        Some("typing" | "collections.abc") => {
                            self.convert_compound_type_spec(attr.attr.to_string(), *subs.slice)
                        }
                        other => {
                            log!(err "unknown: {other:?}");
                            Self::gen_dummy_type_spec(loc)
                        }
                    }
                }
                other => {
                    log!(err "unknown: {other:?}");
                    Self::gen_dummy_type_spec(other.location())
                }
            },
            py_ast::Expr::BinOp(bin) => {
                let loc = bin.location();
                match bin.op {
                    // A | B
                    Operator::BitOr => {
                        let lhs = self.convert_type_spec(*bin.left);
                        let rhs = self.convert_type_spec(*bin.right);
                        TypeSpec::or(lhs, rhs)
                    }
                    _ => Self::gen_dummy_type_spec(loc),
                }
            }
            other => {
                log!(err "unknown: {other:?}");
                Self::gen_dummy_type_spec(other.location())
            }
        }
    }

    fn gen_enclosure_tokens(l_kind: TokenKind, expr_range: PySourceRange) -> (Token, Token) {
        let (l_cont, r_cont, r_kind) = match l_kind {
            TokenKind::LBrace => ("{", "}", TokenKind::RBrace),
            TokenKind::LParen => ("(", ")", TokenKind::RParen),
            TokenKind::LSqBr => ("[", "]", TokenKind::RSqBr),
            _ => unreachable!(),
        };
        let (line_end, c_end) = (
            expr_range.end.unwrap_or(expr_range.start).row.get(),
            expr_range
                .end
                .unwrap_or(expr_range.start)
                .column
                .to_zero_indexed(),
        );
        let l_brace = Token::new(
            l_kind,
            l_cont,
            expr_range.start.row.get(),
            expr_range.start.column.to_zero_indexed(),
        );
        let r_brace = Token::new(r_kind, r_cont, line_end, c_end);
        (l_brace, r_brace)
    }

    fn mutate_expr(expr: Expr) -> Expr {
        let mut_op = Token::new(
            TokenKind::Mutate,
            "!",
            expr.ln_begin().unwrap_or(0),
            expr.col_begin().unwrap_or(0),
        );
        Expr::UnaryOp(UnaryOp::new(mut_op, expr))
    }

    fn convert_const(&mut self, const_: ExprConstant) -> Expr {
        let loc = const_.location();
        match const_.value {
            py_ast::Constant::Int(i) => {
                let kind = if i >= 0.into() {
                    TokenKind::NatLit
                } else {
                    TokenKind::IntLit
                };
                let token = Token::new(
                    kind,
                    i.to_string(),
                    loc.row.get(),
                    loc.column.to_zero_indexed(),
                );
                Expr::Literal(Literal::new(token))
            }
            py_ast::Constant::Float(f) => {
                let token = Token::new(
                    TokenKind::RatioLit,
                    f.to_string(),
                    const_.location().row.get(),
                    const_.location().column.to_zero_indexed(),
                );
                Expr::Literal(Literal::new(token))
            }
            py_ast::Constant::Complex { real: _, imag: _ } => Expr::Dummy(Dummy::new(None, vec![])),
            py_ast::Constant::Str(value) => {
                let kind = if const_
                    .range
                    .end
                    .is_some_and(|end| end.row != const_.range.start.row)
                {
                    TokenKind::DocComment
                } else {
                    TokenKind::StrLit
                };
                let value = format!("\"{value}\"");
                // column - 2 because of the quotes
                let token = Token::new(kind, value, loc.row.get(), loc.column.to_zero_indexed());
                Expr::Literal(Literal::new(token))
            }
            py_ast::Constant::Bool(b) => {
                let cont = if b { "True" } else { "False" };
                Expr::Literal(Literal::new(Token::new(
                    TokenKind::BoolLit,
                    cont,
                    loc.row.get(),
                    loc.column.to_zero_indexed(),
                )))
            }
            py_ast::Constant::None => Expr::Literal(Literal::new(Token::new(
                TokenKind::NoneLit,
                "None",
                const_.location().row.get(),
                const_.location().column.to_zero_indexed(),
            ))),
            py_ast::Constant::Ellipsis => Expr::Literal(Literal::new(Token::new(
                TokenKind::EllipsisLit,
                "...",
                const_.location().row.get(),
                const_.location().column.to_zero_indexed(),
            ))),
            // Bytes, Tuple
            other => {
                log!(err "unknown: {other:?}");
                Expr::Dummy(Dummy::new(None, vec![]))
            }
        }
    }

    fn convert_expr(&mut self, expr: py_ast::Expr) -> Expr {
        match expr {
            py_ast::Expr::Constant(const_) => self.convert_const(const_),
            py_ast::Expr::Name(name) => {
                let ident = self.convert_ident(name.id.to_string(), name.location());
                Expr::Accessor(Accessor::Ident(ident))
            }
            py_ast::Expr::Attribute(attr) => {
                let value = self.convert_expr(*attr.value);
                let name = self.convert_attr_ident(attr.attr.to_string(), attr_name_loc(&value));
                value.attr_expr(name)
            }
            py_ast::Expr::IfExp(if_) => {
                let loc = if_.location();
                let block = self.convert_expr(*if_.body);
                let params = Params::new(vec![], None, vec![], None, None);
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, Block::new(vec![block]), DefId(0));
                let test = self.convert_expr(*if_.test);
                let if_ident = self.convert_ident("if".to_string(), loc);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                let else_block = self.convert_expr(*if_.orelse);
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let else_body =
                    Lambda::new(sig, Token::DUMMY, Block::new(vec![else_block]), DefId(0));
                let args = Args::pos_only(
                    vec![
                        PosArg::new(test),
                        PosArg::new(Expr::Lambda(body)),
                        PosArg::new(Expr::Lambda(else_body)),
                    ],
                    None,
                );
                if_acc.call_expr(args)
            }
            py_ast::Expr::Call(call) => {
                let loc = call.location();
                let end_loc = call.end_location();
                let function = self.convert_expr(*call.func);
                let (pos_args, var_args): (Vec<_>, _) = call
                    .args
                    .into_iter()
                    .partition(|arg| !arg.is_starred_expr());
                let pos_args = pos_args
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let var_args = var_args
                    .into_iter()
                    .map(|ex| {
                        let py_ast::Expr::Starred(star) = ex else {
                            unreachable!()
                        };
                        PosArg::new(self.convert_expr(*star.value))
                    })
                    .next();
                let (kw_args, kw_var): (Vec<_>, _) =
                    call.keywords.into_iter().partition(|kw| kw.arg.is_some());
                let kw_args = kw_args
                    .into_iter()
                    .map(|Keyword { arg, value, range }| {
                        let name = Token::symbol_with_loc(
                            arg.unwrap().to_string(),
                            pyloc_to_ergloc(range),
                        );
                        let ex = self.convert_expr(value);
                        KwArg::new(name, None, ex)
                    })
                    .collect();
                let kw_var = kw_var
                    .into_iter()
                    .map(|Keyword { value, .. }| PosArg::new(self.convert_expr(value)))
                    .next();
                let last_col = end_loc.map_or_else(
                    || {
                        pos_args
                            .last()
                            .and_then(|last| last.col_end())
                            .unwrap_or(function.col_end().unwrap_or(0) + 1)
                    },
                    |loc| loc.column.to_zero_indexed().saturating_sub(1),
                );
                let paren = {
                    let lp = Token::new(
                        TokenKind::LParen,
                        "(",
                        loc.row.get(),
                        function.col_end().unwrap_or(0),
                    );
                    let rp = Token::new(TokenKind::RParen, ")", loc.row.get(), last_col);
                    (lp.loc(), rp.loc())
                };
                let args = Args::new(pos_args, var_args, kw_args, kw_var, Some(paren));
                function.call_expr(args)
            }
            py_ast::Expr::BinOp(bin) => {
                let lhs = self.convert_expr(*bin.left);
                let rhs = self.convert_expr(*bin.right);
                let op = op_to_token(bin.op);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            py_ast::Expr::UnaryOp(un) => {
                let rhs = self.convert_expr(*un.operand);
                let (kind, cont) = match un.op {
                    UnOp::UAdd => (TokenKind::PrePlus, "+"),
                    // UnOp::Not => (TokenKind::PreBitNot, "not"),
                    UnOp::USub => (TokenKind::PreMinus, "-"),
                    UnOp::Invert => (TokenKind::PreBitNot, "~"),
                    _ => return Expr::Dummy(Dummy::new(None, vec![rhs])),
                };
                let op = Token::from_str(kind, cont);
                Expr::UnaryOp(UnaryOp::new(op, rhs))
            }
            // TODO
            py_ast::Expr::BoolOp(mut boole) => {
                let lhs = self.convert_expr(boole.values.remove(0));
                let rhs = self.convert_expr(boole.values.remove(0));
                let (kind, cont) = match boole.op {
                    BoolOp::And => (TokenKind::AndOp, "and"),
                    BoolOp::Or => (TokenKind::OrOp, "or"),
                };
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            // TODO: multiple CmpOps
            py_ast::Expr::Compare(mut cmp) => {
                let lhs = self.convert_expr(*cmp.left);
                let rhs = self.convert_expr(cmp.comparators.remove(0));
                let (kind, cont) = match cmp.ops.remove(0) {
                    CmpOp::Eq => (TokenKind::DblEq, "=="),
                    CmpOp::NotEq => (TokenKind::NotEq, "!="),
                    CmpOp::Lt => (TokenKind::Less, "<"),
                    CmpOp::LtE => (TokenKind::LessEq, "<="),
                    CmpOp::Gt => (TokenKind::Gre, ">"),
                    CmpOp::GtE => (TokenKind::GreEq, ">="),
                    CmpOp::Is => (TokenKind::IsOp, "is!"),
                    CmpOp::IsNot => (TokenKind::IsNotOp, "isnot!"),
                    CmpOp::In => (TokenKind::InOp, "in"),
                    CmpOp::NotIn => (TokenKind::NotInOp, "notin"),
                };
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            py_ast::Expr::Lambda(lambda) => {
                self.grow("<lambda>".to_string(), BlockKind::Function);
                let params = self.convert_params(*lambda.args);
                let body = vec![self.convert_expr(*lambda.body)];
                self.pop();
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let op = Token::from_str(TokenKind::FuncArrow, "->");
                Expr::Lambda(Lambda::new(sig, op, Block::new(body), DefId(0)))
            }
            py_ast::Expr::List(list) => {
                let (l_sqbr, r_sqbr) = Self::gen_enclosure_tokens(TokenKind::LSqBr, list.range);
                let elements = list
                    .elts
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::pos_only(elements, None);
                let arr = Expr::List(List::Normal(NormalList::new(l_sqbr, r_sqbr, elems)));
                Self::mutate_expr(arr)
            }
            py_ast::Expr::ListComp(comp) => {
                let (l_sqbr, r_sqbr) = Self::gen_enclosure_tokens(TokenKind::LSqBr, comp.range);
                let layout = self.convert_expr(*comp.elt);
                let generator = comp.generators.into_iter().next().unwrap();
                let target = self.convert_expr(generator.target);
                let Expr::Accessor(Accessor::Ident(ident)) = target else {
                    log!(err "unimplemented: {target}");
                    let loc = pyloc_to_ergloc(comp.range);
                    return Expr::Dummy(Dummy::new(Some(loc), vec![]));
                };
                let iter = self.convert_expr(generator.iter);
                let guard = generator
                    .ifs
                    .into_iter()
                    .next()
                    .map(|ex| self.convert_expr(ex));
                let generators = vec![(ident, iter)];
                let arr = Expr::List(List::Comprehension(ListComprehension::new(
                    l_sqbr,
                    r_sqbr,
                    Some(layout),
                    generators,
                    guard,
                )));
                Self::mutate_expr(arr)
            }
            py_ast::Expr::Set(set) => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, set.range);
                let elements = set
                    .elts
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::pos_only(elements, None);
                let set = Expr::Set(Set::Normal(NormalSet::new(l_brace, r_brace, elems)));
                Self::mutate_expr(set)
            }
            py_ast::Expr::SetComp(comp) => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, comp.range);
                let layout = self.convert_expr(*comp.elt);
                let generator = comp.generators.into_iter().next().unwrap();
                let target = self.convert_expr(generator.target);
                let Expr::Accessor(Accessor::Ident(ident)) = target else {
                    log!(err "unimplemented: {target}");
                    let loc = pyloc_to_ergloc(comp.range);
                    return Expr::Dummy(Dummy::new(Some(loc), vec![]));
                };
                let iter = self.convert_expr(generator.iter);
                let guard = generator
                    .ifs
                    .into_iter()
                    .next()
                    .map(|ex| self.convert_expr(ex));
                let generators = vec![(ident, iter)];
                let set = Expr::Set(Set::Comprehension(SetComprehension::new(
                    l_brace,
                    r_brace,
                    Some(layout),
                    generators,
                    guard,
                )));
                Self::mutate_expr(set)
            }
            py_ast::Expr::Dict(dict) => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, dict.range);
                let kvs = dict
                    .keys
                    .into_iter()
                    .zip(dict.values)
                    .map(|(k, v)| {
                        KeyValue::new(
                            k.map(|k| self.convert_expr(k))
                                .unwrap_or(Expr::Dummy(Dummy::new(None, vec![]))),
                            self.convert_expr(v),
                        )
                    })
                    .collect::<Vec<_>>();
                let dict = Expr::Dict(Dict::Normal(NormalDict::new(l_brace, r_brace, kvs)));
                Self::mutate_expr(dict)
            }
            py_ast::Expr::DictComp(comp) => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, comp.range);
                let key = self.convert_expr(*comp.key);
                let value = self.convert_expr(*comp.value);
                let kv = KeyValue::new(key, value);
                let generator = comp.generators.into_iter().next().unwrap();
                let target = self.convert_expr(generator.target);
                let Expr::Accessor(Accessor::Ident(ident)) = target else {
                    log!(err "unimplemented: {target}");
                    let loc = pyloc_to_ergloc(comp.range);
                    return Expr::Dummy(Dummy::new(Some(loc), vec![]));
                };
                let iter = self.convert_expr(generator.iter);
                let guard = generator
                    .ifs
                    .into_iter()
                    .next()
                    .map(|ex| self.convert_expr(ex));
                let generators = vec![(ident, iter)];
                let dict = Expr::Dict(Dict::Comprehension(DictComprehension::new(
                    l_brace, r_brace, kv, generators, guard,
                )));
                Self::mutate_expr(dict)
            }
            py_ast::Expr::Tuple(tuple) => {
                let (l, r) = Self::gen_enclosure_tokens(TokenKind::LParen, tuple.range);
                let elements = tuple
                    .elts
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::pos_only(elements, Some((l.loc(), r.loc())));
                Expr::Tuple(Tuple::Normal(NormalTuple::new(elems)))
            }
            py_ast::Expr::Subscript(subs) => {
                let obj = self.convert_expr(*subs.value);
                let method = obj.attr_expr(
                    self.convert_ident("__getitem__".to_string(), subs.slice.location()),
                );
                method.call1(self.convert_expr(*subs.slice))
            }
            // [:] == [slice(None)]
            // [start:] == [slice(start, None)]
            // [:stop] == [slice(stop)]
            // [start:stop] == [slice(start, stop)]
            // [start:stop:step] == [slice(start, stop, step)]
            py_ast::Expr::Slice(slice) => {
                let loc = slice.location();
                let start = slice.lower.map(|ex| self.convert_expr(*ex));
                let stop = slice.upper.map(|ex| self.convert_expr(*ex));
                let step = slice.step.map(|ex| self.convert_expr(*ex));
                let mut args = Args::empty();
                if let Some(start) = start {
                    args.push_pos(PosArg::new(start));
                }
                if let Some(stop) = stop {
                    args.push_pos(PosArg::new(stop));
                }
                if let Some(step) = step {
                    args.push_pos(PosArg::new(step));
                }
                if args.is_empty() {
                    args.push_pos(PosArg::new(Expr::Literal(Literal::new(Token::new(
                        TokenKind::NoneLit,
                        "None",
                        loc.row.get(),
                        loc.column.to_zero_indexed(),
                    )))));
                }
                let slice = self.convert_ident("slice".to_string(), loc);
                slice.call(args).into()
            }
            py_ast::Expr::JoinedStr(string) => {
                if string.values.is_empty() {
                    let loc = string.location();
                    let stringify = self.convert_ident("str".to_string(), loc);
                    return stringify.call(Args::empty()).into();
                } else if string.values.len() == 1 {
                    let loc = string.location();
                    let mut values = string.values;
                    let expr = self.convert_expr(values.remove(0));
                    let stringify = self.convert_ident("str".to_string(), loc);
                    return stringify.call1(expr).into();
                }
                let mut values = vec![];
                for value in string.values {
                    match value {
                        py_ast::Expr::Constant(cons) => {
                            let cons = self.convert_const(cons);
                            values.push(cons);
                        }
                        py_ast::Expr::FormattedValue(form) => {
                            let loc = form.location();
                            let expr = self.convert_expr(*form.value);
                            let stringify = self.convert_ident("str".to_string(), loc);
                            values.push(stringify.call1(expr).into());
                        }
                        _ => {}
                    }
                }
                let fst = values.remove(0);
                values.into_iter().fold(fst, |acc, expr| {
                    let plus = Token::dummy(TokenKind::Plus, "+");
                    Expr::BinOp(BinOp::new(plus, acc, expr))
                })
            }
            py_ast::Expr::FormattedValue(form) => {
                let loc = form.location();
                let expr = self.convert_expr(*form.value);
                let stringify = self.convert_ident("str".to_string(), loc);
                stringify.call1(expr).into()
            }
            py_ast::Expr::NamedExpr(named) => {
                let loc = named.location();
                let target = self.convert_expr(*named.target);
                let target_pat = match &target {
                    Expr::Accessor(Accessor::Ident(ident)) => VarPattern::Ident(ident.clone()),
                    _ => {
                        log!(err "unimplemented: {:?}", target);
                        VarPattern::Ident(Identifier::private("_".into()))
                    }
                };
                let value = self.convert_expr(*named.value);
                let assign = Token::new(
                    TokenKind::Assign,
                    "=",
                    loc.row.get(),
                    loc.column.to_zero_indexed(),
                );
                let def = Def::new(
                    Signature::Var(VarSignature::new(target_pat, None)),
                    DefBody::new(assign, Block::new(vec![value]), DefId(0)),
                );
                Expr::Compound(Compound::new(vec![Expr::Def(def), target]))
            }
            py_ast::Expr::Yield(_) => {
                self.cur_context_mut().return_kind = ReturnKind::Yield;
                log!(err "unimplemented: {:?}", expr);
                Expr::Dummy(Dummy::new(None, vec![]))
            }
            _other => {
                log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::new(None, vec![]))
            }
        }
    }

    fn convert_block(&mut self, block: Suite, kind: BlockKind) -> Block {
        let mut new_block = Vec::new();
        let len = block.len();
        self.block_id_counter += 1;
        self.blocks.push(BlockInfo {
            id: self.block_id_counter,
            kind,
        });
        for (i, stmt) in block.into_iter().enumerate() {
            let is_last = i == len - 1;
            new_block.push(self.convert_statement(stmt, is_last && kind.is_function()));
        }
        self.blocks.pop();
        Block::new(new_block)
    }

    fn check_init_sig(&mut self, sig: &Signature) -> Option<()> {
        match sig {
            Signature::Subr(subr) => {
                if let Some(first) = subr.params.non_defaults.first() {
                    if first.inspect().map(|s| &s[..]) == Some("self") {
                        return Some(());
                    }
                }
                self.errs.push(self_not_found_error(
                    self.cfg.input.clone(),
                    subr.loc(),
                    self.cur_namespace(),
                ));
                Some(())
            }
            Signature::Var(var) => {
                self.errs.push(init_var_error(
                    self.cfg.input.clone(),
                    var.loc(),
                    self.cur_namespace(),
                ));
                None
            }
        }
    }

    // def __init__(self, x: Int, y: Int, z):
    //     self.x = x
    //     self.y = y
    //     self.z = z
    // ↓
    // requirement : {x: Int, y: Int, z: Any}
    // returns     : .__call__(x: Int, y: Int, z: Obj): Self = .unreachable()
    fn extract_init(&mut self, base_type: &mut Option<Expr>, init_def: Def) -> Option<Def> {
        self.check_init_sig(&init_def.sig)?;
        let l_brace = Token::new(
            TokenKind::LBrace,
            "{",
            init_def.ln_begin().unwrap_or(0),
            init_def.col_begin().unwrap_or(0),
        );
        let r_brace = Token::new(
            TokenKind::RBrace,
            "}",
            init_def.ln_end().unwrap_or(0),
            init_def.col_end().unwrap_or(0),
        );
        let Signature::Subr(sig) = init_def.sig else {
            unreachable!()
        };
        let mut fields = vec![];
        for chunk in init_def.body.block {
            #[allow(clippy::single_match)]
            match chunk {
                Expr::ReDef(redef) => {
                    let Accessor::Attr(attr) = redef.attr else {
                        continue;
                    };
                    // if `self.foo == ...`
                    if attr.obj.get_name().map(|s| &s[..]) == Some("self") {
                        // get attribute types
                        let typ = if let Some(t_spec_op) = sig
                            .params
                            .non_defaults
                            .iter()
                            .find(|&param| param.inspect() == Some(attr.ident.inspect()))
                            .and_then(|param| param.t_spec.as_ref())
                            .or_else(|| {
                                sig.params
                                    .defaults
                                    .iter()
                                    .find(|&param| param.inspect() == Some(attr.ident.inspect()))
                                    .and_then(|param| param.sig.t_spec.as_ref())
                            }) {
                            *t_spec_op.t_spec_as_expr.clone()
                        } else if let Some(typ) = redef.t_spec.map(|t_spec| t_spec.t_spec_as_expr) {
                            *typ
                        } else {
                            Expr::from(Accessor::Ident(Identifier::private_with_line(
                                "Any".into(),
                                attr.obj.ln_begin().unwrap_or(0),
                            )))
                        };
                        let sig =
                            Signature::Var(VarSignature::new(VarPattern::Ident(attr.ident), None));
                        let body = DefBody::new(EQUAL, Block::new(vec![typ]), DefId(0));
                        let field_type_def = Def::new(sig, body);
                        fields.push(field_type_def);
                    }
                }
                _ => {}
            }
        }
        if let Some(Expr::Record(Record::Normal(rec))) = base_type.as_mut() {
            let no_exist_fields = fields
                .into_iter()
                .filter(|field| {
                    rec.attrs
                        .iter()
                        .all(|rec_field| rec_field.sig.ident() != field.sig.ident())
                })
                .collect::<Vec<_>>();
            rec.attrs.extend(no_exist_fields);
        } else {
            let record = Record::Normal(NormalRecord::new(
                l_brace,
                r_brace,
                RecordAttrs::new(fields),
            ));
            *base_type = Some(Expr::Record(record));
        }
        let call_ident = Identifier::new(
            VisModifierSpec::Public(ErgLocation::Unknown),
            VarName::from_static("__call__"),
        );
        let class_ident = Identifier::public_with_line(
            DOT,
            self.cur_name().to_string().into(),
            sig.ln_begin().unwrap_or(0),
        );
        let class_ident_expr = Expr::Accessor(Accessor::Ident(class_ident.clone()));
        let class_spec = TypeSpecWithOp::new(COLON, TypeSpec::mono(class_ident), class_ident_expr);
        let mut params = sig.params.clone();
        if params
            .non_defaults
            .first()
            .is_some_and(|param| param.inspect().map(|s| &s[..]) == Some("self"))
        {
            params.non_defaults.remove(0);
        }
        let sig = Signature::Subr(SubrSignature::new(
            set! { Decorator(Expr::static_local("Override")) },
            call_ident,
            TypeBoundSpecs::empty(),
            params,
            Some(class_spec),
        ));
        let unreachable_acc = Identifier::new(
            VisModifierSpec::Public(ErgLocation::Unknown),
            VarName::from_static("exit"),
        );
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        let def = Def::new(sig, body);
        Some(def)
    }

    fn gen_default_init(&self, line: usize) -> Def {
        let call_ident = Identifier::new(
            VisModifierSpec::Public(ErgLocation::Unknown),
            VarName::from_static("__call__"),
        );
        let params = Params::empty();
        let class_ident =
            Identifier::public_with_line(DOT, self.cur_name().to_string().into(), line as u32);
        let class_ident_expr = Expr::Accessor(Accessor::Ident(class_ident.clone()));
        let class_spec = TypeSpecWithOp::new(COLON, TypeSpec::mono(class_ident), class_ident_expr);
        let sig = Signature::Subr(SubrSignature::new(
            set! { Decorator(Expr::static_local("Override")) },
            call_ident,
            TypeBoundSpecs::empty(),
            params,
            Some(class_spec),
        ));
        let unreachable_acc = Identifier::new(
            VisModifierSpec::Public(ErgLocation::Unknown),
            VarName::from_static("exit"),
        );
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        Def::new(sig, body)
    }

    fn extract_method(
        &mut self,
        body: Vec<py_ast::Stmt>,
        inherit: bool,
        class_name: Expr,
    ) -> (Option<Expr>, ClassAttrs) {
        let mut base_type = None;
        let mut attrs = vec![];
        let mut init_is_defined = false;
        let mut call_params_len = None;
        for stmt in body {
            match self.convert_statement(stmt, true) {
                Expr::Def(mut def) => {
                    if inherit {
                        if let Signature::Subr(subr) = &mut def.sig {
                            subr.decorators
                                .insert(Decorator(Expr::static_local("Override")));
                        }
                    }
                    if def.sig.decorators().is_some_and(|decos| {
                        decos.iter().any(|deco| {
                            deco.expr()
                                .get_name()
                                .is_some_and(|name| name == "property")
                        })
                    }) {
                        // class Foo:
                        //     @property
                        //     def foo(self): ...
                        // ↓
                        // class Foo:
                        //     def foo_(self): ...
                        //     foo = Foo(*[]).foo_()
                        let mut args = Args::empty();
                        if call_params_len.as_ref().is_some_and(|&len| len >= 1) {
                            args.set_var_args(PosArg::new(Expr::List(List::Normal(
                                NormalList::new(
                                    Token::dummy(TokenKind::LSqBr, "["),
                                    Token::dummy(TokenKind::RSqBr, "]"),
                                    Args::empty(),
                                ),
                            ))));
                        }
                        let instance = class_name.clone().call(args);
                        let name = def.sig.ident().unwrap().clone();
                        def.sig
                            .ident_mut()
                            .unwrap()
                            .name
                            .rename(format!("{} ", name.inspect()).into());
                        let escaped = def.sig.ident().unwrap().clone();
                        let call = Expr::Call(instance).method_call_expr(escaped, Args::empty());
                        let t_spec = def.sig.t_spec_op_mut().cloned();
                        let sig =
                            Signature::Var(VarSignature::new(VarPattern::Ident(name), t_spec));
                        let var_def =
                            Def::new(sig, DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)));
                        attrs.push(ClassAttr::Def(def));
                        attrs.push(ClassAttr::Def(var_def));
                    } else if def
                        .sig
                        .ident()
                        .is_some_and(|id| &id.inspect()[..] == "__init__")
                    {
                        if let Some(call_def) = self.extract_init(&mut base_type, def) {
                            if let Some(params) = call_def.sig.params() {
                                call_params_len = Some(params.len());
                            }
                            attrs.insert(0, ClassAttr::Def(call_def));
                            init_is_defined = true;
                        }
                    } else {
                        attrs.push(ClassAttr::Def(def));
                    }
                }
                Expr::TypeAscription(type_asc) => {
                    let sig = match type_asc.expr.as_ref() {
                        Expr::Accessor(Accessor::Ident(ident)) => Signature::Var(
                            VarSignature::new(VarPattern::Ident(ident.clone()), None),
                        ),
                        other => {
                            log!(err "{other}");
                            continue;
                        }
                    };
                    let expr = *type_asc.t_spec.t_spec_as_expr;
                    let body = DefBody::new(EQUAL, Block::new(vec![expr]), DefId(0));
                    let def = Def::new(sig, body);
                    match &mut base_type {
                        Some(Expr::Record(Record::Normal(NormalRecord { attrs, .. }))) => {
                            attrs.push(def);
                        }
                        None => {
                            let l_brace = Token::new(
                                TokenKind::LBrace,
                                "{",
                                def.ln_begin().unwrap_or(0),
                                def.col_begin().unwrap_or(0),
                            );
                            let r_brace = Token::new(
                                TokenKind::RBrace,
                                "}",
                                def.ln_end().unwrap_or(0),
                                def.col_end().unwrap_or(0),
                            );
                            let rec = Expr::Record(Record::Normal(NormalRecord::new(
                                l_brace,
                                r_brace,
                                RecordAttrs::new(vec![def]),
                            )));
                            base_type = Some(rec);
                        }
                        _ => {}
                    }
                    // attrs.push(ClassAttr::Decl(type_asc))
                }
                _other => {} // TODO:
            }
        }
        if !init_is_defined && !inherit {
            attrs.insert(0, ClassAttr::Def(self.gen_default_init(0)));
        }
        (base_type, ClassAttrs::new(attrs))
    }

    fn extract_method_list(
        &mut self,
        ident: Identifier,
        body: Vec<py_ast::Stmt>,
        base: Option<py_ast::Expr>,
    ) -> (Option<Expr>, Vec<Methods>) {
        let inherit = base.is_some();
        let class = if let Some(base) = base {
            let base_spec = self.convert_type_spec(base.clone());
            let expr = self.convert_expr(base);
            let loc = expr.loc();
            let base = TypeSpecWithOp::new(COLON, base_spec, expr);
            let args = TypeAppArgs::new(loc, TypeAppArgsKind::SubtypeOf(Box::new(base)), loc);
            TypeSpec::type_app(TypeSpec::mono(ident.clone()), args)
        } else {
            TypeSpec::mono(ident.clone())
        };
        let class_as_expr = Expr::Accessor(Accessor::Ident(ident));
        let (base_type, attrs) = self.extract_method(body, inherit, class_as_expr.clone());
        self.block_id_counter += 1;
        let methods = Methods::new(
            DefId(self.block_id_counter),
            class,
            class_as_expr,
            VisModifierSpec::Public(ErgLocation::Unknown),
            attrs,
        );
        (base_type, vec![methods])
    }

    fn get_type_bounds(&mut self, type_params: Vec<TypeParam>) -> TypeBoundSpecs {
        let mut bounds = TypeBoundSpecs::empty();
        if type_params.is_empty() {
            for ty in self.cur_appeared_type_names() {
                let name = VarName::from_str(ty.clone().into());
                let op = Token::dummy(TokenKind::SubtypeOf, "<:");
                if let Some(tv_info) = self.get_type_var(ty) {
                    let bound = if let Some(bound) = &tv_info.bound {
                        let t_spec = Parser::expr_to_type_spec(bound.clone())
                            .unwrap_or(TypeSpec::Infer(name.token().clone()));
                        let spec = TypeSpecWithOp::new(op, t_spec, bound.clone());
                        TypeBoundSpec::non_default(name, spec)
                    } else if !tv_info.constraints.is_empty() {
                        let op = Token::dummy(TokenKind::Colon, ":");
                        let mut elems = vec![];
                        for constraint in tv_info.constraints.iter() {
                            if let Ok(expr) = Parser::validate_const_expr(constraint.clone()) {
                                elems.push(ConstPosArg::new(expr));
                            }
                        }
                        let t_spec = TypeSpec::Enum(ConstArgs::pos_only(elems, None));
                        let elems = Args::pos_only(
                            tv_info
                                .constraints
                                .iter()
                                .cloned()
                                .map(PosArg::new)
                                .collect(),
                            None,
                        );
                        let expr = Expr::Set(Set::Normal(NormalSet::new(
                            Token::DUMMY,
                            Token::DUMMY,
                            elems,
                        )));
                        let spec = TypeSpecWithOp::new(op, t_spec, expr);
                        TypeBoundSpec::non_default(name, spec)
                    } else {
                        TypeBoundSpec::Omitted(name)
                    };
                    bounds.push(bound);
                }
            }
        }
        for tp in type_params {
            // TODO:
            let Some(tv) = tp.as_type_var() else {
                continue;
            };
            let name = VarName::from_str(tv.name.to_string().into());
            let spec = if let Some(bound) = &tv.bound {
                let op = Token::dummy(TokenKind::SubtypeOf, "<:");
                let spec = self.convert_type_spec(*bound.clone());
                let expr = self.convert_expr(*bound.clone());
                let spec = TypeSpecWithOp::new(op, spec, expr);
                TypeBoundSpec::non_default(name, spec)
            } else {
                TypeBoundSpec::Omitted(name)
            };
            bounds.push(spec);
        }
        bounds
    }

    fn convert_funcdef(&mut self, func_def: py_ast::StmtFunctionDef, is_async: bool) -> Expr {
        let name = func_def.name.to_string();
        let params = *func_def.args;
        let returns = func_def.returns.map(|x| *x);
        // if reassigning of a function referenced by other functions is occurred, it is an error
        if self.get_name(&name).is_some_and(|info| {
            info.defined_times > 0
                && info.defined_in == DefinedPlace::Known(self.cur_namespace())
                && !info.referenced.difference(&set! {name.clone()}).is_empty()
        }) {
            let err = reassign_func_error(
                self.cfg.input.clone(),
                pyloc_to_ergloc(func_def.range),
                self.cur_namespace(),
                &name,
            );
            self.errs.push(err);
            Expr::Dummy(Dummy::new(None, vec![]))
        } else {
            let loc = func_def.range.start;
            let decos = func_def
                .decorator_list
                .into_iter()
                .map(|ex| Decorator(self.convert_expr(ex)))
                .collect::<HashSet<_>>();
            self.register_name_info(&name, NameKind::Function);
            let func_name_loc = PyLocation {
                row: loc.row,
                column: loc.column.saturating_add(4),
            };
            let ident = self.convert_ident(name, func_name_loc);
            let kind = if is_async {
                BlockKind::AsyncFunction
            } else {
                BlockKind::Function
            };
            self.grow(ident.inspect().to_string(), kind);
            let params = self.convert_params(params);
            let return_t = returns
                .or_else(|| {
                    let PyTypeSpec::Func(func) = self.get_cur_scope_t_spec()? else {
                        return None;
                    };
                    func.returns.clone()
                })
                .map(|ret| {
                    let t_spec = self.convert_type_spec(ret.clone());
                    let colon = Token::new(
                        TokenKind::Colon,
                        ":",
                        t_spec.ln_begin().unwrap_or(0),
                        t_spec.col_begin().unwrap_or(0),
                    );
                    TypeSpecWithOp::new(colon, t_spec, self.convert_expr(ret))
                });
            let type_params = if !func_def.type_params.is_empty() {
                func_def.type_params
            } else {
                self.get_cur_scope_t_spec()
                    .and_then(|ty| {
                        if let PyTypeSpec::Func(func) = ty {
                            (!func.type_params.is_empty()).then(|| func.type_params.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or(func_def.type_params)
            };
            let bounds = self.get_type_bounds(type_params);
            let mut sig =
                Signature::Subr(SubrSignature::new(decos, ident, bounds, params, return_t));
            let block = self.convert_block(func_def.body, BlockKind::Function);
            if self.cur_context().return_kind.is_none() {
                let Signature::Subr(subr) = &mut sig else {
                    unreachable!()
                };
                if subr.return_t_spec.is_none() {
                    let none = TypeSpecWithOp::new(
                        Token::dummy(TokenKind::Colon, ":"),
                        TypeSpec::mono(Identifier::private("NoneType".into())),
                        Expr::static_local("NoneType"),
                    );
                    subr.return_t_spec = Some(Box::new(none));
                }
            }
            let body = DefBody::new(EQUAL, block, DefId(0));
            let def = Def::new(sig, body);
            self.pop();
            Expr::Def(def)
        }
    }

    /// ```python
    /// class Foo: pass
    /// ```
    /// ↓
    /// ```erg
    /// Foo = Inheritable Class()
    /// ```
    /// ```python
    /// class Foo(Bar): pass
    /// ```
    /// ↓
    /// ```erg
    /// Foo = Inherit Bar
    /// ```
    fn convert_classdef(&mut self, class_def: py_ast::StmtClassDef) -> Expr {
        let loc = class_def.location();
        let name = class_def.name.to_string();
        let _decos = class_def
            .decorator_list
            .into_iter()
            .map(|deco| self.convert_expr(deco))
            .collect::<Vec<_>>();
        let inherit = class_def.bases.first().cloned();
        let is_inherit = inherit.is_some();
        let mut bases = class_def
            .bases
            .into_iter()
            .map(|base| self.convert_expr(base))
            .collect::<Vec<_>>();
        self.register_name_info(&name, NameKind::Class);
        let class_name_loc = PyLocation {
            row: loc.row,
            column: loc.column.saturating_add(6),
        };
        let ident = self.convert_ident(name, class_name_loc);
        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident.clone()), None));
        self.grow(ident.inspect().to_string(), BlockKind::Class);
        let (base_type, methods) = self.extract_method_list(ident, class_def.body, inherit);
        let classdef = if is_inherit {
            // TODO: multiple inheritance
            let pos_args = vec![PosArg::new(bases.remove(0))];
            let mut args = Args::pos_only(pos_args, None);
            if let Some(rec @ Expr::Record(_)) = base_type {
                args.push_kw(KwArg::new(Token::symbol("Additional"), None, rec));
            }
            let inherit_acc = Expr::Accessor(Accessor::Ident(
                self.convert_ident("Inherit".to_string(), loc),
            ));
            let inherit_call = inherit_acc.call_expr(args);
            let body = DefBody::new(EQUAL, Block::new(vec![inherit_call]), DefId(0));
            let def = Def::new(sig, body);
            ClassDef::new(def, methods)
        } else {
            let pos_args = if let Some(base) = base_type {
                vec![PosArg::new(base)]
            } else {
                vec![]
            };
            let args = Args::pos_only(pos_args, None);
            let class_acc = Expr::Accessor(Accessor::Ident(
                self.convert_ident("Class".to_string(), loc),
            ));
            let class_call = class_acc.call_expr(args);
            let inheritable_acc = Expr::Accessor(Accessor::Ident(
                self.convert_ident("Inheritable".to_string(), loc),
            ));
            let inheritable_call = inheritable_acc.call1(class_call);
            let body = DefBody::new(EQUAL, Block::new(vec![inheritable_call]), DefId(0));
            let def = Def::new(sig, body);
            ClassDef::new(def, methods)
        };
        self.pop();
        Expr::ClassDef(classdef)
    }

    fn convert_for(&mut self, for_: py_ast::StmtFor) -> Expr {
        let loc = for_.location();
        let iter = self.convert_expr(*for_.iter);
        let if_block_id = self.block_id_counter + 1;
        let block = self.convert_for_body(Some(*for_.target), for_.body);
        let for_ident = self.convert_ident("for".to_string(), loc);
        let for_acc = Expr::Accessor(Accessor::Ident(for_ident));
        if for_.orelse.is_empty() {
            for_acc.call2(iter, Expr::Lambda(block))
        } else {
            let else_block = self.convert_block(for_.orelse, BlockKind::Else { if_block_id });
            let params = Params::empty();
            let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
            let op = Token::from_str(TokenKind::FuncArrow, "->");
            let else_block = Lambda::new(sig, op, else_block, DefId(0));
            let args = Args::pos_only(
                vec![
                    PosArg::new(iter),
                    PosArg::new(Expr::Lambda(block)),
                    PosArg::new(Expr::Lambda(else_block)),
                ],
                None,
            );
            for_acc.call_expr(args)
        }
    }

    fn get_t_spec(&self, name: &str) -> Option<&PyTypeSpec> {
        if self.contexts.len() == 1 {
            self.pyi_types.get_type(name)
        } else {
            let class = self.cur_name();
            self.pyi_types.get_class_member_type(class, name)
        }
    }

    fn get_assign_t_spec(
        &mut self,
        name: &py_ast::ExprName,
        expr: &Expr,
    ) -> Option<TypeSpecWithOp> {
        expr.ln_end()
            .and_then(|i| {
                i.checked_sub(1)
                    .and_then(|line| self.comments.get_type(line))
            })
            .cloned()
            .or_else(|| {
                let type_spec = self.get_t_spec(&name.id)?;
                let PyTypeSpec::Var(expr) = type_spec else {
                    return None;
                };
                Some(expr.clone())
            })
            .map(|mut expr| {
                // The range of `expr` is not correct, so we need to change it
                if let py_ast::Expr::Subscript(sub) = &mut expr {
                    sub.range = name.range;
                    *sub.slice.range_mut() = name.range;
                    *sub.value.range_mut() = name.range;
                } else {
                    *expr.range_mut() = name.range;
                }
                let t_as_expr = self.convert_expr(expr.clone());
                TypeSpecWithOp::new(AS, self.convert_type_spec(expr), t_as_expr)
            })
    }

    fn convert_statement(&mut self, stmt: Stmt, dont_call_return: bool) -> Expr {
        match stmt {
            py_ast::Stmt::Expr(stmt) => self.convert_expr(*stmt.value),
            // type-annotated assignment
            py_ast::Stmt::AnnAssign(ann_assign) => {
                let anot = self.convert_expr(*ann_assign.annotation.clone());
                let t_spec = self.convert_type_spec(*ann_assign.annotation);
                let as_op = Token::new(
                    TokenKind::As,
                    "as",
                    t_spec.ln_begin().unwrap_or(0),
                    t_spec.col_begin().unwrap_or(0),
                );
                let t_spec = TypeSpecWithOp::new(as_op, t_spec, anot);
                match *ann_assign.target {
                    py_ast::Expr::Name(name) => {
                        if let Some(value) = ann_assign.value {
                            let expr = self.convert_expr(*value);
                            // must register after convert_expr because value may be contain name (e.g. i = i + 1)
                            let rename =
                                self.register_name_info(name.id.as_str(), NameKind::Variable);
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            match rename {
                                RenameKind::Let => {
                                    let block = Block::new(vec![expr]);
                                    let body = DefBody::new(EQUAL, block, DefId(0));
                                    let sig = Signature::Var(VarSignature::new(
                                        VarPattern::Ident(ident),
                                        Some(t_spec),
                                    ));
                                    let def = Def::new(sig, body);
                                    Expr::Def(def)
                                }
                                RenameKind::Phi => {
                                    let block = Block::new(vec![expr]);
                                    let body = DefBody::new(EQUAL, block, DefId(0));
                                    let sig = Signature::Var(VarSignature::new(
                                        VarPattern::Phi(ident),
                                        Some(t_spec),
                                    ));
                                    let def = Def::new(sig, body);
                                    Expr::Def(def)
                                }
                                RenameKind::Redef => {
                                    let redef =
                                        ReDef::new(Accessor::Ident(ident), Some(t_spec), expr);
                                    Expr::ReDef(redef)
                                }
                            }
                        } else {
                            // no registration because it's just a type ascription
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            let tasc =
                                TypeAscription::new(Expr::Accessor(Accessor::Ident(ident)), t_spec);
                            Expr::TypeAscription(tasc)
                        }
                    }
                    py_ast::Expr::Attribute(attr) => {
                        let value = self.convert_expr(*attr.value);
                        let ident =
                            self.convert_attr_ident(attr.attr.to_string(), attr_name_loc(&value));
                        let attr = value.attr(ident);
                        if let Some(value) = ann_assign.value {
                            let expr = self.convert_expr(*value);
                            let redef = ReDef::new(attr, Some(t_spec), expr);
                            Expr::ReDef(redef)
                        } else {
                            let tasc = TypeAscription::new(Expr::Accessor(attr), t_spec);
                            Expr::TypeAscription(tasc)
                        }
                    }
                    _other => Expr::Dummy(Dummy::new(None, vec![])),
                }
            }
            py_ast::Stmt::Assign(mut assign) => {
                if assign.targets.len() == 1 {
                    let lhs = assign.targets.remove(0);
                    match lhs {
                        py_ast::Expr::Name(name) => {
                            let expr = self.convert_expr(*assign.value);
                            if let Expr::Call(call) = &expr {
                                if let Some("TypeVar") = call.obj.get_name().map(|s| &s[..]) {
                                    let arg = if let Some(Expr::Literal(lit)) =
                                        call.args.get_left_or_key("arg")
                                    {
                                        lit.token.content.trim_matches('\"').to_string()
                                    } else {
                                        name.id.to_string()
                                    };
                                    let mut constraints = vec![];
                                    let mut nth = 1;
                                    while let Some(constr) = call.args.get_nth(nth) {
                                        constraints.push(constr.clone());
                                        nth += 1;
                                    }
                                    if constraints.len() == 1 {
                                        let err = CompileError::syntax_error(
                                            self.cfg.input.clone(),
                                            line!() as usize,
                                            call.args.get_nth(1).unwrap().loc(),
                                            self.cur_namespace(),
                                            "TypeVar must have at least two constrained types"
                                                .into(),
                                            None,
                                        );
                                        self.errs.push(err);
                                    }
                                    let bound = call.args.get_with_key("bound").cloned();
                                    let info = TypeVarInfo::new(arg, constraints, bound);
                                    self.define_type_var(name.id.to_string(), info);
                                }
                            }
                            let rename = self.register_name_info(&name.id, NameKind::Variable);
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            let t_spec = self.get_assign_t_spec(&name, &expr);
                            match rename {
                                RenameKind::Let => {
                                    let block = Block::new(vec![expr]);
                                    let body = DefBody::new(EQUAL, block, DefId(0));
                                    let sig = Signature::Var(VarSignature::new(
                                        VarPattern::Ident(ident),
                                        t_spec,
                                    ));
                                    let def = Def::new(sig, body);
                                    Expr::Def(def)
                                }
                                RenameKind::Phi => {
                                    let block = Block::new(vec![expr]);
                                    let body = DefBody::new(EQUAL, block, DefId(0));
                                    let sig = Signature::Var(VarSignature::new(
                                        VarPattern::Phi(ident),
                                        t_spec,
                                    ));
                                    let def = Def::new(sig, body);
                                    Expr::Def(def)
                                }
                                RenameKind::Redef => {
                                    let redef = ReDef::new(Accessor::Ident(ident), t_spec, expr);
                                    Expr::ReDef(redef)
                                }
                            }
                        }
                        py_ast::Expr::Attribute(attr) => {
                            let value = self.convert_expr(*attr.value);
                            let ident = self
                                .convert_attr_ident(attr.attr.to_string(), attr_name_loc(&value));
                            let attr = value.attr(ident);
                            let expr = self.convert_expr(*assign.value);
                            let adef = ReDef::new(attr, None, expr);
                            Expr::ReDef(adef)
                        }
                        py_ast::Expr::Tuple(tuple) => {
                            let tmp = FRESH_GEN.fresh_varname();
                            let tmp_name =
                                VarName::from_str_and_line(tmp, tuple.location().row.get());
                            let tmp_ident = Identifier::new(
                                VisModifierSpec::Public(ErgLocation::Unknown),
                                tmp_name,
                            );
                            let tmp_expr = Expr::Accessor(Accessor::Ident(tmp_ident.clone()));
                            let sig = Signature::Var(VarSignature::new(
                                VarPattern::Ident(tmp_ident),
                                None,
                            ));
                            let body = DefBody::new(
                                EQUAL,
                                Block::new(vec![self.convert_expr(*assign.value)]),
                                DefId(0),
                            );
                            let tmp_def = Expr::Def(Def::new(sig, body));
                            let mut defs = vec![tmp_def];
                            for (i, elem) in tuple.elts.into_iter().enumerate() {
                                let loc = elem.location();
                                let index = Literal::new(Token::new(
                                    TokenKind::NatLit,
                                    i.to_string(),
                                    loc.row.get(),
                                    loc.column.to_zero_indexed(),
                                ));
                                let (param, mut blocks) =
                                    self.convert_opt_expr_to_param(Some(elem));
                                let sig = Signature::Var(VarSignature::new(
                                    Self::param_pattern_to_var(param.pat),
                                    param.t_spec,
                                ));
                                let method = tmp_expr
                                    .clone()
                                    .attr_expr(self.convert_ident("__getitem__".to_string(), loc));
                                let tuple_acc = method.call1(Expr::Literal(index));
                                let body =
                                    DefBody::new(EQUAL, Block::new(vec![tuple_acc]), DefId(0));
                                let def = Expr::Def(Def::new(sig, body));
                                defs.push(def);
                                defs.append(&mut blocks);
                            }
                            Expr::Dummy(Dummy::new(None, defs))
                        }
                        // a[b] = x
                        // => a.__setitem__(b, x)
                        py_ast::Expr::Subscript(subs) => {
                            let a = self.convert_expr(*subs.value);
                            let slice_loc = subs.slice.location();
                            let b = self.convert_expr(*subs.slice);
                            let x = self.convert_expr(*assign.value);
                            let method = a.attr_expr(
                                self.convert_ident("__setitem__".to_string(), slice_loc),
                            );
                            method.call2(b, x)
                        }
                        other => {
                            log!(err "{other:?} as LHS");
                            Expr::Dummy(Dummy::new(None, vec![]))
                        }
                    }
                } else {
                    let value = self.convert_expr(*assign.value);
                    let mut defs = vec![];
                    for target in assign.targets {
                        match target {
                            py_ast::Expr::Name(name) => {
                                let body =
                                    DefBody::new(EQUAL, Block::new(vec![value.clone()]), DefId(0));
                                let rename = self.register_name_info(&name.id, NameKind::Variable);
                                let ident =
                                    self.convert_ident(name.id.to_string(), name.location());
                                match rename {
                                    RenameKind::Let => {
                                        let sig = Signature::Var(VarSignature::new(
                                            VarPattern::Ident(ident),
                                            None,
                                        ));
                                        let def = Def::new(sig, body);
                                        defs.push(Expr::Def(def));
                                    }
                                    RenameKind::Phi => {
                                        let sig = Signature::Var(VarSignature::new(
                                            VarPattern::Phi(ident),
                                            None,
                                        ));
                                        let def = Def::new(sig, body);
                                        defs.push(Expr::Def(def));
                                    }
                                    RenameKind::Redef => {
                                        let redef =
                                            ReDef::new(Accessor::Ident(ident), None, value.clone());
                                        defs.push(Expr::ReDef(redef));
                                    }
                                }
                            }
                            _other => {
                                defs.push(Expr::Dummy(Dummy::new(None, vec![])));
                            }
                        }
                    }
                    Expr::Dummy(Dummy::new(None, defs))
                }
            }
            py_ast::Stmt::AugAssign(aug_assign) => {
                let op = op_to_token(aug_assign.op);
                match *aug_assign.target {
                    py_ast::Expr::Name(name) => {
                        let val = self.convert_expr(*aug_assign.value);
                        let prev_ident = self.convert_ident(name.id.to_string(), name.location());
                        if self
                            .get_name(name.id.as_str())
                            .is_some_and(|info| info.defined_block_id == self.cur_block_id())
                        {
                            self.register_name_info(&name.id, NameKind::Variable);
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            let bin =
                                BinOp::new(op, Expr::Accessor(Accessor::Ident(prev_ident)), val);
                            let sig =
                                Signature::Var(VarSignature::new(VarPattern::Ident(ident), None));
                            let block = Block::new(vec![Expr::BinOp(bin)]);
                            let body = DefBody::new(EQUAL, block, DefId(0));
                            let def = Def::new(sig, body);
                            Expr::Def(def)
                        } else {
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            let bin =
                                BinOp::new(op, Expr::Accessor(Accessor::Ident(prev_ident)), val);
                            let redef = ReDef::new(Accessor::Ident(ident), None, Expr::BinOp(bin));
                            Expr::ReDef(redef)
                        }
                    }
                    py_ast::Expr::Attribute(attr) => {
                        let assign_value = self.convert_expr(*aug_assign.value);
                        let attr_value = self.convert_expr(*attr.value);
                        let ident = self
                            .convert_attr_ident(attr.attr.to_string(), attr_name_loc(&attr_value));
                        let attr = attr_value.attr(ident);
                        let bin = BinOp::new(op, Expr::Accessor(attr.clone()), assign_value);
                        let redef = ReDef::new(attr, None, Expr::BinOp(bin));
                        Expr::ReDef(redef)
                    }
                    other => {
                        log!(err "{other:?} as LHS");
                        Expr::Dummy(Dummy::new(None, vec![]))
                    }
                }
            }
            py_ast::Stmt::FunctionDef(func_def) => self.convert_funcdef(func_def, false),
            py_ast::Stmt::AsyncFunctionDef(func_def) => {
                let py_ast::StmtAsyncFunctionDef {
                    name,
                    args,
                    body,
                    decorator_list,
                    returns,
                    type_params,
                    range,
                    type_comment,
                } = func_def;
                let func_def = py_ast::StmtFunctionDef {
                    name,
                    args,
                    body,
                    decorator_list,
                    returns,
                    type_params,
                    range,
                    type_comment,
                };
                self.convert_funcdef(func_def, true)
            }
            py_ast::Stmt::ClassDef(class_def) => self.convert_classdef(class_def),
            py_ast::Stmt::For(for_) => self.convert_for(for_),
            py_ast::Stmt::AsyncFor(for_) => {
                let py_ast::StmtAsyncFor {
                    target,
                    iter,
                    body,
                    orelse,
                    range,
                    type_comment,
                } = for_;
                let for_ = py_ast::StmtFor {
                    target,
                    iter,
                    body,
                    orelse,
                    range,
                    type_comment,
                };
                self.convert_for(for_)
            }
            py_ast::Stmt::While(while_) => {
                let loc = while_.location();
                let test = self.convert_expr(*while_.test);
                let params = Params::empty();
                let empty_sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let if_block_id = self.block_id_counter + 1;
                let block = self.convert_block(while_.body, BlockKind::While);
                let body = Lambda::new(empty_sig, Token::DUMMY, block, DefId(0));
                let while_ident = self.convert_ident("while".to_string(), loc);
                let while_acc = Expr::Accessor(Accessor::Ident(while_ident));
                if while_.orelse.is_empty() {
                    while_acc.call2(test, Expr::Lambda(body))
                } else {
                    let else_block =
                        self.convert_block(while_.orelse, BlockKind::Else { if_block_id });
                    let params = Params::empty();
                    let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                    let op = Token::from_str(TokenKind::FuncArrow, "->");
                    let else_body = Lambda::new(sig, op, else_block, DefId(0));
                    let args = Args::pos_only(
                        vec![
                            PosArg::new(test),
                            PosArg::new(Expr::Lambda(body)),
                            PosArg::new(Expr::Lambda(else_body)),
                        ],
                        None,
                    );
                    while_acc.call_expr(args)
                }
            }
            py_ast::Stmt::If(if_) => {
                let loc = if_.location();
                let if_block_id = self.block_id_counter + 1;
                let block = self.convert_block(if_.body, BlockKind::If);
                let params = Params::empty();
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, block, DefId(0));
                let test = self.convert_expr(*if_.test);
                let if_ident = self.convert_ident("if".to_string(), loc);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                if !if_.orelse.is_empty() {
                    let else_block =
                        self.convert_block(if_.orelse, BlockKind::Else { if_block_id });
                    let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                    let else_body = Lambda::new(sig, Token::DUMMY, else_block, DefId(0));
                    let args = Args::pos_only(
                        vec![
                            PosArg::new(test),
                            PosArg::new(Expr::Lambda(body)),
                            PosArg::new(Expr::Lambda(else_body)),
                        ],
                        None,
                    );
                    if_acc.call_expr(args)
                } else {
                    if_acc.call2(test, Expr::Lambda(body))
                }
            }
            py_ast::Stmt::Return(return_) => {
                self.cur_context_mut().return_kind = ReturnKind::Return;
                let loc = return_.location();
                let value = return_
                    .value
                    .map(|val| self.convert_expr(*val))
                    .unwrap_or_else(|| Expr::Tuple(Tuple::Normal(NormalTuple::new(Args::empty()))));
                if dont_call_return {
                    value
                } else {
                    let func_acc = Expr::Accessor(Accessor::Ident(
                        self.convert_ident(self.cur_name().to_string(), loc),
                    ));
                    let return_acc = self.convert_ident("return".to_string(), loc);
                    let return_acc = Expr::Accessor(Accessor::attr(func_acc, return_acc));
                    return_acc.call1(value)
                }
            }
            py_ast::Stmt::Assert(assert) => {
                let loc = assert.location();
                let test = self.convert_expr(*assert.test);
                let args = if let Some(msg) = assert.msg {
                    let msg = self.convert_expr(*msg);
                    Args::pos_only(vec![PosArg::new(test), PosArg::new(msg)], None)
                } else {
                    Args::pos_only(vec![PosArg::new(test)], None)
                };
                let assert_acc = Expr::Accessor(Accessor::Ident(
                    self.convert_ident("assert".to_string(), loc),
                ));
                assert_acc.call_expr(args)
            }
            py_ast::Stmt::Import(import) => {
                let import_loc = import.location();
                let mut imports = vec![];
                for name in import.names {
                    let import_acc = Expr::Accessor(Accessor::Ident(
                        self.convert_ident("__import__".to_string(), import_loc),
                    ));
                    let sym = if name.asname.is_some() {
                        name.name.replace('.', "/")
                    } else {
                        name.name.split('.').next().unwrap().to_string()
                    };
                    let mod_name = Expr::Literal(Literal::new(quoted_symbol(
                        &sym,
                        name.location().row.get(),
                        name.location().column.to_zero_indexed(),
                    )));
                    let call = import_acc.call1(mod_name);
                    let name_loc = name.location();
                    let def = if let Some(alias) = name.asname {
                        self.register_name_info(&alias, NameKind::Variable);
                        let var = VarSignature::new(
                            VarPattern::Ident(self.convert_ident(alias.to_string(), name_loc)),
                            None,
                        );
                        Def::new(
                            Signature::Var(var),
                            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                        )
                    } else {
                        let top_module = name.name.split('.').next().unwrap();
                        self.register_name_info(top_module, NameKind::Variable);
                        let var = VarSignature::new(
                            VarPattern::Ident(
                                self.convert_ident(top_module.to_string(), name.location()),
                            ),
                            None,
                        );
                        Def::new(
                            Signature::Var(var),
                            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                        )
                    };
                    imports.push(Expr::Def(def));
                }
                Expr::Dummy(Dummy::new(None, imports))
            }
            // from module import foo, bar
            py_ast::Stmt::ImportFrom(import_from) => {
                let mut loc = import_from.location();
                loc.column = loc.column.saturating_add(5);
                self.convert_from_import(import_from.module, import_from.names, loc)
            }
            py_ast::Stmt::Try(try_) => {
                let if_block_id = self.block_id_counter + 1;
                let mut chunks = self.convert_block(try_.body, BlockKind::Try);
                for py_ast::ExceptHandler::ExceptHandler(handler) in try_.handlers {
                    let mut block =
                        self.convert_block(handler.body, BlockKind::Else { if_block_id });
                    if let Some(name) = handler.name {
                        let ident = self.convert_ident(name.to_string(), handler.range.start);
                        let t_spec = if let Some(type_) = handler.type_ {
                            let t_spec = self.convert_type_spec(*type_.clone());
                            let as_expr = self.convert_expr(*type_.clone());
                            let as_op = Token::new(
                                TokenKind::As,
                                "as",
                                t_spec.ln_begin().unwrap_or(0),
                                t_spec.col_begin().unwrap_or(0),
                            );
                            let t_spec = TypeSpecWithOp::new(as_op, t_spec, as_expr);
                            Some(t_spec)
                        } else {
                            None
                        };
                        let var = VarSignature::new(VarPattern::Ident(ident), t_spec);
                        let unreachable_acc = Identifier::new(
                            VisModifierSpec::Public(ErgLocation::Unknown),
                            VarName::from_static("exit"),
                        );
                        let body = Expr::Accessor(Accessor::Ident(unreachable_acc))
                            .call_expr(Args::empty());
                        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
                        let def = Def::new(Signature::Var(var), body);
                        block.insert(0, Expr::Def(def));
                    }
                    chunks.extend(block);
                }
                let dummy = chunks
                    .into_iter()
                    .chain(self.convert_block(try_.orelse, BlockKind::Else { if_block_id }))
                    .chain(self.convert_block(try_.finalbody, BlockKind::Else { if_block_id }))
                    .collect();
                Expr::Dummy(Dummy::new(None, dummy))
            }
            py_ast::Stmt::With(mut with) => {
                let loc = with.location();
                let item = with.items.remove(0);
                let context_expr = self.convert_expr(item.context_expr);
                let body = self.convert_for_body(item.optional_vars.map(|x| *x), with.body);
                let with_ident = self.convert_ident("with".to_string(), loc);
                let with_acc = Expr::Accessor(Accessor::Ident(with_ident));
                with_acc.call2(context_expr, Expr::Lambda(body))
            }
            py_ast::Stmt::AsyncWith(mut with) => {
                let loc = with.location();
                let item = with.items.remove(0);
                let context_expr = self.convert_expr(item.context_expr);
                let body = self.convert_for_body(item.optional_vars.map(|x| *x), with.body);
                let with_ident = self.convert_ident("with".to_string(), loc);
                let with_acc = Expr::Accessor(Accessor::Ident(with_ident));
                with_acc.call2(context_expr, Expr::Lambda(body))
            }
            _other => {
                log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::new(None, vec![]))
            }
        }
    }

    fn convert_glob_import(&mut self, location: PyLocation, module: String) -> Expr {
        let import_acc = Expr::Accessor(Accessor::Ident(
            self.convert_ident("__import__".to_string(), location),
        ));
        let sym = if module == "." { "__init__" } else { &module };
        let mod_name = Expr::Literal(Literal::new(quoted_symbol(
            sym,
            location.row.get(),
            location.column.to_zero_indexed(),
        )));
        let call = import_acc.clone().call1(mod_name);
        let var = VarSignature::new(VarPattern::Glob(Token::DUMMY), None);
        Expr::Def(Def::new(
            Signature::Var(var),
            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
        ))
    }

    /**
    ```erg
    from foo import bar # if bar, baz are modules
    # ↓
    .foo = import "foo"
    .bar = import "foo/bar"
    .baz = import "foo/baz"

    from foo import bar, baz # if bar, baz are not modules
    # ↓
    {.bar; .baz} = import "foo"

    from . import bar, baz # if bar, baz are modules
    # ↓
    .bar = import "./bar"
    .baz = import "./baz"

    from . import bar, baz # if bar, baz are not modules
    # ↓
    {.bar; .baz} = import "__init__"
    ```
    */
    fn convert_from_import(
        &mut self,
        module: Option<py_ast::Identifier>,
        names: Vec<Alias>,
        location: PyLocation,
    ) -> Expr {
        let import_acc = Expr::Accessor(Accessor::Ident(
            self.convert_ident("__import__".to_string(), location),
        ));
        let module = module
            .map(|s| s.replace('.', "/"))
            .unwrap_or_else(|| ".".to_string());
        let module_path = Path::new(&module);
        let sym = if module == "." { "__init__" } else { &module };
        let mod_name = Expr::Literal(Literal::new(quoted_symbol(
            sym,
            location.row.get(),
            location.column.to_zero_indexed(),
        )));
        let call = import_acc.clone().call1(mod_name);
        let mut exprs = vec![];
        let mut imports = vec![];
        if names.len() == 1 && names[0].name.as_str() == "*" {
            return self.convert_glob_import(location, module);
        }
        let names_range = PySourceRange {
            start: names[0].location(),
            end: names[names.len() - 1].end_location(),
        };
        for name in names {
            let name_path = self
                .cfg
                .input
                .resolve_py(&module_path.join(name.name.as_str()));
            let true_name = self.convert_ident(name.name.to_string(), name.location());
            let as_loc = name.location();
            let alias = if let Some(alias) = name.asname {
                self.register_name_info(&alias, NameKind::Variable);
                let ident = self.convert_ident(alias.to_string(), as_loc);
                VarSignature::new(VarPattern::Ident(ident), None)
            } else {
                self.register_name_info(&name.name, NameKind::Variable);
                let ident = self.convert_ident(name.name.to_string(), name.location());
                VarSignature::new(VarPattern::Ident(ident), None)
            };
            // from foo import bar, baz (if bar, baz is a module) ==> bar = import "foo/bar"; baz = import "foo/baz"
            if let Ok(mut path) = name_path {
                if path.ends_with("__init__.py") {
                    path.pop();
                }
                let mod_name = path.file_name().unwrap_or_default();
                if name.name.as_str() == mod_name.to_string_lossy().trim_end_matches(".py") {
                    let sym = format!("{module}/{}", name.name);
                    let mod_name = Expr::Literal(Literal::new(quoted_symbol(
                        &sym,
                        location.row.get(),
                        location.column.to_zero_indexed(),
                    )));
                    let call = import_acc.clone().call1(mod_name);
                    let def = Def::new(
                        Signature::Var(alias),
                        DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                    );
                    exprs.push(Expr::Def(def));
                } else {
                    // name.name: Foo, file_name: foo.py
                    imports.push(VarRecordAttr::new(true_name, alias));
                }
            } else {
                imports.push(VarRecordAttr::new(true_name, alias));
            }
        }
        let no_import = imports.is_empty();
        let attrs = VarRecordAttrs::new(imports);
        let braces = pyloc_to_ergloc(names_range);
        let pat = VarRecordPattern::new(braces, attrs);
        let var = VarSignature::new(VarPattern::Record(pat), None);
        let def = Expr::Def(Def::new(
            Signature::Var(var),
            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
        ));
        if no_import {
            Expr::Dummy(Dummy::new(None, exprs))
        } else if exprs.is_empty() {
            def
        } else {
            exprs.push(def);
            Expr::Dummy(Dummy::new(None, exprs))
        }
    }

    pub fn convert_program(mut self, program: ModModule) -> IncompleteArtifact<Module> {
        let program = program
            .body
            .into_iter()
            .map(|stmt| self.convert_statement(stmt, true))
            .collect();
        let module = Desugarer::new().desugar(Module::new(program));
        IncompleteArtifact::new(Some(module), self.errs, self.warns)
    }
}
