use std::path::Path;

use erg_common::config::ErgConfig;
use erg_common::dict::Dict as HashMap;
use erg_common::fresh::FRESH_GEN;
use erg_common::set::Set as HashSet;
use erg_common::traits::{Locational, Stream};
use erg_common::{log, set};
use erg_compiler::artifact::IncompleteArtifact;
use erg_compiler::erg_parser::ast::{
    Accessor, Args, Array, BinOp, Block, ClassAttr, ClassAttrs, ClassDef, ConstAccessor, ConstArgs,
    ConstAttribute, ConstDict, ConstExpr, ConstKeyValue, ConstPosArg, Decorator, Def, DefBody,
    DefId, DefaultParamSignature, Dict, Dummy, Expr, Identifier, KeyValue, KwArg, Lambda,
    LambdaSignature, Literal, Methods, Module, NonDefaultParamSignature, NormalArray, NormalDict,
    NormalRecord, NormalSet, NormalTuple, ParamPattern, ParamTySpec, Params, PosArg,
    PreDeclTypeSpec, ReDef, Record, RecordAttrs, Set, Signature, SubrSignature, SubrTypeSpec,
    Tuple, TupleTypeSpec, TypeAscription, TypeBoundSpecs, TypeSpec, TypeSpecWithOp, UnaryOp,
    VarName, VarPattern, VarRecordAttr, VarRecordAttrs, VarRecordPattern, VarSignature,
    VisModifierSpec,
};
use erg_compiler::erg_parser::desugar::Desugarer;
use erg_compiler::erg_parser::token::{Token, TokenKind, COLON, DOT, EQUAL};
use erg_compiler::erg_parser::Parser;
use erg_compiler::error::{CompileError, CompileErrors};
use rustpython_parser::ast::located::{
    self as py_ast, Alias, Arg, Arguments, BoolOp, CmpOp, ExprConstant, Keyword, Located,
    ModModule, Operator, Stmt, String, Suite, UnaryOp as UnOp,
};
use rustpython_parser::source_code::{
    OneIndexed, SourceLocation as PyLocation, SourceRange as PySourceRange,
};

use crate::ast_util::accessor_name;
use crate::error::*;

pub const ARROW: Token = Token::dummy(TokenKind::FuncArrow, "->");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanShadow {
    Yes,
    No,
}

impl CanShadow {
    pub const fn is_yes(&self) -> bool {
        matches!(self, Self::Yes)
    }
    pub const fn is_no(&self) -> bool {
        matches!(self, Self::No)
    }
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
    For,
    While,
    Try,
    With,
    Function,
    Class,
    Module,
}

impl BlockKind {
    pub const fn is_if(&self) -> bool {
        matches!(self, Self::If)
    }
    pub const fn is_function(&self) -> bool {
        matches!(self, Self::Function)
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
        "list" => "GenericArray".into(),
        "bytes" => "Bytes".into(),
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
        range.end.unwrap().column.get(),
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
    namespace: Vec<String>,
    block_id_counter: usize,
    block_ids: Vec<usize>,
    /// Erg does not allow variables to be defined multiple times, so rename them using this
    names: Vec<HashMap<String, NameInfo>>,
    warns: CompileErrors,
    errs: CompileErrors,
}

impl ASTConverter {
    pub fn new(cfg: ErgConfig, shadowing: ShadowingMode) -> Self {
        Self {
            shadowing,
            cfg,
            namespace: vec![String::from("<module>")],
            block_id_counter: 0,
            block_ids: vec![0],
            names: vec![HashMap::new()],
            warns: CompileErrors::empty(),
            errs: CompileErrors::empty(),
        }
    }

    fn get_name(&self, name: &str) -> Option<&NameInfo> {
        for ns in self.names.iter().rev() {
            if let Some(ni) = ns.get(name) {
                return Some(ni);
            }
        }
        None
    }

    fn get_mut_name(&mut self, name: &str) -> Option<&mut NameInfo> {
        for ns in self.names.iter_mut().rev() {
            if let Some(ni) = ns.get_mut(name) {
                return Some(ni);
            }
        }
        None
    }

    fn define_name(&mut self, name: String, info: NameInfo) {
        self.names.last_mut().unwrap().insert(name, info);
    }

    fn declare_name(&mut self, name: String, info: NameInfo) {
        self.names.first_mut().unwrap().insert(name, info);
    }

    fn grow(&mut self, namespace: String) {
        self.namespace.push(namespace);
        self.names.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.namespace.pop();
        self.names.pop();
    }

    fn cur_block_id(&self) -> usize {
        *self.block_ids.last().unwrap()
    }

    fn cur_namespace(&self) -> String {
        self.namespace.join(".")
    }

    fn register_name_info(&mut self, name: &str, kind: NameKind) -> CanShadow {
        let cur_namespace = self.cur_namespace();
        let cur_block_id = self.cur_block_id();
        if let Some(name_info) = self.get_mut_name(name) {
            if name_info.defined_in == cur_namespace && name_info.defined_block_id == cur_block_id {
                name_info.defined_times += 1;
            }
            if name_info.defined_in.is_unknown() {
                name_info.defined_in = DefinedPlace::Known(cur_namespace);
                name_info.defined_times += 1; // 0 -> 1
            }
            if name_info.defined_block_id == cur_block_id {
                CanShadow::Yes
            } else {
                CanShadow::No
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
            CanShadow::Yes
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
        Identifier::new(VisModifierSpec::Public(dot), name)
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
        Identifier::new(VisModifierSpec::Public(dot), name)
    }

    // Duplicate param names will result in an error at the parser. So we don't need to check it here.
    fn convert_param_pattern(&mut self, arg: String, loc: PyLocation) -> ParamPattern {
        self.register_name_info(&arg, NameKind::Variable);
        let ident = self.convert_ident(arg, loc);
        ParamPattern::VarName(ident.name)
    }

    fn convert_nd_param(&mut self, param: Arg) -> NonDefaultParamSignature {
        let pat = self.convert_param_pattern(param.arg.to_string(), param.location());
        let t_spec = param
            .annotation
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
        fn split_args(params: Arguments) -> (Vec<Arg>, Vec<(Arg, py_ast::Expr)>) {
            let mut args = Vec::new();
            let mut with_defaults = Vec::new();
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
            (args, with_defaults)
        }
        let (non_defaults, defaults) = split_args(params);
        let non_defaults = non_defaults
            .into_iter()
            .map(|p| self.convert_nd_param(p))
            .collect();
        let defaults = defaults
            .into_iter()
            .map(|(kw, default)| self.convert_default_param(kw, default))
            .collect();
        Params::new(non_defaults, None, defaults, None)
    }

    fn convert_for_param(&mut self, name: String, loc: PyLocation) -> NonDefaultParamSignature {
        let pat = self.convert_param_pattern(name, loc);
        let t_spec = None;
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn param_pattern_to_var(pat: ParamPattern) -> VarPattern {
        match pat {
            ParamPattern::VarName(name) => {
                VarPattern::Ident(Identifier::new(VisModifierSpec::Public(DOT), name))
            }
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
                    VisModifierSpec::Public(DOT),
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
        let params = Params::new(vec![param], None, vec![], None);
        self.block_id_counter += 1;
        self.block_ids.push(self.block_id_counter);
        let body = body
            .into_iter()
            .map(|stmt| self.convert_statement(stmt, true))
            .collect::<Vec<_>>();
        self.block_ids.pop();
        let body = block.into_iter().chain(body).collect();
        let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
        let op = Token::from_str(TokenKind::FuncArrow, "->");
        Lambda::new(sig, op, Block::new(body), DefId(0))
    }

    fn convert_ident_type_spec(&mut self, name: String, loc: PyLocation) -> TypeSpec {
        TypeSpec::mono(self.convert_ident(name, loc))
    }

    fn gen_dummy_type_spec(loc: PyLocation) -> TypeSpec {
        TypeSpec::Infer(Token::new(
            TokenKind::UBar,
            "_",
            loc.row.get(),
            loc.column.get() - 1,
        ))
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
                    let expr = self.convert_expr(elem);
                    match Parser::validate_const_expr(expr) {
                        Ok(expr) => {
                            elems.push(ConstPosArg::new(expr));
                        }
                        Err(err) => {
                            let err = CompileError::new(
                                err.into(),
                                self.cfg.input.clone(),
                                self.cur_namespace(),
                            );
                            self.errs.push(err);
                        }
                    }
                }
                let elems = ConstArgs::new(elems, None, vec![], None);
                TypeSpec::Enum(elems)
            }
            // TODO: distinguish from collections.abc.Callable
            "Callable" => {
                let py_ast::Expr::Tuple(mut tuple) = args else {
                    return Self::gen_dummy_type_spec(args.location());
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
                    ARROW,
                    ret,
                ))
            }
            "Iterable" | "Iterator" | "Collection" | "Container" | "Sequence"
            | "MutableSequence" => {
                let elem_t = self.convert_expr(args);
                let elem_t = match Parser::validate_const_expr(elem_t) {
                    Ok(elem_t) => elem_t,
                    Err(err) => {
                        let err = CompileError::new(
                            err.into(),
                            self.cfg.input.clone(),
                            self.cur_namespace(),
                        );
                        self.errs.push(err);
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
                let py_ast::Expr::Tuple(mut tuple) = args else {
                    let err = CompileError::syntax_error(
                        self.cfg.input.clone(),
                        line!() as usize,
                        pyloc_to_ergloc(args.range()),
                        self.cur_namespace(),
                        format!("`{name}` takes 2 types"),
                        None,
                    );
                    self.errs.push(err);
                    return Self::gen_dummy_type_spec(args.location());
                };
                let key_t = self.convert_expr(tuple.elts.remove(0));
                let key_t = match Parser::validate_const_expr(key_t) {
                    Ok(key_t) => key_t,
                    Err(err) => {
                        let err = CompileError::new(
                            err.into(),
                            self.cfg.input.clone(),
                            self.cur_namespace(),
                        );
                        self.errs.push(err);
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let key_t = ConstPosArg::new(key_t);
                let value_t = self.convert_expr(tuple.elts.remove(0));
                let value_t = match Parser::validate_const_expr(value_t) {
                    Ok(value_t) => value_t,
                    Err(err) => {
                        let err = CompileError::new(
                            err.into(),
                            self.cfg.input.clone(),
                            self.cur_namespace(),
                        );
                        self.errs.push(err);
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
            "list" => {
                let len = ConstExpr::Accessor(ConstAccessor::Local(
                    self.convert_ident("_".into(), args.location()),
                ));
                let elem_t = self.convert_expr(args);
                let elem_t = match Parser::validate_const_expr(elem_t) {
                    Ok(elem_t) => elem_t,
                    Err(err) => {
                        let err = CompileError::new(
                            err.into(),
                            self.cfg.input.clone(),
                            self.cur_namespace(),
                        );
                        self.errs.push(err);
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let elem_t = ConstPosArg::new(elem_t);
                let len = ConstPosArg::new(len);
                let global =
                    ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("global".into())));
                let acc = ConstAccessor::Attr(ConstAttribute::new(
                    global,
                    Identifier::private("Array!".into()),
                ));
                TypeSpec::poly(acc, ConstArgs::new(vec![elem_t, len], None, vec![], None))
            }
            "dict" => {
                let py_ast::Expr::Tuple(mut tuple) = args else {
                    return Self::gen_dummy_type_spec(args.location());
                };
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(
                    TokenKind::LBrace,
                    tuple.elts.iter(),
                    tuple.location(),
                );
                let key_t = self.convert_expr(tuple.elts.remove(0));
                let key_t = match Parser::validate_const_expr(key_t) {
                    Ok(key_t) => key_t,
                    Err(err) => {
                        let err = CompileError::new(
                            err.into(),
                            self.cfg.input.clone(),
                            self.cur_namespace(),
                        );
                        self.errs.push(err);
                        ConstExpr::Accessor(ConstAccessor::Local(Identifier::private("Obj".into())))
                    }
                };
                let val_t = self.convert_expr(tuple.elts.remove(0));
                let val_t = match Parser::validate_const_expr(val_t) {
                    Ok(val_t) => val_t,
                    Err(err) => {
                        let err = CompileError::new(
                            err.into(),
                            self.cfg.input.clone(),
                            self.cur_namespace(),
                        );
                        self.errs.push(err);
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
                TypeSpec::poly(acc, ConstArgs::new(vec![dict], None, vec![], None))
            }
            "tuple" => {
                let py_ast::Expr::Tuple(tuple) = args else {
                    return Self::gen_dummy_type_spec(args.location());
                };
                let parens = Self::gen_enclosure_tokens(
                    TokenKind::LParen,
                    tuple.elts.iter(),
                    tuple.location(),
                );
                let tys = tuple
                    .elts
                    .into_iter()
                    .map(|elem| self.convert_type_spec(elem))
                    .collect();
                let tuple = TupleTypeSpec::new(Some(parens), tys);
                TypeSpec::Tuple(tuple)
            }
            _ => Self::gen_dummy_type_spec(args.location()),
        }
    }

    fn convert_type_spec(&mut self, expr: py_ast::Expr) -> TypeSpec {
        #[allow(clippy::collapsible_match)]
        match expr {
            py_ast::Expr::Name(name) => {
                self.convert_ident_type_spec(name.id.to_string(), name.location())
            }
            py_ast::Expr::Constant(cons) => {
                self.convert_ident_type_spec("NoneType".into(), cons.location())
            }
            py_ast::Expr::Attribute(attr) => {
                let namespace = Box::new(self.convert_expr(*attr.value));
                let t = self.convert_ident(attr.attr.to_string(), attr_name_loc(&namespace));
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

    fn gen_enclosure_tokens<'i, Elems>(
        l_kind: TokenKind,
        elems: Elems,
        expr_loc: PyLocation,
    ) -> (Token, Token)
    where
        Elems: Iterator<Item = &'i py_ast::Expr> + ExactSizeIterator,
    {
        let (l_cont, r_cont, r_kind) = match l_kind {
            TokenKind::LBrace => ("{", "}", TokenKind::RBrace),
            TokenKind::LParen => ("(", ")", TokenKind::RParen),
            TokenKind::LSqBr => ("[", "]", TokenKind::RSqBr),
            _ => unreachable!(),
        };
        let (l_end, c_end) = if elems.len() == 0 {
            (expr_loc.row.get(), expr_loc.column.get() - 1)
        } else {
            let last = elems.last().unwrap();
            (last.location().row.get(), last.location().column.get())
        };
        let l_brace = Token::new(
            l_kind,
            l_cont,
            expr_loc.row.get(),
            expr_loc.column.get() - 1,
        );
        let r_brace = Token::new(r_kind, r_cont, l_end, c_end);
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
                let value = format!("\"{value}\"");
                // column - 2 because of the quotes
                let token = Token::new(
                    TokenKind::StrLit,
                    value,
                    loc.row.get(),
                    loc.column.to_zero_indexed(),
                );
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
                let params = Params::new(vec![], None, vec![], None);
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
                let function = self.convert_expr(*call.func);
                let pos_args = call
                    .args
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let kw_args = call
                    .keywords
                    .into_iter()
                    .map(|Keyword { arg, value, range }| {
                        let name = arg.unwrap();
                        let name = Token::symbol_with_loc(name.to_string(), pyloc_to_ergloc(range));
                        let ex = self.convert_expr(value);
                        KwArg::new(name, None, ex)
                    })
                    .collect::<Vec<_>>();
                let last_col = pos_args
                    .last()
                    .and_then(|last| last.col_end())
                    .unwrap_or(function.col_end().unwrap_or(0) + 1);
                let paren = {
                    let lp = Token::new(
                        TokenKind::LParen,
                        "(",
                        loc.row.get(),
                        function.col_end().unwrap_or(0),
                    );
                    let rp = Token::new(TokenKind::RParen, ")", loc.row.get(), last_col);
                    (lp, rp)
                };
                let args = Args::new(pos_args, None, kw_args, Some(paren));
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
                self.grow("<lambda>".to_string());
                let params = self.convert_params(*lambda.args);
                let body = vec![self.convert_expr(*lambda.body)];
                self.pop();
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let op = Token::from_str(TokenKind::FuncArrow, "->");
                Expr::Lambda(Lambda::new(sig, op, Block::new(body), DefId(0)))
            }
            py_ast::Expr::List(list) => {
                let (l_sqbr, r_sqbr) =
                    Self::gen_enclosure_tokens(TokenKind::LSqBr, list.elts.iter(), list.location());
                let elements = list
                    .elts
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::pos_only(elements, None);
                let arr = Expr::Array(Array::Normal(NormalArray::new(l_sqbr, r_sqbr, elems)));
                Self::mutate_expr(arr)
            }
            py_ast::Expr::Set(set) => {
                let (l_brace, r_brace) =
                    Self::gen_enclosure_tokens(TokenKind::LBrace, set.elts.iter(), set.location());
                let elements = set
                    .elts
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::pos_only(elements, None);
                Expr::Set(Set::Normal(NormalSet::new(l_brace, r_brace, elems)))
                // Self::mutate_expr(set)
            }
            py_ast::Expr::Dict(dict) => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(
                    TokenKind::LBrace,
                    dict.values.iter(),
                    dict.location(),
                );
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
            py_ast::Expr::Tuple(tuple) => {
                let elements = tuple
                    .elts
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::pos_only(elements, None);
                Expr::Tuple(Tuple::Normal(NormalTuple::new(elems)))
            }
            py_ast::Expr::Subscript(subs) => {
                let obj = self.convert_expr(*subs.value);
                let method = obj.attr_expr(
                    self.convert_ident("__getitem__".to_string(), subs.slice.location()),
                );
                method.call1(self.convert_expr(*subs.slice))
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
        self.block_ids.push(self.block_id_counter);
        for (i, stmt) in block.into_iter().enumerate() {
            let is_last = i == len - 1;
            new_block.push(self.convert_statement(stmt, is_last && kind.is_function()));
        }
        self.block_ids.pop();
        Block::new(new_block)
    }

    fn check_init_sig(&mut self, sig: &Signature) -> Option<()> {
        match sig {
            Signature::Subr(subr) => {
                if let Some(first) = subr.params.non_defaults.get(0) {
                    if first.inspect().map(|s| &s[..]) == Some("self") {
                        return Some(());
                    }
                }
                self.errs.push(self_not_found_error(
                    self.cfg.input.clone(),
                    subr.loc(),
                    self.namespace.join("."),
                ));
                Some(())
            }
            Signature::Var(var) => {
                self.errs.push(init_var_error(
                    self.cfg.input.clone(),
                    var.loc(),
                    self.namespace.join("."),
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
    // requirement : {x: Int, y: Int, z: Never}
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
        let Signature::Subr(sig) = init_def.sig else { unreachable!() };
        let mut fields = vec![];
        let mut params = vec![];
        for chunk in init_def.body.block {
            #[allow(clippy::single_match)]
            match chunk {
                Expr::ReDef(redef) => {
                    let Accessor::Attr(attr) = redef.attr else { continue; };
                    // if `self.foo == ...`
                    if attr.obj.get_name().map(|s| &s[..]) == Some("self") {
                        let (param_typ_name, arg_typ_name) = if let Some(t_spec_op) = sig
                            .params
                            .non_defaults
                            .iter()
                            .find(|&param| param.inspect() == Some(attr.ident.inspect()))
                            .and_then(|param| param.t_spec.as_ref())
                        {
                            let typ_name = t_spec_op.t_spec.to_string().replace('.', "");
                            (typ_name.clone(), typ_name)
                        } else {
                            ("Obj".to_string(), "Never".to_string()) // accept any type, can be any type
                        };
                        let param_typ_ident = Identifier::public_with_line(
                            DOT,
                            param_typ_name.into(),
                            attr.obj.ln_begin().unwrap_or(0),
                        );
                        let param_typ_spec = TypeSpec::mono(param_typ_ident.clone());
                        let expr = Expr::Accessor(Accessor::Ident(param_typ_ident.clone()));
                        let as_op = Token::new(
                            TokenKind::As,
                            "as",
                            param_typ_spec.ln_begin().unwrap_or(0),
                            param_typ_spec.col_begin().unwrap_or(0),
                        );
                        let param_typ_spec = TypeSpecWithOp::new(as_op, param_typ_spec, expr);
                        let arg_typ_ident = Identifier::public_with_line(
                            DOT,
                            arg_typ_name.into(),
                            attr.obj.ln_begin().unwrap_or(0),
                        );
                        params.push(NonDefaultParamSignature::new(
                            ParamPattern::VarName(attr.ident.name.clone()),
                            Some(param_typ_spec),
                        ));
                        let typ = Expr::Accessor(Accessor::Ident(arg_typ_ident));
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
            VisModifierSpec::Public(DOT),
            VarName::from_static("__call__"),
        );
        let params = Params::new(params, None, vec![], None);
        let class_ident = Identifier::public_with_line(
            DOT,
            self.namespace.last().unwrap().into(),
            sig.ln_begin().unwrap_or(0),
        );
        let class_ident_expr = Expr::Accessor(Accessor::Ident(class_ident.clone()));
        let class_spec = TypeSpecWithOp::new(COLON, TypeSpec::mono(class_ident), class_ident_expr);
        let sig = Signature::Subr(SubrSignature::new(
            set! { Decorator(Expr::static_local("Override")) },
            call_ident,
            TypeBoundSpecs::empty(),
            params,
            Some(class_spec),
        ));
        let unreachable_acc =
            Identifier::new(VisModifierSpec::Public(DOT), VarName::from_static("exit"));
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        let def = Def::new(sig, body);
        Some(def)
    }

    fn gen_default_init(&self, line: usize) -> Def {
        let call_ident = Identifier::new(
            VisModifierSpec::Public(DOT),
            VarName::from_static("__call__"),
        );
        let params = Params::new(vec![], None, vec![], None);
        let class_ident =
            Identifier::public_with_line(DOT, self.namespace.last().unwrap().into(), line as u32);
        let class_ident_expr = Expr::Accessor(Accessor::Ident(class_ident.clone()));
        let class_spec = TypeSpecWithOp::new(COLON, TypeSpec::mono(class_ident), class_ident_expr);
        let sig = Signature::Subr(SubrSignature::new(
            set! { Decorator(Expr::static_local("Override")) },
            call_ident,
            TypeBoundSpecs::empty(),
            params,
            Some(class_spec),
        ));
        let unreachable_acc =
            Identifier::new(VisModifierSpec::Public(DOT), VarName::from_static("exit"));
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        Def::new(sig, body)
    }

    fn extract_method(
        &mut self,
        body: Vec<py_ast::Stmt>,
        inherit: bool,
    ) -> (Option<Expr>, ClassAttrs) {
        let mut base_type = None;
        let mut attrs = vec![];
        let mut init_is_defined = false;
        for stmt in body {
            match self.convert_statement(stmt, true) {
                Expr::Def(mut def) => {
                    if inherit {
                        if let Signature::Subr(subr) = &mut def.sig {
                            subr.decorators
                                .insert(Decorator(Expr::static_local("Override")));
                        }
                    }
                    if def
                        .sig
                        .ident()
                        .is_some_and(|id| &id.inspect()[..] == "__init__")
                    {
                        if let Some(call_def) = self.extract_init(&mut base_type, def) {
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
        inherit: bool,
    ) -> (Option<Expr>, Vec<Methods>) {
        let class = TypeSpec::mono(ident.clone());
        let class_as_expr = Expr::Accessor(Accessor::Ident(ident));
        let (base_type, attrs) = self.extract_method(body, inherit);
        let methods = Methods::new(class, class_as_expr, VisModifierSpec::Public(DOT), attrs);
        (base_type, vec![methods])
    }

    fn convert_funcdef(
        &mut self,
        name: String,
        params: Arguments,
        body: Vec<py_ast::Stmt>,
        decorator_list: Vec<py_ast::Expr>,
        returns: Option<py_ast::Expr>,
        range: PySourceRange,
    ) -> Expr {
        // if reassigning of a function referenced by other functions is occurred, it is an error
        if self.get_name(&name).is_some_and(|info| {
            info.defined_times > 0
                && info.defined_in == DefinedPlace::Known(self.cur_namespace())
                && !info.referenced.difference(&set! {name.clone()}).is_empty()
        }) {
            let err = reassign_func_error(
                self.cfg.input.clone(),
                pyloc_to_ergloc(range),
                self.namespace.join("."),
                &name,
            );
            self.errs.push(err);
            Expr::Dummy(Dummy::new(None, vec![]))
        } else {
            let loc = range.start;
            let decos = decorator_list
                .into_iter()
                .map(|ex| Decorator(self.convert_expr(ex)))
                .collect::<HashSet<_>>();
            self.register_name_info(&name, NameKind::Function);
            let func_name_loc = PyLocation {
                row: loc.row,
                column: loc.column.saturating_add(4),
            };
            let ident = self.convert_ident(name, func_name_loc);
            self.grow(ident.inspect().to_string());
            let params = self.convert_params(params);
            let return_t = returns.map(|ret| {
                let t_spec = self.convert_type_spec(ret.clone());
                let colon = Token::new(
                    TokenKind::Colon,
                    ":",
                    t_spec.ln_begin().unwrap_or(0),
                    t_spec.col_begin().unwrap_or(0),
                );
                TypeSpecWithOp::new(colon, t_spec, self.convert_expr(ret))
            });
            let sig = Signature::Subr(SubrSignature::new(
                decos,
                ident,
                TypeBoundSpecs::empty(),
                params,
                return_t,
            ));
            let block = self.convert_block(body, BlockKind::Function);
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
    fn convert_classdef(
        &mut self,
        name: String,
        body: Vec<py_ast::Stmt>,
        bases: Vec<py_ast::Expr>,
        decorator_list: Vec<py_ast::Expr>,
        loc: PyLocation,
    ) -> Expr {
        let _decos = decorator_list
            .into_iter()
            .map(|deco| self.convert_expr(deco))
            .collect::<Vec<_>>();
        let mut bases = bases
            .into_iter()
            .map(|base| self.convert_expr(base))
            .collect::<Vec<_>>();
        let inherit = !bases.is_empty();
        self.register_name_info(&name, NameKind::Class);
        let class_name_loc = PyLocation {
            row: loc.row,
            column: loc.column.saturating_add(6),
        };
        let ident = self.convert_ident(name, class_name_loc);
        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident.clone()), None));
        self.grow(ident.inspect().to_string());
        let (base_type, methods) = self.extract_method_list(ident, body, inherit);
        let classdef = if inherit {
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

    fn convert_statement(&mut self, stmt: Stmt, dont_call_return: bool) -> Expr {
        match stmt {
            py_ast::Stmt::Expr(stmt) => self.convert_expr(*stmt.value),
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
                            let block = Block::new(vec![self.convert_expr(*value)]);
                            let body = DefBody::new(EQUAL, block, DefId(0));
                            // must register after convert_expr because value may be contain name (e.g. i = i + 1)
                            self.register_name_info(name.id.as_str(), NameKind::Variable);
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            let sig = Signature::Var(VarSignature::new(
                                VarPattern::Ident(ident),
                                Some(t_spec),
                            ));
                            let def = Def::new(sig, body);
                            Expr::Def(def)
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
                            let redef = ReDef::new(attr, expr);
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
                            let can_shadow = self.register_name_info(&name.id, NameKind::Variable);
                            let ident = self.convert_ident(name.id.to_string(), name.location());
                            if can_shadow.is_yes() {
                                let block = Block::new(vec![expr]);
                                let body = DefBody::new(EQUAL, block, DefId(0));
                                let sig = Signature::Var(VarSignature::new(
                                    VarPattern::Ident(ident),
                                    None,
                                ));
                                let def = Def::new(sig, body);
                                Expr::Def(def)
                            } else {
                                let redef = ReDef::new(Accessor::Ident(ident), expr);
                                Expr::ReDef(redef)
                            }
                        }
                        py_ast::Expr::Attribute(attr) => {
                            let value = self.convert_expr(*attr.value);
                            let ident = self
                                .convert_attr_ident(attr.attr.to_string(), attr_name_loc(&value));
                            let attr = value.attr(ident);
                            let expr = self.convert_expr(*assign.value);
                            let adef = ReDef::new(attr, expr);
                            Expr::ReDef(adef)
                        }
                        py_ast::Expr::Tuple(tuple) => {
                            let tmp = FRESH_GEN.fresh_varname();
                            let tmp_name =
                                VarName::from_str_and_line(tmp, tuple.location().row.get());
                            let tmp_ident = Identifier::new(VisModifierSpec::Public(DOT), tmp_name);
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
                                self.register_name_info(&name.id, NameKind::Variable);
                                let ident =
                                    self.convert_ident(name.id.to_string(), name.location());
                                let sig = Signature::Var(VarSignature::new(
                                    VarPattern::Ident(ident),
                                    None,
                                ));
                                let def = Expr::Def(Def::new(sig, body));
                                defs.push(def);
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
                            let redef = ReDef::new(Accessor::Ident(ident), Expr::BinOp(bin));
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
                        let redef = ReDef::new(attr, Expr::BinOp(bin));
                        Expr::ReDef(redef)
                    }
                    other => {
                        log!(err "{other:?} as LHS");
                        Expr::Dummy(Dummy::new(None, vec![]))
                    }
                }
            }
            py_ast::Stmt::FunctionDef(func_def) => self.convert_funcdef(
                func_def.name.to_string(),
                *func_def.args,
                func_def.body,
                func_def.decorator_list,
                func_def.returns.map(|x| *x),
                func_def.range,
            ),
            py_ast::Stmt::ClassDef(class_def) => {
                let class_loc = class_def.location();
                self.convert_classdef(
                    class_def.name.to_string(),
                    class_def.body,
                    class_def.bases,
                    class_def.decorator_list,
                    class_loc,
                )
            }
            py_ast::Stmt::For(for_) => {
                let loc = for_.location();
                let iter = self.convert_expr(*for_.iter);
                let block = self.convert_for_body(Some(*for_.target), for_.body);
                let for_ident = self.convert_ident("for".to_string(), loc);
                let for_acc = Expr::Accessor(Accessor::Ident(for_ident));
                for_acc.call2(iter, Expr::Lambda(block))
            }
            py_ast::Stmt::While(while_) => {
                let loc = while_.location();
                let test = self.convert_expr(*while_.test);
                let params = Params::new(vec![], None, vec![], None);
                let empty_sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let block = self.convert_block(while_.body, BlockKind::While);
                let body = Lambda::new(empty_sig, Token::DUMMY, block, DefId(0));
                let while_ident = self.convert_ident("while".to_string(), loc);
                let while_acc = Expr::Accessor(Accessor::Ident(while_ident));
                while_acc.call2(test, Expr::Lambda(body))
            }
            py_ast::Stmt::If(if_) => {
                let loc = if_.location();
                let block = self.convert_block(if_.body, BlockKind::If);
                let params = Params::new(vec![], None, vec![], None);
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, block, DefId(0));
                let test = self.convert_expr(*if_.test);
                let if_ident = self.convert_ident("if".to_string(), loc);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                if !if_.orelse.is_empty() {
                    let else_block = self.convert_block(if_.orelse, BlockKind::If);
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
                let loc = return_.location();
                let value = return_
                    .value
                    .map(|val| self.convert_expr(*val))
                    .unwrap_or_else(|| Expr::Tuple(Tuple::Normal(NormalTuple::new(Args::empty()))));
                if dont_call_return {
                    value
                } else {
                    let func_acc = Expr::Accessor(Accessor::Ident(
                        self.convert_ident(self.namespace.last().unwrap().clone(), loc),
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
                let loc = import.location();
                let mut imports = vec![];
                for name in import.names {
                    let import_acc = Expr::Accessor(Accessor::Ident(
                        self.convert_ident("__import__".to_string(), loc),
                    ));
                    let cont = if name.asname.is_some() {
                        format!("\"{}\"", name.name.replace('.', "/"))
                    } else {
                        format!("\"{}\"", name.name.split('.').next().unwrap())
                    };
                    let mod_name = Expr::Literal(Literal::new(Token::new(
                        TokenKind::StrLit,
                        cont,
                        name.location().row.get(),
                        name.location().column.get() - 1,
                    )));
                    let call = import_acc.call1(mod_name);
                    let loc = name.location();
                    let def = if let Some(alias) = name.asname {
                        self.register_name_info(&alias, NameKind::Variable);
                        let var = VarSignature::new(
                            VarPattern::Ident(self.convert_ident(alias.to_string(), loc)),
                            None,
                        );
                        Def::new(
                            Signature::Var(var),
                            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                        )
                    } else {
                        self.register_name_info(&name.name, NameKind::Variable);
                        let var = VarSignature::new(
                            VarPattern::Ident(
                                self.convert_ident(name.name.to_string(), name.location()),
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
                let loc = import_from.location();
                self.convert_from_import(import_from.module, import_from.names, loc)
            }
            py_ast::Stmt::Try(try_) => {
                let chunks = self.convert_block(try_.body, BlockKind::Try).into_iter();
                let dummy = chunks
                    .chain(self.convert_block(try_.orelse, BlockKind::Try))
                    .chain(self.convert_block(try_.finalbody, BlockKind::Try))
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
            _other => {
                log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::new(None, vec![]))
            }
        }
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
        let cont = if module == "." {
            "\"__init__\"".to_string()
        } else {
            format!("\"{module}\"")
        };
        let mod_name = Expr::Literal(Literal::new(Token::new(
            TokenKind::StrLit,
            cont,
            location.row.get(),
            location.column.to_zero_indexed(),
        )));
        let call = import_acc.clone().call1(mod_name);
        let mut exprs = vec![];
        let mut imports = vec![];
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
            if let Ok(_path) = name_path {
                let cont = format!("\"{module}/{}\"", name.name);
                let mod_name = Expr::Literal(Literal::new(Token::new(
                    TokenKind::StrLit,
                    cont,
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
                imports.push(VarRecordAttr::new(true_name, alias));
            }
        }
        let attrs = VarRecordAttrs::new(imports);
        let pat = VarRecordPattern::new(Token::DUMMY, attrs, Token::DUMMY);
        let var = VarSignature::new(VarPattern::Record(pat), None);
        let def = Expr::Def(Def::new(
            Signature::Var(var),
            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
        ));
        if exprs.is_empty() {
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
