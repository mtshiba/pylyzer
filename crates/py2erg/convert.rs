use erg_common::config::ErgConfig;
use erg_common::dict::Dict as HashMap;
use erg_common::fresh::fresh_varname;
use erg_common::set::Set as HashSet;
use erg_common::traits::{Locational, Stream};
use erg_common::{log, set};
use erg_compiler::artifact::IncompleteArtifact;
use erg_compiler::erg_parser::ast::{
    Accessor, Args, Array, BinOp, Block, ClassAttr, ClassAttrs, ClassDef, ConstArgs, Decorator,
    Def, DefBody, DefId, DefaultParamSignature, Dict, Dummy, Expr, Identifier, KeyValue, Lambda,
    LambdaSignature, Literal, Methods, Module, NonDefaultParamSignature, NormalArray, NormalDict,
    NormalRecord, NormalSet, NormalTuple, ParamPattern, Params, PosArg, PreDeclTypeSpec, ReDef,
    Record, RecordAttrs, Set, Signature, SimpleTypeSpec, SubrSignature, Tuple, TypeAscription,
    TypeBoundSpecs, TypeSpec, TypeSpecWithOp, UnaryOp, VarName, VarPattern, VarSignature,
};
use erg_compiler::erg_parser::token::{Token, TokenKind, COLON, DOT, EQUAL};
use erg_compiler::error::CompileErrors;
use rustpython_parser::ast::Location as PyLocation;
use rustpython_parser::ast::{
    BooleanOperator, Comparison, ExpressionType, Located, Number, Operator, Parameter, Parameters,
    Program, StatementType, StringGroup, Suite, UnaryOperator,
};

use crate::error::*;

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

/// Variables are automatically rewritten with `py_compatible`,
/// but types are rewritten here because they are complex components used inseparably in the Erg system.
fn escape_name(name: String) -> String {
    match &name[..] {
        "int" => "Int".into(),
        "float" => "Float".into(),
        "str" => "Str".into(),
        "bool" => "Bool".into(),
        "list" => "GenericArray".into(),
        // "range" => "GenericRange".into(),
        "dict" => "GenericDict".into(),
        "set" => "GenericSet".into(),
        "tuple" => "GenericTuple".into(),
        "type" => "Type".into(),
        "ModuleType" => "GeneticModule".into(),
        "Obj" => "Obj\0".into(),
        "Int" => "Int\0".into(),
        "Nat" => "Nat\0".into(),
        "Ratio" => "Ratio\0".into(),
        "Float" => "Float\0".into(),
        "Bool" => "Bool\0".into(),
        "Str" => "Str\0".into(),
        "NoneType" => "NoneType\0".into(),
        "Code" => "Code\0".into(),
        "Frame" => "Frame\0".into(),
        "Error" => "Error\0".into(),
        "Inf" => "Inf\0".into(),
        "NegInf" => "NegInf\0".into(),
        "Type" => "Type\0".into(),
        "ClassType" => "ClassType\0".into(),
        "TraitType" => "TraitType\0".into(),
        "Patch" => "Patch\0".into(),
        "NotImplemented" => "NotImplemented\0".into(),
        "Ellipsis" => "Ellipsis\0".into(),
        "Never" => "Never\0".into(),
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
        Operator::BitOr => (TokenKind::BitOr, "|"),
        Operator::BitXor => (TokenKind::BitXor, "^"),
        Operator::BitAnd => (TokenKind::BitAnd, "&"),
        Operator::FloorDiv => (TokenKind::FloorDiv, "//"),
        Operator::MatMult => (TokenKind::AtSign, "@"),
    };
    Token::from_str(kind, cont)
}

pub fn pyloc_to_ergloc(loc: PyLocation, cont_len: usize) -> erg_common::error::Location {
    erg_common::error::Location::range(
        loc.row() as u32,
        loc.column() as u32,
        loc.row() as u32,
        (loc.column() + cont_len) as u32,
    )
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
/// ```
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
    names: HashMap<String, NameInfo>,
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
            names: HashMap::new(),
            warns: CompileErrors::empty(),
            errs: CompileErrors::empty(),
        }
    }

    fn get_name(&self, name: &str) -> Option<&NameInfo> {
        self.names.get(name)
    }

    fn get_mut_name(&mut self, name: &str) -> Option<&mut NameInfo> {
        self.names.get_mut(name)
    }

    fn cur_block_id(&self) -> usize {
        *self.block_ids.last().unwrap()
    }

    fn cur_namespace(&self) -> String {
        self.namespace.last().unwrap().clone()
    }

    fn register_name_info(&mut self, name: &str, kind: NameKind) {
        let cur_namespace = self.cur_namespace();
        if let Some(name_info) = self.names.get_mut(name) {
            if name_info.defined_in == cur_namespace {
                name_info.defined_times += 1;
            } else if name_info.defined_in.is_unknown() {
                name_info.defined_in = DefinedPlace::Known(cur_namespace);
                name_info.defined_times += 1;
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
            self.names.insert(String::from(name), info);
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
            self.names.insert(name.clone(), info);
            name
        };
        let token = Token::new(
            TokenKind::Symbol,
            cont,
            loc.row() as u32,
            loc.column() as u32 - 1,
        );
        let name = VarName::new(token);
        let dot = Token::new(
            TokenKind::Dot,
            ".",
            loc.row() as u32,
            loc.column() as u32 - 1,
        );
        Identifier::new(Some(dot), name)
    }

    // TODO: module member mangling
    fn convert_attr_ident(&mut self, name: String, loc: PyLocation) -> Identifier {
        let token = Token::new(
            TokenKind::Symbol,
            name,
            loc.row() as u32,
            loc.column() as u32 - 1,
        );
        let name = VarName::new(token);
        let dot = Token::new(
            TokenKind::Dot,
            ".",
            loc.row() as u32,
            loc.column() as u32 - 1,
        );
        Identifier::new(Some(dot), name)
    }

    fn convert_param_pattern(&mut self, arg: String, loc: PyLocation) -> ParamPattern {
        self.register_name_info(&arg, NameKind::Variable);
        let ident = self.convert_ident(arg, loc);
        ParamPattern::VarName(ident.name)
    }

    fn convert_nd_param(&mut self, param: Parameter) -> NonDefaultParamSignature {
        let pat = self.convert_param_pattern(param.arg, param.location);
        let t_spec = param
            .annotation
            .map(|anot| self.convert_type_spec(*anot))
            .map(|t_spec| TypeSpecWithOp::new(COLON, t_spec));
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn _convert_default_param(_param: Parameter) -> DefaultParamSignature {
        todo!()
    }

    // TODO: defaults
    fn convert_params(&mut self, args: Box<Parameters>) -> Params {
        let non_defaults = args
            .args
            .into_iter()
            .map(|p| self.convert_nd_param(p))
            .collect();
        // let defaults = args. args.defaults.into_iter().map(convert_default_param).collect();
        Params::new(non_defaults, None, vec![], None)
    }

    fn convert_for_param(&mut self, name: String, loc: PyLocation) -> NonDefaultParamSignature {
        let pat = self.convert_param_pattern(name, loc);
        let t_spec = None;
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn param_pattern_to_var(pat: ParamPattern) -> VarPattern {
        match pat {
            ParamPattern::VarName(name) => VarPattern::Ident(Identifier::new(Some(DOT), name)),
            ParamPattern::Discard(token) => VarPattern::Discard(token),
            other => todo!("{other}"),
        }
    }

    /// (i, j) => $1 (i = $1[0]; j = $1[1])
    fn convert_expr_to_param(
        &mut self,
        expr: Located<ExpressionType>,
    ) -> (NonDefaultParamSignature, Vec<Expr>) {
        match expr.node {
            ExpressionType::Identifier { name } => {
                (self.convert_for_param(name, expr.location), vec![])
            }
            ExpressionType::Tuple { elements } => {
                let tmp = fresh_varname();
                let tmp_name =
                    VarName::from_str_and_line((&tmp).into(), expr.location.row() as u32);
                let tmp_expr = Expr::Accessor(Accessor::Ident(Identifier::new(
                    Some(DOT),
                    tmp_name.clone(),
                )));
                let mut block = vec![];
                for (i, elem) in elements.into_iter().enumerate() {
                    let index = Literal::new(Token::new(
                        TokenKind::NatLit,
                        i.to_string(),
                        elem.location.row() as u32,
                        elem.location.column() as u32 - 1,
                    ));
                    let (param, mut blocks) = self.convert_expr_to_param(elem);
                    let sig = Signature::Var(VarSignature::new(
                        Self::param_pattern_to_var(param.pat),
                        param.t_spec.map(|t| t.t_spec),
                    ));
                    let method = tmp_expr.clone().attr_expr(
                        self.convert_ident("__Tuple_getitem__".to_string(), expr.location),
                    );
                    let args = Args::new(vec![PosArg::new(Expr::Lit(index))], vec![], None);
                    let tuple_acc = method.call_expr(args);
                    let body = DefBody::new(EQUAL, Block::new(vec![tuple_acc]), DefId(0));
                    let def = Expr::Def(Def::new(sig, body));
                    block.push(def);
                    block.append(&mut blocks);
                }
                let pat = ParamPattern::VarName(tmp_name);
                (NonDefaultParamSignature::new(pat, None), block)
            }
            _other => {
                let token = Token::new(
                    TokenKind::UBar,
                    "_",
                    expr.location.row() as u32,
                    expr.location.column() as u32 - 1,
                );
                (
                    NonDefaultParamSignature::new(ParamPattern::Discard(token), None),
                    vec![],
                )
            }
        }
    }

    fn convert_for_body(&mut self, lhs: Located<ExpressionType>, body: Suite) -> Lambda {
        let (param, block) = self.convert_expr_to_param(lhs);
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
        let op = Token::from_str(TokenKind::ProcArrow, "=>");
        Lambda::new(sig, op, Block::new(body), DefId(0))
    }

    fn convert_type_spec(&mut self, expr: Located<ExpressionType>) -> TypeSpec {
        match expr.node {
            ExpressionType::Identifier { name } => TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(
                SimpleTypeSpec::new(self.convert_ident(name, expr.location), ConstArgs::empty()),
            )),
            _other => TypeSpec::Infer(Token::new(
                TokenKind::UBar,
                "_",
                expr.location.row() as u32,
                expr.location.column() as u32 - 1,
            )),
        }
    }

    fn gen_enclosure_tokens<'i, Elems>(
        l_kind: TokenKind,
        elems: Elems,
        expr_loc: PyLocation,
    ) -> (Token, Token)
    where
        Elems: Iterator<Item = &'i Located<ExpressionType>> + ExactSizeIterator,
    {
        let (l_cont, r_cont, r_kind) = match l_kind {
            TokenKind::LBrace => ("{", "}", TokenKind::RBrace),
            TokenKind::LParen => ("(", ")", TokenKind::RParen),
            TokenKind::LSqBr => ("[", "]", TokenKind::RSqBr),
            _ => unreachable!(),
        };
        let (l_end, c_end) = if elems.len() == 0 {
            (expr_loc.row(), expr_loc.column() - 1)
        } else {
            let last = elems.last().unwrap();
            (last.location.row(), last.location.column())
        };
        let l_brace = Token::new(
            l_kind,
            l_cont,
            expr_loc.row() as u32,
            expr_loc.column() as u32 - 1,
        );
        let r_brace = Token::new(r_kind, r_cont, l_end as u32, c_end as u32);
        (l_brace, r_brace)
    }

    fn _mutate_expr(expr: Expr) -> Expr {
        let mut_op = Token::new(
            TokenKind::Mutate,
            "!",
            expr.ln_begin().unwrap_or(0),
            expr.col_begin().unwrap_or(0),
        );
        Expr::UnaryOp(UnaryOp::new(mut_op, expr))
    }

    fn convert_expr(&mut self, expr: Located<ExpressionType>) -> Expr {
        match expr.node {
            ExpressionType::Number { value } => {
                let (kind, cont) = match value {
                    Number::Integer { value } => (TokenKind::IntLit, value.to_string()),
                    Number::Float { value } => (TokenKind::RatioLit, value.to_string()),
                    Number::Complex { .. } => {
                        return Expr::Dummy(Dummy::new(vec![]));
                    }
                };
                let token = Token::new(
                    kind,
                    cont,
                    expr.location.row() as u32,
                    expr.location.column() as u32 - 1,
                );
                Expr::Lit(Literal::new(token))
            }
            ExpressionType::String { value } => {
                let StringGroup::Constant{ value } = value else {
                    return Expr::Dummy(Dummy::new(vec![]));
                };
                let value = format!("\"{value}\"");
                // column - 2 because of the quotes
                let token = Token::new(
                    TokenKind::StrLit,
                    value,
                    expr.location.row() as u32,
                    expr.location.column() as u32 - 2,
                );
                Expr::Lit(Literal::new(token))
            }
            ExpressionType::False => Expr::Lit(Literal::new(Token::new(
                TokenKind::BoolLit,
                "False",
                expr.location.row() as u32,
                expr.location.column() as u32 - 1,
            ))),
            ExpressionType::True => Expr::Lit(Literal::new(Token::new(
                TokenKind::BoolLit,
                "True",
                expr.location.row() as u32,
                expr.location.column() as u32 - 1,
            ))),
            ExpressionType::None => Expr::Lit(Literal::new(Token::new(
                TokenKind::NoneLit,
                "None",
                expr.location.row() as u32,
                expr.location.column() as u32 - 1,
            ))),
            ExpressionType::Ellipsis => Expr::Lit(Literal::new(Token::new(
                TokenKind::EllipsisLit,
                "Ellipsis",
                expr.location.row() as u32,
                expr.location.column() as u32 - 1,
            ))),
            ExpressionType::IfExpression { test, body, orelse } => {
                let block = self.convert_expr(*body);
                let params = Params::new(vec![], None, vec![], None);
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, Block::new(vec![block]), DefId(0));
                let test = self.convert_expr(*test);
                let if_ident = self.convert_ident("if".to_string(), expr.location);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                let else_block = self.convert_expr(*orelse);
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let else_body =
                    Lambda::new(sig, Token::DUMMY, Block::new(vec![else_block]), DefId(0));
                let args = Args::new(
                    vec![
                        PosArg::new(test),
                        PosArg::new(Expr::Lambda(body)),
                        PosArg::new(Expr::Lambda(else_body)),
                    ],
                    vec![],
                    None,
                );
                if_acc.call_expr(args)
            }
            ExpressionType::Call {
                function,
                args,
                keywords: _,
            } => {
                let function = self.convert_expr(*function);
                let pos_args = args
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let args = Args::new(pos_args, vec![], None);
                function.call_expr(args)
            }
            ExpressionType::Binop { a, op, b } => {
                let lhs = self.convert_expr(*a);
                let rhs = self.convert_expr(*b);
                let op = op_to_token(op);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            ExpressionType::Unop { op, a } => {
                let rhs = self.convert_expr(*a);
                let (kind, cont) = match op {
                    UnaryOperator::Pos => (TokenKind::PrePlus, "+"),
                    // UnaryOperator::Not => (TokenKind::PreBitNot, "not"),
                    UnaryOperator::Neg => (TokenKind::PreMinus, "-"),
                    UnaryOperator::Inv => (TokenKind::PreBitNot, "~"),
                    _ => return Expr::Dummy(Dummy::new(vec![rhs])),
                };
                let op = Token::from_str(kind, cont);
                Expr::UnaryOp(UnaryOp::new(op, rhs))
            }
            // TODO
            ExpressionType::BoolOp { op, mut values } => {
                let lhs = self.convert_expr(values.remove(0));
                let rhs = self.convert_expr(values.remove(0));
                let (kind, cont) = match op {
                    BooleanOperator::And => (TokenKind::AndOp, "and"),
                    BooleanOperator::Or => (TokenKind::OrOp, "or"),
                };
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            // TODO: multiple comparisons
            ExpressionType::Compare { mut vals, mut ops } => {
                let lhs = self.convert_expr(vals.remove(0));
                let rhs = self.convert_expr(vals.remove(0));
                let (kind, cont) = match ops.remove(0) {
                    Comparison::Equal => (TokenKind::Equal, "=="),
                    Comparison::NotEqual => (TokenKind::NotEq, "!="),
                    Comparison::Less => (TokenKind::Less, "<"),
                    Comparison::LessOrEqual => (TokenKind::LessEq, "<="),
                    Comparison::Greater => (TokenKind::Gre, ">"),
                    Comparison::GreaterOrEqual => (TokenKind::GreEq, ">="),
                    Comparison::Is => (TokenKind::IsOp, "is!"),
                    Comparison::IsNot => (TokenKind::IsNotOp, "isnot!"),
                    Comparison::In => (TokenKind::InOp, "in"),
                    Comparison::NotIn => (TokenKind::NotInOp, "notin"),
                };
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            ExpressionType::Identifier { name } => {
                let ident = self.convert_ident(name, expr.location);
                Expr::Accessor(Accessor::Ident(ident))
            }
            ExpressionType::Attribute { value, name } => {
                let obj = self.convert_expr(*value);
                let name = self.convert_attr_ident(name, expr.location);
                obj.attr_expr(name)
            }
            ExpressionType::Lambda { args, body } => {
                self.namespace.push("<lambda>".to_string());
                let params = self.convert_params(args);
                let body = vec![self.convert_expr(*body)];
                self.namespace.pop();
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let op = Token::from_str(TokenKind::ProcArrow, "=>");
                Expr::Lambda(Lambda::new(sig, op, Block::new(body), DefId(0)))
            }
            ExpressionType::List { elements } => {
                let (l_sqbr, r_sqbr) =
                    Self::gen_enclosure_tokens(TokenKind::LSqBr, elements.iter(), expr.location);
                let elements = elements
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::new(elements, vec![], None);
                Expr::Array(Array::Normal(NormalArray::new(l_sqbr, r_sqbr, elems)))
                // Self::mutate_expr(arr)
            }
            ExpressionType::Set { elements } => {
                let (l_brace, r_brace) =
                    Self::gen_enclosure_tokens(TokenKind::LBrace, elements.iter(), expr.location);
                let elements = elements
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::new(elements, vec![], None);
                Expr::Set(Set::Normal(NormalSet::new(l_brace, r_brace, elems)))
                // Self::mutate_expr(set)
            }
            ExpressionType::Dict { elements } => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(
                    TokenKind::LBrace,
                    elements.iter().map(|(_, v)| v),
                    expr.location,
                );
                let kvs = elements
                    .into_iter()
                    .map(|(k, v)| {
                        KeyValue::new(
                            k.map(|k| self.convert_expr(k))
                                .unwrap_or(Expr::Dummy(Dummy::empty())),
                            self.convert_expr(v),
                        )
                    })
                    .collect::<Vec<_>>();
                Expr::Dict(Dict::Normal(NormalDict::new(l_brace, r_brace, kvs)))
                // Self::mutate_expr(dict)
            }
            ExpressionType::Tuple { elements } => {
                let elements = elements
                    .into_iter()
                    .map(|ex| PosArg::new(self.convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::new(elements, vec![], None);
                Expr::Tuple(Tuple::Normal(NormalTuple::new(elems)))
            }
            ExpressionType::Subscript { a, b } => {
                let obj = self.convert_expr(*a);
                let method =
                    obj.attr_expr(self.convert_ident("__getitem__".to_string(), expr.location));
                let args = Args::new(vec![PosArg::new(self.convert_expr(*b))], vec![], None);
                method.call_expr(args)
            }
            _other => {
                log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::empty())
            }
        }
    }

    fn convert_block(&mut self, block: Vec<Located<StatementType>>, kind: BlockKind) -> Block {
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
    // {x: Int, y: Int, z: Never}, .__call__(x: Int, y: Int, z: Obj): Self = .unreachable()
    fn extract_init(&mut self, init_def: Def) -> Option<(Expr, Def)> {
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
                    let Accessor::Attr(attr) = redef.attr else { break; };
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
                        let param_typ_spec = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(
                            SimpleTypeSpec::new(param_typ_ident.clone(), ConstArgs::empty()),
                        ));
                        let param_typ_spec = TypeSpecWithOp::new(COLON, param_typ_spec);
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
        let record = Record::Normal(NormalRecord::new(
            l_brace,
            r_brace,
            RecordAttrs::new(fields),
        ));
        let call_ident = Identifier::new(Some(DOT), VarName::from_static("__call__"));
        let params = Params::new(params, None, vec![], None);
        let class_ident = Identifier::public_with_line(
            DOT,
            self.namespace.last().unwrap().into(),
            sig.ln_begin().unwrap_or(0),
        );
        let class_spec = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(
            class_ident,
            ConstArgs::empty(),
        )));
        let sig = Signature::Subr(SubrSignature::new(
            HashSet::new(),
            call_ident,
            TypeBoundSpecs::empty(),
            params,
            Some(class_spec),
        ));
        let unreachable_acc = Identifier::new(Some(DOT), VarName::from_static("exit"));
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        let def = Def::new(sig, body);
        Some((Expr::Record(record), def))
    }

    fn gen_default_init(&self, line: usize) -> Def {
        let call_ident = Identifier::new(Some(DOT), VarName::from_static("__call__"));
        let params = Params::new(vec![], None, vec![], None);
        let class_ident =
            Identifier::public_with_line(DOT, self.namespace.last().unwrap().into(), line as u32);
        let class_spec = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(
            class_ident,
            ConstArgs::empty(),
        )));
        let sig = Signature::Subr(SubrSignature::new(
            HashSet::new(),
            call_ident,
            TypeBoundSpecs::empty(),
            params,
            Some(class_spec),
        ));
        let unreachable_acc = Identifier::new(Some(DOT), VarName::from_static("exit"));
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        Def::new(sig, body)
    }

    fn extract_method(&mut self, body: Vec<Located<StatementType>>) -> (Option<Expr>, ClassAttrs) {
        let mut base_type = None;
        let mut attrs = vec![];
        let mut init_is_defined = false;
        for stmt in body {
            let chunk = self.convert_statement(stmt, true);
            match chunk {
                Expr::Def(def) => {
                    if def
                        .sig
                        .ident()
                        .map(|id| &id.inspect()[..] == "__init__")
                        .unwrap_or(false)
                    {
                        if let Some((base_t, init_def)) = self.extract_init(def) {
                            base_type = Some(base_t);
                            attrs.push(ClassAttr::Def(init_def));
                            init_is_defined = true;
                        }
                    } else {
                        attrs.push(ClassAttr::Def(def));
                    }
                }
                Expr::TypeAsc(type_asc) => attrs.push(ClassAttr::Decl(type_asc)),
                _other => {} // TODO:
            }
        }
        if !init_is_defined {
            attrs.push(ClassAttr::Def(self.gen_default_init(0)));
        }
        (base_type, ClassAttrs::new(attrs))
    }

    fn extract_method_list(
        &mut self,
        ident: Identifier,
        body: Vec<Located<StatementType>>,
    ) -> (Option<Expr>, Vec<Methods>) {
        let class = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(
            ident,
            ConstArgs::empty(),
        )));
        let (base_type, attrs) = self.extract_method(body);
        let methods = Methods::new(class, DOT, attrs);
        (base_type, vec![methods])
    }

    fn convert_classdef(
        &mut self,
        name: String,
        body: Vec<Located<StatementType>>,
        bases: Vec<Located<ExpressionType>>,
        decorator_list: Vec<Located<ExpressionType>>,
        loc: PyLocation,
    ) -> Expr {
        let _decos = decorator_list
            .into_iter()
            .map(|deco| self.convert_expr(deco))
            .collect::<Vec<_>>();
        let _bases = bases
            .into_iter()
            .map(|base| self.convert_expr(base))
            .collect::<Vec<_>>();
        self.register_name_info(&name, NameKind::Class);
        let ident = self.convert_ident(name, loc);
        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident.clone()), None));
        self.namespace.push(ident.inspect().to_string());
        let (base_type, methods) = self.extract_method_list(ident, body);
        let pos_args = if let Some(base) = base_type {
            vec![PosArg::new(base)]
        } else {
            vec![]
        };
        let args = Args::new(pos_args, vec![], None);
        let class_acc = Expr::Accessor(Accessor::Ident(
            self.convert_ident("Class".to_string(), loc),
        ));
        let class_call = class_acc.call_expr(args);
        let body = DefBody::new(EQUAL, Block::new(vec![class_call]), DefId(0));
        let def = Def::new(sig, body);
        let classdef = ClassDef::new(def, methods);
        self.namespace.pop();
        Expr::ClassDef(classdef)
    }

    fn convert_statement(&mut self, stmt: Located<StatementType>, dont_call_return: bool) -> Expr {
        match stmt.node {
            StatementType::Expression { expression } => self.convert_expr(expression),
            StatementType::AnnAssign {
                target,
                annotation,
                value,
            } => {
                let t_spec = self.convert_type_spec(*annotation);
                match target.node {
                    ExpressionType::Identifier { name } => {
                        if let Some(value) = value {
                            let block = Block::new(vec![self.convert_expr(value)]);
                            let body = DefBody::new(EQUAL, block, DefId(0));
                            // must register after convert_expr because value may be contain name (e.g. i = i + 1)
                            self.register_name_info(&name, NameKind::Variable);
                            let ident = self.convert_ident(name, stmt.location);
                            let sig = Signature::Var(VarSignature::new(
                                VarPattern::Ident(ident),
                                Some(t_spec),
                            ));
                            let def = Def::new(sig, body);
                            Expr::Def(def)
                        } else {
                            // no registration because it's just a type ascription
                            let ident = self.convert_ident(name, stmt.location);
                            let tasc = TypeAscription::new(
                                Expr::Accessor(Accessor::Ident(ident)),
                                COLON,
                                t_spec,
                            );
                            Expr::TypeAsc(tasc)
                        }
                    }
                    ExpressionType::Attribute { value: attr, name } => {
                        let attr = self
                            .convert_expr(*attr)
                            .attr(self.convert_attr_ident(name, target.location));
                        if let Some(value) = value {
                            let expr = self.convert_expr(value);
                            let redef = ReDef::new(attr, expr);
                            Expr::ReDef(redef)
                        } else {
                            let tasc = TypeAscription::new(Expr::Accessor(attr), COLON, t_spec);
                            Expr::TypeAsc(tasc)
                        }
                    }
                    _other => Expr::Dummy(Dummy::empty()),
                }
            }
            StatementType::Assign { mut targets, value } => {
                if targets.len() == 1 {
                    let lhs = targets.remove(0);
                    match lhs.node {
                        ExpressionType::Identifier { name } => {
                            let block = Block::new(vec![self.convert_expr(value)]);
                            let body = DefBody::new(EQUAL, block, DefId(0));
                            self.register_name_info(&name, NameKind::Variable);
                            let ident = self.convert_ident(name, stmt.location);
                            let sig =
                                Signature::Var(VarSignature::new(VarPattern::Ident(ident), None));
                            let def = Def::new(sig, body);
                            Expr::Def(def)
                        }
                        ExpressionType::Attribute { value: attr, name } => {
                            let attr = self
                                .convert_expr(*attr)
                                .attr(self.convert_attr_ident(name, lhs.location));
                            let expr = self.convert_expr(value);
                            let adef = ReDef::new(attr, expr);
                            Expr::ReDef(adef)
                        }
                        ExpressionType::Tuple { elements } => {
                            let tmp = fresh_varname();
                            let tmp_name = VarName::from_str_and_line(
                                (&tmp).into(),
                                stmt.location.row() as u32,
                            );
                            let tmp_ident = Identifier::new(Some(DOT), tmp_name);
                            let tmp_expr = Expr::Accessor(Accessor::Ident(tmp_ident.clone()));
                            let sig = Signature::Var(VarSignature::new(
                                VarPattern::Ident(tmp_ident),
                                None,
                            ));
                            let body = DefBody::new(
                                EQUAL,
                                Block::new(vec![self.convert_expr(value)]),
                                DefId(0),
                            );
                            let tmp_def = Expr::Def(Def::new(sig, body));
                            let mut defs = vec![tmp_def];
                            for (i, elem) in elements.into_iter().enumerate() {
                                let index = Literal::new(Token::new(
                                    TokenKind::NatLit,
                                    i.to_string(),
                                    elem.location.row() as u32,
                                    elem.location.column() as u32 - 1,
                                ));
                                let (param, mut blocks) = self.convert_expr_to_param(elem);
                                let sig = Signature::Var(VarSignature::new(
                                    Self::param_pattern_to_var(param.pat),
                                    param.t_spec.map(|t| t.t_spec),
                                ));
                                let method =
                                    tmp_expr.clone().attr_expr(self.convert_ident(
                                        "__Tuple_getitem__".to_string(),
                                        stmt.location,
                                    ));
                                let args =
                                    Args::new(vec![PosArg::new(Expr::Lit(index))], vec![], None);
                                let tuple_acc = method.call_expr(args);
                                let body =
                                    DefBody::new(EQUAL, Block::new(vec![tuple_acc]), DefId(0));
                                let def = Expr::Def(Def::new(sig, body));
                                defs.push(def);
                                defs.append(&mut blocks);
                            }
                            Expr::Dummy(Dummy::new(defs))
                        }
                        _other => Expr::Dummy(Dummy::empty()),
                    }
                } else {
                    let value = self.convert_expr(value);
                    let mut defs = vec![];
                    for target in targets {
                        match target.node {
                            ExpressionType::Identifier { name } => {
                                let body =
                                    DefBody::new(EQUAL, Block::new(vec![value.clone()]), DefId(0));
                                self.register_name_info(&name, NameKind::Variable);
                                let ident = self.convert_ident(name, stmt.location);
                                let sig = Signature::Var(VarSignature::new(
                                    VarPattern::Ident(ident),
                                    None,
                                ));
                                let def = Expr::Def(Def::new(sig, body));
                                defs.push(def);
                            }
                            _other => {
                                defs.push(Expr::Dummy(Dummy::empty()));
                            }
                        }
                    }
                    Expr::Dummy(Dummy::new(defs))
                }
            }
            StatementType::AugAssign { target, op, value } => {
                let op = op_to_token(op);
                match target.node {
                    ExpressionType::Identifier { name } => {
                        let val = self.convert_expr(*value);
                        let prev_ident = self.convert_ident(name.clone(), stmt.location);
                        if self
                            .get_name(&name)
                            .map(|info| info.defined_block_id == self.cur_block_id())
                            .unwrap_or(false)
                        {
                            self.register_name_info(&name, NameKind::Variable);
                            let ident = self.convert_ident(name.clone(), stmt.location);
                            let bin =
                                BinOp::new(op, Expr::Accessor(Accessor::Ident(prev_ident)), val);
                            let sig =
                                Signature::Var(VarSignature::new(VarPattern::Ident(ident), None));
                            let block = Block::new(vec![Expr::BinOp(bin)]);
                            let body = DefBody::new(EQUAL, block, DefId(0));
                            let def = Def::new(sig, body);
                            Expr::Def(def)
                        } else {
                            let ident = self.convert_ident(name.clone(), stmt.location);
                            let bin =
                                BinOp::new(op, Expr::Accessor(Accessor::Ident(prev_ident)), val);
                            let redef = ReDef::new(Accessor::Ident(ident), Expr::BinOp(bin));
                            Expr::ReDef(redef)
                        }
                    }
                    ExpressionType::Attribute { value: attr, name } => {
                        let val = self.convert_expr(*value);
                        let attr = self
                            .convert_expr(*attr)
                            .attr(self.convert_attr_ident(name, target.location));
                        let bin = BinOp::new(op, Expr::Accessor(attr.clone()), val);
                        let adef = ReDef::new(attr, Expr::BinOp(bin));
                        Expr::ReDef(adef)
                    }
                    _other => Expr::Dummy(Dummy::empty()),
                }
            }
            StatementType::FunctionDef {
                is_async: _,
                name,
                args,
                body,
                decorator_list,
                returns,
            } => {
                // if reassigning of a function referenced by other functions is occurred, it is an error
                if self
                    .get_name(&name)
                    .map(|info| {
                        info.defined_times > 0
                            && !info.referenced.difference(&set! {name.clone()}).is_empty()
                    })
                    .unwrap_or(false)
                {
                    let err = reassign_func_error(
                        self.cfg.input.clone(),
                        pyloc_to_ergloc(stmt.location, name.len()),
                        self.namespace.join("."),
                        &name,
                    );
                    self.errs.push(err);
                    Expr::Dummy(Dummy::empty())
                } else {
                    let decos = decorator_list
                        .into_iter()
                        .map(|ex| Decorator(self.convert_expr(ex)))
                        .collect::<HashSet<_>>();
                    self.register_name_info(&name, NameKind::Function);
                    let ident = self.convert_ident(name, stmt.location);
                    self.namespace.push(ident.inspect().to_string());
                    let params = self.convert_params(args);
                    let return_t = returns.map(|ret| self.convert_type_spec(ret));
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
                    self.namespace.pop();
                    Expr::Def(def)
                }
            }
            StatementType::ClassDef {
                name,
                body,
                bases,
                keywords: _,
                decorator_list,
            } => self.convert_classdef(name, body, bases, decorator_list, stmt.location),
            StatementType::For {
                is_async: _,
                target,
                iter,
                body,
                orelse: _,
            } => {
                let iter = self.convert_expr(*iter);
                let block = self.convert_for_body(*target, body);
                let for_ident = self.convert_ident("for".to_string(), stmt.location);
                let for_acc = Expr::Accessor(Accessor::Ident(for_ident));
                for_acc.call_expr(Args::new(
                    vec![PosArg::new(iter), PosArg::new(Expr::Lambda(block))],
                    vec![],
                    None,
                ))
            }
            StatementType::While {
                test,
                body,
                orelse: _,
            } => {
                let test = self.convert_expr(test);
                let params = Params::new(vec![], None, vec![], None);
                let empty_sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let block = self.convert_block(body, BlockKind::While);
                let body = Lambda::new(empty_sig, Token::DUMMY, block, DefId(0));
                let while_ident = self.convert_ident("while".to_string(), stmt.location);
                let while_acc = Expr::Accessor(Accessor::Ident(while_ident));
                while_acc.call_expr(Args::new(
                    vec![PosArg::new(test), PosArg::new(Expr::Lambda(body))],
                    vec![],
                    None,
                ))
            }
            StatementType::If { test, body, orelse } => {
                let block = self.convert_block(body, BlockKind::If);
                let params = Params::new(vec![], None, vec![], None);
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, block, DefId(0));
                let test = self.convert_expr(test);
                let if_ident = self.convert_ident("if".to_string(), stmt.location);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                if let Some(orelse) = orelse {
                    let else_block = self.convert_block(orelse, BlockKind::If);
                    let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                    let else_body = Lambda::new(sig, Token::DUMMY, else_block, DefId(0));
                    let args = Args::new(
                        vec![
                            PosArg::new(test),
                            PosArg::new(Expr::Lambda(body)),
                            PosArg::new(Expr::Lambda(else_body)),
                        ],
                        vec![],
                        None,
                    );
                    if_acc.call_expr(args)
                } else {
                    let args = Args::new(
                        vec![PosArg::new(test), PosArg::new(Expr::Lambda(body))],
                        vec![],
                        None,
                    );
                    if_acc.call_expr(args)
                }
            }
            StatementType::Return { value } => {
                let value = value
                    .map(|val| self.convert_expr(val))
                    .unwrap_or_else(|| Expr::Tuple(Tuple::Normal(NormalTuple::new(Args::empty()))));
                if dont_call_return {
                    value
                } else {
                    let func_acc = Expr::Accessor(Accessor::Ident(
                        self.convert_ident(self.namespace.last().unwrap().clone(), stmt.location),
                    ));
                    let return_acc = self.convert_ident("return".to_string(), stmt.location);
                    let return_acc = Expr::Accessor(Accessor::attr(func_acc, return_acc));
                    return_acc.call_expr(Args::new(vec![PosArg::new(value)], vec![], None))
                }
            }
            StatementType::Assert { test, msg } => {
                let test = self.convert_expr(test);
                let args = if let Some(msg) = msg {
                    let msg = self.convert_expr(msg);
                    Args::new(vec![PosArg::new(test), PosArg::new(msg)], vec![], None)
                } else {
                    Args::new(vec![PosArg::new(test)], vec![], None)
                };
                let assert_acc = Expr::Accessor(Accessor::Ident(
                    self.convert_ident("assert".to_string(), stmt.location),
                ));
                assert_acc.call_expr(args)
            }
            StatementType::Import { names } => {
                let mut imports = vec![];
                for name in names {
                    let import_acc = Expr::Accessor(Accessor::Ident(
                        self.convert_ident("__import__".to_string(), stmt.location),
                    ));
                    let cont = format!("\"{}\"", name.symbol.split('.').next().unwrap());
                    let mod_name = Expr::Lit(Literal::new(Token::new(
                        TokenKind::StrLit,
                        cont,
                        stmt.location.row() as u32,
                        stmt.location.column() as u32 - 1,
                    )));
                    let args = Args::new(vec![PosArg::new(mod_name)], vec![], None);
                    let call = import_acc.call_expr(args);
                    let def = if let Some(alias) = name.alias {
                        self.register_name_info(&alias, NameKind::Variable);
                        let var = VarSignature::new(
                            VarPattern::Ident(self.convert_ident(alias, stmt.location)),
                            None,
                        );
                        Def::new(
                            Signature::Var(var),
                            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                        )
                    } else {
                        self.register_name_info(&name.symbol, NameKind::Variable);
                        let var = VarSignature::new(
                            VarPattern::Ident(self.convert_ident(name.symbol, stmt.location)),
                            None,
                        );
                        Def::new(
                            Signature::Var(var),
                            DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                        )
                    };
                    imports.push(Expr::Def(def));
                }
                Expr::Dummy(Dummy::new(imports))
            }
            StatementType::ImportFrom {
                level: _,
                module,
                names,
            } => {
                let import_acc = Expr::Accessor(Accessor::Ident(
                    self.convert_ident("__import__".to_string(), stmt.location),
                ));
                let cont = format!("\"{}\"", module.clone().unwrap());
                let mod_name = Expr::Lit(Literal::new(Token::new(
                    TokenKind::StrLit,
                    cont,
                    stmt.location.row() as u32,
                    stmt.location.column() as u32 - 1,
                )));
                let args = Args::new(vec![PosArg::new(mod_name)], vec![], None);
                let call = import_acc.call_expr(args);
                self.register_name_info(module.as_ref().unwrap(), NameKind::Variable);
                let mod_ident = self.convert_ident(module.unwrap(), stmt.location);
                let mod_expr = Expr::Accessor(Accessor::Ident(mod_ident.clone()));
                let var = VarSignature::new(VarPattern::Ident(mod_ident), None);
                let moddef = Expr::Def(Def::new(
                    Signature::Var(var),
                    DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)),
                ));
                let mut imports = vec![];
                for name in names {
                    let ident = self.convert_ident(name.symbol.clone(), stmt.location);
                    let var = if let Some(alias) = name.alias {
                        self.register_name_info(&alias, NameKind::Variable);
                        VarSignature::new(
                            VarPattern::Ident(self.convert_ident(alias, stmt.location)),
                            None,
                        )
                    } else {
                        self.register_name_info(&name.symbol, NameKind::Variable);
                        VarSignature::new(
                            VarPattern::Ident(
                                self.convert_ident(name.symbol.clone(), stmt.location),
                            ),
                            None,
                        )
                    };
                    let attr = mod_expr.clone().attr_expr(ident);
                    let def = Def::new(
                        Signature::Var(var),
                        DefBody::new(EQUAL, Block::new(vec![attr]), DefId(0)),
                    );
                    imports.push(Expr::Def(def));
                }
                let imports = Dummy::new(
                    vec![moddef]
                        .into_iter()
                        .chain(imports.into_iter())
                        .collect(),
                );
                Expr::Dummy(imports)
            }
            StatementType::Try {
                body,
                handlers: _,
                orelse,
                finalbody,
            } => {
                let chunks = self.convert_block(body, BlockKind::Try).into_iter();
                let dummy = match (orelse, finalbody) {
                    (Some(orelse), Some(finalbody)) => chunks
                        .chain(self.convert_block(orelse, BlockKind::Try).into_iter())
                        .chain(self.convert_block(finalbody, BlockKind::Try).into_iter())
                        .collect(),
                    (Some(orelse), None) => chunks
                        .chain(self.convert_block(orelse, BlockKind::Try).into_iter())
                        .collect(),
                    (None, Some(finalbody)) => chunks
                        .chain(self.convert_block(finalbody, BlockKind::Try).into_iter())
                        .collect(),
                    (None, None) => chunks.collect(),
                };
                Expr::Dummy(Dummy::new(dummy))
            }
            StatementType::With {
                is_async: _,
                mut items,
                body,
            } => {
                let item = items.remove(0);
                let context_expr = self.convert_expr(item.context_expr);
                let body = self.convert_for_body(item.optional_vars.unwrap(), body);
                let with_ident = self.convert_ident("with".to_string(), stmt.location);
                let with_acc = Expr::Accessor(Accessor::Ident(with_ident));
                with_acc.call_expr(Args::new(
                    vec![PosArg::new(context_expr), PosArg::new(Expr::Lambda(body))],
                    vec![],
                    None,
                ))
            }
            _other => {
                log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::empty())
            }
        }
    }

    pub fn convert_program(mut self, program: Program) -> IncompleteArtifact<Module> {
        let program = program
            .statements
            .into_iter()
            .map(|stmt| self.convert_statement(stmt, true))
            .collect();
        IncompleteArtifact::new(Some(Module::new(program)), self.errs, self.warns)
    }
}
