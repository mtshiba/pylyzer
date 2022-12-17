use erg_common::fresh::fresh_varname;
use erg_common::traits::{Locational};
use rustpython_parser::ast::{StatementType, ExpressionType, Located, Program, Number, StringGroup, Operator, BooleanOperator, UnaryOperator, Suite, Parameters, Parameter, Comparison};
use rustpython_parser::ast::Location as PyLocation;

use erg_common::set::Set as HashSet;
use erg_compiler::erg_parser::token::{Token, TokenKind, EQUAL, COLON, DOT};
use erg_compiler::erg_parser::ast::{
    Expr, Module, Signature, VarSignature, VarPattern, Params, Identifier, VarName, DefBody, DefId, Block, Def, Literal, Args, PosArg, Accessor, ClassAttrs, ClassAttr, RecordAttrs,
    BinOp, Lambda, LambdaSignature, TypeBoundSpecs, TypeSpec, SubrSignature, Decorator, NonDefaultParamSignature, DefaultParamSignature, ParamPattern, TypeSpecWithOp,
    Tuple, NormalTuple, Array, NormalArray, Set, NormalSet, Dict, NormalDict, PreDeclTypeSpec, SimpleTypeSpec, ConstArgs, AttrDef, UnaryOp, KeyValue, Dummy, TypeAscription, ClassDef, Record, Methods, NormalRecord,
};

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

/// Variables are automatically rewritten with `python_compatible_mode`,
/// but types are rewritten here because they are complex components used inseparably in the Erg system.
fn escape_name(name: String) -> String {
    match &name[..] {
        "int" => "Int".into(),
        "float" => "Float".into(),
        "str" => "Str".into(),
        "bool" => "Bool".into(),
        "list" => "GenericArray".into(),
        "range" => "GenericRange".into(),
        "dict" => "GenericDict".into(),
        "set" => "GenericSet".into(),
        "tuple" => "GenericTuple".into(),
        "type" => "Type".into(),
        "ModuleType" => "GeneticModule".into(),
        _ => name,
    }
}

fn _de_escape_name(name: String) -> String {
    match &name[..] {
        "Int" => "int".into(),
        "Float" => "float".into(),
        "Str" => "str".into(),
        "Bool" => "bool".into(),
        "GenericArray" => "list".into(),
        "GenericRange" => "range".into(),
        "GenericDict" => "dict".into(),
        "GenericSet" => "set".into(),
        "GenericTuple" => "tuple".into(),
        "Type" => "type".into(),
        "GenericModule" => "ModuleType".into(),
        _ => name,
    }
}

#[derive(Debug, Default)]
pub struct ASTConverter {
    namespace: Vec<String>,
}

impl ASTConverter {
    pub fn new() -> Self {
        Self {
            namespace: vec![String::from("<module>")],
        }
    }

    fn convert_ident(name: String, loc: PyLocation) -> Identifier {
        let token = Token::new(TokenKind::Symbol, escape_name(name), loc.row(), loc.column() - 1);
        let name = VarName::new(token);
        let dot = Token::new(TokenKind::Dot, ".", loc.row(), loc.column() - 1);
        Identifier::new(Some(dot), name)
    }

    fn convert_param_pattern(arg: String, loc: PyLocation) -> ParamPattern {
        let token = Token::new(TokenKind::Symbol, arg, loc.row(), loc.column() - 1);
        let name = VarName::new(token);
        ParamPattern::VarName(name)
    }

    fn convert_nd_param(param: Parameter) -> NonDefaultParamSignature {
        let pat = Self::convert_param_pattern(param.arg, param.location);
        let t_spec = param.annotation
            .map(|anot| Self::convert_type_spec(*anot))
            .map(|t_spec| TypeSpecWithOp::new(COLON, t_spec));
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn _convert_default_param(_param: Parameter) -> DefaultParamSignature {
        todo!()
    }

    // TODO: defaults
    fn convert_params(args: Box<Parameters>) -> Params {
        let non_defaults = args.args.into_iter().map(Self::convert_nd_param).collect();
        // let defaults = args. args.defaults.into_iter().map(convert_default_param).collect();
        Params::new(non_defaults, None, vec![], None)
    }

    fn convert_for_param(name: String, loc: PyLocation) -> NonDefaultParamSignature {
        let pat = Self::convert_param_pattern(name, loc);
        let t_spec = None;
        NonDefaultParamSignature::new(pat, t_spec)
    }

    fn param_pattern_to_var(pat: ParamPattern) -> VarPattern {
        match pat {
            ParamPattern::VarName(name) => VarPattern::Ident(Identifier::new(Some(DOT), name)),
            _ => todo!(),
        }
    }

    /// (i, j) => $1 (i = $1[0]; j = $1[1])
    fn convert_expr_to_param(expr: Located<ExpressionType>) -> (NonDefaultParamSignature, Vec<Expr>) {
        match expr.node {
            ExpressionType::Identifier { name } => (Self::convert_for_param(name, expr.location), vec![]),
            ExpressionType::Tuple { elements } => {
                let tmp = fresh_varname();
                let tmp_name = VarName::from_str_and_line((&tmp).into(), expr.location.row());
                let tmp_expr = Expr::Accessor(Accessor::Ident(Identifier::new(Some(DOT), tmp_name.clone())));
                let mut block = vec![];
                for (i, elem) in elements.into_iter().enumerate() {
                    let index = Literal::new(Token::new(TokenKind::NatLit, i.to_string(), elem.location.row(), elem.location.column() - 1));
                    let (param, mut blocks) = Self::convert_expr_to_param(elem);
                    let sig = Signature::Var(VarSignature::new(Self::param_pattern_to_var(param.pat), param.t_spec.map(|t| t.t_spec)));
                    let method = tmp_expr.clone().attr_expr(Self::convert_ident("__Tuple_getitem__".to_string(), expr.location));
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
                let token = Token::new(TokenKind::UBar, "_", expr.location.row(), expr.location.column() - 1);
                (NonDefaultParamSignature::new(ParamPattern::Discard(token), None), vec![])
            },
        }
    }

    fn convert_for_body(&mut self, lhs: Located<ExpressionType>, body: Suite) -> Lambda {
        let (param, block) = Self::convert_expr_to_param(lhs);
        let params = Params::new(vec![param], None, vec![], None);
        let body = body.into_iter().map(|stmt| self.convert_statement(stmt, true)).collect::<Vec<_>>();
        let body = block.into_iter().chain(body).collect();
        let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
        let op = Token::from_str(TokenKind::ProcArrow, "=>");
        Lambda::new(sig, op, Block::new(body), DefId(0))
    }

    fn convert_type_spec(expr: Located<ExpressionType>) -> TypeSpec {
        match expr.node {
            ExpressionType::Identifier { name } =>
                TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(Self::convert_ident(name, expr.location), ConstArgs::empty()))),
            _other => TypeSpec::Infer(Token::new(TokenKind::UBar, "_", expr.location.row(), expr.location.column() - 1)),
        }
    }

    fn gen_enclosure_tokens<'i, Elems>(l_kind: TokenKind, elems: Elems, expr_loc: PyLocation) -> (Token, Token)
        where Elems: Iterator<Item=&'i Located<ExpressionType>> + ExactSizeIterator {
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
        let l_brace = Token::new(l_kind, l_cont, expr_loc.row(), expr_loc.column() - 1);
        let r_brace = Token::new(r_kind, r_cont, l_end, c_end);
        (l_brace, r_brace)
    }

    fn convert_expr(expr: Located<ExpressionType>) -> Expr {
        match expr.node {
            ExpressionType::Number { value } => {
                let (kind, cont) = match value {
                    Number::Integer { value } => (TokenKind::IntLit, value.to_string()),
                    Number::Float { value } => (TokenKind::RatioLit, value.to_string()),
                    Number::Complex { .. } => { return Expr::Dummy(Dummy::new(vec![])); },
                };
                let token = Token::new(kind, cont, expr.location.row(), expr.location.column() - 1);
                Expr::Lit(Literal::new(token))
            }
            ExpressionType::String { value } => {
                let StringGroup::Constant{ value } = value else {
                    return Expr::Dummy(Dummy::new(vec![]));
                };
                let value = format!("\"{value}\"");
                // column - 2 because of the quotes
                let token = Token::new(TokenKind::StrLit, value, expr.location.row(), expr.location.column() - 2);
                Expr::Lit(Literal::new(token))
            }
            ExpressionType::False => {
                Expr::Lit(Literal::new(Token::new(TokenKind::BoolLit, "False", expr.location.row(), expr.location.column() - 1)))
            }
            ExpressionType::True => {
                Expr::Lit(Literal::new(Token::new(TokenKind::BoolLit, "True", expr.location.row(), expr.location.column() - 1)))
            }
            ExpressionType::None => {
                Expr::Lit(Literal::new(Token::new(TokenKind::NoneLit, "None", expr.location.row(), expr.location.column() - 1)))
            }
            ExpressionType::Ellipsis => {
                Expr::Lit(Literal::new(Token::new(TokenKind::EllipsisLit, "Ellipsis", expr.location.row(), expr.location.column() - 1)))
            }
            ExpressionType::IfExpression { test, body, orelse } => {
                let block = Self::convert_expr(*body);
                let params = Params::new(vec![], None, vec![], None);
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, Block::new(vec![block]), DefId(0));
                let test = Self::convert_expr(*test);
                let if_ident = Self::convert_ident("if".to_string(), expr.location);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                let else_block = Self::convert_expr(*orelse);
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let else_body = Lambda::new(sig, Token::DUMMY, Block::new(vec![else_block]), DefId(0));
                let args = Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body)), PosArg::new(Expr::Lambda(else_body))], vec![], None);
                if_acc.call_expr(args)
            }
            ExpressionType::Call { function, args, keywords: _ } => {
                let function = Self::convert_expr(*function);
                let pos_args = args.into_iter().map(|ex| PosArg::new(Self::convert_expr(ex))).collect::<Vec<_>>();
                let args = Args::new(pos_args, vec![], None);
                function.call_expr(args)
            }
            ExpressionType::Binop { a, op, b } => {
                let lhs = Self::convert_expr(*a);
                let rhs = Self::convert_expr(*b);
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
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            ExpressionType::Unop { op, a } => {
                let rhs = Self::convert_expr(*a);
                let (kind, cont) = match op {
                    UnaryOperator::Pos => (TokenKind::PrePlus, "+"),
                    // UnaryOperator::Not => (TokenKind::PreBitNot, "not"),
                    UnaryOperator::Neg => (TokenKind::PreMinus, "-"),
                    UnaryOperator::Inv => (TokenKind::Minus, "~"),
                    _ => { return Expr::Dummy(Dummy::new(vec![rhs])) }
                };
                let op = Token::from_str(kind, cont);
                Expr::UnaryOp(UnaryOp::new(op, rhs))
            }
            // TODO
            ExpressionType::BoolOp { op, mut values } => {
                let lhs = Self::convert_expr(values.remove(0));
                let rhs = Self::convert_expr(values.remove(0));
                let (kind, cont) = match op {
                    BooleanOperator::And => (TokenKind::AndOp, "and"),
                    BooleanOperator::Or => (TokenKind::OrOp, "or"),
                };
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            // TODO: multiple comparisons
            ExpressionType::Compare { mut vals, mut ops } => {
                let lhs = Self::convert_expr(vals.remove(0));
                let rhs = Self::convert_expr(vals.remove(0));
                let (kind, cont) = match ops.remove(0) {
                    Comparison::Equal => (TokenKind::Equal, "=="),
                    Comparison::NotEqual => (TokenKind::NotEq, "!="),
                    Comparison::Less => (TokenKind::Less, "<"),
                    Comparison::LessOrEqual => (TokenKind::LessEq, "<="),
                    Comparison::Greater => (TokenKind::Gre, ">"),
                    Comparison::GreaterOrEqual => (TokenKind::GreEq, ">="),
                    Comparison::Is => (TokenKind::IsOp, "is"),
                    Comparison::IsNot => (TokenKind::IsNotOp, "isnot"),
                    Comparison::In => (TokenKind::InOp, "in"),
                    Comparison::NotIn => (TokenKind::NotInOp, "notin"),
                };
                let op = Token::from_str(kind, cont);
                Expr::BinOp(BinOp::new(op, lhs, rhs))
            }
            ExpressionType::Identifier { name } => {
                let ident = Self::convert_ident(name, expr.location);
                Expr::Accessor(Accessor::Ident(ident))
            }
            ExpressionType::Attribute { value, name } => {
                let obj = Self::convert_expr(*value);
                let name = Self::convert_ident(name, expr.location);
                Expr::Accessor(Accessor::attr(obj, name))
            }
            ExpressionType::Lambda { args, body } => {
                let params = Self::convert_params(args);
                let body = vec![Self::convert_expr(*body)];
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let op = Token::from_str(TokenKind::ProcArrow, "=>");
                Expr::Lambda(Lambda::new(sig, op, Block::new(body), DefId(0)))
            }
            ExpressionType::List { elements } => {
                let (l_sqbr, r_sqbr) = Self::gen_enclosure_tokens(TokenKind::LSqBr, elements.iter(), expr.location);
                let elements = elements.into_iter()
                    .map(|ex| PosArg::new(Self::convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::new(elements, vec![], None);
                Expr::Array(Array::Normal(NormalArray::new(l_sqbr, r_sqbr, elems)))
            }
            ExpressionType::Set { elements } => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, elements.iter(), expr.location);
                let elements = elements.into_iter()
                    .map(|ex| PosArg::new(Self::convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::new(elements, vec![], None);
                Expr::Set(Set::Normal(NormalSet::new(l_brace, r_brace, elems)))
            }
            ExpressionType::Dict { elements } => {
                let (l_brace, r_brace) = Self::gen_enclosure_tokens(TokenKind::LBrace, elements.iter().map(|(_, v)| v), expr.location);
                let kvs = elements.into_iter()
                    .map(|(k, v)|
                        KeyValue::new(k.map(Self::convert_expr).unwrap_or(Expr::Dummy(Dummy::empty())), Self::convert_expr(v))
                    ).collect::<Vec<_>>();
                Expr::Dict(Dict::Normal(NormalDict::new(l_brace, r_brace, kvs)))
            }
            ExpressionType::Tuple { elements } => {
                let elements = elements.into_iter()
                    .map(|ex| PosArg::new(Self::convert_expr(ex)))
                    .collect::<Vec<_>>();
                let elems = Args::new(elements, vec![], None);
                Expr::Tuple(Tuple::Normal(NormalTuple::new(elems)))
            }
            ExpressionType::Subscript { a, b } => {
                let obj = Self::convert_expr(*a);
                let method = obj.attr_expr(Self::convert_ident("__getitem__".to_string(), expr.location));
                let args = Args::new(vec![PosArg::new(Self::convert_expr(*b))], vec![], None);
                method.call_expr(args)
            }
            _other => {
                erg_common::log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::empty())
            },
        }
    }

    fn convert_block(&mut self, block: Vec<Located<StatementType>>, kind: BlockKind) -> Block {
        let mut new_block = Vec::new();
        let len = block.len();
        for (i, stmt) in block.into_iter().enumerate() {
            let is_last = i == len - 1;
            new_block.push(self.convert_statement(stmt, is_last && kind.is_function()));
        }
        Block::new(new_block)
    }

    // def __init__(self, x: Int, y: Int):
    //     self.x = x
    //     self.y = y
    // ↓
    // {x: Int, y: Int}, .__call__(x: Int, y: Int): Self = .unreachable()
    fn extract_init(&self, def: Def) -> (Expr, Def) {
        let l_brace = Token::new(TokenKind::LBrace, "{", def.ln_begin().unwrap(), def.col_begin().unwrap());
        let r_brace = Token::new(TokenKind::RBrace, "}", def.ln_end().unwrap(), def.col_end().unwrap());
        let Signature::Subr(sig) = def.sig else { unreachable!() };
        let mut fields = vec![];
        let mut params = vec![];
        for chunk in def.body.block {
            #[allow(clippy::single_match)]
            match chunk {
                Expr::AttrDef(adef) => {
                    let Accessor::Attr(attr) = adef.attr else { break; };
                    if attr.obj.get_name().map(|s| &s[..]) == Some("self") {
                        let typ_name = if let Some(t_spec_op) = sig.params.non_defaults.iter()
                            .find(|&param| param.inspect() == Some(attr.ident.inspect()))
                            .and_then(|param| param.t_spec.as_ref()) {
                                t_spec_op.t_spec.to_string().replace('.', "")
                            } else {
                                "Never".to_string()
                            };
                        let typ_ident = Identifier::public_with_line(DOT, typ_name.into(), attr.obj.ln_begin().unwrap());
                        let typ_spec = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(typ_ident.clone(), ConstArgs::empty())));
                        let typ_spec = TypeSpecWithOp::new(COLON, typ_spec);
                        params.push(NonDefaultParamSignature::new(ParamPattern::VarName(attr.ident.name.clone()), Some(typ_spec)));
                        let typ = Expr::Accessor(Accessor::Ident(typ_ident));
                        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(attr.ident), None));
                        let body = DefBody::new(EQUAL, Block::new(vec![typ]), DefId(0));
                        let field_type_def = Def::new(sig, body);
                        fields.push(field_type_def);
                    }
                }
                _ => {}
            }
        }
        let record = Record::Normal(NormalRecord::new(l_brace, r_brace, RecordAttrs::new(fields)));
        let call_ident = Identifier::new(Some(DOT), VarName::from_static("__call__"));
        let params = Params::new(params, None, vec![], None);
        let class_ident = Identifier::public_with_line(DOT, self.namespace.last().unwrap().into(), sig.ln_begin().unwrap());
        let class_spec = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(class_ident, ConstArgs::empty())));
        let sig = Signature::Subr(SubrSignature::new(HashSet::new(), call_ident, TypeBoundSpecs::empty(), params, Some(class_spec)));
        let unreachable_acc = Identifier::new(Some(DOT), VarName::from_static("exit"));
        let body = Expr::Accessor(Accessor::Ident(unreachable_acc)).call_expr(Args::empty());
        let body = DefBody::new(EQUAL, Block::new(vec![body]), DefId(0));
        let def = Def::new(sig, body);
        (Expr::Record(record), def)
    }

    fn extract_method(&mut self, body: Vec<Located<StatementType>>) -> (Expr, ClassAttrs) {
        let mut base_type = Expr::Tuple(Tuple::Normal(NormalTuple::new(Args::empty())));
        let mut attrs = vec![];
        for stmt in body {
            let chunk = self.convert_statement(stmt, true);
            match chunk {
                Expr::Def(def) => {
                    if def.is_subr() && &def.sig.ident().unwrap().inspect()[..] == "__init__" {
                        let (base_t, call_def) = self.extract_init(def);
                        base_type = base_t;
                        attrs.push(ClassAttr::Def(call_def));
                    } else {
                        attrs.push(ClassAttr::Def(def));
                    }
                },
                Expr::TypeAsc(type_asc) => attrs.push(ClassAttr::Decl(type_asc)),
                _other => {} // TODO:
            }
        }
        (base_type, ClassAttrs::new(attrs))
    }

    fn extract_method_list(&mut self, ident: Identifier, body: Vec<Located<StatementType>>) -> (Expr, Vec<Methods>) {
        let class = TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(ident, ConstArgs::empty())));
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
        self.namespace.push(name.clone());
        let _decos = decorator_list.into_iter().map(Self::convert_expr).collect::<Vec<_>>();
        let _bases = bases.into_iter().map(Self::convert_expr).collect::<Vec<_>>();
        let ident = Self::convert_ident(name, loc);
        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident.clone()), None));
        let (base_type, methods) = self.extract_method_list(ident, body);
        let args = Args::new(vec![PosArg::new(base_type)], vec![], None);
        let class_acc = Expr::Accessor(Accessor::Ident(Self::convert_ident("Class".to_string(), loc)));
        let class_call = class_acc.call_expr(args);
        let body = DefBody::new(EQUAL, Block::new(vec![class_call]), DefId(0));
        let def = Def::new(sig, body);
        let classdef = ClassDef::new(def, methods);
        self.namespace.pop();
        Expr::ClassDef(classdef)
    }

    fn convert_statement(&mut self, stmt: Located<StatementType>, dont_call_return: bool) -> Expr {
        match stmt.node {
            StatementType::Expression { expression } => Self::convert_expr(expression),
            StatementType::AnnAssign { target, annotation, value } => {
                let t_spec = Self::convert_type_spec(*annotation);
                match target.node {
                    ExpressionType::Identifier { name } => {
                        let ident = Self::convert_ident(name, stmt.location);
                        if let Some(value) = value {
                            let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident), Some(t_spec)));
                            let block = Block::new(vec![Self::convert_expr(value)]);
                            let body = DefBody::new(EQUAL, block, DefId(0));
                            let def = Def::new(sig, body);
                            Expr::Def(def)
                        } else {
                            let tasc = TypeAscription::new(Expr::Accessor(Accessor::Ident(ident)), COLON, t_spec);
                            Expr::TypeAsc(tasc)
                        }
                    }
                    ExpressionType::Attribute { value: attr, name } => {
                        let attr = Self::convert_expr(*attr).attr(Self::convert_ident(name, target.location));
                        if let Some(value) = value {
                            let expr = Self::convert_expr(value);
                            let adef = AttrDef::new(attr, expr);
                            Expr::AttrDef(adef)
                        } else {
                            let tasc = TypeAscription::new(Expr::Accessor(attr), COLON, t_spec);
                            Expr::TypeAsc(tasc)
                        }
                    }
                    _other => Expr::Dummy(Dummy::empty()),
                }
            }
            StatementType::Assign { mut targets, value } => {
                let lhs = targets.remove(0);
                match lhs.node {
                    ExpressionType::Identifier { name } => {
                        let ident = Self::convert_ident(name, stmt.location);
                        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident), None));
                        let block = Block::new(vec![Self::convert_expr(value)]);
                        let body = DefBody::new(EQUAL, block, DefId(0));
                        let def = Def::new(sig, body);
                        Expr::Def(def)
                    }
                    ExpressionType::Attribute { value: attr, name } => {
                        let attr = Self::convert_expr(*attr).attr(Self::convert_ident(name, lhs.location));
                        let expr = Self::convert_expr(value);
                        let adef = AttrDef::new(attr, expr);
                        Expr::AttrDef(adef)
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
                returns
            } => {
                self.namespace.push(name.clone());
                let decos = decorator_list.into_iter().map(|ex| Decorator(Self::convert_expr(ex))).collect::<HashSet<_>>();
                let ident = Self::convert_ident(name, stmt.location);
                let params = Self::convert_params(args);
                let return_t = returns.map(Self::convert_type_spec);
                let sig = Signature::Subr(SubrSignature::new(decos, ident, TypeBoundSpecs::empty(), params, return_t));
                let block = self.convert_block(body, BlockKind::Function);
                let body = DefBody::new(EQUAL, block, DefId(0));
                let def = Def::new(sig, body);
                self.namespace.pop();
                Expr::Def(def)
            }
            StatementType::ClassDef { name, body, bases, keywords: _, decorator_list } => {
                self.convert_classdef(name, body, bases, decorator_list, stmt.location)
            }
            StatementType::For { is_async: _, target, iter, body, orelse: _ } => {
                let block = self.convert_for_body(*target, body);
                let iter = Self::convert_expr(*iter);
                let for_ident = Self::convert_ident("for".to_string(), stmt.location);
                let for_acc = Expr::Accessor(Accessor::Ident(for_ident));
                for_acc.call_expr(Args::new(vec![PosArg::new(iter), PosArg::new(Expr::Lambda(block))], vec![], None))
            }
            StatementType::While { test, body, orelse: _ } => {
                let block = self.convert_block(body, BlockKind::While);
                let params = Params::new(vec![], None, vec![], None);
                let body = Lambda::new(LambdaSignature::new(params, None, TypeBoundSpecs::empty()), Token::DUMMY, block, DefId(0));
                let test = Self::convert_expr(test);
                let while_ident = Self::convert_ident("while".to_string(), stmt.location);
                let while_acc = Expr::Accessor(Accessor::Ident(while_ident));
                while_acc.call_expr(Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body))], vec![], None))
            }
            StatementType::If { test, body, orelse } => {
                let block = self.convert_block(body, BlockKind::If);
                let params = Params::new(vec![], None, vec![], None);
                let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
                let body = Lambda::new(sig, Token::DUMMY, block, DefId(0));
                let test = Self::convert_expr(test);
                let if_ident = Self::convert_ident("if".to_string(), stmt.location);
                let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
                if let Some(orelse) = orelse {
                    let else_block = self.convert_block(orelse, BlockKind::If);
                    let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                    let else_body = Lambda::new(sig, Token::DUMMY, else_block, DefId(0));
                    let args = Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body)), PosArg::new(Expr::Lambda(else_body))], vec![], None);
                    if_acc.call_expr(args)
                } else {
                    let args = Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body))], vec![], None);
                    if_acc.call_expr(args)
                }
            }
            StatementType::Return { value } => {
                let value = value.map(Self::convert_expr)
                    .unwrap_or_else(||Expr::Tuple(Tuple::Normal(NormalTuple::new(Args::empty()))));
                if dont_call_return {
                    value
                } else {
                    let func_acc = Expr::Accessor(Accessor::Ident(Self::convert_ident(self.namespace.last().unwrap().clone(), stmt.location)));
                    let return_acc = Self::convert_ident("return".to_string(), stmt.location);
                    let return_acc = Expr::Accessor(Accessor::attr(func_acc, return_acc));
                    erg_common::log!(err "{return_acc}");
                    return_acc.call_expr(Args::new(vec![PosArg::new(value)], vec![], None))
                }
            }
            StatementType::Assert { test, msg } => {
                let test = Self::convert_expr(test);
                let args = if let Some(msg) = msg {
                    let msg = Self::convert_expr(msg);
                    Args::new(vec![PosArg::new(test), PosArg::new(msg)], vec![], None)
                } else {
                    Args::new(vec![PosArg::new(test)], vec![], None)
                };
                let assert_acc = Expr::Accessor(Accessor::Ident(Self::convert_ident("assert".to_string(), stmt.location)));
                assert_acc.call_expr(args)
            }
            StatementType::Import { names } => {
                let mut imports = vec![];
                for name in names {
                    let import_acc = Expr::Accessor(Accessor::Ident(Self::convert_ident("__import__".to_string(), stmt.location)));
                    let cont = format!("\"{}\"", name.symbol);
                    let mod_name = Expr::Lit(Literal::new(Token::new(TokenKind::StrLit, cont, stmt.location.row(), stmt.location.column() - 1)));
                    let args = Args::new(vec![PosArg::new(mod_name)], vec![], None);
                    let call = import_acc.call_expr(args);
                    let def = if let Some(alias) = name.alias {
                        let var = VarSignature::new(VarPattern::Ident(Self::convert_ident(alias, stmt.location)), None);
                        Def::new(Signature::Var(var), DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)))
                    } else {
                        let var = VarSignature::new(VarPattern::Ident(Self::convert_ident(name.symbol, stmt.location)), None);
                        Def::new(Signature::Var(var), DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)))
                    };
                    imports.push(Expr::Def(def));
                }
                Expr::Dummy(Dummy::new(imports))
            }
            _other => {
                erg_common::log!(err "unimplemented: {:?}", _other);
                Expr::Dummy(Dummy::empty())
            },
        }
    }

    pub fn convert_program(mut self, program: Program) -> Module {
        let program = program.statements.into_iter()
            .map(|stmt| self.convert_statement(stmt, true))
            .collect();
        Module::new(program)
    }
}
