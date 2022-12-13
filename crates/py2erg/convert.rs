use erg_common::traits::Stream;
use rustpython_parser::ast::{StatementType, ExpressionType, Located, Program, Number, StringGroup, Operator, BooleanOperator, UnaryOperator, Suite, Parameters, Parameter};
use rustpython_parser::ast::Location as PyLocation;

use erg_common::set::Set as HashSet;
use erg_compiler::erg_parser::token::{Token, TokenKind, EQUAL, COLON};
use erg_compiler::erg_parser::ast::{
    Expr, Module, Signature, VarSignature, VarPattern, Params, Identifier, VarName, DefBody, DefId, Block, Def, Literal, Args, PosArg, Accessor,
    BinOp, Lambda, LambdaSignature, TypeBoundSpecs, TypeSpec, SubrSignature, Decorator, NonDefaultParamSignature, DefaultParamSignature, ParamPattern, TypeSpecWithOp,
    Tuple, NormalTuple, Array, NormalArray, Set, NormalSet, Dict, NormalDict, PreDeclTypeSpec, SimpleTypeSpec, ConstArgs, AttrDef, UnaryOp, KeyValue, Dummy, TypeAscription
};

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
    let pat = convert_param_pattern(param.arg, param.location);
    let t_spec = param.annotation
        .map(|anot| convert_type_spec(*anot))
        .map(|t_spec| TypeSpecWithOp::new(COLON, t_spec));
    NonDefaultParamSignature::new(pat, t_spec)
}

fn _convert_default_param(_param: Parameter) -> DefaultParamSignature {
    todo!()
}

// TODO: defaults
fn convert_params(args: Box<Parameters>) -> Params {
    let non_defaults = args.args.into_iter().map(convert_nd_param).collect();
    // let defaults = args. args.defaults.into_iter().map(convert_default_param).collect();
    Params::new(non_defaults, None, vec![], None)
}

fn convert_for_param(name: String, loc: PyLocation) -> NonDefaultParamSignature {
    let pat = convert_param_pattern(name, loc);
    let t_spec = None;
    NonDefaultParamSignature::new(pat, t_spec)
}

fn convert_for_body(lhs: Located<ExpressionType>, body: Suite) -> Lambda {
    let ExpressionType::Identifier { name } = lhs.node else { todo!() };
    let param = convert_for_param(name, lhs.location);
    let params = Params::new(vec![param], None, vec![], None);
    let body = body.into_iter().map(convert_statement).collect::<Vec<_>>();
    let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
    let op = Token::from_str(TokenKind::ProcArrow, "=>");
    Lambda::new(sig, op, Block::new(body), DefId(0))
}

fn convert_type_spec(expr: Located<ExpressionType>) -> TypeSpec {
    match expr.node {
        ExpressionType::Identifier { name } =>
            TypeSpec::PreDeclTy(PreDeclTypeSpec::Simple(SimpleTypeSpec::new(convert_ident(name, expr.location), ConstArgs::empty()))),
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
        ExpressionType::Call { function, args, keywords: _ } => {
            let function = convert_expr(*function);
            let pos_args = args.into_iter().map(|ex| PosArg::new(convert_expr(ex))).collect::<Vec<_>>();
            let args = Args::new(pos_args, vec![], None);
            function.call_expr(args)
        }
        ExpressionType::Binop { a, op, b } => {
            let lhs = convert_expr(*a);
            let rhs = convert_expr(*b);
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
            let rhs = convert_expr(*a);
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
            let lhs = convert_expr(values.remove(0));
            let rhs = convert_expr(values.remove(0));
            let (kind, cont) = match op {
                BooleanOperator::And => (TokenKind::AndOp, "and"),
                BooleanOperator::Or => (TokenKind::OrOp, "or"),
            };
            let op = Token::from_str(kind, cont);
            Expr::BinOp(BinOp::new(op, lhs, rhs))
        }
        ExpressionType::Identifier { name } => {
            let ident = convert_ident(name, expr.location);
            Expr::Accessor(Accessor::Ident(ident))
        }
        ExpressionType::Attribute { value, name } => {
            let obj = convert_expr(*value);
            let name = convert_ident(name, expr.location);
            Expr::Accessor(Accessor::attr(obj, name))
        }
        ExpressionType::Lambda { args, body } => {
            let params = convert_params(args);
            let body = vec![convert_expr(*body)];
            let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
            let op = Token::from_str(TokenKind::ProcArrow, "=>");
            Expr::Lambda(Lambda::new(sig, op, Block::new(body), DefId(0)))
        }
        ExpressionType::List { elements } => {
            let (l_sqbr, r_sqbr) = gen_enclosure_tokens(TokenKind::LSqBr, elements.iter(), expr.location);
            let elements = elements.into_iter()
                .map(|ex| PosArg::new(convert_expr(ex)))
                .collect::<Vec<_>>();
            let elems = Args::new(elements, vec![], None);
            Expr::Array(Array::Normal(NormalArray::new(l_sqbr, r_sqbr, elems)))
        }
        ExpressionType::Set { elements } => {
            let (l_brace, r_brace) = gen_enclosure_tokens(TokenKind::LBrace, elements.iter(), expr.location);
            let elements = elements.into_iter()
                .map(|ex| PosArg::new(convert_expr(ex)))
                .collect::<Vec<_>>();
            let elems = Args::new(elements, vec![], None);
            Expr::Set(Set::Normal(NormalSet::new(l_brace, r_brace, elems)))
        }
        ExpressionType::Dict { elements } => {
            let (l_brace, r_brace) = gen_enclosure_tokens(TokenKind::LBrace, elements.iter().map(|(_, v)| v), expr.location);
            let kvs = elements.into_iter()
                .map(|(k, v)|
                    KeyValue::new(k.map(convert_expr).unwrap_or(Expr::Dummy(Dummy::empty())), convert_expr(v))
                ).collect::<Vec<_>>();
            Expr::Dict(Dict::Normal(NormalDict::new(l_brace, r_brace, kvs)))
        }
        ExpressionType::Tuple { elements } => {
            let elements = elements.into_iter()
                .map(|ex| PosArg::new(convert_expr(ex)))
                .collect::<Vec<_>>();
            let elems = Args::new(elements, vec![], None);
            Expr::Tuple(Tuple::Normal(NormalTuple::new(elems)))
        }
        ExpressionType::Subscript { a, b } => {
            let obj = convert_expr(*a);
            let method = obj.attr_expr(convert_ident("__getitem__".to_string(), expr.location));
            let args = Args::new(vec![PosArg::new(convert_expr(*b))], vec![], None);
            method.call_expr(args)
        }
        _other => {
            erg_common::log!(err "unimplemented: {:?}", _other);
            Expr::Dummy(Dummy::empty())
        },
    }
}

fn convert_block(block: Vec<Located<StatementType>>) -> Block {
    Block::new(block.into_iter().map(convert_statement).collect::<Vec<_>>())
}

fn convert_statement(stmt: Located<StatementType>) -> Expr {
    match stmt.node {
        StatementType::Expression { expression } => convert_expr(expression),
        StatementType::AnnAssign { target, annotation, value } => {
            let t_spec = convert_type_spec(*annotation);
            match target.node {
                ExpressionType::Identifier { name } => {
                    let ident = convert_ident(name, stmt.location);
                    if let Some(value) = value {
                        let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident), Some(t_spec)));
                        let block = Block::new(vec![convert_expr(value)]);
                        let body = DefBody::new(EQUAL, block, DefId(0));
                        let def = Def::new(sig, body);
                        Expr::Def(def)
                    } else {
                        let tasc = TypeAscription::new(Expr::Accessor(Accessor::Ident(ident)), COLON, t_spec);
                        Expr::TypeAsc(tasc)
                    }
                }
                ExpressionType::Attribute { value: attr, name } => {
                    let attr = convert_expr(*attr).attr(convert_ident(name, target.location));
                    if let Some(value) = value {
                        let expr = convert_expr(value);
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
                    let ident = convert_ident(name, stmt.location);
                    let sig = Signature::Var(VarSignature::new(VarPattern::Ident(ident), None));
                    let block = Block::new(vec![convert_expr(value)]);
                    let body = DefBody::new(EQUAL, block, DefId(0));
                    let def = Def::new(sig, body);
                    Expr::Def(def)
                }
                ExpressionType::Attribute { value: attr, name } => {
                    let attr = convert_expr(*attr).attr(convert_ident(name, lhs.location));
                    let expr = convert_expr(value);
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
            let decos = decorator_list.into_iter().map(|ex| Decorator(convert_expr(ex))).collect::<HashSet<_>>();
            let ident = convert_ident(name, stmt.location);
            let params = convert_params(args);
            let return_t = returns.map(convert_type_spec);
            let sig = Signature::Subr(SubrSignature::new(decos, ident, TypeBoundSpecs::empty(), params, return_t));
            let block = convert_block(body);
            let body = DefBody::new(EQUAL, block, DefId(0));
            let def = Def::new(sig, body);
            Expr::Def(def)
        }
        // TODO
        StatementType::ClassDef { name: _, body, bases, keywords: _, decorator_list } => {
            let exprs = decorator_list.into_iter().map(convert_expr).collect::<Vec<_>>();
            let bases = bases.into_iter().map(convert_expr).collect::<Vec<_>>();
            // let decos = decorator_list.into_iter().map(|ex| Decorator(convert_expr(ex))).collect::<Set<_>>();
            // let ident = convert_ident(name, stmt.location);
            // let params = Params::new(vec![], None, vec![], None);
            let mut block = convert_block(body);
            block.extend(exprs);
            block.extend(bases);
            // let body = DefBody::new(EQUAL, block, DefId(0));
            // let def = Def::new(sig, body);
            Expr::Dummy(Dummy::new(block.into_iter().collect()))
        }
        StatementType::For { is_async: _, target, iter, body, orelse: _ } => {
            let block = convert_for_body(*target, body);
            let iter = convert_expr(*iter);
            let for_ident = convert_ident("for!".to_string(), stmt.location);
            let for_acc = Expr::Accessor(Accessor::Ident(for_ident));
            for_acc.call_expr(Args::new(vec![PosArg::new(iter), PosArg::new(Expr::Lambda(block))], vec![], None))
        }
        StatementType::While { test, body, orelse: _ } => {
            let block = convert_block(body);
            let params = Params::new(vec![], None, vec![], None);
            let body = Lambda::new(LambdaSignature::new(params, None, TypeBoundSpecs::empty()), Token::DUMMY, block, DefId(0));
            let test = convert_expr(test);
            let while_ident = convert_ident("while!".to_string(), stmt.location);
            let while_acc = Expr::Accessor(Accessor::Ident(while_ident));
            while_acc.call_expr(Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body))], vec![], None))
        }
        StatementType::If { test, body, orelse } => {
            let block = convert_block(body);
            let params = Params::new(vec![], None, vec![], None);
            let sig = LambdaSignature::new(params.clone(), None, TypeBoundSpecs::empty());
            let body = Lambda::new(sig, Token::DUMMY, block, DefId(0));
            let test = convert_expr(test);
            let if_ident = convert_ident("if!".to_string(), stmt.location);
            let if_acc = Expr::Accessor(Accessor::Ident(if_ident));
            if let Some(orelse) = orelse {
                let else_block = convert_block(orelse);
                let sig = LambdaSignature::new(params, None, TypeBoundSpecs::empty());
                let else_body = Lambda::new(sig, Token::DUMMY, else_block, DefId(0));
                let args = Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body)), PosArg::new(Expr::Lambda(else_body))], vec![], None);
                if_acc.call_expr(args)
            } else {
                let args = Args::new(vec![PosArg::new(test), PosArg::new(Expr::Lambda(body))], vec![], None);
                if_acc.call_expr(args)
            }
        }
        // This is fine for static analysis only.
        StatementType::Return { value } => {
            value.map(convert_expr)
                .unwrap_or_else(||Expr::Tuple(Tuple::Normal(NormalTuple::new(Args::empty()))))
        }
        StatementType::Import { names } => {
            let mut imports = vec![];
            for name in names {
                let import_acc = Expr::Accessor(Accessor::Ident(convert_ident("__import__".to_string(), stmt.location)));
                let cont = format!("\"{}\"", name.symbol);
                let mod_name = Expr::Lit(Literal::new(Token::new(TokenKind::StrLit, cont, stmt.location.row(), stmt.location.column() - 1)));
                let args = Args::new(vec![PosArg::new(mod_name)], vec![], None);
                let call = import_acc.call_expr(args);
                let def = if let Some(alias) = name.alias {
                    let var = VarSignature::new(VarPattern::Ident(convert_ident(alias, stmt.location)), None);
                    Def::new(Signature::Var(var), DefBody::new(EQUAL, Block::new(vec![call]), DefId(0)))
                } else {
                    let var = VarSignature::new(VarPattern::Ident(convert_ident(name.symbol, stmt.location)), None);
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

pub fn convert_program(program: Program) -> Module {
    Module::new(program.statements.into_iter().map(convert_statement).collect())
}
