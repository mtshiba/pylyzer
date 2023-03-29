use rustpython_parser::ast::{
    BooleanOperator, Comparison, Comprehension, ComprehensionKind, ExpressionType, Keyword,
    Located, Number, Operator, Parameter, Parameters, StringGroup, UnaryOperator, Varargs,
};

fn clone_number(num: &Number) -> Number {
    match num {
        Number::Integer { value } => Number::Integer {
            value: value.clone(),
        },
        Number::Float { value } => Number::Float { value: *value },
        Number::Complex { real, imag } => Number::Complex {
            real: *real,
            imag: *imag,
        },
    }
}

fn clone_string_group(group: &StringGroup) -> StringGroup {
    match group {
        StringGroup::Constant { value } => StringGroup::Constant {
            value: value.clone(),
        },
        StringGroup::FormattedValue {
            value,
            conversion,
            spec,
        } => StringGroup::FormattedValue {
            value: Box::new(clone_loc_expr(value)),
            conversion: *conversion,
            spec: spec.as_deref().map(|sp| Box::new(clone_string_group(sp))),
        },
        StringGroup::Joined { values } => StringGroup::Joined {
            values: values.iter().map(clone_string_group).collect::<Vec<_>>(),
        },
    }
}

fn clone_unary_op(op: &UnaryOperator) -> UnaryOperator {
    match op {
        UnaryOperator::Not => UnaryOperator::Not,
        UnaryOperator::Inv => UnaryOperator::Inv,
        UnaryOperator::Pos => UnaryOperator::Pos,
        UnaryOperator::Neg => UnaryOperator::Neg,
    }
}

fn clone_bin_op(op: &Operator) -> Operator {
    match op {
        Operator::Add => Operator::Add,
        Operator::Sub => Operator::Sub,
        Operator::Mult => Operator::Mult,
        Operator::MatMult => Operator::MatMult,
        Operator::Div => Operator::Div,
        Operator::Mod => Operator::Mod,
        Operator::Pow => Operator::Pow,
        Operator::LShift => Operator::LShift,
        Operator::RShift => Operator::RShift,
        Operator::BitOr => Operator::BitOr,
        Operator::BitXor => Operator::BitXor,
        Operator::BitAnd => Operator::BitAnd,
        Operator::FloorDiv => Operator::FloorDiv,
    }
}

fn clone_comp_op(op: &Comparison) -> Comparison {
    match op {
        Comparison::Equal => Comparison::Equal,
        Comparison::NotEqual => Comparison::NotEqual,
        Comparison::Less => Comparison::Less,
        Comparison::LessOrEqual => Comparison::LessOrEqual,
        Comparison::Greater => Comparison::Greater,
        Comparison::GreaterOrEqual => Comparison::GreaterOrEqual,
        Comparison::Is => Comparison::Is,
        Comparison::IsNot => Comparison::IsNot,
        Comparison::In => Comparison::In,
        Comparison::NotIn => Comparison::NotIn,
    }
}

fn clone_bool_op(op: &BooleanOperator) -> BooleanOperator {
    match op {
        BooleanOperator::And => BooleanOperator::And,
        BooleanOperator::Or => BooleanOperator::Or,
    }
}

fn clone_param(param: &Parameter) -> Parameter {
    Parameter {
        location: param.location,
        arg: param.arg.clone(),
        annotation: param
            .annotation
            .as_deref()
            .map(|a| Box::new(clone_loc_expr(a))),
    }
}

fn clone_varargs(varargs: &Varargs) -> Varargs {
    match varargs {
        Varargs::None => Varargs::None,
        Varargs::Unnamed => Varargs::Unnamed,
        Varargs::Named(name) => Varargs::Named(clone_param(name)),
    }
}

fn clone_params(params: &Parameters) -> Parameters {
    Parameters {
        posonlyargs_count: params.posonlyargs_count,
        args: params.args.iter().map(clone_param).collect::<Vec<_>>(),
        vararg: clone_varargs(&params.vararg),
        kwonlyargs: params
            .kwonlyargs
            .iter()
            .map(clone_param)
            .collect::<Vec<_>>(),
        kw_defaults: params
            .kw_defaults
            .iter()
            .map(|def| def.as_ref().map(clone_loc_expr))
            .collect::<Vec<_>>(),
        kwarg: clone_varargs(&params.kwarg),
        defaults: params
            .defaults
            .iter()
            .map(clone_loc_expr)
            .collect::<Vec<_>>(),
    }
}

fn clone_kw(keyword: &Keyword) -> Keyword {
    Keyword {
        name: keyword.name.clone(),
        value: clone_loc_expr(&keyword.value),
    }
}

fn clone_comprehension_kind(kind: &ComprehensionKind) -> ComprehensionKind {
    match kind {
        ComprehensionKind::Dict { key, value } => ComprehensionKind::Dict {
            key: clone_loc_expr(key),
            value: clone_loc_expr(value),
        },
        ComprehensionKind::List { element } => ComprehensionKind::List {
            element: clone_loc_expr(element),
        },
        ComprehensionKind::Set { element } => ComprehensionKind::Set {
            element: clone_loc_expr(element),
        },
        ComprehensionKind::GeneratorExpression { element } => {
            ComprehensionKind::GeneratorExpression {
                element: clone_loc_expr(element),
            }
        }
    }
}

pub fn clone_loc_expr(expr: &Located<ExpressionType>) -> Located<ExpressionType> {
    Located {
        node: clone_expr(&expr.node),
        location: expr.location,
    }
}

pub fn clone_expr(expr: &ExpressionType) -> ExpressionType {
    match expr {
        ExpressionType::None => ExpressionType::None,
        ExpressionType::Ellipsis => ExpressionType::Ellipsis,
        ExpressionType::True => ExpressionType::True,
        ExpressionType::False => ExpressionType::False,
        ExpressionType::Identifier { name } => ExpressionType::Identifier { name: name.clone() },
        ExpressionType::Number { value } => ExpressionType::Number {
            value: clone_number(value),
        },
        ExpressionType::String { value } => ExpressionType::String {
            value: clone_string_group(value),
        },
        ExpressionType::Attribute { value, name } => ExpressionType::Attribute {
            value: Box::new(clone_loc_expr(value)),
            name: name.clone(),
        },
        ExpressionType::Subscript { a, b } => ExpressionType::Subscript {
            a: Box::new(clone_loc_expr(a)),
            b: Box::new(clone_loc_expr(b)),
        },
        ExpressionType::Slice { elements } => ExpressionType::Slice {
            elements: elements.iter().map(clone_loc_expr).collect::<Vec<_>>(),
        },
        ExpressionType::Bytes { value } => ExpressionType::Bytes {
            value: value.clone(),
        },
        ExpressionType::Call {
            function,
            args,
            keywords,
        } => ExpressionType::Call {
            function: Box::new(clone_loc_expr(function)),
            args: args.iter().map(clone_loc_expr).collect::<Vec<_>>(),
            keywords: keywords.iter().map(clone_kw).collect::<Vec<_>>(),
        },
        ExpressionType::Unop { op, a } => ExpressionType::Unop {
            op: clone_unary_op(op),
            a: Box::new(clone_loc_expr(a)),
        },
        ExpressionType::Binop { a, op, b } => ExpressionType::Binop {
            a: Box::new(clone_loc_expr(a)),
            op: clone_bin_op(op),
            b: Box::new(clone_loc_expr(b)),
        },
        ExpressionType::Compare { vals, ops } => ExpressionType::Compare {
            ops: ops.iter().map(clone_comp_op).collect::<Vec<_>>(),
            vals: vals.iter().map(clone_loc_expr).collect::<Vec<_>>(),
        },
        ExpressionType::BoolOp { op, values } => ExpressionType::BoolOp {
            op: clone_bool_op(op),
            values: values.iter().map(clone_loc_expr).collect::<Vec<_>>(),
        },
        ExpressionType::Lambda { args, body } => ExpressionType::Lambda {
            args: Box::new(clone_params(args)),
            body: Box::new(clone_loc_expr(body)),
        },
        ExpressionType::IfExpression { test, body, orelse } => ExpressionType::IfExpression {
            test: Box::new(clone_loc_expr(test)),
            body: Box::new(clone_loc_expr(body)),
            orelse: Box::new(clone_loc_expr(orelse)),
        },
        ExpressionType::Dict { elements } => ExpressionType::Dict {
            elements: elements
                .iter()
                .map(|(key, value)| (key.as_ref().map(clone_loc_expr), clone_loc_expr(value)))
                .collect::<Vec<_>>(),
        },
        ExpressionType::Set { elements } => ExpressionType::Set {
            elements: elements.iter().map(clone_loc_expr).collect::<Vec<_>>(),
        },
        ExpressionType::List { elements } => ExpressionType::List {
            elements: elements.iter().map(clone_loc_expr).collect::<Vec<_>>(),
        },
        ExpressionType::Tuple { elements } => ExpressionType::Tuple {
            elements: elements.iter().map(clone_loc_expr).collect::<Vec<_>>(),
        },
        ExpressionType::Yield { value } => ExpressionType::Yield {
            value: value.as_ref().map(|val| Box::new(clone_loc_expr(val))),
        },
        ExpressionType::YieldFrom { value } => ExpressionType::YieldFrom {
            value: Box::new(clone_loc_expr(value)),
        },
        ExpressionType::Await { value } => ExpressionType::Await {
            value: Box::new(clone_loc_expr(value)),
        },
        ExpressionType::NamedExpression { left, right } => ExpressionType::NamedExpression {
            left: Box::new(clone_loc_expr(left)),
            right: Box::new(clone_loc_expr(right)),
        },
        ExpressionType::Starred { value } => ExpressionType::Starred {
            value: Box::new(clone_loc_expr(value)),
        },
        ExpressionType::Comprehension { kind, generators } => ExpressionType::Comprehension {
            kind: Box::new(clone_comprehension_kind(kind)),
            generators: generators
                .iter()
                .map(|gen| Comprehension {
                    location: gen.location,
                    target: clone_loc_expr(&gen.target),
                    iter: clone_loc_expr(&gen.iter),
                    ifs: gen.ifs.iter().map(clone_loc_expr).collect::<Vec<_>>(),
                    is_async: gen.is_async,
                })
                .collect::<Vec<_>>(),
        },
    }
}
