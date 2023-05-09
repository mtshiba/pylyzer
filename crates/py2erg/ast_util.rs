use erg_common::log;
use rustpython_parser::ast::{
    BooleanOperator, Comparison, ExpressionType, Keyword, Number, StringGroup,
};

pub fn number_to_string(num: &Number) -> String {
    match num {
        Number::Integer { value } => value.to_string(),
        Number::Float { value } => value.to_string(),
        Number::Complex { real, imag } => format!("{real}+{imag}j"),
    }
}

pub fn keyword_length(keyword: &Keyword) -> usize {
    if let Some(name) = &keyword.name {
        name.len() + 1 + length(&keyword.value.node)
    } else {
        length(&keyword.value.node)
    }
}

pub fn string_length(string: &StringGroup) -> usize {
    match string {
        StringGroup::Constant { value } => value.len(),
        StringGroup::Joined { values } => values.iter().map(string_length).sum(),
        other => {
            log!(err "{other:?}");
            0
        }
    }
}

pub fn comp_to_string(comp: &Comparison) -> String {
    match comp {
        Comparison::In => "in".to_string(),
        Comparison::NotIn => "not in".to_string(),
        Comparison::Is => "is".to_string(),
        Comparison::IsNot => "is not".to_string(),
        Comparison::Less => "<".to_string(),
        Comparison::Greater => ">".to_string(),
        Comparison::Equal => "==".to_string(),
        Comparison::NotEqual => "!=".to_string(),
        Comparison::LessOrEqual => "<=".to_string(),
        Comparison::GreaterOrEqual => ">=".to_string(),
    }
}

pub fn length(expr: &ExpressionType) -> usize {
    match expr {
        ExpressionType::Identifier { name } => name.len(),
        ExpressionType::Number { value } => number_to_string(value).len(),
        ExpressionType::String { value } => string_length(value),
        ExpressionType::Attribute { value, name } => length(&value.node) + name.len() + 1,
        ExpressionType::Subscript { a, b } => length(&a.node) + length(&b.node) + 2,
        ExpressionType::Tuple { elements }
        | ExpressionType::List { elements }
        | ExpressionType::Set { elements } => {
            if let (Some(first), Some(last)) = (elements.first(), elements.last()) {
                2 + last.location.column() - first.location.column()
            } else {
                2
            }
        }
        ExpressionType::Call {
            function,
            args,
            keywords,
        } => {
            let args_len = if let (Some(first), Some(last)) = (args.first(), args.last()) {
                last.location.column() - first.location.column()
            } else {
                0
            };
            let kw_len = if let (Some(first), Some(last)) = (keywords.first(), keywords.last()) {
                last.value.location.column() - first.value.location.column()
            } else {
                0
            };
            length(&function.node) + args_len + kw_len + 2 // ()
        }
        ExpressionType::Unop { op: _, a } => 1 + length(&a.node),
        ExpressionType::Binop { a, op: _, b } => length(&a.node) + 3 + length(&b.node),
        ExpressionType::BoolOp { op, values } => match op {
            BooleanOperator::And => values
                .iter()
                .map(|elem| length(&elem.node))
                .fold(0, |acc, x| acc + x + 3),
            BooleanOperator::Or => values
                .iter()
                .map(|elem| length(&elem.node))
                .fold(0, |acc, x| acc + x + 2),
        },
        ExpressionType::Compare { vals, ops } => vals
            .iter()
            .zip(ops.iter())
            .map(|(elem, op)| length(&elem.node) + comp_to_string(op).len())
            .fold(0, |acc, x| acc + x + 2),
        ExpressionType::IfExpression { test, body, orelse } => {
            // x if y else z
            length(&test.node) + 4 + length(&body.node) + 6 + length(&orelse.node)
        }
        ExpressionType::Lambda { args: _, body } => {
            // lambda x: y
            // TODO:
            7 + 1 + length(&body.node)
        }
        ExpressionType::Await { value } => 5 + length(&value.node),
        ExpressionType::Yield { value } => 5 + value.as_ref().map(|x| length(&x.node)).unwrap_or(0),
        ExpressionType::NamedExpression { left, right } => {
            // x := y
            length(&left.node) + 4 + length(&right.node)
        }
        ExpressionType::Starred { value } => 1 + length(&value.node),
        ExpressionType::False => 5,
        ExpressionType::True | ExpressionType::None => 4,
        ExpressionType::Ellipsis => 8,
        other => {
            log!(err "{other:?}");
            0
        }
    }
}

pub fn accessor_name(expr: ExpressionType) -> Option<String> {
    match expr {
        ExpressionType::Identifier { name } => Some(name),
        ExpressionType::Attribute { value, name } => {
            accessor_name(value.node).map(|value| format!("{value}.{name}"))
        }
        _ => None,
    }
}
