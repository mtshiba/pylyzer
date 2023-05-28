use erg_common::log;
use rustpython_parser::ast::{Boolop, Cmpop, Constant, ExpressionType, Keyword, StringGroup};

pub fn number_to_string(num: &Constant) -> String {
    num.to_string()
}

pub fn comp_to_string(comp: &Cmpop) -> String {
    cmpop.as_str()
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
