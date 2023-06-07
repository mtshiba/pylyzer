use rustpython_parser::ast::located::Expr;

pub fn accessor_name(expr: Expr) -> Option<String> {
    match expr {
        Expr::Name(name) => Some(name.id.to_string()),
        Expr::Attribute(attr) => {
            accessor_name(*attr.value).map(|value| format!("{value}.{}", attr.attr))
        }
        _ => None,
    }
}
