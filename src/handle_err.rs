use erg_common::error::ErrorKind;
use erg_common::style::{remove_style, StyledString, Color};
use erg_compiler::error::{CompileErrors, CompileError};
use erg_compiler::context::Context;

pub(crate) fn filter_errors(ctx: &Context, errors: CompileErrors) -> CompileErrors {
    errors.into_iter().filter_map(|error| filter_error(ctx, error)).collect()
}

fn filter_error(ctx: &Context, error: CompileError) -> Option<CompileError> {
    match error.core.kind {
        ErrorKind::VisibilityError => None,
        ErrorKind::NameError => Some(map_name_error(ctx, error)),
        ErrorKind::AssignError => handle_assign_error(error),
        _ => Some(error),
    }
}

fn map_name_error(ctx: &Context, mut error: CompileError) -> CompileError {
    let hint = error.core.sub_messages.iter_mut().find_map(|sub| sub.hint.as_mut());
    if let Some(hint) = hint {
        if let Some(name) = hint.split("exists a similar name variable: ").last() {
            let name = remove_style(name);
            if let Some((_, vi)) = ctx.get_var_info(&name) {
                if let Some(py_name) = &vi.py_name {
                    *hint = format!("exists a similar name variable: {}", StyledString::new(&py_name[..], Some(Color::Green), None));
                }
            }
        }
    }
    error
}

fn handle_assign_error(error: CompileError) -> Option<CompileError> {
    if error.core.main_message.ends_with("cannot be assigned more than once") {
        None
    } else {
        Some(error)
    }
}
