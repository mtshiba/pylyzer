use erg_common::error::ErrorKind;
use erg_common::log;
use erg_common::traits::Stream;
use erg_common::Str;
// use erg_common::style::{remove_style, StyledString, Color};
use erg_compiler::context::ModuleContext;
use erg_compiler::error::{CompileError, CompileErrors, CompileWarning, CompileWarnings};

pub fn default_implementations(op: &str) -> Option<&str> {
    match op {
        "`==`" => Some("__eq__"),
        "`!=`" => Some("__ne__"),
        _ => None,
    }
}

pub(crate) fn filter_errors(ctx: &ModuleContext, errors: CompileErrors) -> CompileErrors {
    errors
        .into_iter()
        .filter_map(|error| filter_error(ctx, error))
        .collect()
}

fn filter_error(_ctx: &ModuleContext, error: CompileError) -> Option<CompileError> {
    match error.core.kind {
        ErrorKind::FeatureError => {
            log!(err "this error is ignored:");
            log!(err "{error}");
            None
        }
        ErrorKind::VisibilityError => None,
        // exclude doc strings
        ErrorKind::UnusedWarning => {
            let code = error.input.reread_lines(
                error.core.loc.ln_begin().unwrap_or(1) as usize,
                error.core.loc.ln_end().unwrap_or(1) as usize,
            );
            if code[0].trim().starts_with("\"\"\"") {
                None
            } else {
                Some(error)
            }
        }
        // ErrorKind::AssignError => handle_assign_error(error),
        _ => Some(error),
    }
}

pub(crate) fn downgrade_errors(
    ctx: &ModuleContext,
    errors: CompileErrors,
) -> (CompileErrors, CompileWarnings) {
    let mut errs = CompileErrors::empty();
    let mut warns = CompileWarnings::empty();
    for error in errors {
        match downgrade_error(ctx, error) {
            Ok(err) => errs.push(err),
            Err(warn) => warns.push(warn),
        }
    }
    (errs, warns)
}

#[allow(clippy::result_large_err)]
fn downgrade_error(
    _ctx: &ModuleContext,
    mut error: CompileError,
) -> Result<CompileError, CompileWarning> {
    match error.core.kind {
        ErrorKind::TypeError => {
            // TODO: trim escape sequences
            let callee = Str::rc(
                error
                    .core
                    .main_message
                    .trim_start_matches("the type of ")
                    .trim_end_matches(" is mismatched"),
            );
            if let Some(op) = callee.find_sub(&["`==`", "`!=`"]) {
                error.core.main_message = format!(
                    "this object does not implement `{}`",
                    default_implementations(op).unwrap()
                );
                error.core.kind = ErrorKind::TypeWarning;
                Err(error)
            } else {
                Ok(error)
            }
        }
        _ => Ok(error),
    }
}
