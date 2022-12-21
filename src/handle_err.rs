use erg_common::error::ErrorKind;
use erg_common::log;
// use erg_common::style::{remove_style, StyledString, Color};
use erg_compiler::error::{CompileErrors, CompileError};
use erg_compiler::context::Context;

pub(crate) fn filter_errors(ctx: &Context, errors: CompileErrors) -> CompileErrors {
    errors.into_iter().filter_map(|error| filter_error(ctx, error)).collect()
}

fn filter_error(_ctx: &Context, error: CompileError) -> Option<CompileError> {
    match error.core.kind {
        ErrorKind::FeatureError => {
            log!(err "this error is ignored:");
            log!(err "{error}");
            None
        }
        ErrorKind::VisibilityError => None,
        // ErrorKind::AssignError => handle_assign_error(error),
        _ => Some(error),
    }
}
