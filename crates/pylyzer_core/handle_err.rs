use erg_common::error::ErrorKind;
use erg_common::log;
use erg_common::style::{remove_style, StyledStr};
// use erg_common::style::{remove_style, StyledString, Color};
use erg_compiler::context::ModuleContext;
use erg_compiler::error::{CompileError, CompileErrors};

pub(crate) fn filter_errors(ctx: &ModuleContext, errors: CompileErrors) -> CompileErrors {
    errors
        .into_iter()
        .filter_map(|error| filter_error(ctx, error))
        .collect()
}

fn handle_name_error(error: CompileError) -> Option<CompileError> {
    let main = &error.core.main_message;
    if main.contains("is already declared")
        || main.contains("cannot be assigned more than once")
        || {
            main.contains(" is not defined") && {
                let name = StyledStr::destyle(main.trim_end_matches(" is not defined"));
                error
                    .core
                    .get_hint()
                    .is_some_and(|hint| hint.contains(name))
            }
        }
    {
        None
    } else {
        Some(error)
    }
}

fn filter_error(_ctx: &ModuleContext, mut error: CompileError) -> Option<CompileError> {
    match error.core.kind {
        ErrorKind::FeatureError => {
            log!(err "this error is ignored:");
            log!(err "{error}");
            None
        }
        ErrorKind::InheritanceError => None,
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
                for sub in error.core.sub_messages.iter_mut() {
                    if let Some(hint) = &mut sub.hint {
                        *hint = remove_style(hint);
                        *hint = hint.replace("use discard function", "bind to `_` (`_ = ...`)");
                    }
                }
                Some(error)
            }
        }
        ErrorKind::NameError | ErrorKind::AssignError => handle_name_error(error),
        _ => Some(error),
    }
}
