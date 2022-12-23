use erg_common::switch_lang;
use erg_common::config::Input;
use erg_common::error::{Location, ErrorCore, ErrorKind, SubMessage};
use erg_compiler::error::CompileError;

pub(crate) fn reassign_func_error(
    input: Input,
    loc: Location,
    caused_by: String,
    name: &str,
) -> CompileError {
    CompileError::new(
        ErrorCore::new(
            vec![SubMessage::only_loc(loc)],
            switch_lang!(
                "japanese" => format!("{name}は既に宣言され、参照されています。このような関数に再代入するのは望ましくありません"),
                "simplified_chinese" => format!("{name}已声明，已被引用。不建议再次赋值"),
                "traditional_chinese" => format!("{name}已宣告，已被引用。不建議再次賦值"),
                "english" => format!("{name} has already been declared and referenced. It is not recommended to reassign such a function"),
            ),
            1,
            ErrorKind::AssignError,
            loc,
        ),
        input,
        caused_by,
    )
}
