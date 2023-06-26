use erg_common::error::{ErrorCore, ErrorKind, Location, SubMessage};
use erg_common::io::Input;
use erg_common::switch_lang;
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

pub(crate) fn self_not_found_error(input: Input, loc: Location, caused_by: String) -> CompileError {
    CompileError::new(
        ErrorCore::new(
            vec![SubMessage::only_loc(loc)],
            switch_lang!(
                "japanese" => format!("このメソッドは第一引数にselfを取るべきですが、見つかりませんでした"),
                "simplified_chinese" => format!("该方法应该有第一个参数self，但是没有找到"),
                "traditional_chinese" => format!("該方法應該有第一個參數self，但是沒有找到"),
                "english" => format!("This method should have the first parameter `self`, but it was not found"),
            ),
            2,
            ErrorKind::NameError,
            loc,
        ),
        input,
        caused_by,
    )
}

pub(crate) fn init_var_error(input: Input, loc: Location, caused_by: String) -> CompileError {
    CompileError::new(
        ErrorCore::new(
            vec![SubMessage::only_loc(loc)],
            switch_lang!(
                "japanese" => format!("`__init__`はメソッドです。メンバ変数として宣言するべきではありません"),
                "simplified_chinese" => format!("__init__是方法。不能宣告为变量"),
                "traditional_chinese" => format!("__init__是方法。不能宣告為變量"),
                "english" => format!("`__init__` should be a method. It should not be defined as a member variable"),
            ),
            3,
            ErrorKind::NameError,
            loc,
        ),
        input,
        caused_by,
    )
}
