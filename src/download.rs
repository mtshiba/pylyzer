use std::error::Error;

use erg_common::env::erg_path;
use erg_common::python_util::exec_py_code;

#[allow(unused)]
pub(crate) fn download_dependencies() -> Result<(), Box<dyn Error>> {
    if erg_path().exists() {
        return Ok(());
    }
    println!("Erg standard library not found. Installing...");
    let install_script_url = "https://github.com/mtshiba/ergup/raw/main/ergup.py";
    let code = ureq::get(install_script_url).call()?.into_string()?;
    exec_py_code(&code, &[])?;
    Ok(())
}
