use std::path::Path;
use std::fs::{copy, read_dir, create_dir_all};

use erg_common::env::{erg_path, python_site_packages};

fn copy_dir(from: impl AsRef<Path>, to: impl AsRef<Path>) -> std::io::Result<()> {
    let from = from.as_ref();
    let to = to.as_ref();
    if !from.exists() {
        return Ok(());
    }
    if !to.exists() {
        create_dir_all(to)?;
    }
    for entry in read_dir(from)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            copy_dir(entry.path(), to.join(entry.file_name()))?;
        } else {
            copy(entry.path(), to.join(entry.file_name()))?;
        }
    }
    Ok(())
}

#[allow(unused)]
pub(crate) fn copy_dot_erg() {
    if erg_path().exists() {
        return;
    }
    for site_packages in python_site_packages() {
        if site_packages.join(".erg").exists() {
            println!("Copying site-package/.erg to {}", erg_path().display());
            copy_dir(site_packages.join(".erg"), erg_path())
                .expect("Failed to copy .erg");
        }
    }
}
