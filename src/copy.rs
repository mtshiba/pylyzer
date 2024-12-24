use std::fs::{copy, create_dir_all, read_dir, remove_file, DirEntry};
use std::path::Path;

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

pub(crate) fn copy_dot_erg() {
    if erg_path().exists() {
        return;
    }
    for site_packages in python_site_packages() {
        if site_packages.join(".erg").exists() {
            println!("Copying site-package/.erg to {}", erg_path().display());
            copy_dir(site_packages.join(".erg"), erg_path()).expect("Failed to copy .erg");
        }
    }
}

pub(crate) fn clear_cache() {
    for dir in read_dir(".").expect("Failed to read dir") {
        let Ok(dir) = dir else {
            continue;
        };
        rec_clear_cache(dir);
    }
    for site_packages in python_site_packages() {
        for pkg in site_packages
            .read_dir()
            .expect("Failed to read site-packages")
        {
            let Ok(pkg) = pkg else {
                continue;
            };
            rec_clear_cache(pkg);
        }
    }
}

fn rec_clear_cache(pkg: DirEntry) {
    if pkg.file_type().expect("Failed to get file type").is_dir() {
        let cache = if pkg.path().ends_with("__pycache__") {
            pkg.path()
        } else {
            pkg.path().join("__pycache__")
        };
        if cache.exists() {
            let Ok(dir) = cache.read_dir() else {
                return;
            };
            for cache_file in dir {
                let Ok(cache_file) = cache_file else {
                    continue;
                };
                if cache_file.file_name().to_string_lossy().ends_with(".d.er") {
                    println!("Removing cache file {}", cache_file.path().display());
                    remove_file(cache_file.path()).expect("Failed to remove cache file");
                }
            }
        }
        let Ok(dir) = pkg.path().read_dir() else {
            return;
        };
        for entry in dir {
            let Ok(entry) = entry else {
                continue;
            };
            rec_clear_cache(entry);
        }
    }
}
