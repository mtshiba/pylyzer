use std::io::Write;
use std::path::PathBuf;

use erg_common::config::Input;
use erg_common::log;
use erg_compiler::hir::{Expr, HIR};
use erg_compiler::ty::HasType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Succeed,
    Failed,
}

impl CheckStatus {
    pub const fn is_succeed(&self) -> bool {
        matches!(self, CheckStatus::Succeed)
    }
    pub const fn is_failed(&self) -> bool {
        matches!(self, CheckStatus::Failed)
    }
}

pub struct DeclFile {
    pub filename: String,
    pub code: String,
}

fn escape_type(typ: String) -> String {
    typ.replace('%', "Type_")
}

pub fn gen_decl_er(hir: HIR, status: CheckStatus) -> DeclFile {
    let mut code = if status.is_failed() {
        "# failed\n".to_string()
    } else {
        "# succeed\n".to_string()
    };
    for chunk in hir.module.into_iter() {
        match chunk {
            Expr::Def(def) => {
                let typ = def.sig.ident().ref_t().to_string();
                let typ = escape_type(typ);
                let decl = format!(".{}: {typ}", def.sig.ident().inspect());
                code += &decl;
            }
            Expr::ClassDef(def) => {
                let decl = format!(".{}: ClassType", def.sig.ident().inspect());
                code += &decl;
            }
            _ => {}
        }
        code.push('\n');
    }
    log!("code:\n{code}");
    let filename = hir.name.replace(".py", ".d.er");
    DeclFile { filename, code }
}

pub fn dump_decl_er(input: Input, hir: HIR, status: CheckStatus) {
    let file = gen_decl_er(hir, status);
    let mut path = if let Input::File(path) = input {
        path
    } else {
        PathBuf::new()
    };
    path.pop();
    path.push("__pycache__");
    let pycache_dir = path.as_path();
    if !pycache_dir.exists() {
        std::fs::create_dir(pycache_dir).unwrap();
    }
    let f = std::fs::File::create(pycache_dir.join(file.filename)).unwrap();
    let mut f = std::io::BufWriter::new(f);
    f.write_all(file.code.as_bytes()).unwrap();
}
