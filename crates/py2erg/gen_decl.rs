use std::fs::File;
use std::io::{BufWriter, Write};

use erg_common::config::Input;
use erg_common::log;
use erg_compiler::context::register::PylyzerStatus;
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

pub fn gen_decl_er(input: &Input, hir: HIR, status: CheckStatus) -> DeclFile {
    let timestamp = if let Some(file) = input.path() {
        let metadata = std::fs::metadata(file).unwrap();
        metadata.modified().unwrap()
    } else {
        std::time::SystemTime::now()
    };
    let status = PylyzerStatus {
        succeed: status.is_succeed(),
        file: input.unescaped_path().into(),
        timestamp,
    };
    let mut code = format!("{status}\n");
    for chunk in hir.module.into_iter() {
        match chunk {
            Expr::Def(def) => {
                let name = def.sig.ident().inspect().replace('\0', "");
                let typ = def.sig.ident().ref_t().to_string();
                let typ = escape_type(typ);
                let decl = format!(".{name}: {typ}");
                code += &decl;
            }
            Expr::ClassDef(def) => {
                let name = def.sig.ident().inspect().replace('\0', "");
                let decl = format!(".{name}: ClassType");
                code += &decl;
            }
            _ => {}
        }
        code.push('\n');
    }
    log!("code:\n{code}");
    let filename = input.unescaped_filename().replace(".py", ".d.er");
    DeclFile { filename, code }
}

pub fn dump_decl_er(input: Input, hir: HIR, status: CheckStatus) {
    let file = gen_decl_er(&input, hir, status);
    let mut dir = input.dir();
    dir.push("__pycache__");
    let pycache_dir = dir.as_path();
    if !pycache_dir.exists() {
        std::fs::create_dir(pycache_dir).unwrap();
    }
    let f = File::create(pycache_dir.join(file.filename)).unwrap();
    let mut f = BufWriter::new(f);
    f.write_all(file.code.as_bytes()).unwrap();
}
