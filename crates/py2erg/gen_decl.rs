use std::io::Write;
use std::path::PathBuf;

use erg_common::config::Input;
use erg_compiler::hir::{HIR, Expr};
use erg_compiler::ty::HasType;

pub struct DeclFile {
    pub filename: String,
    pub code: String,
}

fn escape_type(typ: String) -> String {
    typ.replace('%', "Type_")
}

pub fn gen_decl_er(hir: HIR) -> DeclFile {
    let mut code = "".to_string();
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
    let filename = hir.name.replace(".py", ".d.er");
    DeclFile { filename, code }
}

pub fn dump_decl_er(input: Input, hir: HIR) {
    let file = gen_decl_er(hir);
    let mut path = if let Input::File(path) = input { path } else { PathBuf::new() };
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
