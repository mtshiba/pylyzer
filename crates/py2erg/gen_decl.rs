use std::fs::File;
use std::io::{BufWriter, Write};

use erg_common::config::Input;
use erg_common::log;
use erg_compiler::context::register::PylyzerStatus;
use erg_compiler::hir::{Expr, HIR};
use erg_compiler::ty::value::{GenTypeObj, TypeObj};
use erg_compiler::ty::{HasType, Type};

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
    typ.replace('%', "Type_").replace("<module>", "")
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
        gen_chunk_decl("", chunk, &mut code);
    }
    log!("code:\n{code}");
    let filename = input.unescaped_filename().replace(".py", ".d.er");
    DeclFile { filename, code }
}

fn gen_chunk_decl(namespace: &str, chunk: Expr, code: &mut String) {
    match chunk {
        Expr::Def(def) => {
            let mut name = def
                .sig
                .ident()
                .inspect()
                .replace('\0', "")
                .replace('%', "___");
            let typ = def.sig.ident().ref_t().to_string();
            let typ = escape_type(typ);
            // Erg can automatically import nested modules
            // `import http.client` => `http = pyimport "http"`
            let decl = if def.sig.ident().ref_t().is_py_module() {
                name = name.split('.').next().unwrap().to_string();
                format!(
                    "{namespace}.{name} = pyimport {}",
                    def.sig.ident().ref_t().typarams()[0]
                )
            } else {
                format!("{namespace}.{name}: {typ}")
            };
            *code += &decl;
        }
        Expr::ClassDef(def) => {
            let class_name = def
                .sig
                .ident()
                .inspect()
                .replace('\0', "")
                .replace('%', "___");
            let namespace = format!("{namespace}.{class_name}");
            let decl = format!(".{class_name}: ClassType");
            *code += &decl;
            code.push('\n');
            if let GenTypeObj::Subclass(class) = &def.obj {
                let sup = class.sup.as_ref().typ().to_string();
                let sup = escape_type(sup);
                let decl = format!(".{class_name} <: {sup}\n");
                *code += &decl;
            }
            if let Some(TypeObj::Builtin {
                t: Type::Record(rec),
                ..
            }) = def.obj.base_or_sup()
            {
                for (attr, t) in rec.iter() {
                    let typ = escape_type(t.to_string());
                    let decl = format!("{namespace}.{}: {typ}\n", attr.symbol);
                    *code += &decl;
                }
            }
            if let Some(TypeObj::Builtin {
                t: Type::Record(rec),
                ..
            }) = def.obj.additional()
            {
                for (attr, t) in rec.iter() {
                    let typ = escape_type(t.to_string());
                    let decl = format!("{namespace}.{}: {typ}\n", attr.symbol);
                    *code += &decl;
                }
            }
            for attr in def.methods.into_iter() {
                gen_chunk_decl(&namespace, attr, code);
            }
        }
        Expr::Dummy(dummy) => {
            for chunk in dummy.into_iter() {
                gen_chunk_decl(namespace, chunk, code);
            }
        }
        _ => {}
    }
    code.push('\n');
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
