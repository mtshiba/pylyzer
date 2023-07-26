use std::fs::File;
use std::io::{BufWriter, Write};

use erg_common::io::Input;
use erg_common::log;
use erg_common::traits::LimitedDisplay;
use erg_compiler::context::register::{CheckStatus, PylyzerStatus};
use erg_compiler::hir::{Expr, HIR};
use erg_compiler::ty::value::{GenTypeObj, TypeObj};
use erg_compiler::ty::{HasType, Type};

pub struct DeclFile {
    pub filename: String,
    pub code: String,
}

fn escape_type(typ: String) -> String {
    typ.replace('%', "Type_").replace("<module>", "")
}

pub struct DeclFileGenerator {
    filename: String,
    namespace: String,
    code: String,
}

impl DeclFileGenerator {
    pub fn new(input: &Input, status: CheckStatus) -> Self {
        let (timestamp, hash) = {
            let py_file_path = input.path();
            let metadata = std::fs::metadata(py_file_path).unwrap();
            let dummy_hash = metadata.len();
            (metadata.modified().unwrap(), dummy_hash)
        };
        let status = PylyzerStatus {
            status,
            file: input.path().into(),
            timestamp,
            hash,
        };
        let code = format!("{status}\n");
        Self {
            filename: input.filename().replace(".py", ".d.er"),
            namespace: "".to_string(),
            code,
        }
    }

    pub fn gen_decl_er(mut self, hir: HIR) -> DeclFile {
        for chunk in hir.module.into_iter() {
            self.gen_chunk_decl(chunk);
        }
        log!("code:\n{}", self.code);
        DeclFile {
            filename: self.filename,
            code: self.code,
        }
    }

    fn gen_chunk_decl(&mut self, chunk: Expr) {
        match chunk {
            Expr::Def(def) => {
                let mut name = def
                    .sig
                    .ident()
                    .inspect()
                    .replace('\0', "")
                    .replace('%', "___");
                let ref_t = def.sig.ident().ref_t();
                let typ = ref_t.replace_failure().to_string_unabbreviated();
                let typ = escape_type(typ);
                // Erg can automatically import nested modules
                // `import http.client` => `http = pyimport "http"`
                let decl = if ref_t.is_py_module() {
                    name = name.split('.').next().unwrap().to_string();
                    format!(
                        "{}.{name} = pyimport {}",
                        self.namespace,
                        ref_t.typarams()[0]
                    )
                } else {
                    format!("{}.{name}: {typ}", self.namespace)
                };
                self.code += &decl;
            }
            Expr::ClassDef(def) => {
                let class_name = def
                    .sig
                    .ident()
                    .inspect()
                    .replace('\0', "")
                    .replace('%', "___");
                let src = format!("{}.{class_name}", self.namespace);
                let stash = std::mem::replace(&mut self.namespace, src);
                let decl = format!(".{class_name}: ClassType");
                self.code += &decl;
                self.code.push('\n');
                if let GenTypeObj::Subclass(class) = &def.obj {
                    let sup = class.sup.as_ref().typ().to_string_unabbreviated();
                    let sup = escape_type(sup);
                    let decl = format!(".{class_name} <: {sup}\n");
                    self.code += &decl;
                }
                if let Some(TypeObj::Builtin {
                    t: Type::Record(rec),
                    ..
                }) = def.obj.base_or_sup()
                {
                    for (attr, t) in rec.iter() {
                        let typ = escape_type(t.to_string_unabbreviated());
                        let decl = format!("{}.{}: {typ}\n", self.namespace, attr.symbol);
                        self.code += &decl;
                    }
                }
                if let Some(TypeObj::Builtin {
                    t: Type::Record(rec),
                    ..
                }) = def.obj.additional()
                {
                    for (attr, t) in rec.iter() {
                        let typ = escape_type(t.to_string_unabbreviated());
                        let decl = format!("{}.{}: {typ}\n", self.namespace, attr.symbol);
                        self.code += &decl;
                    }
                }
                for attr in def.methods.into_iter() {
                    self.gen_chunk_decl(attr);
                }
                self.namespace = stash;
            }
            Expr::Dummy(dummy) => {
                for chunk in dummy.into_iter() {
                    self.gen_chunk_decl(chunk);
                }
            }
            _ => {}
        }
        self.code.push('\n');
    }
}

pub fn reserve_decl_er(input: Input) {
    let mut dir = input.dir();
    dir.push("__pycache__");
    let pycache_dir = dir.as_path();
    if !pycache_dir.exists() {
        std::fs::create_dir(pycache_dir).unwrap();
    }
    let filename = input.filename();
    let mut path = pycache_dir.join(filename);
    path.set_extension("d.er");
    if !path.exists() {
        let _f = File::create(path).unwrap();
    }
}

pub fn dump_decl_er(input: Input, hir: HIR, status: CheckStatus) {
    let decl_gen = DeclFileGenerator::new(&input, status);
    let file = decl_gen.gen_decl_er(hir);
    let mut dir = input.dir();
    dir.push("__pycache__");
    let pycache_dir = dir.as_path();
    let f = File::options()
        .write(true)
        .open(pycache_dir.join(file.filename))
        .unwrap();
    let mut f = BufWriter::new(f);
    f.write_all(file.code.as_bytes()).unwrap();
}
