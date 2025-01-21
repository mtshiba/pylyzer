use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use erg_common::pathutil::{mod_name, NormalizedPathBuf};
use erg_common::set::Set;
use erg_common::traits::LimitedDisplay;
use erg_common::{log, Str};
use erg_compiler::build_package::{CheckStatus, PylyzerStatus};
use erg_compiler::context::ControlKind;
use erg_compiler::hir::{ClassDef, Expr, HIR};
use erg_compiler::module::SharedModuleCache;
use erg_compiler::ty::value::{GenTypeObj, TypeObj};
use erg_compiler::ty::{HasType, Type};

pub struct DeclFile {
    pub filename: String,
    pub code: String,
}
pub struct DeclFileGenerator {
    filename: String,
    namespace: String,
    imported: Set<Str>,
    code: String,
}

impl DeclFileGenerator {
    pub fn new(path: &NormalizedPathBuf, status: CheckStatus) -> std::io::Result<Self> {
        let (timestamp, hash) = {
            let metadata = std::fs::metadata(path)?;
            let dummy_hash = metadata.len();
            (metadata.modified()?, dummy_hash)
        };
        let status = PylyzerStatus {
            status,
            file: path.to_path_buf(),
            timestamp,
            hash,
        };
        let code = format!("{status}\n");
        Ok(Self {
            filename: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .replace(".py", ".d.er"),
            namespace: "".to_string(),
            imported: Set::new(),
            code,
        })
    }

    pub fn gen_decl_er(mut self, hir: &HIR) -> DeclFile {
        for chunk in hir.module.iter() {
            self.gen_chunk_decl(chunk);
        }
        log!("code:\n{}", self.code);
        DeclFile {
            filename: self.filename,
            code: self.code,
        }
    }

    fn escape_type(&self, typ: String) -> String {
        typ.replace('%', "Type_")
            .replace("<module>", "")
            .replace('/', ".")
            .trim_start_matches(self.filename.trim_end_matches(".d.er"))
            .trim_start_matches(&self.namespace)
            .to_string()
    }

    // e.g. `x: foo.Bar` => `foo = pyimport "foo"; x: foo.Bar`
    fn prepare_using_type(&mut self, typ: &Type) {
        let namespace = Str::rc(
            typ.namespace()
                .split('/')
                .next()
                .unwrap()
                .split('.')
                .next()
                .unwrap(),
        );
        if namespace != self.namespace
            && !namespace.is_empty()
            && self.imported.insert(namespace.clone())
        {
            self.code += &format!("{namespace} = pyimport \"{namespace}\"\n");
        }
    }

    fn gen_chunk_decl(&mut self, chunk: &Expr) {
        match chunk {
            Expr::Def(def) => {
                let mut name = def
                    .sig
                    .ident()
                    .inspect()
                    .replace('\0', "")
                    .replace(['%', '*'], "___");
                let ref_t = def.sig.ident().ref_t();
                self.prepare_using_type(ref_t);
                let typ = self.escape_type(ref_t.replace_failure().to_string_unabbreviated());
                // Erg can automatically import nested modules
                // `import http.client` => `http = pyimport "http"`
                let decl = if ref_t.is_py_module() && ref_t.typarams()[0].is_str_value() {
                    name = name.split('.').next().unwrap().to_string();
                    let full_path_str = ref_t.typarams()[0].to_string_unabbreviated();
                    let mod_name = mod_name(Path::new(full_path_str.trim_matches('"')));
                    let imported = if self.imported.insert(mod_name.clone()) {
                        format!("{}.{mod_name} = pyimport \"{mod_name}\"", self.namespace)
                    } else {
                        "".to_string()
                    };
                    if self.imported.insert(name.clone().into()) {
                        format!(
                            "{}.{name} = pyimport \"{mod_name}\"\n{imported}",
                            self.namespace,
                        )
                    } else {
                        imported
                    }
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
                    .replace(['%', '*'], "___");
                let src = format!("{}.{class_name}", self.namespace);
                let stash = std::mem::replace(&mut self.namespace, src);
                let decl = format!(".{class_name}: ClassType");
                self.code += &decl;
                self.code.push('\n');
                if let GenTypeObj::Subclass(class) = def.obj.as_ref() {
                    let sup = class
                        .sup
                        .as_ref()
                        .typ()
                        .replace_failure()
                        .to_string_unabbreviated();
                    self.prepare_using_type(class.sup.typ());
                    let sup = self.escape_type(sup);
                    let decl = format!(".{class_name} <: {sup}\n");
                    self.code += &decl;
                }
                if let Some(TypeObj::Builtin {
                    t: Type::Record(rec),
                    ..
                }) = def.obj.base_or_sup()
                {
                    for (attr, t) in rec.iter() {
                        self.prepare_using_type(t);
                        let typ = self.escape_type(t.replace_failure().to_string_unabbreviated());
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
                        self.prepare_using_type(t);
                        let typ = self.escape_type(t.replace_failure().to_string_unabbreviated());
                        let decl = format!("{}.{}: {typ}\n", self.namespace, attr.symbol);
                        self.code += &decl;
                    }
                }
                for attr in ClassDef::get_all_methods(&def.methods_list) {
                    self.gen_chunk_decl(attr);
                }
                self.namespace = stash;
            }
            Expr::Dummy(dummy) => {
                for chunk in dummy.iter() {
                    self.gen_chunk_decl(chunk);
                }
            }
            Expr::Compound(compound) => {
                for chunk in compound.iter() {
                    self.gen_chunk_decl(chunk);
                }
            }
            Expr::Call(call)
                if call
                    .obj
                    .show_acc()
                    .is_some_and(|acc| ControlKind::try_from(&acc[..]).is_ok()) =>
            {
                for arg in call.args.iter() {
                    self.gen_chunk_decl(arg);
                }
            }
            Expr::Lambda(lambda) => {
                for arg in lambda.body.iter() {
                    self.gen_chunk_decl(arg);
                }
            }
            _ => {}
        }
        self.code.push('\n');
    }
}

fn dump_decl_er(path: &NormalizedPathBuf, hir: &HIR, status: CheckStatus) -> std::io::Result<()> {
    let decl_gen = DeclFileGenerator::new(path, status)?;
    let file = decl_gen.gen_decl_er(hir);
    let Some(dir) = path.parent().and_then(|p| p.canonicalize().ok()) else {
        return Ok(());
    };
    let cache_dir = dir.join("__pycache__");
    if !cache_dir.exists() {
        let _ = create_dir_all(&cache_dir);
    }
    let path = cache_dir.join(file.filename);
    if !path.exists() {
        File::create(&path)?;
    }
    let f = File::options().write(true).open(path)?;
    let mut f = BufWriter::new(f);
    f.write_all(file.code.as_bytes())
}

pub fn dump_decl_package(modules: &SharedModuleCache) {
    for (path, module) in modules.raw_iter() {
        if let Some(hir) = module.hir.as_ref() {
            let _ = dump_decl_er(path, hir, module.status);
        }
    }
}
