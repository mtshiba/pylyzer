from pathlib import Path
import os
import shlex
from glob import glob
import tomllib
import shutil

from setuptools import setup, Command
from setuptools_rust import RustBin

class Clean(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        # super().run()
        for d in ["build", "dist", "src/pylyzer.egg-info"]:
            shutil.rmtree(d, ignore_errors=True)

with open("README.md", encoding="utf-8", errors="ignore") as fp:
    long_description = fp.read()

with open("Cargo.toml", "rb") as fp:
    toml = tomllib.load(fp)
    name = toml["package"]["name"]
    description = toml["package"]["description"]
    version = toml["workspace"]["package"]["version"]
    license = toml["workspace"]["package"]["license"]
    url = toml["workspace"]["package"]["repository"]

cargo_args = ["--no-default-features"]

home = os.path.expanduser("~")
file_and_dirs = glob(".erg/lib/**", recursive=True, root_dir=home)
paths = [Path(home + "/" + path) for path in file_and_dirs if os.path.isfile(home + "/" + path)]
files = [(str(path).removesuffix("/" + path.name).removeprefix(home), str(path)) for path in paths]
data_files = {}
for key, value in files:
    if key in data_files:
        data_files[key].append(value)
    else:
        data_files[key] = [value]
data_files = list(data_files.items())

setup(
    name=name,
    author="mtshiba",
    author_email="sbym1346@gmail.com",
    url=url,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version,
    license=license,
    python_requires=">=3",
    rust_extensions=[
        RustBin("pylyzer", args=cargo_args, cargo_manifest_args=["--locked"])
    ],
    cmdclass={
        "clean": Clean,
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Topic :: Software Development :: Quality Assurance",
    ],
    data_files=data_files,
)
