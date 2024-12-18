# pylyzer ⚡

![pylyzer_logo_with_letters](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/pylyzer-logo-with-letters.png)

<a href="https://marketplace.visualstudio.com/items?itemName=pylyzer.pylyzer" target="_blank" rel="noreferrer noopener nofollow"><img src="https://img.shields.io/visual-studio-marketplace/v/pylyzer.pylyzer?style=flat&amp;label=VS%20Marketplace&amp;logo=visual-studio-code" alt="vsm-version"></a>
<a href="https://github.com/mtshiba/pylyzer/releases"><img alt="Build status" src="https://img.shields.io/github/v/release/mtshiba/pylyzer.svg"></a>
<a href="https://github.com/mtshiba/pylyzer/actions/workflows/rust.yml"><img alt="Build status" src="https://github.com/mtshiba/pylyzer/actions/workflows/rust.yml/badge.svg"></a>

`pylyzer` is a static code analyzer / language server for Python, written in Rust.

## Installation

### pip

```bash
pip install pylyzer
```

### cargo (Rust package manager)

```bash
cargo install pylyzer --locked
```

### build from source

```bash
git clone https://github.com/mtshiba/pylyzer.git
cargo install --path . --locked
```

Make sure that `cargo`/`rustc` is up-to-date, as pylyzer may be written with the latest (stable) language features.

### [GitHub Releases](https://github.com/mtshiba/pylyzer/releases/latest)

## How to use

### Check a single file

```sh
pylyzer file.py
```

### Check an entire package

If you don't specify a file path, pylyzer will automatically search for the entry point.

```sh
pylyzer
```

### Start the language server

This option is used when an LSP-aware editor requires arguments to start pylyzer.

```sh
pylyzer --server
```

For other options, check [the manual](https://mtshiba.github.io/pylyzer/options/options/).

## What is the advantage over pylint, pyright, pytype, etc.?

* Performance 🌟

On average, pylyzer can inspect Python scripts more than __100 times faster__ than pytype and pyright [<sup id="f1">1</sup>](#1). This is largely due to the fact that pylyzer is implemented in Rust.

![performance](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/performance.png)

* Reports readability 📖

While pytype/pyright's error reports are illegible, pylyzer shows where the error occurred and provides clear error messages.

### pyright

![pyright_report](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/pyright_report.png)

### pylyzer 😃

![report](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/report.png)

* Rich LSP support 📝

pylyzer as a language server supports various features, such as completion and renaming (The language server is an adaptation of the Erg Language Server (ELS). For more information on the implemented features, please see [here](https://github.com/erg-lang/erg/tree/main/crates/els#readme)).

![lsp_support](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/lsp_support.png)

![autoimport](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/autoimport.gif)

## VSCode extension

You can install the VSCode extension from the [Marketplace](https://marketplace.visualstudio.com/items?itemName=pylyzer.pylyzer) or from the command line:

```sh
code --install-extension pylyzer.pylyzer
```

## What is the difference from [Ruff](https://github.com/astral-sh/ruff)?

[Ruff](https://github.com/astral-sh/ruff), like pylyzer, is a static code analysis tool for Python written in Rust, but Ruff is a linter and pylyzer is a type checker & language server.
pylyzer does not perform linting & formatting, and Ruff does not perform type checking.

## How it works

pylyzer uses the type checker of [the Erg programming language](https://erg-lang.org) internally.
This language is a transpiled language that targets Python, and has a static type system.

pylyzer converts Python ASTs to Erg ASTs and passes them to Erg's type checker. It then displays the results with appropriate modifications.

## Limitations

* pylyzer's type inspector only assumes (potentially) statically typed code, so you cannot check any code uses reflections, such as `exec`, `setattr`, etc.

* pylyzer (= Erg's type system) has its own type declarations for the Python standard APIs. Typing of all APIs is not complete and may result in an error that such an API does not exist.

* Since pylyzer's type checking is conservative, you may encounter many (possibly false positive) errors. We are working on fixing this, but if you are concerned about editor errors, please turn off the diagnostics feature.

## TODOs

* [x] type checking
    * [x] variable
    * [x] operator
    * [x] function/method
    * [x] class
    * [ ] `async/await`
* [x] type inference
    * [x] variable
    * [x] operator
    * [x] function/method
    * [x] class
* [x] builtin modules analysis
* [x] local scripts analysis
* [x] local packages analysis
* [x] LSP features
    * [x] diagnostics
    * [x] completion
    * [x] rename
    * [x] hover
    * [x] goto definition
    * [x] signature help
    * [x] find references
    * [x] document symbol
    * [x] call hierarchy
* [x] collection types
    * [x] `list`
    * [x] `dict`
    * [x] `tuple`
    * [x] `set`
* [ ] `typing`
    * [x] `Union`
    * [x] `Optional`
    * [x] `Literal`
    * [x] `Callable`
    * [x] `Any`
    * [x] `TypeVar`
    * [ ] `TypedDict`
    * [ ] `ClassVar`
    * [ ] `Generic`
    * [ ] `Protocol`
    * [ ] `Final`
    * [ ] `Annotated`
    * [ ] `TypeAlias`
    * [ ] `TypeGuard`
    * [x] type parameter syntax
    * [x] type narrowing
    * [ ] others
* [ ] `collections.abc`
    * [x] `Iterable`
    * [x] `Iterator`
    * [x] `Mapping`
    * [x] `Sequence`
    * [ ] others
* [x] type assertion (`typing.cast`)
* [x] type narrowing (`is`, `isinstance`)
* [x] `pyi` (stub) files support
* [ ] glob pattern file check
* [x] type comment (`# type: ...`)
* [x] virtual environment support
* [x] package manager support
    * [x] `pip`
    * [x] `poetry`
    * [x] `uv`

## Join us!

We are looking for contributors to help us improve pylyzer. If you are interested in contributing and have any questions, please feel free to contact us.

* [Discord (Erg language)](https://discord.gg/kQBuaSUS46)
    * [#pylyzer](https://discord.com/channels/1006946336433774742/1056815981168697354)
* [GitHub discussions](https://github.com/mtshiba/pylyzer/discussions)

---

<span id="1" style="font-size:x-small"><sup>1</sup> The performance test was conducted on MacBook (Early 2016) with 1.1 GHz Intel Core m3 processor and 8 GB 1867 MHz LPDDR3 memory.[↩](#f1)</span>
