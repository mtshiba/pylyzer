# pylyzer

<a href="https://marketplace.visualstudio.com/items?itemName=pylyzer.pylyzer" target="_blank" rel="noreferrer noopener nofollow"><img src="https://img.shields.io/visual-studio-marketplace/v/pylyzer.pylyzer?style=flat&amp;label=VS%20Marketplace&amp;logo=visual-studio-code" alt="vsm-version"></a>

`pylyzer` is a static code analyzer / language server for Python written in Rust.

## Installation

### pip

```bash
pip install pylyzer
```

If installed this way, type resolution for the standard libraries is not available.
If you want to use it, you need to install [Erg](https://github.com/mtshiba/ergup).

### cargo (rust package manager)

```bash
cargo install pylyzer
```

or download the binary from [the releases page](https://github.com/mtshiba/pylyzer/releases).

## What is the advantage over pylint, pyright, pytype, etc.?

* Performance

pylyzer can inspect Python scripts on average __100 times faster__ than pytype and pyright. This is largely due to the fact that pylyzer is implemented in Rust, whereas pytype is implemented in Python.

![performance](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/performance.png)

* Detailed analysis

pylyzer can do more than the usual type testing. For example, it can detect out-of-bounds accesses to lists and accesses to nonexistent keys in dicts.

![analysis](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/analysis.png)

* Reports readability

While pytype/pyright's error reports are illegible, pylyzer shows where the error occurred and provides clear error messages.

![reports](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/reports.png)

## How it works

pylyzer uses the type checker of [the Erg programming language](https://erg-lang.org) internally.
This language is a transpiled language that targets Python, and has a static type system.

pylyzer converts Python ASTs to Erg ASTs and passes them to Erg's type checker. It then displays the results with appropriate modifications.

## TODOs

* [x] type checking
* [x] type inferring
* [x] builtin modules resolving (partially)
* [x] local scripts resolving
* [ ] local packages resolving
