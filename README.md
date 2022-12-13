# pype

`pype` is a static code analyzer / language server for Python written in Rust.

## Installation

### pip

```bash
pip install pype
```

### cargo (rust package manager)

```bash
cargo install pype
```

or download the binary from [the releases page](https://github.com/mtshiba/pype/releases).

## What is the advantage over pylint, pyright, pytype, etc.?

* Performance: pype can inspect Python scripts on average __100 times faster__ than pytype and pyright. This is largely due to the fact that pype is implemented in Rust, whereas pytype is implemented in Python.

![performance](https://raw.githubusercontent.com/mtshiba/pype/main/images/performance.png)

* Detailed analysis

pype can do more than the usual type testing. For example, it can detect out-of-bounds accesses to lists and accesses to nonexistent keys in dicts.

![analysis](https://raw.githubusercontent.com/mtshiba/pype/main/images/analysis.png)

* Reports readability: While pytype's error reports are crude, showing only that an error has occurred, pype shows where the error occurred and provides clear error messages.

![reports](https://raw.githubusercontent.com/mtshiba/pype/main/images/reports.png)

## How it works

pype uses the type checker of [the Erg programming language](https://erg-lang.org) internally.
This language is a transpiled language that targets Python, and has a static type system.

pype converts Python ASTs to Erg ASTs and passes them to Erg's type checker. It then displays the results with appropriate modifications.
