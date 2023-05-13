# vscode-pylyzer

![pylyzer_logo_with_letters](https://raw.githubusercontent.com/mtshiba/pylyzer/main/images/pylyzer-logo-with-letters.png)

<a href="https://marketplace.visualstudio.com/items?itemName=pylyzer.pylyzer" target="_blank" rel="noreferrer noopener nofollow"><img src="https://img.shields.io/visual-studio-marketplace/v/pylyzer.pylyzer?style=flat&amp;label=VS%20Marketplace&amp;logo=visual-studio-code" alt="vsm-version"></a>
<a href="https://github.com/mtshiba/pylyzer/releases"><img alt="Build status" src="https://img.shields.io/github/v/release/mtshiba/pylyzer.svg"></a>
<a href="https://github.com/mtshiba/pylyzer/actions/workflows/rust.yml"><img alt="Build status" src="https://github.com/mtshiba/pylyzer/actions/workflows/rust.yml/badge.svg"></a>

`pylyzer` is a static code analyzer / language server for Python, written in Rust.

## Requirements

You need to have the [pylyzer](https://github.com/mtshiba/pylyzer) installed on your system.

To install it, run the following command:

```console
pip install pylyzer
```

or

```console
cargo install pylyzer
```

## Commands

| Command | Description |
| - | - |
| pylyzer.restartLanguageServer | Restart the language server |

## Settings

| Setting | Description | Default |
| - | - | - |
| pylyzer.diagnostics | Enable diagnostics | true |
| pylyzer.inlayHints | Enable inlay hints (this feature is unstable) | false |
| pylyzer.smartCompletion | Enable smart completion (see [ELS features](https://github.com/erg-lang/erg/blob/main/crates/els/doc/features.md))| true |
