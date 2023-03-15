# vscode-pylyzer

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
| pylyzer.checkOnType | Perform checking each time any character is entered. This improves the accuracy of completion, etc., but may slow down the execution | false |
