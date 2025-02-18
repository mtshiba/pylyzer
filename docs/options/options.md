# command line options

## --server

Launch as a language server.

## --clear-cache

Clear the cache files.

## --dump-decl

Dump a type declarations file (d.er) after type checking.

```bash
$ pylyzer --dump-decl test.py
Start checking: test.py
All checks OK: test.py

$ ls
test.py  test.d.er
```

## -c/--code

Check code from the command line.

```bash
$ pylyzer -c "print('hello world')"
Start checking: string
All checks OK: string
```

## --disable

Disable a default LSP feature.
Default (disableable) features are:

* codeAction
* codeLens
* completion
* diagnostics
* findReferences
* gotoDefinition
* hover
* inlayHint
* rename
* semanticTokens
* signatureHelp
* documentLink

## --verbose

Print process information verbosely.

## --no-infer-fn-type

When a function type is not specified, no type inference is performed and the function type is assumed to be `Any`.

## --fast-error-report

Simplify error reporting by eliminating to search for similar variables when a variable does not exist.

## --hurry

Enable `--no-infer-fn-type` and `--fast-error-report`.

## --do-not-show-ext-errors

Do not show errors from external libraries.

## --do-not-respect-pyi

If specified, the actual `.py` types will be respected over the `.pyi` types.
Applying this option may slow down the analysis.
