# command line options

## --server

Launch as a language server.

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
