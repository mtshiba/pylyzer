# Pylyzer-specific errors

## E0001: Reassignment of a function referenced by other functions

```python
def g(): return f()

def f(): return 1
def f(): return "a" # E0001: Reassignment of a function referenced by other functions

print(g())
```

## E0002: `__init__` doesn't have a first parameter named `self`

```python
class C:
    def __init__(a): pass # E0002
```

## E0003: `__init__` as a member variable

```python
class C:
    __init__ = 1 # E0003
```
