# Pylyzer-specific errors

## E0001: Reassignment of a function referenced by other functions

```python
def g(): return f()

def f(): return 1
def f(): return "a" # E0001: Reassignment of a function referenced by other functions

print(g())
```
