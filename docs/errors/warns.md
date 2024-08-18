# Pylyzer-specific warnings

## W0188: Used value

```python
def f(x): return x

f(1) # W0188: UnusedWarning: the evaluation result of the expression (: {1, }) is not used
```
