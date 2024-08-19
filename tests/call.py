print("aaa", sep = ";", end = "") # OK
print("a", sep=1) # ERR
print("a", foo=None) # ERR

def f(x, y=1):
    return x + y

print(f(1, 2)) # OK
print(f(1)) # OK
print(f(1, y="a")) # ERR

def g(first, second):
    pass

g(**{"first": "bar", "second": 1}) # OK
g(**[1, 2]) # ERR
g(1, *[2]) # OK
g(*[1, 2]) # OK
g(1, 2, *[3, 4]) # ERR
g(*1) # ERR
g(*[1], **{"second": 1}) # OK

_ = f(1, *[2]) # OK
_ = f(**{"x": 1, "y": 2}) # OK
