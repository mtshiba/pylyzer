def f(x: int):
    for i in [1, 2, 3]:
        x += i
        print(x)
    return x

i: int = f(1)

def g(x: int):
    if True:
        x = "a" # ERR
    return x

def h(x: str):
    if True:
        x = "a" # OK
    return x
