i: int = 0
i: str = "a" # OK

if True:
    i = 1 # ERR
else:
    i = 2 # ERR

while False:
    i = 3 # ERR

def f(x: int):
    i = 1 # OK
    return x + i

if True:
    pass
elif True:
    for i in []: pass
    pass
elif True:
    for i in []: pass
    pass

if True:
    pass
elif True:
    with open("") as x:
        pass
    pass
elif True:
    with open("") as x:
        pass
    pass

if True:
    left, right = 1, 2
if True:
    left, _ = 1, 2
