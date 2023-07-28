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
