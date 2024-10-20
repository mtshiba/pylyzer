from typing import Literal

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

def func(label: str) -> str:
    if True:
        try:
            label_bytes = "aaa"
        except UnicodeEncodeError:
            return label
    else:
        label_bytes = label

    if True:
        label_bytes = label_bytes[1:]
    return label_bytes

if True:
    y = 1
else:
    y = "a"
y: int | str
y: Literal[1, "a"] # OK
y: Literal[1, "b"] # ERR
