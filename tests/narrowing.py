def f(x: int | None):
    x + 1 # ERR
    if x != None:
        print(x + 1) # OK
    if isinstance(x, int):
        print(x + 1) # OK
    return None

f(1)

from typing import Optional

x: Optional[int] = None
if x is not None:
    x += 1
