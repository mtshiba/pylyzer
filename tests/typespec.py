from typing import Union, Optional, Literal

i: Union[int, str] = 1 # OK
j: Union[int, str] = "aa" # OK
k: Union[list[int], str] = 1 # ERR
l: Union[list[int], str] = [1] # OK
o: Optional[int] = None # OK
p: Optional[int] = "a" # ERR
weekdays: Literal[1, 2, 3, 4, 5, 6, 7] = 1 # OK
weekdays: Literal[1, 2, 3, 4, 5, 6, 7] = 8 # ERR

def f(x: Union[int, str]) -> None:
    pass

f(1) # OK
f(None) # ERR
