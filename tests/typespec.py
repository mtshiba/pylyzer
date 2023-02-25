from typing import Union, Optional

i: Union[int, str] = 1 # OK
j: Union[int, str] = "aa" # OK
k: Union[list[int], str] = 1 # ERR
l: Union[list[int], str] = [1] # OK
o: Optional[int] = None # OK
p: Optional[int] = "a" # ERR

def f(x: Union[int, str]) -> None:
    pass

f(1) # OK
f(None) # ERR
