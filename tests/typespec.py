from typing import Union, Optional, Literal, Callable
from collections.abc import Iterable, Mapping

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

def g(x: int) -> int:
    return x

_: Callable[[Union[int, str]], None] = f # OK
_: Callable[[Union[int, str]], None] = g # ERR

_: Iterable[int] = [1] # OK
_: Iterable[int] = {1} # OK
_: Iterable[int] = (1, 2) # OK
_: Iterable[int] = ["a"] # ERR

_: Mapping[str, int] = {"a": 1, "c": 2} # OK
_: Mapping[str, int] = {1: "a", 2: "b"} # ERR

def f(x: Union[int, str, None]):
    pass
# OK
f(1)
f("a")
f(None)
