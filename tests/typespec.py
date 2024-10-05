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
_: tuple[int, ...] = (1, 2, 3)
_: list[tuple[int, ...]] = [(1, 2, 3)]
_: dict[str, dict[str, Union[int, str]]] = {"a": {"b": 1}}
_: dict[str, dict[str, list[int]]] = {"a": {"b": [1]}}
_: dict[str, dict[str, dict[str, int]]] = {"a": {"b": {"c": 1}}}
_: dict[str, dict[str, Optional[int]]] = {"a": {"b": 1}}
_: dict[str, dict[str, Literal[1, 2]]] = {"a": {"b": 1}}
_: dict[str, dict[str, Callable[[int], int]]] = {"a": {"b": abs}}
_: dict[str, dict[str, Callable[[int], None]]] = {"a": {"b": print}}
_: dict[str, dict[str, Opional[int]]] = {"a": {"b": 1}} # ERR
_: dict[str, dict[str, Union[int, str]]] = {"a": {"b": None}} # ERR
_: dict[str, dict[str, list[int]]] = {"a": {"b": ["c"]}} # ERR
_: dict[str, dict[str, Callable[[int], int]]] = {"a": {"b": print}} # ERR
_: dict[str, dict[str, Optional[int]]] = {"a": {"b": "c"}} # ERR
_: dict[str, dict[str, Literal[1, 2]]] = {"a": {"b": 3}} # ERR
_: list[tuple[int, ...]] = [(1, "a", 3)] # ERR

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

i1 = 1 # type: int
# ERR
i2 = 1 # type: str
i3 = 1 # type: ignore
i3 + "a" # OK
