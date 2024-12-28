from typing import Callable, Mapping

_: Mapping[int, str, str] = ...  # ERR
_: Mapping[int] = ...  # ERR
_: Callable[[int, str]] = ...  # ERR
_: Callable[int] = ...  # ERR
_: dict[int] = ...  # ERR
_: dict[int, int, int] = ...  # ERR
