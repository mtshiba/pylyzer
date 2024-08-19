from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U", bound=int)
IS = TypeVar("IS", int, str)
def id(x: T) -> T:
    return x

def id_int(x: U) -> U:
    return x

def id_int_or_str(x: IS) -> IS:
    return x

_ = id(1) + 1 # OK
_ = id("a") + "b" # OK
_ = id_int(1) # OK
_ = id_int("a") # ERR
_ = id_int_or_str(1) # OK
_ = id_int_or_str("a") # OK
_ = id_int_or_str(None) # ERR

def id2[T](x: T) -> T:
    return x

def id_int2[T: int](x: T) -> T:
    return x

_ = id2(1) + 1  # OK
_ = id2("a") + "b"  # OK
_ = id_int2(1) # OK
_ = id_int2("a") # ERR
