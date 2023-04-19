import typing

s = "a"

assert isinstance(s, int)  # ERR

# force cast to int
i = typing.cast(int, s)
print(i + 1) # OK
