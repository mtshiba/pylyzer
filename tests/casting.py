import typing

s = "a"

assert isinstance(s, int)  # ERR

# force cast to int
i = typing.cast(int, s)
print(i + 1) # OK

l = typing.cast(list[str], [1, 2, 3])
_ = map(lambda x: x + "a", l)  # OK

d = typing.cast(dict[str, int], [1, 2, 3])
_ = map(lambda x: d["a"] + 1, d)  # OK

t = typing.cast(tuple[str, str], [1, 2, 3])
_ = map(lambda x: x + "a", t)  # OK
