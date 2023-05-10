i_arr = [0]

i_arr.append(1)
i_arr.append("a") # ERR

union_arr: list[int | str] = []
union_arr.append(1)
union_arr.append("a") # OK
union_arr.append(None) # ERR

dic = {"a": 1}
dic["b"] = 2
_ = dic["a"]
_ = dic["b"]
_ = dic["c"] # ERR

dic2: dict[str, int] = {"a": 1}
_ = dic2["c"] # OK

t: tuple[int, str] = (1, "a")
_ = t[0] == 1 # OK
_ = t[1] == 1 # ERR

def f(s: Str): return None
for i in getattr(1, "aaa", ()):
    f(i)
