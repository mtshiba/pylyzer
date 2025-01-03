def add(x, y):
    return x + y

print(add(1, 2))
print(add(1, "a")) # ERR
add.x = 1 # ERR

def add2(x: int, y: int) -> str: # ERR
    return x + y

print(add2(1, 2))

for i in [1, 2, 3]:
    j = i + "aa" # ERR
    print(j)

a: int # OK
a = 1
a: str # ERR
a = "aa" if True else "bb"
a: str # OK

while "aaa": # ERR
    a += 1 # ERR
    break

class C:
    x = 1 + "a" # ERR

dic = {"a": 1, "b": 2}
print(dic["c"]) # ERR

def f(d1, d2: dict[str, int]):
    _ = d1["b"] # OK
    _ = d2["a"] # OK
    _ = d2[1] # ERR
    dic = {"a": 1}
    _ = dic["b"] # ERR

i, j = 1, 2
assert i == 1
assert j == 2

with open("test.py") as f:
    for line in f.readlines():
        print("line: " + line)

print(x := 1)
print(x)
