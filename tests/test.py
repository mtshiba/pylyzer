def add(x, y):
    return x + y

print(add(1, 2))
print(add(1, "a"))
add.x = 1

def add2(x: int, y: int) -> str:
    return x + y

print(add2(1, 2))

for i in [1, 2, 3]:
    j = i + "aa"
    print(j)

while "aaa":
    print("invalid")
    break

class C:
    x = 1 + "a"

dic = {"a": 1, "b": 2}
print(dic["c"])

a = [1, 2, 3]
print(a[4])

a_  = "aa" if True else "bb"