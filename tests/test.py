def add(x, y):
    return x + y

print(add(1, 2))
print(add(1, "a")) # ERR
add.x = 1 # ERR

def add2(x: int, y: int) -> str: # ERR
    return x + y

print(add2(1, 2))

# ERR
for i in [1, 2, 3]:
    j = i + "aa"
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

arr = [1, 2, 3]
print(arr[4]) # ERR

# OK
for i in range(3):
    print(arr[i])
# ERR
for i in range(4):
    print(arr[i])

i, j = 1, 2
assert i == 1
assert j == 2

with open("test.py") as f:
    for line in f.readlines():
        print("line: " + line)
