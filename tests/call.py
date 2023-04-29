print("aaa", sep = ";", end = "") # OK
print("a", sep=1) # ERR
print("a", foo=None) # ERR

def f(x, y=1):
    return x + y

print(f(1, 2)) # OK
print(f(1)) # OK
print(f(1, y="a")) # ERR
