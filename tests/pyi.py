x = 1
x + "a" # OK, because x: Any

def f(x, y):
    return x + y

class C:
    y = 1
    def __init__(self, x):
        self.x = x
    def f(self, x):
        return self.x + x

print(f(1, 2)) # OK
print(f("a", "b")) # ERR*2
c = C(1)
print(c.f(2)) # OK
print(c.f("a")) # ERR
_ = C("a") # ERR

def g(x):
    pass

print(g(c)) # OK
print(g(1)) # ERR
