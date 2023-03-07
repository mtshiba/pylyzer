class Empty: pass
emp = Empty()

class x(): pass
y = x()
# multiple class definitions are allowed
class x(): pass
y = x()

class C:
    def __init__(self, x: int, y): # y: Obj
        self.x = x
        self.y = y # y: Never
    def __add__(self, other: C):
        return C(self.x + other.x, self.y + other.y)
    def method(self):
        return self.x

c = C(1, 2)
assert c.x == 1
# OK, c.y == "a" is also OK (cause the checker doesn't know the type of C.y)
assert c.y == 2
assert c.z == 3 # ERR
d = c + c
assert d.x == 2
assert d.x == "a" # ERR
a = c.method() # OK
_: int = a + 1
b = C("a").method() # ERR

class D:
    c: int
    def __add__(self, other: D):
        return D(self.c + other.c)
    def __sub__(self, other: C):
        return D(self.c - other.x)
    def __neg__(self):
        return D(-self.c)
    def __gt__(self, other: D):
        return self.c > other.c
    def __init__(self, c):
        self.c = c

class E(D):
    def __add__(self, other: E):
        return E(self.c + other.c)

c1 = D(1).c + 1
d = D(1) + D(2)
err = C(1, 2) + D(1) # ERR
ok = D(1) - C(1, 2) # OK
assert D(1) > D(0)
c = -d # OK
e = E(1)
