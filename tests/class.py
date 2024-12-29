from typing import Self

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
    def id(self) -> Self:
        return self
    def id2(self) -> "C":
        return self

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
assert c.id() == c

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
    def invalid(self):
        return self.d # ERR: E object has no attribute `d`

c1 = D(1).c + 1
d = D(1) + D(2)
err = C(1, 2) + D(1) # ERR
ok = D(1) - C(1, 2) # OK
assert D(1) > D(0)
c = -d # OK
e = E(1)

class F:
    def __init__(self, x: int, y: int = 1, z: int = 2):
        self.x = x
        self.y = y
        self.z = z

_ = F(1)
_ = F(1, 2)
_ = F(1, z=1, y=2)

class G(DoesNotExist):  # ERR
    def foo(self):
        return 1

g = G()
assert g.foo() == 1

class Value:
    value: object

class H(Value):
    value: int

    def __init__(self, value):
        self.value = value

    def incremented(self):
        return H(self.value + 1)

class MyList(list):
    @staticmethod
    def try_new(lis) -> "MyList" | None:
        if isinstance(lis, list):
            return MyList(lis)
        else:
            return None

class Implicit:
    def __init__(self):
        self.foo = False

    def set_foo(self):
        self.foo = True
