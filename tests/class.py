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
    def method(self):
        return self.x

c = C(1, 2)
assert c.x == 1
assert c.y == 2 # OK, c.y == "a" is also OK
a = c.method() # OK
_: int = a + 1
b = C("a").method() # ERR

class D:
    c: int
    def __init__(self, c):
        self.c = c

c1 = D(1).c + 1
