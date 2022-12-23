class Empty: pass
emp = Empty()

class x(): pass
y = x()
# multiple class definitions are allowed
class x(): pass
y = x()

class C:
    def __init__(x: int):
        self.x = x
    def method(self):
        return self.x

c = C(1)
assert c.x == 1
a = c.method() # OK
_: int = a + 1
b = C("a").method() # ERR
