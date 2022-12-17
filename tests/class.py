class C:
    def __init__(x: int):
        self.x = x
    def method(self):
        return self.x

a = C(1).method() # OK
_: int = a + 1
b = C("a").method() # ERR
