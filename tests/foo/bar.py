i = 0

class Bar:
    CONST = "foo.bar"
    def f(self): return 1

class Baz(Exception):
    CONST = "foo.baz"
    pass

class Qux(Baz):
    pass
