class Foo:
    x: int
    def __init__(self, x):
        self.x = x

    @property
    def foo(self):
        return self.x

f = Foo(1)
print(f.foo + "a")  # ERR
