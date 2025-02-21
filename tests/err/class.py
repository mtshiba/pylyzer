class Foo:
    def invalid_append(self):
        paths: list[str] = []
        paths.append(self)  # ERR

class Bar:
    foos: list[Foo]

    def __init__(self, foos: list[Foo]) -> None:
        self.foos = foos

    def add_foo(self, foo: Foo):
        self.foos.append(foo)

    def invalid_add_foo(self):
        self.foos.append(1) # ERR

_ = Bar([Bar([])]) # ERR
