class Foo:
    def invalid_append(self):
        paths: list[str] = []
        paths.append(self)  # ERR
