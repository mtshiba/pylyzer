from collections.abc import Sequence

class Vec(Sequence):
    x: list[int]

    def __init__(self):
        self.x = []

    def __getitem__(self, i: int) -> int:
        return self.x[i]

    def __iter__(self):
        return iter(self.x)

    def __len__(self) -> int:
        return len(self.x)

    def __contains__(self, i: int) -> bool:
        return i in self.x
