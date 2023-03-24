import http.client

test = 1

def add(a, b):
    return a + b

class C:
    x: int
    const = 1
    def __init__(self, x):
        self.x = x
    def method(self, y): return self.x + y

class D(C):
    y: int
    def __init__(self, x, y):
        self.x = x
        self.y = y
