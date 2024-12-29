def imaginary(x):
    return x.imag

assert imaginary(1) == 0
assert imaginary(1.0) <= 0.0
print(imaginary("a")) # ERR

class C:
    def method(self, x): return x
def call_method(obj, x):
    return obj.method(x)

c = C()
assert call_method(c, 1) == 1
assert call_method(c, 1) == "a" # ERR
print(call_method(1, 1)) # ERR
print(call_method(c)) # ERR

def x_and_y(a):
    z: int = a.y
    return a.x + z

class A:
    x: int
    y: int

    def __init__(self, x, y):
        self.x = x
        self.y = y

class B:
    x: int

    def __init__(self, x):
        self.x = x

a = A(1, 2)
assert x_and_y(a) == 3
b = B(3)
_ = x_and_y(b)  # ERR: B object has no attribute `y`
