def imaginary(x):
    x.imag

assert imaginary(1) == 0
assert imaginary(1.0) <= 0.0
print(imaginary("a")) # ERR

class C:
    def method(self, x): return x
def call_method(obj, x):
    obj.method(x)

c = C()
assert call_method(c, 1) == 1
assert call_method(c, 1) == "a" # ERR
print(call_method(1, 1)) # ERR
print(call_method(c)) # ERR
