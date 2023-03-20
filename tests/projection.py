def imaginary(x):
    x.imag

assert imaginary(1) == 0
assert imaginary(1.0) <= 0.0
print(imaginary("a")) # ERR
