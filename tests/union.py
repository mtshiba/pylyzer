s: str | bytes = ""
s2 = s.capitalize()
s3 = s2.center(1)

s4: str | bytes | bytearray = ""
_ = s4.__len__()

def f(x: str | bytes):
    return x.isalnum()

def check(s: str | bytes | bytearray):
    if isinstance(s, (bytes, bytearray)):
        pass
