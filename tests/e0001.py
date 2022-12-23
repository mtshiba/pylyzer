def a(): return 1
def a(): return "a" # OK

print(a())

def g(): return f()

def f(): return 1
def f(): return "a" # E0001: Reassignment of a function referenced by other functions
