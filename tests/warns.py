# W0188: unused value

1 # Warn

def f(): return "a"
f() # Warn

def f(): return None
f() # OK
