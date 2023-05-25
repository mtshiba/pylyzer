import export
import foo
from foo import bar
from foo import baz
import random
from random import randint as rdi
from datetime import datetime, timedelta
import datetime as dt
from http.client import HTTPResponse
import http

i = random.randint(0, 1)
print(i + 1)
rdi(0, 1, 2) # ERR

print(export.test)
print(export.add(1, 2))
assert export.add("a", "b") == 1 # ERR

c = export.C(1)
assert c.const == 1
assert c.x == 2
assert c.method(2) == 3

d = export.D(1, 2)
assert d.x == 1
assert d.y == 2

assert foo.i == 0

from glob import glob
print(glob("*"))
glob = None
assert glob == None

max_date = datetime.max
max_delta = timedelta.max
assert dt.datetime.max == max_date

Resp = http.client.HTTPResponse
assert export.http.client.HTTPResponse == Resp
