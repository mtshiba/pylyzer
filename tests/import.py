import export
import random
from random import randint as rdi
from datetime import datetime, timedelta
import datetime as dt
from http.client import HTTPResponse

i = random.randint(0, 1)
print(i + 1)
rdi(0, 1, 2) # ERR

print(export.test)
print(export.add(1, 2))
assert export.add("a", "b") == 1 # ERR

from glob import glob
print(glob("*"))
glob = None
assert glob == None

max_date = datetime.max
max_delta = timedelta.max

assert dt.datetime.max == max_date
