import export
import random
from random import randint as rdi

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
