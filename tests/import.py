import export
import random

i = random.randint(0, 1)
print(i + 1)

print(export.test)
print(export.add(1, 2))
assert export.add("a", "b") == 1
