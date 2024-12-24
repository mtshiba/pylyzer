b = False
if True:
    b = True
if True:
    b = "a" # ERR

counter = 100 # counter: Literal[100]
while counter > 0:
    counter -= 1  # counter: Int
    counter -= 1.0  # counter: Float
