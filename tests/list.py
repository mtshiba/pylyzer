l = [1, 2, 3]
_ = l[1:2]
print(l[2])
print(l["a"]) # ERR

# OK
for i in range(3):
    print(l[i])
# ERR
for i in "abcd":
    print(l[i])

lis = "a,b,c".split(",") if True is not None else []
if "a" in lis:
    lis.remove("a") # OK
