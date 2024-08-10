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
