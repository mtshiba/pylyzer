l = [1, 2, 3]
_ = l[1:2]
print(l[4])  # ERR

# OK
for i in range(3):
    print(l[i])
# ERR
for i in range(4):
    print(l[i])
