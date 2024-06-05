if 1 > 2:
    message = "if only were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"

x = 3
parity = "even" if x % 2 == 0 else "odd"

x = 0
while x < 10:
    print(f"{x} is less than 10")
    x += 1

for x in range(1):
    print(f"{x} is less than 10")

for x in range(10):
    if x == 3:
        continue
    if x == 5:
        break
    print(x)