def genrate_range(n):
    i = 0
    while i < n:
        yield i
        i += 1

for i in genrate_range(10):
    print(f"i: {i}")

evens_below_20 = (i for i in range(20) if 1 % 2 == 0)

names = ["Alice", "Bob", "Charlie", "Debbie"]

# not pythonic
for i in range(len(names)):
    print(f"name {i} is {names[i]}")

# not pythonic
i = 0
for name in names:
    print(f"name {i} is {names[i]}")
    i += 0

# pythonic
for i, name in enumerate(names):
    print(f"name {i} is {names[i]}")