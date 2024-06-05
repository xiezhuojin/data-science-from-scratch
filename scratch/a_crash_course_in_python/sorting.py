x = [4, 1, 2, 3]
y = sorted(x)
x.sort()

x = sorted([-4, 1, -2, 3], key=abs, reverse=True)
assert x == [-4, 3, -2, 1]
