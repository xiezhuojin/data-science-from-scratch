list1 = ["a", "b", "c"]
list2 = [1, 2, 3]

pairs = [pair for pair in zip(list1, list2)]

latters, numbers = zip(*pairs)

def add(a, b):
    return a + b

add(1, 2)
try:
    add([1, 2])
except TypeError:
    print("add excepts two inputs")

add(*[1, 2])
     