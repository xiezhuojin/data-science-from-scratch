even_numbers = [x for x in range(5) if x % 2 == 0]
sqaures = [x ** 2 for x in range(5)]
even_sqaures = [x ** 2 for x in even_numbers]

sqaure_dict = {x: x ** 2 for x in range(5)}
sqaure_set = {x ** 2 for x in [1, -1]}

zeros = [0 for _ in even_numbers]

pairs = [(x, y)
        for x in range(10)
        for y in range(10)]

incresing_pairs = [(x, y)
                   for x in range(10)
                   for y in range(x + 1, 10)]