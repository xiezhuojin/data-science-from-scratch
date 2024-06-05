import random

random.seed(10)
four_uniform_randoms = [random.random() for _ in range(4)]

random.seed(10)
print(random.random())
random.seed(10)
print(random.random())

up_to_twn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
random.shuffle(up_to_twn)
print(up_to_twn)

my_best_friend = random.choice(["Alice", "Bob", "Charlie"])

lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
print(winning_numbers)

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print(four_with_replacement)