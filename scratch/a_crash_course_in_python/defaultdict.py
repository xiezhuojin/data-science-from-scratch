from collections import defaultdict

document = "Imagine that you’re trying to count the words in a document.  An obvious approach is to create a dictionary in which the keys are words and the values are counts.  As you check each word, you can increment its count if it’s already in the dictionary and add it to the dictionary if it’s not".split(",. ")

word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts = 1

word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1

word_counts = defaultdict(int)
for word in document:
    word_counts[word] += 1

dd_list = defaultdict(list)
dd_list[2].append(1)

dd_dict = defaultdict(dict)
dd_dict["Joel"]["City"] = "Seattle"

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1