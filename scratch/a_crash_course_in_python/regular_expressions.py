import re


# re.match checks whether the beginning of a string matches a regular expression,
# while re.search checks any part of a stirng matches a regular expression.
re_example = [
    not re.match("a", "cat"),
    re.search("a", "cat"),
    not re.search("c", "dog"),
    3 == len(re.split("[ab]", "carbs")),
    "R-D-" == re.sub("[0-9]", "-", "R2D2")
]

assert all(re_example)
print(re_example)