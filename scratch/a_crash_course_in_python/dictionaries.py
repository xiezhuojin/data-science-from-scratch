empty_dict = {}
empty_dict2 = dict()
grades = {"Joel": 80, "Tim": 95}

joels_grade = grades["Joel"]
try:
    kates_grade = grades["Kate"]
except KeyError:
    print("no grade for Kate!")

joel_has_grade = "Joel" in grades
kate_has_grade = "Kate" in grades

joels_grade = grades.get("Joel", 0)
kates_grade = grades.get("Kate", 0)
no_ones_grade = grades.get("No One")

grades["Tim"] = 99
grades["Kate"] = 100
num_students = len(grades)
assert num_students == 3

tweet = {
    "user" : "joelgrus",
    "text" : "Data Science is Awesome",
    "retweet_count" : 100,
    "hashtags" : ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}

tweet_keys = tweet.keys()
tweet_values = tweet.values()
tweet_items = tweet.items()

assert "user" in tweet_keys
assert "user" in tweet
assert "joelgrus" in tweet_values
