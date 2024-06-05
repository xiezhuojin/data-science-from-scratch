from finding_key_connectors import *


def get_friends(friendships: DefaultDict[int, List[int]], user_id: int, level: int) -> List[int]:
    if level < 0 or not friendships[user_id]:
        return [user_id]

    level_of_friends = [[user_id]]
    existed_friends = set()
    for i in range(level):
        last_level_of_friends = level_of_friends[i]
        existed_friends = existed_friends.union(last_level_of_friends)
        current_level_of_friends = [ff for f in last_level_of_friends for ff in friendships[f] if ff not in existed_friends]
        current_level_of_friends = list(set(current_level_of_friends))
        level_of_friends.append(current_level_of_friends)
    return level_of_friends[level]

# print(f"foaf: {get_friends(friendships, 0, 2)}")

def friends_of_friends(friendships: DefaultDict[int, List[int]], user) -> Counter:
    user_id = user["id"]
    friends_friends = []
    friends = friendships[user_id]
    for friend in friends:
        for friend_friend in friendships[friend]:
            if friend_friend not in friends and friend_friend != user_id:
                friends_friends.append(friend_friend)
    return Counter(friends_friends)

# print(friends_of_friends(friendships, users[3]))

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

def data_scientists_who_like(target_interest: str) -> List[int]:
    return [user_id for user_id, interest in interests if interest == target_interest]

user_ids_with_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_with_interest[interest].append(user_id)

interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

# pprint(user_ids_with_interest)
# pprint(interests_by_user_id)

def most_common_interests_with(user):
    user_id = user["id"]
    common_interests_users = []
    for interest in interests_by_user_id[user_id]:
        for user in user_ids_with_interest[interest]:
            if user != user_id:
                common_interests_users.append(user)
    return Counter(common_interests_users).most_common(1)

pprint(most_common_interests_with(users[0]))