from typing import DefaultDict, List
from collections import defaultdict, Counter
from pprint import pprint

users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# create a friendship dict (key: user id, value: set of user ids)
friendships = defaultdict(list)

for user_id, friend_id in friendship_pairs:
    friendships[user_id].append(friend_id)
    friendships[friend_id].append(user_id)

def number_of_friends(friendships: DefaultDict[int, List[int]], user_id: int) -> int:
    return len(friendships[user_id])

total_connections = sum((number_of_friends(friendships, user_id) for user_id in friendships))
# print(f"total connecitons: {total_connections}")

avg_connections = total_connections / len(friendships.keys())
# print(f"avg connections: {avg_connections}")

# network metric degree centrality
friend_num_by_id = Counter({user_id: len(friends) for user_id, friends in friendships.items()})

# print(f"num of friends by id: {friend_num_by_id}")
# print(friend_num_by_id.most_common())
