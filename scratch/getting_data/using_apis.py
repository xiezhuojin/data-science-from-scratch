import json
from collections import Counter

from bs4 import BeautifulSoup
import requests
from dateutil.parser import parse


# JSON and XML
serialized = """{ "title" : "Data Science Book",
                  "author" : "Joel Grus",
                  "publicationYear" : 2019,
                  "topics" : [ "data", "science", "data science"] }"""

# parse the JSON to create a Python dict
deserialized = json.loads(serialized)
assert deserialized["publicationYear"] == 2019
assert "data science" in deserialized["topics"]

serialized = """
<Book>
  <Title>Data Science Book</Title>
  <Author>Joel Grus</Author>
  <PublicationYear>2014</PublicationYear>
  <Topics>
    <Topic>data</Topic>
    <Topic>science</Topic>
    <Topic>data science</Topic>
  </Topics>
</Book>"""

soup = BeautifulSoup(serialized, "xml")
assert len(soup("Topic")) == 3


# Using an Unauthenticated API

# We'll start by taking a look at GitHub's API, with which you can do some simple
# things unauthenticated:
github_user = "xiezhuojin"
endpoint = f"https://api.github.com/users/{github_user}/repos"
repos = requests.get(endpoint).json()

# At this point repos is a list of Python dicts, each representing a public
# repository in my GitHub account.
dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)

last_5_repositories = sorted(repos, key=lambda r: r["pushed_at"], reverse=True)[:5]
last_5_languages = [repo["language"] for repo in last_5_repositories]