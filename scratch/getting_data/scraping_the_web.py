from typing import Dict, Set
import re

from bs4 import BeautifulSoup
import requests

url = "https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

# For example, to find the first <p> tag (and its contents), you can use:
first_paragraph = soup.find("p")

# You can get the text contents of a Tag using its text property
first_paragraph_id = first_paragraph["id"] # raise KeyError if no 'id
first_paragraph_id2 = first_paragraph.get("id") # return None if no 'id

# You can get multiple tags at once as follows:
all_paragraphs = soup.find_all("p")
parapgraphs_with_ids = [p for p in all_paragraphs if p.get("id")]

# Frequently, you' ll want to find tags with a specific class:
important_paragraphs = soup.get("p", {"class": "important"})
important_paragraphs2 = soup("p", "important")
important_paragraphs3 = [p for p in soup.find_all("p")
                         if "important" in p.get("class", [])]

# For example, if you want to find every <span> element that is contained inside
# a <div> element, you could do this:
spans_inside_divs = [span
                     for div in soup("div")
                     for span in div("span")]


# Example: Keeping Tabs on Congress
url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a["href"]
            for a in soup("a")
            if a.has_attr("href")]

# Must start with http:// or https:/
# Must end with .house.gov or .house.gov/
regex = r"^https?://.*\.house\.gov/?$"
good_urls = [url for url in all_urls if re.match(regex, url)]
good_urls = list(set(good_urls))

press_releases: Dict[str, Set[str]] = {}

for house_url in good_urls:
    html = requests.get(house_url).text
    soup = BeautifulSoup(html, "html5lib")
    pr_links = {a["href"]
               for a in soup("a")
               if "press_releases" in a.text.lower()}
    print(f"{house_url}: {pr_links}")
    press_releases[house_url] = pr_links

def paragraph_mentions(text: str, keyword: str) -> bool:
    """
    Returns True if a <p> inside the text mentions {keyword}
    """

    soup = BeautifulSoup(text, "html5lib")
    paragraphs = [p.get_text() for p in soup("p")]

    return any(keyword.lower() in paragraph.lower()
               for paragraph in paragraphs)

for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"
        text = requests.get(url).text

        if paragraph_mentions(text, "data"):
            print(f"{house_url}")
            break