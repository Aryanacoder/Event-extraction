import feedparser
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter

# URL of the RSS feed
rss_url = "https://news.google.com/rss/search?q=war"

# Parse the RSS feed
feed = feedparser.parse(rss_url)

# Extract URLs from the RSS feed entries
article_urls = [entry.link for entry in feed.entries]

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def fetch_article_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch article: {url} - {e}")
        return None

def parse_article_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.text for para in paragraphs])
    return article_text

def extract_locations(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return locations

# Dictionary to store location counts
location_counter = Counter()

# Fetch, parse, and extract locations from all articles
for url in article_urls:
    html_content = fetch_article_content(url)
    if html_content:
        article_text = parse_article_content(html_content)
        locations = extract_locations(article_text)
        location_counter.update(locations)

# Check if any locations were found
if location_counter:
    # Get the most common location
    most_common_location, count = location_counter.most_common(1)[0]
    print(f"The most frequently mentioned location is: {most_common_location} (mentioned {count} times)")
else:
    print("No locations were found in the articles.")
