import feedparser
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime, timedelta
import time
import logging
import backoff
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define search queries for disaster types
disaster_queries = [
    "cyclone"
]

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, requests.exceptions.HTTPError), max_tries=5)
def fetch_article_content(url):
    try:
        response = requests.get(url, timeout=10)  # Set a timeout to avoid hanging
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.content
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - {url}")
        raise
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred: {conn_err} - {url}")
        raise
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred: {timeout_err} - {url}")
        raise
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error occurred: {req_err} - {url}")
        raise

def parse_article_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([para.text for para in paragraphs])
    return article_text

def extract_locations(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return locations

@backoff.on_exception(backoff.expo, GeocoderTimedOut, max_tries=5)
def location_to_coordinates(location):
    geolocator = Nominatim(user_agent="your_unique_user_agent")
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return (location_data.latitude, location_data.longitude)
        else:
            return "Location not found"
    except GeocoderTimedOut:
        return "Geocoding service timed out"
    except GeocoderServiceError as e:
        return f"Geocoding service error: {e}"

# Time range for the last 6 hours
current_time = datetime.utcnow()
time_6_hours_ago = current_time - timedelta(hours=6)

# Fetch and process articles for each disaster type
all_articles = []

for query in disaster_queries:
    rss_url = f"https://news.google.com/rss/search?q={query}"
    feed = feedparser.parse(rss_url)
    
    for entry in feed.entries:
        published_time = datetime.fromtimestamp(time.mktime(entry.published_parsed))
        if published_time >= time_6_hours_ago:
            url = entry.link
            try:
                html_content = fetch_article_content(url)
                if html_content:
                    article_text = parse_article_content(html_content)
                    all_articles.append(article_text)
            except Exception as e:
                logger.error(f"Failed to process article: {url} - {e}")

# Extract locations from all articles
all_locations = []
for article in all_articles:
    locations = extract_locations(article)
    all_locations.extend(locations)

# Create a single string with all locations for TF-IDF processing
location_texts = [" ".join(all_locations)]

# Apply TF-IDF to find the most relevant location
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(location_texts)
tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

# Determine the most relevant location based on TF-IDF scores
most_relevant_location = max(tfidf_scores, key=tfidf_scores.get)

print(f"\nThe most relevant location is: {most_relevant_location}")

# Geocode the most relevant location
coordinates = location_to_coordinates(most_relevant_location)
print(f"The coordinates of {most_relevant_location} are: {coordinates}")
