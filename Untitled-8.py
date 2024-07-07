import torch
import spacy
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

# Load pre-trained BERT tokenizer and model for NER
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

# Create a pipeline for NER
nlp_ner = pipeline('ner', model=model, tokenizer=tokenizer)
def extract_ner_entities(text, nlp):
    ner_results = nlp(text)
    entities = {}
    for entity in ner_results:
        entity_type = entity['entity'].split('_')[1]  # Get the entity type
        entity_text = entity['word']
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(entity_text)
    return entities

def extract_impact_variables(article_text):
    # Use BERT for NER to extract relevant entities
    entities = extract_ner_entities(article_text, nlp_ner)
    
    # Implement logic to parse the entities and extract the following variables:
    # deaths, injured, affected, homeless
    # For example, if 'NUMBER' entity is followed by 'FATALITIES', 'INJURIES', etc.
    
    deaths = sum([int(num) for num in entities.get('NUMBER', []) if 'death' in article_text.lower()])
    injured = sum([int(num) for num in entities.get('NUMBER', []) if 'injur' in article_text.lower()])
    affected = sum([int(num) for num in entities.get('NUMBER', []) if 'affect' in article_text.lower()])
    homeless = sum([int(num) for num in entities.get('NUMBER', []) if 'homeless' in article_text.lower()])
    
    return deaths, injured, affected, homeless
import math
import feedparser
import requests
from bs4 import BeautifulSoup
from collections import Counter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime, timedelta
import time
import logging
import backoff

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define search queries for disaster types
disaster_queries = [
    "earthquake",
    "flood",
    "hurricane",
    "war",
    "explosion",
    "fire",
    "tornado",
    "tsunami"
]

# Load the spaCy model
nlp_spacy = spacy.load('en_core_web_sm')

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
    doc = nlp_spacy(text)
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

# Continuous severity classification function based on human impact variables
def continuous_severity_classification(deaths, injured, affected, homeless, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
    severity_score = (alpha * math.log1p(deaths) +
                      beta * math.log1p(injured) +
                      gamma * math.log1p(affected) +
                      delta * math.log1p(homeless))
    return severity_score

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

# Extract locations and calculate severity score for each article
all_locations = []
severity_scores = []
for article in all_articles:
    locations = extract_locations(article)
    all_locations.extend(locations)
    
    # Extract impact variables using BERT
    deaths, injured, affected, homeless = extract_impact_variables(article)
    
    severity_score = continuous_severity_classification(deaths, injured, affected, homeless)
    severity_scores.append((severity_score, locations))

# Determine the most relevant location based on severity scores
most_relevant_location = None
if severity_scores:
    # Sort by severity score
    severity_scores.sort(reverse=True, key=lambda x: x[0])
    most_relevant_location = severity_scores[0][1][0]  # Get the location with the highest severity score

print(f"\nThe most relevant location based on severity is: {most_relevant_location}")

# Geocode the most relevant location
coordinates = location_to_coordinates(most_relevant_location)
print(f"The coordinates of {most_relevant_location} are: {coordinates}")

# Placeholder function to extract impact variables from articles
def extract_impact_variables(article_text):
    # Implement your own logic to extract the following variables from the article text:
    # deaths, injured, affected, homeless
    # This is a placeholder example returning dummy values
    deaths = 0
    injured = 0
    affected = 0
    homeless = 0
    return deaths, injured, affected, homeless
