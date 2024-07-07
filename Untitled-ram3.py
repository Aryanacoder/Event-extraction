import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress only the single InsecureRequestWarning from urllib3
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Function to clean and parse tweet date
def parse_tweet_date(date_str):
    return datetime.strptime(date_str, '%d %b %Y %H:%M')

# Define the search query for natural disasters
query = "war"
nitter_instance = "https://nitter.net"  # You can choose a different instance if needed

# Get the current time and the time 6 hours ago
now = datetime.utcnow()
since_time = now - timedelta(hours=60)

# Function to fetch tweets from Nitter
def get_tweets(query):
    url = f"{nitter_instance}/search?f=tweets&q={query}"
    response = requests.get(url, verify=False)  # Disable SSL verification
    
    # Debug: Print the response status and content
    print(f"Request URL: {url}")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content Length: {len(response.content)}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Debug: Print the HTML content to check if it contains tweets
    print(soup.prettify()[:1000])  # Print the first 1000 characters of the HTML content
    
    tweets = soup.find_all('div', class_='timeline-item')
    
    # Debug: Print the number of tweets found
    print(f"Number of tweets found: {len(tweets)}")
    
    tweet_data = []
    
    for tweet in tweets:
        text = tweet.find('div', class_='tweet-content').text.strip()
        date_str = tweet.find('span', class_='tweet-date').text.strip()
        date = parse_tweet_date(date_str)
        
        if date >= since_time:
            twitter_link = nitter_instance + tweet.find('a', class_='tweet-link')['href']
            location = tweet.find('span', class_='tweet-location').text.strip() if tweet.find('span', 'tweet-location') else 'N/A'
            likes = int(tweet.find('span', class_='tweet-likes').text.strip()) if tweet.find('span', 'tweet-likes') else 0
            comments = int(tweet.find('span', class_='tweet-comments').text.strip()) if tweet.find('span', 'tweet-comments') else 0
            
            tweet_info = {
                'twitter_link': twitter_link,
                'text': text,
                'date': date,
                'location': location,
                'likes': likes,
                'comments': comments
            }
            tweet_data.append(tweet_info)
    
    return pd.DataFrame(tweet_data)

# Fetch tweets
data = get_tweets(query)

# Display the data
print(data)

# Save the data to a CSV file
data.to_csv('natural_disasters_tweets.csv', index=False)
