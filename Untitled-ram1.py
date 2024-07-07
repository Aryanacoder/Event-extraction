import requests
from bs4 import BeautifulSoup

def fetch_news(disaster_type):
    # Constructing Google News URL for the disaster type
    base_url = "https://news.google.com"
    search_url = f"{base_url}/search?q={disaster_type}+when:1d&hl=en-US&gl=US&ceid=US:en"

    # Fetching the page
    response = requests.get(search_url)
    if response.status_code != 200:
        print("Failed to retrieve the web page.")
        return []

    # Parsing the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []

    # Extracting news items
    for item in soup.find_all('div', attrs={"class": "xrnccd"}):
        title_element = item.find('h3', attrs={"class": "ipQwMb ekueJc RD0gLb"})
        source_element = item.find('a', attrs={"class": "wEwyrc AVN2gc uQIVzc Sksgp"})
        link_element = item.find('a', attrs={"class": "VDXfz"})

        if title_element and source_element and link_element:
            title = title_element.text
            source = source_element.text
            link = base_url + link_element['href'][1:]
            articles.append((title, source, link))
            

    return articles

# Example usage:
disaster_type = "earthquake"  # You can change this to 'flood', 'landslide', etc.
news_articles = fetch_news(disaster_type)
for title, source, link in news_articles:
    print(f"Title: {title}\nSource: {source}\nLink: {link}\n")

