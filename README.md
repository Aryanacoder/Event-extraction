

# Event Extraction Project from Google News

## Overview
This project is designed to extract and analyze news events related to natural disasters from Google News. It uses advanced data clustering techniques and geolocation tools to identify, categorize, and extract precise locations of events reported in real-time news articles. This project is particularly useful for researchers, data analysts, and organizations that need to monitor and respond to natural disasters efficiently.

## Features
- **Event Extraction**: Automatically scrapes news articles from Google News related to specified types of natural disasters such as earthquakes, cyclones, and tornadoes.
- **Data Clustering**: Utilizes clustering algorithms to categorize news based on similarity in content, ensuring that similar events are grouped for better analysis and reporting.
- **Location Extraction**: Employs the Geopy library to accurately determine the geographical coordinates of the locations mentioned in the news articles.
- **Exact Location Finder**: Integrates advanced geolocation techniques to refine the location data, providing precise coordinates for effective mapping and visualization.

## Technology Stack
- Python: Primary programming language used for the project.
- BeautifulSoup: Used for web scraping tasks.
- Scikit-Learn: Applied for clustering news articles.
- Geopy: Utilized for extracting and refining geographical data.
- Pandas: Used for data manipulation and cleaning.
- Requests: Library used for making HTTP requests to Google News.

## Project Structure
```
event-extraction/
│
├── src/                     # Source files for the project
│   ├── scraper.py           # Script to scrape news articles
│   ├── cluster.py           # Script for clustering articles
│   ├── location.py          # Script to extract and refine locations
│   └── main.py              # Main script to run the processes
│
├── data/                    # Data storage for scraped articles
│   ├── raw_data.csv         # Raw news data
│   └── processed_data.csv   # Data after processing and clustering
│
├── notebooks/               # Jupyter notebooks for analysis
│   └── analysis.ipynb       # Notebook for detailed data analysis
│
├── requirements.txt         # Python dependencies for the project
└── README.md                # Project documentation
```

## Setup
### Prerequisites
Ensure you have Python 3.8+ installed on your system. You will also need pip to install dependencies.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/event-extraction.git
   cd event-extraction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the event extraction process, follow these steps:

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

This will execute the scraping, clustering, and location extraction processes sequentially and store the output in the `data/` directory.

## Configuration
- **Modify Clustering Parameters**: Adjust the parameters in `cluster.py` to change the clustering behavior.
- **Customize Location Extraction**: Edit `location.py` to alter the geolocation extraction specifics.

## Contributing
Contributions to improve the project are welcome. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b new-feature`
3. Make your changes and commit them: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

This README file provides a detailed guide on my project, from an introduction to the technology stack, setup instructions, usage details, and contribution guidelines. Adjust the repository links and specific configuration details as necessary to match your project's setup.
