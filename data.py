import requests
import json
import datetime
import time

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
# Replace this with your GNews API key
GNEWS_API_KEY = "YOUR_API_KEY"
QUERY = "Apple"
LANGUAGE = "en"
MAX_ARTICLES_PER_REQUEST = 25

# Defining date range (last year)
END_DATE = datetime.date.today()
START_DATE = END_DATE - datetime.timedelta(days=365)

OUTPUT_FILENAME = "apple_news.json"


# ----------------------------------------------------
# FUNCTION FETCHING DATA FOR GIVEN DAY
# ----------------------------------------------------
def get_gnews_for_day(date_str, query, api_key, language, max_articles):
    date_start_iso = f"{date_str}T00:00:00Z"
    date_end_iso = f"{date_str}T23:59:59Z"

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": language,
        "from": date_start_iso,
        "to": date_end_iso,
        "max": max_articles,
        "token": api_key,
        "country": ""
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        print(f"  -> Day {date_str}: Found {len(articles)} articles")
        return articles

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {date_str}: {e}")
        if response.status_code == 429:
            print("Reached API request limit")
            raise
        return []
    except Exception as e:
        print(f"Error in {date_str}: {e}")
        return []


# ----------------------------------------------------
# MAIN LOOP LOADING DATA
# ----------------------------------------------------

# List for storing all articles
all_articles = []

# Creating a list of dates for iteration
date_range = [START_DATE + datetime.timedelta(days=i) for i in range((END_DATE - START_DATE).days)]

print(
    f"Starting fetching data for query: '{QUERY}' in date range {START_DATE} to {END_DATE} ({len(date_range)} days)...")

for date in date_range:
    date_str = date.strftime('%Y-%m-%d')

    # Loading data from given day
    articles = get_gnews_for_day(date_str, QUERY, GNEWS_API_KEY, LANGUAGE, MAX_ARTICLES_PER_REQUEST)

    # Adding date to each article
    for article in articles:
        all_articles.append(article)

    # Adding 1 second delay for safety (to avoid crossing API request limit)
    time.sleep(1)

print(f"\nLoading ended. Total number of articles: {len(all_articles)}")

# ----------------------------------------------------
# SAVING IN JSON FILE
# ----------------------------------------------------
try:
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=4)
    print(f"File '{OUTPUT_FILENAME}' saved successfully")
except Exception as e:
    print(f"Error: {e}")
