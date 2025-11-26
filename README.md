# News Sentiment and Stock Returns Analysis (AAPL)
## Project Summary
This project analyzes the correlation between news sentiment regarding the Apple company (AAPL) and its daily stock returns using a three-step Python workflow.
| Script | Purpose | Key Action |
|--------|---------|------------|
| data.py | Data Acquisition | Fetches raw news headlines for "Apple" via the GNews API. |
| filter.py | Data Cleaning | Filters headlines to exclude non-company topics (e.g., fruit, cooking, celebrities) using specific inclusion/exclusion terms. |
| sentiment.py | Analysis & Output | 1. Calculates daily mean sentiment (VADER NLP). 2. Fetches AAPL returns (yfinance). 3. Performs correlation and OLS regression (current & lagged sentiment). 4. Generates plots.|

## Setup and Installation
### Prerequisites & Dependencies
Install dependencies:
```
pip install requests pandas yfinance nltk matplotlib statsmodels scipy
```
GNews API Key is required for data.py.
Configure API Key: Update the GNEWS_API_KEY in data.py.
## Execution 
### Steps
Run the scripts in sequential order:
* Fetch Data:  python data.py
* Filter Data:  python filter.py
* Analyze & Visualize:  python sentiment.py

### Output
The final script (sentiment.py) prints regression statistics to the console and saves four key plots (including time series and scatter plots) to the project directory.

You can read my paper here:
https://drive.google.com/file/d/14LvP3D8CLshd40PmyYgsGORMfIUFZvsC/view?usp=drive_link
