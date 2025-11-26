import nltk
import yfinance as yf
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import scipy.stats as stats

# ----------------------------------------------------
# INITIAL SETUP
# ----------------------------------------------------
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(e)

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
ticker = "AAPL"
start_date = "2024-11-10"
end_date = "2025-11-10"

# PARAMETERS FOR EXTREME SENTIMENT ANALYSIS
POSITIVE_THRESHOLD = 0.35
NEGATIVE_THRESHOLD = 0.10

# ----------------------------------------------------
# 1. MARKET DATA (Yahoo Finance)
# ----------------------------------------------------
print(f"Fetching market data for {ticker}...")
df_prices = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
df_returns = df_prices[['Close']].pct_change().dropna()
df_returns.columns = ['returns']

print(f"Number of daily returns observations: {len(df_returns)}")

# ----------------------------------------------------
# 2. NEWS DATA FROM JSON FILE
# ----------------------------------------------------
json_file_path = 'apple_news_filtered.json'
print(f"\nLoading headlines from JSON file for Apple")

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

df_raw = pd.DataFrame(data)

# Combining title and description
df_raw['combined_text'] = df_raw['title'].astype(str).fillna('') + ' ' + df_raw['description'].astype(str).fillna('')

# Extracting publishing date
df_raw['date'] = pd.to_datetime(df_raw['publishedAt']).dt.date

# Selecting only the useful columns
df_news = df_raw.rename(columns={'combined_text': 'text'})[['date', 'text']]

print(f"Loaded {len(df_raw)} headlines from JSON")

# ----------------------------------------------------
# 3. SENTIMENT ANALYSIS (NLTK VADER)
# ----------------------------------------------------
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    if pd.isna(text) or text is None:
        # Returning only 0.0 for missing data
        return 0.0

    # Returning the compound sentiment score
    return analyzer.polarity_scores(str(text))['compound']


# Applying the VADER sentiment function and storing the compound score directly.
df_news['sentiment_score'] = df_news['text'].apply(get_sentiment)

# Daily aggregation: mean compound sentiment
df_daily_sentiment = df_news.groupby('date')['sentiment_score'].mean().reset_index()
df_daily_sentiment.set_index('date', inplace=True)

# Changing column names
df_daily_sentiment.columns = ['sentiment_mean']

# Creating binary variables (Dummy Variables) based on sentiment
df_daily_sentiment['Positive_Dummy'] = (df_daily_sentiment['sentiment_mean'] > POSITIVE_THRESHOLD).astype(int)
df_daily_sentiment['Negative_Dummy'] = (df_daily_sentiment['sentiment_mean'] < NEGATIVE_THRESHOLD).astype(int)

# ----------------------------------------------------
# 4. MERGE DATASETS - WITH LAG
# ----------------------------------------------------
df_returns.index = pd.to_datetime(df_returns.index).date
df_daily_sentiment.index = pd.to_datetime(df_daily_sentiment.index).date

df_merged = pd.merge(df_returns, df_daily_sentiment, left_index=True, right_index=True, how='inner')
df_merged.index = pd.to_datetime(df_merged.index)

# LAGGED VARIABLES
df_merged['sentiment_lag1'] = df_merged['sentiment_mean'].shift(1)

df_merged = df_merged.dropna()

print(f"\nSize of merged dataset: {len(df_merged)} days")
print(df_merged.head())

# ----------------------------------------------------
# 5. CORRELATION ANALYSIS
# ----------------------------------------------------
# --- PEARSON CORRELATION (LINEAR) ---
corr_current_mean_p = df_merged['returns'].corr(df_merged['sentiment_mean'])
corr_lagged_mean_p = df_merged['returns'].corr(df_merged['sentiment_lag1'])

# --- SPEARMAN CORRELATION (RANK/MONOTONIC) ---
corr_current_mean_s = stats.spearmanr(df_merged['returns'], df_merged['sentiment_mean'])[0]
corr_lagged_mean_s = stats.spearmanr(df_merged['returns'], df_merged['sentiment_lag1'])[0]

# Displaying both Pearson and Spearman
print(f"\nPearson Correlation (Linear relationship):")
print(f"  - Compound Mean (t) vs Returns: {corr_current_mean_p:.3f}")
print(f"  - Compound Mean (t-1) vs Returns: {corr_lagged_mean_p:.3f}")

print(f"\nSpearman Correlation (Rank relationship):")
print(f"  - Compound Mean (t) vs Returns: {corr_current_mean_s:.3f}")
print(f"  - Compound Mean (t-1) vs Returns: {corr_lagged_mean_s:.3f}")

# Note: The original Pearson dummy calculations are kept below for context
corr_dummy_neg = df_merged['returns'].corr(df_merged['Negative_Dummy'])
corr_dummy_pos = df_merged['returns'].corr(df_merged['Positive_Dummy'])
print(f"\nPearson Correlation (Dummy variables):")
print(f"  - Dummy Negative vs Returns: {corr_dummy_neg:.3f}")
print(f"  - Dummy Positive vs Returns: {corr_dummy_pos:.3f}")

# ----------------------------------------------------
# 6. DESCRIPTIVE STATISTICS (Including Quartiles)
# ----------------------------------------------------
print("\nDescriptive Statistics for Daily Sentiment (sentiment_mean):")
sentiment_stats = df_merged['sentiment_mean'].describe(percentiles=[.25, .75])

# Displaying only key statistics
print("--------------------------------------------------")
print(f"Number of observations (Count): {sentiment_stats['count']:.0f}")
print(f"Mean: {sentiment_stats['mean']:.4f}")
print(f"Standard Deviation (Std): {sentiment_stats['std']:.4f}")
print(f"Minimum (Min): {sentiment_stats['min']:.4f}")
print(f"First Quartile (Q1 - 25%): {sentiment_stats['25%']:.4f}")
print(f"Median (50%): {sentiment_stats['50%']:.4f}")
print(f"Third Quartile (Q3 - 75%): {sentiment_stats['75%']:.4f}")
print(f"Maximum (Max): {sentiment_stats['max']:.4f}")
print("--------------------------------------------------")

# ----------------------------------------------------
# 7. REGRESSION ANALYSIS
# ----------------------------------------------------
y = df_merged['returns']

# Model 1: continuous sentiment (current day, t)
X1 = sm.add_constant(df_merged['sentiment_mean'])
model_current_mean = sm.OLS(y, X1).fit()
print("\n--- Model 1: Regression - Continuous Sentiment (t) ---")
print(model_current_mean.summary())

# Model 2: lagged continuous sentiment (t-1)
X2 = sm.add_constant(df_merged['sentiment_lag1'])
model_lagged_mean = sm.OLS(y, X2).fit()
print("\n--- Model 2: Regression - Continuous Sentiment (t-1) ---")
print(model_lagged_mean.summary())

# Model 3: current extreme sentiment (t)
X3 = sm.add_constant(df_merged[['Positive_Dummy', 'Negative_Dummy']])
model_current_dummy = sm.OLS(y, X3).fit()
print("\n--- Model 3: Regression - Extreme Sentiment (DUMMY t) ---")
print(model_current_dummy.summary())

# ----------------------------------------------------
# 8. VISUALIZATION
# ----------------------------------------------------

# Box plot for sentiment
plt.figure(figsize=(6, 8))
plt.boxplot(df_merged['sentiment_mean'].dropna(), vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(marker='o', color='gray', alpha=0.6))
plt.title(f"{ticker}: Distribution of Daily Sentiment (Compound Mean)")
plt.ylabel("Sentiment score (Compound)")
plt.xticks([1], ['Sentiment'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{ticker}_Daily_Sentiment_Box_Plot", dpi=300, bbox_inches='tight')

# Scatter plot for the current binary variable (Positive_Dummy)
plt.figure(figsize=(10, 5))
plt.scatter(df_merged['Positive_Dummy'], df_merged['returns'], color='green')
plt.title(f"{ticker}: Daily Returns vs. Positive binary variable (Dummy t)")
plt.xlabel(f"Positive binary variable (1 = Sentiment > {POSITIVE_THRESHOLD})")
plt.ylabel("Daily return")
plt.xticks([0, 1])
plt.grid(True)
plt.savefig(f"{ticker}_Positive_Binary_Scatter", dpi=300, bbox_inches='tight')

# Scatter plot showing daily returns in relation to mean sentiment
plt.figure(figsize=(10, 5))
plt.scatter(df_merged['sentiment_mean'], df_merged['returns'], color='purple')
plt.title(f"{ticker}: Daily Returns vs. Sentiment")
plt.xlabel("Sentiment Score")
plt.ylabel("Daily Return")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{ticker}_Daily_Sentiment_Scatter", dpi=300, bbox_inches='tight')

# Time series: daily returns vs mean sentiment
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df_merged.index, df_merged['returns'], color='blue', label='Daily returns')
ax2 = ax1.twinx()
ax2.plot(df_merged.index, df_merged['sentiment_mean'], color='red', label='Sentiment (Daily mean)')
ax1.set_ylabel('Returns', color='blue')
ax2.set_ylabel('Sentiment score', color='red')
plt.title(f"{ticker} Returns and sentiment in time")
fig.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{ticker}_Daily_Sentiment_Time_series", dpi=300, bbox_inches='tight')
