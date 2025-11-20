import os
import json
import datetime
from datetime import date, timedelta
import warnings
import http.client, urllib.parse
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================

# Stock & news config (only Reliance for now)
COMPANY_SYMBOL = os.getenv("COMPANY_SYMBOL", "RELIANCE.NS")
SEARCH_QUERY = os.getenv("SEARCH_QUERY", "reliance")

# Mediastack API key (set as env var locally; later from Secret Manager in Cloud Run)
MEDIASTACK_API_TOKEN = os.getenv("MEDIASTACK_API_TOKEN")

# Dates
TODAY = str(date.today())
YESTERDAY = str(date.today() - timedelta(days=1))

# File paths (same structure as your notebook)
DATA_DIR = os.getenv("DATA_DIR", "./data")  # you can change this
os.makedirs(DATA_DIR, exist_ok=True)

STOCK_HISTORY_FILE = os.path.join(DATA_DIR, "reliance_stock_history.csv")
NEWS_FILE = os.path.join(DATA_DIR, "reliance_news.json")
NEWS_SENTIMENT_FILE = os.path.join(DATA_DIR, "reliance_news_sentiment.csv")

# ===================== FINBERT SETUP =====================

print("Loading FinBERT model... (first time will be slow)")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("FinBERT loaded.")

# =========================================================
# 1. STOCK HISTORY FUNCTIONS
# =========================================================

def create_stock_history_dataset():
    """Create initial stock history file with last 1 day of data."""
    ticker_object = yf.Ticker(COMPANY_SYMBOL)
    df = ticker_object.history(period="1d").reset_index()
    return df


def update_stock_history_dataset():
    """Update existing stock history CSV with today's data (or overwrite last row)."""
    ticker_object = yf.Ticker(COMPANY_SYMBOL)

    # load existing
    history = pd.read_csv(STOCK_HISTORY_FILE)

    # Normalize existing dates
    history["Date"] = (
    history["Date"]
        .astype(str)
        .str.replace(r"\s*\+.*$", "", regex=True)  )


    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")

    # get today's data
    today_data = ticker_object.history(period="1d").reset_index()

    if today_data.empty:
        print("No new stock data returned from yfinance.")
        return history

    # Normalize today's date
    today_data["Date"] = (
    today_data["Date"]
        .astype(str)
        .str.replace(r"\+\d{2}:\d{2}$", "", regex=True))

    today_data["Date"] = pd.to_datetime(today_data["Date"])

    today_date_str = str(today_data.loc[0, "Date"]).split()[0]
    last_date_str = history["Date"].dt.strftime("%Y-%m-%d").iloc[-1]

    if today_date_str == last_date_str:
        print("Today's stock data already exists, updating last row.")
        history.iloc[-1, :] = today_data.iloc[-1].tolist()
    else:
        print("Appending new stock row for", today_date_str)
        history.loc[len(history)] = today_data.iloc[-1].tolist()

    return history

def process_stock_history():
    if not os.path.exists(STOCK_HISTORY_FILE):
        print("Stock history file not found, creating new one...")
        df = create_stock_history_dataset()
    else:
        print("Updating existing stock history file...")
        df = update_stock_history_dataset()

    df.to_csv(STOCK_HISTORY_FILE, index=False)
    print(f"Saved stock history to {STOCK_HISTORY_FILE}")


# =========================================================
# 2. NEWS FUNCTIONS
# =========================================================

def fetch_latest_news():
    """Fetch yesterday's news from Mediastack."""
    if not MEDIASTACK_API_TOKEN:
        raise RuntimeError(
            "MEDIASTACK_API_TOKEN env var is not set. "
            "Set it before running this script."
        )

    conn = http.client.HTTPConnection("api.mediastack.com")
    params = urllib.parse.urlencode(
        {
            "keywords": SEARCH_QUERY,
            "access_key": MEDIASTACK_API_TOKEN,
            "sort": "published_desc",
            "limit": 10,
            "languages": "en",
            "country": "in",
            "date": YESTERDAY,
        }
    )

    conn.request("GET", f"/v1/news?{params}")
    res = conn.getresponse().read()
    data = json.loads(res.decode("utf-8"))
    articles = data.get("data", [])
    print(f"Fetched {len(articles)} articles from Mediastack.")
    return articles


def create_news_dataset():
    """Create JSON file with latest articles only."""
    articles = fetch_latest_news()
    news_obj = {"articles": articles}
    return news_obj, articles


def update_news_dataset():
    """
    Update existing news JSON:
    - If yesterday's news already exists, do nothing.
    - Else fetch latest & append.
    """
    with open(NEWS_FILE, "r", encoding="utf-8") as f:
        news_obj = json.load(f)

    existing_articles = news_obj.get("articles", [])
    news_inserted = any(
        art.get("published_at", "").split("T")[0] == YESTERDAY
        for art in existing_articles
    )

    current_articles = None
    if news_inserted:
        print("Yesterday's news already exists in JSON. Not fetching new news.")
    else:
        print("Fetching new news to append to JSON...")
        current_articles = fetch_latest_news()
        existing_articles += current_articles
        news_obj["articles"] = existing_articles

    return news_obj, current_articles


def process_news():
    """Create or update reliance_news.json and return today's new articles."""
    if not os.path.exists(NEWS_FILE):
        print("News JSON not found, creating new one...")
        news_obj, current_articles = create_news_dataset()
    else:
        print("Updating existing news JSON...")
        news_obj, current_articles = update_news_dataset()

    with open(NEWS_FILE, "w") as f:
        json.dump(news_obj, f, indent=2)

    print(f"Saved news JSON to {NEWS_FILE}")
    return current_articles  # may be None if no new articles


# =========================================================
# 3. NEWS SENTIMENT FUNCTIONS
# =========================================================

def process_news_sentiment(current_articles):
    """
    Create or update reliance_news_sentiment.csv using FinBERT
    for only the new articles fetched today.
    """
    # Load existing sentiment CSV or create empty DF
    if os.path.exists(NEWS_SENTIMENT_FILE):
        sentiments_df = pd.read_csv(NEWS_SENTIMENT_FILE)
    else:
        sentiments_df = pd.DataFrame(
            columns=[
                "published_at",
                "title",
                "description",
                "url",
                "sentiment",
                "sentiment_score",
            ]
        )

    if not current_articles:
        print("No new articles to score for sentiment today.")
        sentiments_df.to_csv(NEWS_SENTIMENT_FILE, index=False)
        print(f"Sentiment file saved (unchanged) to {NEWS_SENTIMENT_FILE}")
        return

    print(f"Running FinBERT on {len(current_articles)} new articles...")
    texts = []
    for art in current_articles:
        title = art.get("title") or ""
        desc = art.get("description") or ""
        texts.append(f"{title} {desc}".strip())

    predictions = classifier(texts)
    for art, pred in zip(current_articles, predictions):
        sentiments_df.loc[len(sentiments_df)] = {
            "published_at": art.get("published_at"),
            "title": art.get("title"),
            "description": art.get("description"),
            "url": art.get("url"),
            "sentiment": pred["label"],
            "sentiment_score": float(pred["score"]),
        }

    sentiments_df.to_csv(NEWS_SENTIMENT_FILE, index=False)
    print(f"Updated sentiment CSV saved to {NEWS_SENTIMENT_FILE}")


# =========================================================
# MAIN ENTRYPOINT
# =========================================================

def run_daily_update():
    print("=== FinSight Daily Update ===")
    print("Date:", TODAY)

    # 1. Update stock history
    process_stock_history()

    # 2. Update news JSON and get today's new articles
    current_articles = process_news()

    # 3. Update news sentiment CSV using FinBERT
    process_news_sentiment(current_articles)

    print("=== Daily update completed. ===")


if __name__ == "__main__":
    run_daily_update()
