# server/main.py

import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from production_agent.agent import root_agent

# =========================
# Env & paths
# =========================

load_dotenv()

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

STOCK_FILE = os.path.join(DATA_DIR, "reliance_stock_history.csv")
NEWS_SENT_FILE = os.path.join(DATA_DIR, "reliance_news_sentiment.csv")

# =========================
# ADK Agent setup (Runner inside main.py)
# =========================

_session_service = InMemorySessionService()
_runner = Runner(
    agent=root_agent,
    app_name="finsight_app",
    session_service=_session_service,
)

_USER_ID = "finsight_user"
_session = _runner.session_service.create_session(
    app_name="finsight_app",
    user_id=_USER_ID,
)
_SESSION_ID = _session.id


def run_finsight_agent(prompt: str) -> str:
    """
    Call the ADK agent with the given prompt and
    return the final text response.
    """
    last_text = ""

    for event in _runner.run(
        user_id=_USER_ID,
        session_id=_SESSION_ID,
        new_message=prompt,
    ):
        if getattr(event, "content", None) and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    last_text = part.text

    if not last_text:
        raise RuntimeError("FinSight agent returned no text")

    return last_text


# =========================
# FastAPI setup
# =========================

app = FastAPI(
    title="FinSight AI Backend",
    description="Reads Reliance dataset and uses ADK FinSight agent to analyze news.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Pydantic models
# =========================

class NewsRequest(BaseModel):
    headline: str
    description: str | None = ""


class StockData(BaseModel):
    last_close: float | None = None
    prev_close: float | None = None
    daily_change_percent: float | None = None


class HistoricalSentiment(BaseModel):
    last_7_days: Dict[str, int] = {}
    overall_trend: str = "Unknown"


class AnalyzeNewsResponse(BaseModel):
    stock: str
    explanation: str
    stock_data: StockData
    historical_sentiment: HistoricalSentiment
    disclaimer: str


# =========================
# Dataset helpers
# =========================

def load_stock_data() -> StockData:
    if not os.path.exists(STOCK_FILE):
        return StockData()

    df = pd.read_csv(STOCK_FILE)
    if df.empty:
        return StockData()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.sort_values("Date")

    last = df.iloc[-1]
    last_close = float(last["Close"])

    prev_close = None
    change_pct = None
    if len(df) > 1:
        prev = df.iloc[-2]
        prev_close = float(prev["Close"])
        if prev_close != 0:
            change_pct = ((last_close - prev_close) / prev_close) * 100.0

    return StockData(
        last_close=last_close,
        prev_close=prev_close,
        daily_change_percent=change_pct,
    )


def load_historical_sentiment(days: int = 7) -> HistoricalSentiment:
    if not os.path.exists(NEWS_SENT_FILE):
        return HistoricalSentiment()

    df = pd.read_csv(NEWS_SENT_FILE)
    if df.empty or "published_at" not in df.columns:
        return HistoricalSentiment()

    df["published_at"] = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = df[df["published_at"] >= cutoff]

    if recent.empty:
        return HistoricalSentiment(overall_trend="No recent data")

    counts = recent["sentiment"].value_counts().to_dict()

    if not counts:
        trend = "No recent data"
    else:
        top = max(counts, key=counts.get).upper()
        if "POS" in top:
            trend = "Mostly positive"
        elif "NEG" in top:
            trend = "Mostly negative"
        else:
            trend = "Mostly neutral"

    return HistoricalSentiment(last_7_days=counts, overall_trend=trend)


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_dir": DATA_DIR,
        "stock_file_exists": os.path.exists(STOCK_FILE),
        "news_sentiment_file_exists": os.path.exists(NEWS_SENT_FILE),
    }


@app.post("/analyze-news", response_model=AnalyzeNewsResponse)
def analyze_news(payload: NewsRequest):
    headline = (payload.headline or "").strip()
    description = (payload.description or "").strip()

    if not headline:
        raise HTTPException(status_code=400, detail="headline is required")

    stock = load_stock_data()
    hist = load_historical_sentiment(days=7)

    context_prompt = f"""
You are FinSight, analyzing a Reliance news item.

Context:
- Last close price: {stock.last_close}
- Previous close price: {stock.prev_close}
- Daily change (%): {stock.daily_change_percent}
- Recent 7-day sentiment counts: {hist.last_7_days}
- Overall sentiment trend: {hist.overall_trend}

News:
Headline: {headline}
Description: {description}

Please follow your FinSight instructions:
- Restate the news simply
- Explain sentiment (positive/negative/neutral)
- Explain a likely market stance (bullish/bearish/neutral)
- Do NOT give financial advice
- End with: "This analysis is for educational purposes only, not financial advice."
"""

    try:
        explanation = run_finsight_agent(context_prompt)
    except Exception as e:
        print("Error running FinSight agent:", e)
        raise HTTPException(status_code=500, detail="Agent analysis failed")

    return AnalyzeNewsResponse(
        stock="RELIANCE.NS",
        explanation=explanation,
        stock_data=stock,
        historical_sentiment=hist,
        disclaimer="This analysis is for educational purposes only, not financial advice.",
    )
