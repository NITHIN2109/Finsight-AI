import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from dotenv import load_dotenv
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import litellm
from production_agent.agent import GEMMA_MODEL, API_BASE, production_agent

# =========================
# Env & paths
# =========================

load_dotenv()

# Default to ../data relative to this file if DATA_DIR not set
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)
print("Using DATA_DIR:", DATA_DIR)

STOCK_FILE = os.path.join(DATA_DIR, "reliance_stock_history.csv")
NEWS_SENT_FILE = os.path.join(DATA_DIR, "reliance_news_sentiment.csv")

# =========================
# LLM / Agent config
# =========================

FIN_SIGHT_SYSTEM_PROMPT = production_agent.instruction

print(f"[main.py] Using model: ollama_chat/{GEMMA_MODEL}")
print(f"[main.py] Using API_BASE: {API_BASE}")


def run_finsight_agent(prompt: str) -> str:
    """
    Call the Gemma model via LiteLLM + Ollama.
    Completely stateless: every call is a fresh completion.
    """
    try:
        response = litellm.completion(
            model=f"ollama_chat/{GEMMA_MODEL}",
            messages=[
                {"role": "system", "content": FIN_SIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            api_base=API_BASE,
        )
    except Exception as e:
        print("[run_finsight_agent] Error calling LiteLLM/Ollama:", e)
        raise

    try:
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("[run_finsight_agent] Unexpected response format:", response)
        raise RuntimeError("Model returned an unexpected response format") from e


# =========================
# FastAPI setup
# =========================

app = FastAPI(
    title="FinSight AI Backend",
    description="Reads Reliance dataset and uses Gemma (via Ollama) to analyze news.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic models
# =========================

class NewsRequest(BaseModel):
    # Frontend sends only one field: { "News": "..." } or { "news": "..." }
    News: str | None = None
    news: str | None = None


class StockData(BaseModel):
    last_close: float | None = None
    prev_close: float | None = None
    daily_change_percent: float | None = None


class HistoricalSentiment(BaseModel):
    last_7_days: Dict[str, int] = {}
    overall_trend: str = "Unknown"


class MarketStance(BaseModel):
    label: str              # "bullish" / "bearish" / "neutral"
    score: float | None = None  # 0–1


class FinSightLLMResponse(BaseModel):
    news_verdict: str           # "real" or "fake"
    sentiment: str              # "positive" / "negative" / "neutral"
    sentiment_score: float | None = None
    market_stance: MarketStance
    reasons: List[str]


# =========================
# Helpers
# =========================

def load_stock_data() -> StockData:
    if not os.path.exists(STOCK_FILE):
        print(f"[load_stock_data] Stock file not found: {STOCK_FILE}")
        return StockData()

    df = pd.read_csv(STOCK_FILE)
    if df.empty:
        print("[load_stock_data] Stock file is empty")
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
        print(f"[load_historical_sentiment] Sentiment file not found: {NEWS_SENT_FILE}")
        return HistoricalSentiment()

    df = pd.read_csv(NEWS_SENT_FILE)
    if df.empty or "published_at" not in df.columns:
        print("[load_historical_sentiment] Empty sentiment file or missing published_at")
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


def extract_json_from_text(text: str) -> str:
    """
    Clean LLM output:
    - remove ```...``` fences if present
    - keep only substring from first '{' to last '}'
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove markdown code-fence lines
    lines = cleaned.splitlines()
    lines = [ln for ln in lines if not ln.strip().startswith("```")]
    cleaned = "\n".join(lines).strip()

    # Narrow down to the JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        cleaned = cleaned[start : end + 1]

    return cleaned.strip()


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


@app.post("/analyze-news", response_model=FinSightLLMResponse)
def analyze_news(payload: NewsRequest):
    # Get the text from either "News" or "news"
    raw_news = payload.News or payload.news or ""
    news = raw_news.strip()

    if not news:
        raise HTTPException(status_code=400, detail="News text is empty")

    # Context from datasets (used only in prompt)
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
{news}
"""

    try:
        llm_raw = run_finsight_agent(context_prompt)
        print("[analyze-news] Raw LLM output:", llm_raw)

        cleaned = extract_json_from_text(llm_raw)
        print("[analyze-news] Cleaned JSON string:", cleaned)

        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("[analyze-news] JSON parse error:", e)
        raise HTTPException(
            status_code=500,
            detail="Model did not return valid JSON",
        )
    except Exception as e:
        print("Error running FinSight agent:", e)
        raise HTTPException(status_code=500, detail="Agent analysis failed")

    return FinSightLLMResponse(**parsed)
