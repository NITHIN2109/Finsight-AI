# server/main.py

import os
import json
from typing import List, Optional, Dict
from difflib import SequenceMatcher

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from production_agent.agent import run_finsight  # Gemini wrapper

# ============================================================
# ENV & PATHS
# ============================================================

load_dotenv()

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)
print("Using DATA_DIR:", DATA_DIR)

RELIANCE_JSON = os.path.join(DATA_DIR, "reliance_news.json")

MEDIASTACK_API_KEY = os.getenv("MEDIASTACK_API_KEY", "")

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FinSight AI Backend",
    description="Reliance Fake-News Checker (Local DB + Mediastack + Gemini)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELS
# ============================================================

class NewsRequest(BaseModel):
    News: Optional[str] = None
    news: Optional[str] = None


class MarketStance(BaseModel):
    label: str
    score: Optional[float]


class FinSightLLMResponse(BaseModel):
    news_verdict: str
    sentiment: str
    sentiment_score: Optional[float]
    market_stance: MarketStance
    reasons: List[str]

# ============================================================
# LOCAL RELIANCE NEWS HELPERS
# ============================================================

def load_local_reliance_news() -> List[dict]:
    """Supports JSON formats: {"articles": [...]}, {"data": [...]}, or list."""
    if not os.path.exists(RELIANCE_JSON):
        return []
    try:
        with open(RELIANCE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if isinstance(data.get("articles"), list):
                return data["articles"]
            if isinstance(data.get("data"), list):
                return data["data"]

        if isinstance(data, list):
            return data

        return []
    except:
        return []


def find_local_article(query: str, threshold: float = 0.6) -> Optional[Dict]:
    """Return article dict if similarity high enough."""
    db = load_local_reliance_news()
    if not db:
        return None

    q = query.lower().strip()
    best_score = 0.0
    best_article = None

    for item in db:
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or item.get("headline") or "").lower().strip()
        if not title:
            continue

        score = SequenceMatcher(None, q, title).ratio()
        if score > best_score:
            best_score = score
            best_article = item

    if best_article and best_score >= threshold:
        return best_article

    return None

# ============================================================
# MEDIATSTACK CHECK
# ============================================================

def _shorten_query(query: str) -> str:
    words = query.lower().split()[:10]
    if "reliance" not in words:
        words.append("reliance")
    return " ".join(words)


def search_mediastack(query: str, threshold: float = 0.6) -> Optional[bool]:
    """Return True (match), False (no match), or None if API failed."""
    if not MEDIASTACK_API_KEY:
        return None

    url = "http://api.mediastack.com/v1/news"
    params = {
        "access_key": MEDIASTACK_API_KEY,
        "keywords": _shorten_query(query),
        "languages": "en",
        "limit": 10,
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()

        articles = data.get("data") or data.get("articles") or []
        if not articles:
            return False

        q = query.lower().strip()
        best = 0.0

        for art in articles:
            title = (art.get("title") or "").lower().strip()
            if not title:
                continue

            score = SequenceMatcher(None, q, title).ratio()
            best = max(best, score)

            if score >= threshold:
                return True

        return False

    except:
        return None  # API failed → let Gemini decide

# ============================================================
# ROUTES
# ============================================================

@app.post("/analyze-news", response_model=FinSightLLMResponse)
def analyze_news(payload: NewsRequest):

    news = (payload.News or payload.news or "").strip()
    if not news:
        raise HTTPException(400, "News text empty")

    # 1️⃣ Check LOCAL
    local_article = find_local_article(news)

    # 2️⃣ Check Mediastack only if not found locally
    mediastack_result = None
    if not local_article:
        mediastack_result = search_mediastack(news)

    # 3️⃣ Build Gemini input based on cases
    if local_article:
        # CASE A: FOUND IN LOCAL DB
        gemini_input = f"""
User news:
{news}

[VERIFIED_LOCAL_DB_RECORD]
{json.dumps(local_article, ensure_ascii=False)}

This article WAS FOUND in our verified local Reliance news database.
Treat the news as REAL.
Return your final JSON only.
"""
    elif mediastack_result is True:
        # CASE B: FOUND IN MEDIASTACK
        gemini_input = f"""
User news:
{news}

[MATCHED_ON_MEDIASTACK]
This news was found online on Mediastack.
Treat it as LIKELY REAL, but verify carefully.
Return only the final JSON.
"""
    elif mediastack_result is False:
        # CASE C: NOT found in local AND NOT found in Mediastack → send to Gemini
        gemini_input = f"""
User news:
{news}

[NOT_FOUND_ANYWHERE]
This news was NOT found in our local Reliance DB and NOT found on Mediastack.
There are chances this news is FAKE.
You decide REAL or FAKE based on the text.
Return ONLY the final JSON.
"""
    else:
        # CASE D: Mediastack API failed → fail-open, ask Gemini to judge
        gemini_input = f"""
User news:
{news}

[MAYBE_REAL]
Online verification could not be performed. Please analyze the text carefully.
in the absence of any verification, decide if the news is REAL or FAKE based on the content.
Return ONLY the final JSON.
"""

    # 4️⃣ Call Gemini
    try:
        raw = run_finsight(gemini_input)
        parsed = json.loads(raw)
        return FinSightLLMResponse(**parsed)
    except Exception as e:
        print("[Gemini Error]", e)
        raise HTTPException(500, "Gemini LLM failed")
