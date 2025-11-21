# production_agent/agent.py

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyC0oScY6NDn3vL0NTFscEL4b3QYmO5Qh3I")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

FIN_PROMPT = """
You are FinSight. Output ONLY JSON.

OUT OF CONTEXT RULE:
If the news does NOT contain these words:
- reliance
- ril
- reliance jio
- jio
- reliance retail
- reliance industries

Return EXACTLY this:

{
  "news_verdict": "out_of_context",
  "sentiment": "neutral",
  "sentiment_score": 0.0,
  "market_stance": { "label": "neutral", "score": 0.0 },
  "reasons": ["The news is not about Reliance."]
}

WHEN NEWS *IS* ABOUT RELIANCE:
Return JSON with exactly:

{
  "news_verdict": "real or fake",
  "sentiment": "positive or negative or neutral",
  "sentiment_score": 0-1 float,
  "market_stance": {
    "label": "bullish or bearish or neutral",
    "score": 0-1 float
  },
  "reasons": ["short reason 1", "short reason 2"]
}

Rules:
- No markdown.
- No text outside JSON.
- No extra fields.
"""

def run_finsight(news: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{FIN_PROMPT}\nNews: {news}",
        # 👇 IMPORTANT: in google-genai this is `config`, not `generation_config`
        config={
            "temperature": 0.1,
            "max_output_tokens": 256,
            "response_mime_type": "application/json",
        },
    )
    # For google-genai, the JSON string is in response.text
    return response.text
