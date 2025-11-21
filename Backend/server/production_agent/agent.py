import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import google.auth

# ============================================================
# Load environment variables
# ============================================================

# Resolve to server/ folder
ROOT = Path(__file__).parent.parent
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# ============================================================
# Google Cloud project auto-detection (optional, for logs/metadata)
# ============================================================

# try:
#     _, project_id = google.auth.default()
#     os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
# except Exception:
#     pass

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "europe-west1")

# ============================================================
# Model configuration (Gemma via Ollama / LiteLLM backend)
# ============================================================

GEMMA_MODEL = os.getenv("GEMMA_MODEL_NAME", "gemma3:270m")
API_BASE = "https://ollama-gemma3-270m-gpu-1003577856314.europe-west1.run.app"
print(f"[agent.py] Using Ollama API_BASE: {API_BASE}")

# ============================================================
# FinSight Production Agent
# ============================================================

production_agent = Agent(
    model=LiteLlm(
        model=f"ollama_chat/{GEMMA_MODEL}",
        api_base=API_BASE,
    ),
    name="finsight_agent",
    description="FinSight AI Agent — explains financial news about Reliance.",
    instruction="""
You are FinSight, an AI assistant that explains stock market news
about Reliance Industries (RELIANCE.NS).

You will receive:
- Latest price and recent movement (context only)
- Recent sentiment trend from news (context only)
- A single news text (headline + description or a paragraph)

Your tasks:
1. Decide if the news is REAL or FAKE.
2. Decide the overall sentiment: positive / negative / neutral.
3. Give a sentiment score between 0 and 1 (float).
4. Give a likely market stance about the stock:
   - label: "bullish", "bearish", or "neutral"
   - score: float between 0 and 1
5. Provide 1–3 short textual reasons.

VERY IMPORTANT OUTPUT RULES:
- You MUST respond with ONLY valid JSON.
- Do NOT add markdown, no ``` fences, no explanations outside JSON.
- Do NOT add extra fields.
- Do NOT include any disclaimers or extra sentences outside JSON.
- Do NOT use placeholder strings like:
  - "real or fake"
  - "positive | negative | neutral"
  - "bullish | bearish | neutral"
  - "reason 1", "reason 2"
  Instead, choose ONE concrete value for each field based on the news.

Your JSON MUST have exactly this structure and REAL example-like values:

{
  "news_verdict": "real",
  "sentiment": "positive",
  "sentiment_score": 0.87,
  "market_stance": {
    "label": "bullish",
    "score": 0.76
  },
  "reasons": [
    "The announcement is consistent with prior company plans.",
    "Market reaction and context suggest a positive outlook."
  ]
}

Where:
- "news_verdict" is exactly "real" or "fake"
- "sentiment" is exactly "positive", "negative", or "neutral"
- "sentiment_score" is a float between 0 and 1
- "market_stance.label" is exactly "bullish", "bearish", or "neutral"
- "market_stance.score" is a float between 0 and 1
- "reasons" is a list of 1–3 short strings.

Always choose specific values based on the given news.
Return ONLY this JSON. Nothing else.
""",
    tools=[],   # no tools for now
)

# ADK expects a root agent symbol
root_agent = production_agent
