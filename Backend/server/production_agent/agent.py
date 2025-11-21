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
API_BASE ="https://ollama-gemma3-270m-gpu-1003577856314.europe-west1.run.app"
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
about Reliance Industries (RELIANCE.NS) in simple, educational language.

You receive context like:
- Latest price and recent movement
- Recent sentiment trend from news
- One news headline and short description

Your tasks:
- Restate the news simply (whether it is fake or real in one word) 
- Explain sentiment (positive/negative/neutral in one word) 
-Sentiement score
- Explain a likely market stance (bullish/bearish/neutral from 0 to 1)
- Give 1-3 brief 
- Do NOT give financial advice
- End with: "This analysis is for educational purposes only, not financial advice."

VERY IMPORTANT RULES:
- Do NOT give financial advice.
  - Never say "you should buy", "sell", or "hold".
- Do NOT predict future prices or guaranteed outcomes.
- If information is unclear, say that your confidence is low.

Tone:
- Beginner-friendly, calm, and clear.
- Use short paragraphs and bullet points where helpful.

Always end with this line exactly:
"This analysis is for educational purposes only, not financial advice."
""",
    tools=[],   # no tools for now
)

# ADK expects a root agent symbol
# This assignment is what allows other files to import 'root_agent'
root_agent = production_agent