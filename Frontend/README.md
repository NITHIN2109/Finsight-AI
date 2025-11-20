# Finsight.AI — News Analyzer (Frontend)

Simple static frontend to paste news text and call a backend `/analyze-news` API. The UI renders the returned JSON (credibility, sentiment, market stance, disclaimer) in an easy-to-read format.

How it works
- Paste news text into the textarea and click `Analyze`.
- By default the page will POST `{"text": "..."}` to `/analyze-news` on the same origin.
- Response should be JSON as in the project spec; the UI will render it.

Local testing
1. Open `index.html` directly in a browser for basic UI; use the "Use mock response" checkbox to test without a backend.
2. To serve locally (recommended) run a simple HTTP server from this folder. In PowerShell:

```powershell
# Windows PowerShell
cd 'd:\Finsight - AI- UI'
python -m http.server 8000; # serves at http://localhost:8000
```

3. When serving from a local server, disable the mock checkbox to call your backend. Ensure your backend endpoint `POST /analyze-news` is reachable and allows CORS if it's on a different origin.

Notes for backend integration
- The frontend POSTs JSON: `{ "text": "..." }`.
- The expected response shape is:

```json
{
  "stock": "RELIANCE.NS",
  "credibility": { "label": "Likely real", "score": 0.78, "reason": "..." },
  "sentiment": { "label": "Positive", "score": 0.86, "reason": "..." },
  "market_stance": { "label": "Bullish", "reason": "..." },
  "disclaimer": "..."
}
```

- For production, ensure your backend is secured and avoid returning overly deterministic or legally actionable statements. The UI intentionally uses educational wording ("Market stance (educational)").

Next steps I can take
- Wire this to your existing backend and test with real responses.
- Improve visuals or add animations.
- Add simple request logging and retry/backoff.
