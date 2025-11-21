const el = (id) => document.getElementById(id);
const newsText = el("newsText");
const analyzeBtn = el("analyzeBtn");
const status = el("status");
const result = el("result");

const API_URL = "http://127.0.0.1:8000/analyze-news"; 
// Replace with your Cloud Run backend URL

function setStatus(text) {
  status.textContent = text;
}

function renderResponse(json) {
  result.classList.remove("hidden");
  result.innerHTML = "";

  // -------- NEWS VERDICT --------
  const verdictCard = document.createElement("div");
  verdictCard.className = "card";
  verdictCard.innerHTML = `
    <div class="label">News Credibility</div>
    <div class="value">${json.news_verdict || "—"}</div>
  `;
  result.appendChild(verdictCard);

  // -------- SENTIMENT --------
  const sentCard = document.createElement("div");
  sentCard.className = "card";

  const sentColor =
    json.sentiment === "positive"
      ? "good"
      : json.sentiment === "negative"
      ? "bad"
      : "neutral";

  sentCard.innerHTML = `
    <div class="label">Sentiment</div>
    <div class="row" style="margin-top:6px">
      <div class="value sentiment ${sentColor}">
        ${json.sentiment || "neutral"}
      </div>
      <div style="margin-left:auto;color:var(--muted)">
        ${Math.round((json.sentiment_score || 0) * 100)}%
      </div>
    </div>
  `;
  result.appendChild(sentCard);

  // -------- MARKET STANCE --------
  const stanceCard = document.createElement("div");
  stanceCard.className = "card";

  stanceCard.innerHTML = `
    <div class="label">Market Stance</div>
    <div class="value">${json.market_stance?.label || "neutral"}</div>
    <div class="cred-reason">Confidence: ${(json.market_stance?.score || 0).toFixed(2)}</div>
  `;
  result.appendChild(stanceCard);

  // -------- REASONS --------
  const reasonsCard = document.createElement("div");
  reasonsCard.className = "card";

  const reasonsHTML = (json.reasons || [])
    .map((r) => `<li>${r}</li>`)
    .join("");

  reasonsCard.innerHTML = `
    <div class="label">Reasons</div>
    <ul class="cred-reason" style="margin-top:8px">${reasonsHTML}</ul>
  `;
  result.appendChild(reasonsCard);

  // // -------- RAW JSON --------
  // const raw = document.createElement("div");
  // raw.className = "raw-json";
  // raw.textContent = JSON.stringify(json, null, 2);
  // result.appendChild(raw);
}

async function analyze() {
  const text = newsText.value.trim();
  if (!text) {
    setStatus("Please paste some news text.");
    return;
  }

  analyzeBtn.disabled = true;
  setStatus("Analyzing...");
  result.classList.add("hidden");

  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ news:text }),
    });

    if (!resp.ok) throw new Error(`Server error ${resp.status}`);

    const data = await resp.json();
    renderResponse(data);
    setStatus("Analysis complete.");
  } catch (err) {
    setStatus("Error: " + err.message);
    result.classList.remove("hidden");
    result.innerHTML = `
      <div class="card">
        <div class="label">Error</div>
        <div class="value">${err.message}</div>
        <div class="cred-reason">Check if the backend API is reachable.</div>
      </div>`;
  } finally {
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", analyze);
