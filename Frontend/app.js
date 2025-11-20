const el = (id)=>document.getElementById(id)
const newsText = el('newsText')
const analyzeBtn = el('analyzeBtn')
const status = el('status')
const result = el('result')

function setStatus(text){ status.textContent = text }

function renderResponse(json){
  result.classList.remove('hidden')
  result.innerHTML = ''

  const stockCard = document.createElement('div'); stockCard.className='card'
  stockCard.innerHTML = `<div class="row"><div>
    <div class="label">Stock</div>
    <div class="value">${json.stock || '—'}</div>
  </div><div style="margin-left:auto"><span class="badge">${json.market_stance?.label || '—'}</span></div></div>`
  result.appendChild(stockCard)

  const credCard = document.createElement('div'); credCard.className='card'
  const cscore = typeof json.credibility?.score === 'number' ? json.credibility.score : 0
  const pcolor = cscore>0.66? '#16a34a' : cscore>0.4? '#f59e0b' : '#ef4444'
  credCard.innerHTML = `<div class="label">News credibility</div>
  <div class="row" style="align-items:center;margin-top:6px">
    <div class="value">${json.credibility?.label || 'Unclear'}</div>
    <div style="flex:1;margin-left:12px">
      <div class="progress"><i style="width:${Math.round(cscore*100)}%;background:${pcolor}"></i></div>
    </div>
    <div style="width:48px;text-align:right;color:var(--muted)">${Math.round(cscore*100)}%</div>
  </div>
  <div class="cred-reason">${json.credibility?.reason || ''}</div>`
  result.appendChild(credCard)

  const sentCard = document.createElement('div'); sentCard.className='card'
  const sLabel = (json.sentiment?.label||'Neutral')
  const scls = sLabel.toLowerCase()==='positive'? 'good' : sLabel.toLowerCase()==='negative'? 'bad' : 'neutral'
  const sentimentEmoji = sLabel.toLowerCase()==='positive'? '😊' : sLabel.toLowerCase()==='negative'? '⚠️' : '😐'
  sentCard.innerHTML = `<div class="label">Sentiment</div>
  <div class="row" style="margin-top:6px">
    <div class="value ${'sentiment '+scls}">${sentimentEmoji} ${sLabel}</div>
    <div style="margin-left:auto;color:var(--muted)">${Math.round((json.sentiment?.score||0)*100)}%</div>
  </div>
  <div class="cred-reason">${json.sentiment?.reason || ''}</div>`
  result.appendChild(sentCard)

  const stanceCard = document.createElement('div'); stanceCard.className='card'
  stanceCard.innerHTML = `<div class="label">Market stance (educational)</div>
  <div style="margin-top:6px" class="value">${json.market_stance?.label || 'Sideways'}</div>
  <div class="cred-reason">${json.market_stance?.reason || ''}</div>
  <div class="disclaimer">${json.disclaimer || ''}</div>`
  result.appendChild(stanceCard)

  const raw = document.createElement('div'); raw.className='raw-json'
  raw.textContent = JSON.stringify(json, null, 2)
  result.appendChild(raw)
}

async function analyze(){
  const text = newsText.value.trim()
  if(!text){ setStatus('Please paste some news text or enable mock.'); return }
  analyzeBtn.disabled = true
  setStatus('Analyzing...')
  result.classList.add('hidden')

  try{
    const resp = await fetch('/analyze-news', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    })
    if(!resp.ok) throw new Error(`Server returned ${resp.status}`)
    const data = await resp.json()
    renderResponse(data)
    setStatus('Analysis complete')
  }catch(err){
    setStatus('Error: '+err.message)
    result.classList.remove('hidden')
    result.innerHTML = `<div class="card"><div class="label">Error</div><div class="value">${err.message}</div>
      <div class="cred-reason">If you intended to call a backend make sure `/analyze-news` is reachable and allows CORS.</div></div>`
  }finally{
    analyzeBtn.disabled = false
  }
}

analyzeBtn.addEventListener('click', analyze)
newsText.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && (e.metaKey||e.ctrlKey)) analyze() })
