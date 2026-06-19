"""
app.py
Simple web interface for the fact-checking retrieval system.
Type a claim, get relevant Wikipedia passages with support scores.
"""

import json
import os
import re
import sys
from flask import Flask, render_template_string, request, jsonify

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from nli import NLISupportScorer, get_scorer
from pyserini.search.lucene import LuceneSearcher
from retrieval import get_candidates, select_top_sentences

# Prevent OpenMP/JVM conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_DIR = os.path.join(DATA_DIR, "bm25_index")
PASSAGES_JSONL = os.path.join(DATA_DIR, "passages", "passages.jsonl")

# Load models once at startup
print("Loading BM25 index...")
searcher = LuceneSearcher(INDEX_DIR)

print("Loading NLI model...")
scorer = get_scorer()

# Load passage lookup
print("Loading passages...")
passage_lookup = {}
with open(PASSAGES_JSONL) as f:
    for line in f:
        doc = json.loads(line)
        passage_lookup[doc["id"]] = doc

print(f"Ready! {len(passage_lookup)} passages loaded.")


def clean_query(text: str) -> str:
    """Remove URLs from query text."""
    return re.sub(r"https?://[^\s,\]\)\"]+", "", str(text)).strip()


def search_claim(claim: str, top_k: int = 10):
    """Search for relevant passages and score them."""
    query = clean_query(claim)
    # Candidate retrieval (use BM25 results and fusion; FAISS optional)
    candidates = get_candidates(query, searcher=searcher, k_bm25=top_k, final_k=top_k, fusion='rrf')

    results = []
    pairs = []
    meta = []

    # prepare sentence-level pairs for batch scoring
    for rank, cand in enumerate(candidates):
        pid = cand['id']
        doc = passage_lookup.get(pid, {})
        passage_text = doc.get("contents", "")
        source_url = doc.get("source_url", "")

        # select top sentences per passage
        sents = select_top_sentences(passage_text, query, top_m=3)
        if not sents:
            # fallback to whole passage
            pairs.append((query, passage_text))
            meta.append({"passage_id": pid, "sentence": passage_text, "bm25_score": cand.get('bm25_score'), "dense_score": cand.get('dense_score'), "source_url": source_url, "rank": rank + 1})
        else:
            for sent, sscore in sents:
                pairs.append((query, sent))
                meta.append({"passage_id": pid, "sentence": sent, "sent_score": sscore, "bm25_score": cand.get('bm25_score'), "dense_score": cand.get('dense_score'), "source_url": source_url, "rank": rank + 1})

    # batch compute logits and decisions
    if pairs:
        logits_list = scorer.score_batch_logits(pairs)
    else:
        logits_list = []

    # aggregate per passage: use max entailment probability among sentences
    per_passage = {}
    for m, logits in zip(meta, logits_list):
        pid = m['passage_id']
        decision = scorer.decide_label(logits)
        entail = decision['entail_prob']
        entry = per_passage.get(pid)
        record = {
            "sentence": m.get('sentence'),
            "entail_prob": entail,
            "contra_prob": decision.get('contra_prob'),
            "margin": decision.get('margin'),
            "logits": decision.get('logits'),
            "sent_score": m.get('sent_score'),
            "bm25_score": m.get('bm25_score'),
            "dense_score": m.get('dense_score'),
            "source_url": m.get('source_url'),
            "rank": m.get('rank'),
        }
        if entry is None or entail > entry['entail_prob']:
            per_passage[pid] = record

    # build results list preserving candidate rank order
    for cand in candidates:
        pid = cand['id']
        entry = per_passage.get(pid)
        if entry is None:
            # no evidence scored, fallback
            results.append({
                "rank": None,
                "passage_id": pid,
                "bm25_score": cand.get('bm25_score'),
                "nli_score": None,
                "nli_label": "NO_EVIDENCE",
                "passage_text": passage_lookup.get(pid, {}).get('contents', '')[:500],
                "source_url": passage_lookup.get(pid, {}).get('source_url', ''),
            })
        else:
            results.append({
                "rank": entry.get('rank'),
                "passage_id": pid,
                "bm25_score": round(entry.get('bm25_score') or 0.0, 4),
                "nli_score": round(entry.get('entail_prob'), 4),
                "nli_label": 'Supports' if entry.get('entail_prob') >= 0.45 else ('Neutral' if entry.get('entail_prob') >= 0.15 else 'Contradicts'),
                "passage_text": (entry.get('sentence') or passage_lookup.get(pid, {}).get('contents', ''))[:500] + ("..." if len((entry.get('sentence') or passage_lookup.get(pid, {}).get('contents', ''))) > 500 else ""),
                "source_url": entry.get('source_url'),
                "raw_logits": entry.get('logits'),
                "margin": entry.get('margin'),
            })

    # build detailed evidence list and confidence
    evidence = []
    for pid, e in per_passage.items():
        evidence.append({
            "passage_id": pid,
            "sentence": e.get('sentence'),
            "entail_prob": float(e.get('entail_prob')),
            "contra_prob": float(e.get('contra_prob')) if e.get('contra_prob') is not None else None,
            "margin": float(e.get('margin')) if e.get('margin') is not None else None,
            "logits": e.get('logits'),
            "bm25_score": float(e.get('bm25_score')) if e.get('bm25_score') is not None else None,
            "dense_score": float(e.get('dense_score')) if e.get('dense_score') is not None else None,
            "source_url": e.get('source_url'),
            "rank": int(e.get('rank')) if e.get('rank') is not None else None,
        })

    # confidence: max entailment probability across evidence (0..1)
    confidence = 0.0
    if evidence:
        confidence = max([item["entail_prob"] for item in evidence])

    # sort evidence by entail_prob desc
    evidence = sorted(evidence, key=lambda x: x["entail_prob"], reverse=True)

    return {"results": results, "evidence": evidence, "confidence": round(float(confidence), 4)}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Community Notes Fact Checker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 {
            text-align: center;
            padding: 30px 0 10px;
            font-size: 2.2em;
            background: linear-gradient(90deg, #2196F3, #4CAF50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        #claim-input {
            flex: 1;
            padding: 15px 20px;
            font-size: 16px;
            border: 2px solid #333;
            border-radius: 12px;
            background: #1e1e3a;
            color: #fff;
            outline: none;
            transition: border-color 0.3s;
        }
        #claim-input:focus { border-color: #2196F3; }
        #search-btn {
            padding: 15px 30px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        #search-btn:hover { transform: scale(1.02); }
        #search-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-card {
            background: #1e1e3a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid #333;
            transition: border-color 0.3s;
        }
        .result-card:hover { border-color: #444; }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .rank {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .scores {
            display: flex;
            gap: 15px;
        }
        .score-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
        }
        .bm25 { background: #FF9800; color: #000; }
        .nli-high { background: #4CAF50; color: white; }
        .nli-mid { background: #FF9800; color: #000; }
        .nli-low { background: #F44336; color: white; }
        .passage-text {
            line-height: 1.6;
            margin-bottom: 10px;
            color: #ccc;
        }
        .source-link {
            color: #2196F3;
            text-decoration: none;
            font-size: 13px;
        }
        .source-link:hover { text-decoration: underline; }
        .loading {
            text-align: center;
            padding: 40px;
            font-size: 18px;
            color: #888;
        }
        .spinner {
            border: 3px solid #333;
            border-top: 3px solid #2196F3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error {
            background: #3a1e1e;
            color: #F44336;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        .stats {
            text-align: center;
            color: #888;
            margin-bottom: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Community Notes Fact Checker</h1>
        <p class="subtitle">Enter a claim to find supporting or contradicting Wikipedia evidence</p>

        <div class="search-box">
            <input type="text" id="claim-input" placeholder="e.g., Carrot cake originated from English pudding recipes..." 
                   onkeypress="if(event.key==='Enter') searchClaim()">
            <button id="search-btn" onclick="searchClaim()">Search</button>
        </div>

        <div id="results"></div>
    </div>

    <script>
        function nliClass(score) {
            if (score >= 0.5) return 'nli-high';
            if (score >= 0.2) return 'nli-mid';
            return 'nli-low';
        }

        function nliLabel(score) {
            if (score >= 0.5) return 'Supports ✓';
            if (score >= 0.2) return 'Neutral ~';
            return 'Contradicts ✗';
        }

        async function searchClaim() {
            const input = document.getElementById('claim-input');
            const btn = document.getElementById('search-btn');
            const resultsDiv = document.getElementById('results');
            const claim = input.value.trim();

            if (!claim) return;

            btn.disabled = true;
            btn.textContent = 'Searching...';
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Searching through 6,588 passages...</div>';

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({claim: claim})
                });

                const data = await response.json();

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="error">${data.error}</div>`;
                } else {
                    let html = `<div class="stats">Found ${data.results.length} relevant passages</div>`;

                    data.results.forEach(r => {
                        html += `
                            <div class="result-card">
                                <div class="result-header">
                                    <span class="rank">#${r.rank}</span>
                                    <div class="scores">
                                        <span class="score-badge bm25">BM25: ${r.bm25_score}</span>
                                        <span class="score-badge ${nliClass(r.nli_score)}">${nliLabel(r.nli_score)} (${r.nli_score})</span>
                                    </div>
                                </div>
                                <p class="passage-text">${r.passage_text}</p>
                                <a href="${r.source_url}" target="_blank" class="source-link">📄 ${r.source_url}</a>
                            </div>
                        `;
                    });

                    resultsDiv.innerHTML = html;
                }
            } catch (e) {
                resultsDiv.innerHTML = `<div class="error">Error: ${e.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Search';
            }
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    claim = data.get("claim", "").strip()

    if not claim:
        return jsonify({"error": "Please enter a claim to fact-check."})

    if len(claim) < 10:
        return jsonify({"error": "Claim too short. Please enter at least 10 characters."})

    try:
        resp = search_claim(claim, top_k=10)
        # resp contains: results, evidence, confidence
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Community Notes Fact Checker")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)