"""
retrieval.py

Provides hybrid retrieval utilities combining Pyserini (BM25/Lucene) and a FAISS index.
Includes score normalization, reciprocal rank fusion (RRF), simple sentence splitting,
and a lightweight sentence scoring function for sentence-level evidence selection.

This module intentionally avoids requiring an encoder for query embeddings — the
`dense_search` function accepts a precomputed `query_embedding`. If you have a query
encoder available, pass the embedding to `get_candidates` to enable dense retrieval.

Functions:
- init_bm25_searcher(index_dir)
- load_faiss(index_path, embeddings_path, passage_ids_path)
- bm25_search(searcher, query, k)
- dense_search(faiss_index, query_emb, k)
- normalize_scores_zscore / minmax
- rrf_fusion
- get_candidates(...)
- split_sentences(text)
- select_top_sentences(passage_text, claim, top_m=3)

"""
from __future__ import annotations

import os
import json
import math
from typing import List, Tuple, Optional

try:
    from pyserini.search.lucene import LuceneSearcher
except Exception:
    LuceneSearcher = None

try:
    import faiss
    import numpy as np
except Exception:
    faiss = None
    np = None


def init_bm25_searcher(index_dir: str):
    if LuceneSearcher is None:
        raise RuntimeError("pyserini not available in environment")
    return LuceneSearcher(index_dir)


def load_faiss(index_path: str, embeddings_path: str, passage_ids_path: str):
    if faiss is None:
        raise RuntimeError("faiss or numpy not available in environment")
    idx = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)
    with open(passage_ids_path, encoding="utf-8") as f:
        passage_ids = json.load(f)
    return idx, embeddings, passage_ids


def bm25_search(searcher, query: str, k: int = 50):
    hits = searcher.search(query, k=k)
    results = []
    for rank, h in enumerate(hits):
        results.append({"id": h.docid, "score": float(h.score), "rank": rank + 1})
    return results


def dense_search(faiss_index, query_emb, k: int = 50):
    if faiss_index is None or query_emb is None:
        return []
    D, I = faiss_index.search(query_emb.reshape(1, -1).astype('float32'), k)
    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        results.append({"index": int(idx), "score": float(dist), "rank": rank + 1})
    return results


def _zscore(arr):
    import numpy as _np
    a = _np.array(arr, dtype=float)
    mu = a.mean()
    sd = a.std() if a.std() > 0 else 1.0
    return ((a - mu) / sd).tolist()


def _minmax(arr):
    import numpy as _np
    a = _np.array(arr, dtype=float)
    lo = a.min()
    hi = a.max()
    if hi - lo <= 0:
        return [0.5] * len(a)
    return (((a - lo) / (hi - lo))).tolist()


def rrf_fusion(bm25_list: List[dict], dense_list: List[dict], k: int = 50, k_rrf: int = 60):
    """Reciprocal Rank Fusion of two ranked lists. Expects lists of dicts with 'id' or 'index' and 'rank'."""
    scores = {}
    # BM25 ids are docids; dense_list indices are numeric positions — caller must map indices to ids
    for i, it in enumerate(bm25_list):
        docid = it.get('id')
        r = i + 1
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (k_rrf + r)
    for i, it in enumerate(dense_list):
        docid = it.get('id') or it.get('index')
        r = i + 1
        scores[docid] = scores.get(docid, 0.0) + 1.0 / (k_rrf + r)
    # return sorted by fused score
    fused = sorted([{"id": k, "fused_score": v} for k, v in scores.items()], key=lambda x: x['fused_score'], reverse=True)
    return fused[:k]


def get_candidates(query: str,
                   searcher=None,
                   faiss_index=None,
                   embeddings=None,
                   passage_ids: Optional[list] = None,
                   k_bm25: int = 50,
                   k_dense: int = 50,
                   final_k: int = 20,
                   fusion: str = 'rrf') -> List[dict]:
    """
    Return candidate passages with BM25, dense scores and fused score.
    If `faiss_index` is provided, `embeddings` must be provided and caller should pass query embedding.
    """
    bm25 = []
    dense = []
    if searcher is not None:
        try:
            bm25 = bm25_search(searcher, query, k=k_bm25)
        except Exception:
            bm25 = []

    if faiss_index is not None and embeddings is not None:
        # embeddings expected shape (num_passages, dim); here we require a query embedding passed via `query` param
        # caller can pass a JSON-like dict: {"query_emb": np.array([...])}
        q_emb = None
        if isinstance(query, dict) and 'query_emb' in query:
            q_emb = query['query_emb']
        if q_emb is not None:
            dense_raw = dense_search(faiss_index, q_emb, k=k_dense)
            # map dense indices to passage ids
            dense = []
            for it in dense_raw:
                idx = it['index']
                pid = passage_ids[idx] if passage_ids and idx < len(passage_ids) else str(idx)
                dense.append({"id": pid, "score": it['score'], "rank": it['rank']})

    # simple fusion
    fused = []
    if fusion == 'rrf':
        fused = rrf_fusion(bm25, dense, k=final_k)
    else:
        # weighted fusion with minmax normalization
        ids = set([h['id'] for h in bm25] + [h['id'] for h in dense])
        bm_scores = {h['id']: h['score'] for h in bm25}
        de_scores = {h['id']: h['score'] for h in dense}
        all_bm_vals = [v for v in bm_scores.values()] or [0.0]
        all_de_vals = [v for v in de_scores.values()] or [0.0]
        bm_norm = dict(zip(list(bm_scores.keys()), _minmax(list(bm_scores.values()))))
        de_norm = dict(zip(list(de_scores.keys()), _minmax(list(de_scores.values()))))
        for pid in ids:
            s_b = bm_norm.get(pid, 0.0)
            s_d = de_norm.get(pid, 0.0)
            fused.append({"id": pid, "fused_score": 0.5 * s_b + 0.5 * s_d})
        fused = sorted(fused, key=lambda x: x['fused_score'], reverse=True)[:final_k]

    # enrich fused results with parent scores if available
    out = []
    for it in fused:
        pid = it['id']
        parent_b = next((h for h in bm25 if h['id'] == pid), None)
        parent_d = next((h for h in dense if h['id'] == pid), None)
        out.append({
            'id': pid,
            'fused_score': float(it.get('fused_score', 0.0)),
            'bm25_score': float(parent_b['score']) if parent_b else None,
            'dense_score': float(parent_d['score']) if parent_d else None,
        })
    return out


def split_sentences(text: str) -> List[str]:
    """Very small sentence splitter: split on punctuation followed by space and capital letter or linebreaks."""
    import re
    if not text:
        return []
    # naive splitter
    parts = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ').strip())
    return [p.strip() for p in parts if p.strip()]


def select_top_sentences(passage_text: str, claim: str, top_m: int = 3) -> List[Tuple[str, float]]:
    """
    Select top sentences by simple token overlap score with claim.
    Returns list of (sentence, score) sorted by score desc.
    """
    import re
    def tokenize(s):
        return re.findall(r"\w+", s.lower())

    claim_toks = set(tokenize(claim))
    sents = split_sentences(passage_text)
    scored = []
    for s in sents:
        toks = tokenize(s)
        if not toks:
            continue
        overlap = len(claim_toks.intersection(toks))
        score = overlap / math.sqrt(len(toks))  # normalize a bit
        scored.append((s, float(score)))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored[:top_m]
