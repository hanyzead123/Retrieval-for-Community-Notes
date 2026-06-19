"""
retrieve.py
Build a Pyserini BM25 index over the passage corpus and retrieve
top-k passages for each Community Note query.

Supports two modes:
  --mode bm25   : BM25-only retrieval (default)
  --mode hybrid : BM25 + dense retrieval with Reciprocal Rank Fusion
"""

import argparse
import json
import os
import re
import subprocess

# Prevent OpenMP/JVM conflict between FAISS and Pyserini
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASSAGES_JSONL = os.path.join(DATA_DIR, "passages", "passages.jsonl")
INDEX_DIR = os.path.join(DATA_DIR, "bm25_index")
NOTES_PATH = os.path.join(DATA_DIR, "notes_filtered.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

DENSE_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "dense_embeddings.npy")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
PASSAGE_IDS_PATH = os.path.join(DATA_DIR, "passage_ids.json")

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BM25_TOP_K = 10  # default; overridden by --top-k


def build_index():
    """Build a Lucene index over the passages JSONL using Pyserini."""
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print(f"Index already exists at {INDEX_DIR}, skipping build.")
        return

    print("Building BM25 index...")
    subprocess.run(
        [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", os.path.dirname(PASSAGES_JSONL),
            "--index", INDEX_DIR,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "4",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ],
        check=True,
    )
    print(f"Index built at {INDEX_DIR}")


def build_dense_index():
    """Encode passages with a sentence-transformer and build a FAISS index.

    Caches embeddings, FAISS index, and passage IDs to disk.
    Skips if all cached files already exist.
    """
    if (os.path.exists(DENSE_EMBEDDINGS_PATH)
            and os.path.exists(FAISS_INDEX_PATH)
            and os.path.exists(PASSAGE_IDS_PATH)):
        print("Dense index already cached, skipping build.")
        return

    import faiss
    from sentence_transformers import SentenceTransformer

    print("Building dense index...")
    model = SentenceTransformer(DENSE_MODEL_NAME)

    passages = []
    passage_ids = []
    with open(PASSAGES_JSONL, "r") as f:
        for line in f:
            doc = json.loads(line)
            passage_ids.append(doc["id"])
            passages.append(doc["contents"])

    print(f"Encoding {len(passages)} passages...")
    embeddings = model.encode(passages, show_progress_bar=True,
                              batch_size=256, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS flat inner-product index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save to disk
    np.save(DENSE_EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(PASSAGE_IDS_PATH, "w") as f:
        json.dump(passage_ids, f)

    print(f"Dense index built: {len(passages)} passages, dim={dim}")


def retrieve_dense(queries, top_k):
    """Retrieve top-k passages per query using the cached FAISS index.

    Returns a list of lists of (passage_id, score) tuples.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(DENSE_MODEL_NAME)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(PASSAGE_IDS_PATH, "r") as f:
        passage_ids = json.load(f)

    query_embeddings = model.encode(queries, show_progress_bar=True,
                                    batch_size=256, normalize_embeddings=True)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)

    scores, indices = index.search(query_embeddings, top_k)

    results = []
    for i in range(len(queries)):
        hits = []
        for j in range(top_k):
            idx = indices[i][j]
            if idx == -1:
                break
            hits.append((passage_ids[idx], float(scores[i][j])))
        results.append(hits)
    return results


def merge_results(bm25_hits, dense_hits, top_k=20):
    """Merge BM25 and dense hits using Reciprocal Rank Fusion.

    rrf_score = sum(1 / (60 + rank)) across both systems.
    Deduplicates by passage_id, returns top-k by fused score.
    """
    rrf_scores = {}

    for rank, (pid, _score) in enumerate(bm25_hits):
        rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (60 + rank + 1)

    for rank, (pid, _score) in enumerate(dense_hits):
        rrf_scores[pid] = rrf_scores.get(pid, 0) + 1.0 / (60 + rank + 1)

    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_pids[:top_k]


def retrieve_bm25(notes, top_k):
    """Retrieve top-k passages for each note using BM25.

    Returns list of (note_row, query, [(passage_id, score), ...]).
    """
    searcher = LuceneSearcher(INDEX_DIR)
    results = []
    for _, row in tqdm(notes.iterrows(), total=len(notes), desc="BM25 retrieval"):
        query = re.sub(r"https?://[^\s,\]\)\"]+", "", str(row["summary"])).strip()
        hits = searcher.search(query, k=top_k)
        hit_list = [(hit.docid, hit.score) for hit in hits]
        results.append((row, query, hit_list))
    return results


def run_bm25(notes, top_k=BM25_TOP_K):
    """Run BM25-only retrieval and write results."""
    bm25_results = retrieve_bm25(notes, top_k)
    searcher = LuceneSearcher(INDEX_DIR)

    output = []
    for row, query, hits in bm25_results:
        for rank, (pid, score) in enumerate(hits):
            doc = json.loads(searcher.doc(pid).raw())
            output.append({
                "note_id": str(row["noteId"]),
                "query": query,
                "rank": rank + 1,
                "passage_id": pid,
                "score": score,
                "passage_text": doc["contents"],
                "source_url": doc.get("source_url", ""),
            })
    return output


def run_hybrid(notes, top_k=BM25_TOP_K):
    """Run hybrid (BM25 + dense) retrieval with RRF merge."""
    bm25_results = retrieve_bm25(notes, top_k)

    queries = [
        re.sub(r"https?://[^\s,\]\)\"]+", "", str(row["summary"])).strip()
        for _, row in notes.iterrows()
    ]
    dense_results = retrieve_dense(queries, top_k)

    # Load passage texts via searcher for output
    searcher = LuceneSearcher(INDEX_DIR)

    output = []
    for i, (row, query, bm25_hits) in enumerate(bm25_results):
        dense_hits = dense_results[i]
        merged = merge_results(bm25_hits, dense_hits, top_k=top_k)

        for rank, (pid, rrf_score) in enumerate(merged):
            doc = json.loads(searcher.doc(pid).raw())
            output.append({
                "note_id": str(row["noteId"]),
                "query": query,
                "rank": rank + 1,
                "passage_id": pid,
                "score": rrf_score,
                "passage_text": doc["contents"],
                "source_url": doc.get("source_url", ""),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description="Retrieve passages for Community Notes")
    parser.add_argument("--mode", choices=["bm25", "hybrid"], default="bm25",
                        help="Retrieval mode: bm25 (default) or hybrid (BM25 + dense)")
    parser.add_argument("--top-k", type=int, default=BM25_TOP_K,
                        help=f"Passages to retrieve per note (default: {BM25_TOP_K})")
    args = parser.parse_args()

    # Always need BM25 index
    build_index()

    notes = pd.read_parquet(NOTES_PATH)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"{args.mode}_results.jsonl")

    if args.mode == "bm25":
        results = run_bm25(notes, top_k=args.top_k)
    elif args.mode == "hybrid":
        build_dense_index()
        results = run_hybrid(notes, top_k=args.top_k)

    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results to {results_path}")


if __name__ == "__main__":
    main()
