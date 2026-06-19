"""
rerank.py
Cross-encoder reranking of retrieval results.

Reads a retrieval results JSONL, scores each (query, passage) pair with a
cross-encoder (ms-marco-MiniLM-L-6-v2), re-sorts per note by that score,
and writes the top-k reranked results.
"""

import argparse
import json
import os
from collections import defaultdict

from sentence_transformers import CrossEncoder
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_INPUT = os.path.join(RESULTS_DIR, "hybrid_results.jsonl")
MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 10


def load_results(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rerank(rows: list[dict], model: CrossEncoder, top_k: int) -> list[dict]:
    by_note = defaultdict(list)
    for r in rows:
        by_note[r["note_id"]].append(r)

    output = []
    for note_rows in tqdm(by_note.values(), desc="Reranking"):
        query = note_rows[0]["query"]
        pairs = [(query, r["passage_text"]) for r in note_rows]
        scores = model.predict(pairs, show_progress_bar=False)

        for r, score in zip(note_rows, scores):
            r["ce_score"] = float(score)

        sorted_rows = sorted(note_rows, key=lambda x: x["ce_score"], reverse=True)
        for rank, r in enumerate(sorted_rows[:top_k], 1):
            r["rank"] = rank
            output.append(r)

    return output


def main():
    parser = argparse.ArgumentParser(description="Cross-encoder reranking of retrieval results")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT,
                        help=f"Retrieval results JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: derived from input name)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Passages to keep per note after reranking (default: {DEFAULT_TOP_K})")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}. Run retrieve.py first.")

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(RESULTS_DIR, f"{base.replace('_results', '_reranked_results')}.jsonl")

    rows = load_results(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    model = CrossEncoder(MODEL_ID)
    output = rerank(rows, model, args.top_k)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        for r in output:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(output)} reranked results to {out_path}")


if __name__ == "__main__":
    main()
