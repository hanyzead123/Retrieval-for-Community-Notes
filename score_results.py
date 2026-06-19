"""score_results.py
Compute NLI entailment (`nli_score`) for a retrieval JSONL and write a new scored file.

Usage:
  python score_results.py --input results/bm25_results.jsonl --batch-size 64

Produces: <input_basename>_nli_results.jsonl in the same folder.
"""
import argparse
import json
import os
from tqdm import tqdm

from nli import get_scorer


def load_passage_lookup():
    # attempt to load passages lookup if needed (lazy)
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    PASSAGES_JSONL = os.path.join(DATA_DIR, "passages", "passages.jsonl")
    lookup = {}
    if os.path.exists(PASSAGES_JSONL):
        with open(PASSAGES_JSONL, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                lookup[str(doc.get("id"))] = doc
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Retrieval JSONL input file")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    out_path = os.path.splitext(args.input)[0] + "_nli_results.jsonl"
    scorer = get_scorer()
    passage_lookup = load_passage_lookup()

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    pairs = []
    for r in rows:
        query = r.get("query") or r.get("claim") or ""
        passage = r.get("passage_text")
        if not passage:
            pid = str(r.get("passage_id") or r.get("passageId") or "")
            doc = passage_lookup.get(pid, {})
            passage = doc.get("contents") or doc.get("passage_text") or ""
        pairs.append((query, passage))

    # compute in batches
    scores = []
    B = args.batch_size
    for i in tqdm(range(0, len(pairs), B), desc="Scoring batches"):
        batch = pairs[i : i + B]
        sc = scorer.score_batch(batch)
        scores.extend(sc)

    # write out new file with nli_score added
    with open(out_path, "w", encoding="utf-8") as out:
        for r, s in zip(rows, scores):
            r["nli_score"] = round(float(s), 4)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote scored results to {out_path}")


if __name__ == "__main__":
    main()
