"""
evaluate.py
Compute Recall@k and SupportScore@k from an NLI results file.

- Recall@k: proportion of queries where the top-k retrieved sources contain
  at least one gold (cited) source. Gold sources = URLs extracted from the note summary.
- SupportScore@k: mean NLI entailment score over the top-k passages per query
  (how well excerpts support the claim; uses roberta-large-mnli scores already in the file).
"""

import argparse
import json
import os
import re

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NOTES_PATH = os.path.join(DATA_DIR, "notes_filtered.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_NLI_RESULTS = os.path.join(RESULTS_DIR, "bm25_nli_results.jsonl")


def extract_urls(summary: str) -> list[str]:
    """Pull URLs out of the summary text (same logic as preprocess)."""
    return re.findall(r"https?://[^\s,\]\)\"]+", str(summary))


def normalize_url(url: str) -> str:
    """
    Normalize a Wikipedia URL to a canonical article slug for robust matching.

    Handles: http/https, www/mobile subdomains, percent-encoding vs underscores,
    fragment identifiers, query parameters, and case in the article title.
    Falls back to lowercased stripped URL for non-Wikipedia URLs.
    """
    from urllib.parse import urlparse, unquote

    u = (url or "").strip().rstrip("/")
    if not u:
        return u

    try:
        parsed = urlparse(u)
        host = parsed.hostname or ""
        # Normalize Wikipedia URLs to just the article slug
        if "wikipedia.org" in host:
            # Strip fragment and query, decode percent-encoding, replace spaces with _
            path = unquote(parsed.path).strip("/")
            # path is typically "wiki/Article_Title"
            slug = path.lower().replace(" ", "_")
            return slug
    except Exception:
        pass

    return u.lower()


def load_gold_sources_by_note(notes_path: str) -> dict[str, set[str]]:
    """Load notes and return for each note_id the set of normalized cited URLs."""
    if notes_path.endswith(".tsv") or notes_path.endswith(".csv"):
        sep = "\t" if notes_path.endswith(".tsv") else ","
        df = pd.read_csv(notes_path, sep=sep, dtype=str)
    else:
        df = pd.read_parquet(notes_path)
    gold = {}
    for _, row in df.iterrows():
        nid = str(row["noteId"])
        urls = extract_urls(row["summary"])
        gold[nid] = {normalize_url(u) for u in urls}
    return gold


def load_nli_results(path: str) -> list[dict]:
    """Load NLI results JSONL."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def recall_at_k(nli_rows: list[dict], gold_by_note: dict[str, set[str]], k: int) -> float:
    """
    Recall@k = (1/|Q|) * sum_q 1[G(q) ∩ R_k(q) ≠ ∅].
    R_k(q) = set of first k unique source_urls (by rank) for query q.
    """
    from collections import defaultdict
    # Group by note_id, keep rows sorted by rank
    by_note = defaultdict(list)
    for r in nli_rows:
        by_note[r["note_id"]].append(r)
    for nid in by_note:
        by_note[nid].sort(key=lambda x: x["rank"])

    hits = 0
    n_queries = 0
    for nid, rows in by_note.items():
        G = gold_by_note.get(nid)
        if not G:
            continue
        # Build R_k(q): first k unique sources (by order of appearance)
        seen = set()
        R_k = set()
        for r in rows:
            src = normalize_url(r.get("source_url", ""))
            if not src:
                continue
            if src not in seen:
                seen.add(src)
                R_k.add(src)
                if len(R_k) >= k:
                    break
        if G & R_k:
            hits += 1
        n_queries += 1

    return hits / n_queries if n_queries else 0.0


def support_score_at_k(nli_rows: list[dict], k: int) -> float:
    """
    SupportScore@k = (1/|Q|) * sum_q [ mean of nli_score over top-k passages for q ].
    Uses the precomputed nli_score (roberta-large-mnli entailment) in the file.
    """
    from collections import defaultdict
    by_note = defaultdict(list)
    for r in nli_rows:
        by_note[r["note_id"]].append(r)
    for nid in by_note:
        by_note[nid].sort(key=lambda x: x["rank"])

    scores = []
    for nid, rows in by_note.items():
        top_k = [r for r in rows if r["rank"] <= k]
        if not top_k:
            continue
        nli_scores = [r["nli_score"] for r in top_k if "nli_score" in r]
        if not nli_scores:
            continue
        scores.append(sum(nli_scores) / len(nli_scores))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Compute Recall@k and SupportScore@k from NLI results."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_NLI_RESULTS,
        help=f"Path to NLI results JSONL (default: {DEFAULT_NLI_RESULTS})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for Recall@k and SupportScore@k (default: 5)",
    )
    parser.add_argument(
        "--notes",
        default=NOTES_PATH,
        help=f"Path to notes parquet for gold URLs (default: {NOTES_PATH})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"NLI results not found at {args.input}. Run test.py after retrieve.py."
        )
    if not os.path.exists(args.notes):
        raise FileNotFoundError(
            f"Notes not found at {args.notes}. Run preprocess.py first."
        )

    nli_rows = load_nli_results(args.input)
    # Quick check: ensure NLI scores are present in the input file
    has_nli = any("nli_score" in r for r in nli_rows)
    gold_by_note = load_gold_sources_by_note(args.notes)
    k = args.k

    if not has_nli:
        print(f"Input file {args.input} contains no 'nli_score' values.")
        print("SupportScore requires precomputed NLI entailment scores.\n")
        print("Options:\n"
              "  1) Generate scored results using the provided scorer script:\n"
              "       python score_results.py --input results/bm25_results.jsonl\n"
              "     This will write a file named results/bm25_results_nli_results.jsonl with 'nli_score' fields.\n"
              "  2) Or pass an already-scored file (e.g. results/bm25_nli_results.jsonl) to --input.\n")
        # still compute recall (which doesn't need nli_score) but report SupportScore as 0.0
        recall = recall_at_k(nli_rows, gold_by_note, k)
        print(f"Recall@{k}\t\t{recall:.4f}")
        print(f"SupportScore@{k}\t{0.0:.4f}")
        return

    recall = recall_at_k(nli_rows, gold_by_note, k)
    support = support_score_at_k(nli_rows, k)

    print(f"Recall@{k}\t\t{recall:.4f}")
    print(f"SupportScore@{k}\t{support:.4f}")


if __name__ == "__main__":
    main()
