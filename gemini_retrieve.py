"""
gemini_retrieve.py
Use Gemini's parametric knowledge to retrieve Wikipedia URLs for Community Notes.

For each note summary, asks Gemini for the top-k most relevant Wikipedia article
URLs. Outputs results/gemini_results.jsonl with fields: note_id, query, rank, source_url.
Recall@k evaluation only (no passage text or NLI score).
"""

import argparse
import json
import os
import re
import time
import warnings

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()

URL_RE = re.compile(r"https?://[^\s,\]\)\"]+")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def clean_query(summary: str) -> str:
    return URL_RE.sub("", str(summary)).strip()


def parse_urls(text: str) -> list[str]:
    """Extract a JSON array of URLs from Gemini's response."""
    # Find the first [...] block
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        urls = json.loads(match.group())
        return [u for u in urls if isinstance(u, str) and u.startswith("http")]
    except json.JSONDecodeError:
        return []


def parse_url_passage_pairs(text: str) -> list[dict]:
    """Extract a JSON array of {url, passage} dicts from Gemini's response."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
        result = []
        for item in items:
            if isinstance(item, dict) and item.get("url", "").startswith("http"):
                result.append({
                    "url": item["url"],
                    "passage": str(item.get("passage", "")).strip(),
                })
        return result
    except json.JSONDecodeError:
        return []


def build_prompt(query: str, k: int) -> str:
    return (
        f"Given this Community Note summary, list the {k} most relevant Wikipedia article URLs "
        "that support or verify the claims made. Return only a JSON array of Wikipedia URLs "
        'in order of relevance. Example: ["https://en.wikipedia.org/wiki/X", ...]\n\n'
        f"Community Note: {query}"
    )


def build_prompt_with_passages(query: str, k: int) -> str:
    return (
        f"Given this Community Note summary, identify the {k} most relevant Wikipedia articles "
        "that support or verify the claims made. For each article, provide the Wikipedia URL and "
        "a verbatim excerpt (1-3 sentences) from that article that is most relevant to the claim. "
        'Return only a JSON array like: [{"url": "https://en.wikipedia.org/wiki/X", "passage": "relevant excerpt..."}, ...]\n\n'
        f"Community Note: {query}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve Wikipedia URLs from Gemini for Community Notes"
    )
    parser.add_argument(
        "--notes", default="data/notes-small.reduced.tsv",
        help="Path to notes TSV (default: data/notes-small.reduced.tsv)"
    )
    parser.add_argument(
        "--output", default="results/gemini_results.jsonl",
        help="Output JSONL path (default: results/gemini_results.jsonl)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of Wikipedia URLs to request per note (default: 5)"
    )
    parser.add_argument(
        "--model", default="gemini-2.0-flash",
        help="Gemini model name (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many notes (for testing)"
    )
    parser.add_argument(
        "--with-passages", action="store_true",
        help="Ask Gemini for passage text alongside URLs (enables NLI/SupportScore)"
    )
    args = parser.parse_args()

    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = os.environ.get("GEMINI_API_KEY")

    if project:
        # Vertex AI — uses Application Default Credentials (gcloud auth application-default login)
        client = genai.Client(vertexai=True, project=project, location=location)
    elif api_key and api_key != "your-key-here":
        # AI Studio fallback
        client = genai.Client(api_key=api_key)
    else:
        raise SystemExit(
            "ERROR: Set GOOGLE_CLOUD_PROJECT in .env for Vertex AI, "
            "or GEMINI_API_KEY for AI Studio."
        )

    notes = pd.read_csv(args.notes, sep="\t", dtype=str)
    if args.limit:
        notes = notes.head(args.limit)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows_written = 0
    skipped = 0

    with open(args.output, "w") as out_f:
        for _, row in tqdm(notes.iterrows(), total=len(notes), desc="Gemini retrieval"):
            note_id = str(row["noteId"])
            query = clean_query(row.get("summary", ""))
            if not query:
                skipped += 1
                continue

            if args.with_passages:
                prompt = build_prompt_with_passages(query, args.k)
            else:
                prompt = build_prompt(query, args.k)
            try:
                response = client.models.generate_content(
                    model=args.model, contents=prompt
                )
                if args.with_passages:
                    pairs = parse_url_passage_pairs(response.text)
                else:
                    pairs = [{"url": u, "passage": None} for u in parse_urls(response.text)]
            except Exception as exc:
                warnings.warn(f"note {note_id}: API error — {exc}")
                skipped += 1
                time.sleep(0.03)
                continue

            if not pairs:
                warnings.warn(f"note {note_id}: no valid Wikipedia URLs in response")
                skipped += 1
                time.sleep(0.03)
                continue

            for rank, item in enumerate(pairs[: args.k], start=1):
                record = {
                    "note_id": note_id,
                    "query": query,
                    "rank": rank,
                    "source_url": item["url"],
                }
                if item["passage"] is not None:
                    record["passage_text"] = item["passage"]
                out_f.write(json.dumps(record) + "\n")
                rows_written += 1

            time.sleep(0.03)

    print(f"Saved {rows_written} rows to {args.output} ({skipped} notes skipped)")


if __name__ == "__main__":
    main()
