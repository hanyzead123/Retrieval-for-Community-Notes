"""
preprocess.py
Load Community Notes TSV, filter to notes with cited URLs,
fetch Wikipedia article text via the MediaWiki API (Option C: hybrid expansion
with depth 1, 10 links per seed), deduplicate URLs and passages, chunk,
and save passages as JSONL for Pyserini.
"""

import json
import os
import re
import time
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_RAW_TSV = os.path.join(DATA_DIR, "notes-small.reduced.tsv")
NOTES_OUT = os.path.join(DATA_DIR, "notes_filtered.parquet")
PASSAGES_DIR = os.path.join(DATA_DIR, "passages")
PASSAGES_OUT = os.path.join(PASSAGES_DIR, "passages.jsonl")

CHUNK_WORDS = 200
CHUNK_OVERLAP = 50
MIN_TEXT_WORDS = 20
HTTP_TIMEOUT_SECS = 30
USER_AGENT = "CS224N-Final-Project/1.0 (MediaWiki API preprocessing)"

# Option C hybrid expansion: depth 1, up to this many outlinks per seed article
EXPANSION_LINK_LIMIT = 10
# Main namespace only (0 = articles)
MW_NAMESPACE_MAIN = 0


def load_and_filter_notes(path: str) -> pd.DataFrame:
    """Load the TSV and keep only notes that cite at least one URL."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Keep rows where summary contains a URL
    df = df[df["summary"].str.contains(r"https?://", na=False)].copy()
    df = df.dropna(subset=["summary"])
    df = df.reset_index(drop=True)
    print(f"Filtered to {len(df)} notes with URLs")
    return df


def extract_urls(summary: str) -> list[str]:
    """Pull URLs out of the summary text."""
    return re.findall(r"https?://[^\s,\]\)\"]+", str(summary))


def normalize_url_for_dedup(url: str) -> str:
    """Normalize URL for use as a dedup key (strip fragment, trailing slash, lowercase host)."""
    u = (url or "").strip().rstrip("/")
    parsed = urlparse(u)
    if not parsed.scheme or not parsed.netloc:
        return u
    host = normalize_wikipedia_host(parsed.netloc)
    path = (parsed.path or "/").split("#", 1)[0].rstrip("/") or "/"
    return f"{parsed.scheme or 'https'}://{host}{path}"


def normalize_wikipedia_host(host: str) -> str:
    """Normalize Wikipedia hosts so mobile URLs use the canonical API host."""
    host = host.lower().split(":", 1)[0]
    return host.replace(".m.wikipedia.org", ".wikipedia.org")


def is_wikipedia_url(url: str) -> bool:
    """Return True if the URL points to a Wikipedia page."""
    host = normalize_wikipedia_host(urlparse(url).netloc)
    return host == "wikipedia.org" or host.endswith(".wikipedia.org")


def wikipedia_api_endpoint(url: str) -> str | None:
    """Build the MediaWiki API endpoint for a Wikipedia article URL."""
    parsed = urlparse(url)
    host = normalize_wikipedia_host(parsed.netloc)
    if not host or not is_wikipedia_url(url):
        return None
    scheme = parsed.scheme or "https"
    return f"{scheme}://{host}/w/api.php"


def wikipedia_title_from_url(url: str) -> str | None:
    """Extract the article title from a standard Wikipedia URL."""
    parsed = urlparse(url)
    if not is_wikipedia_url(url):
        return None

    if parsed.path.startswith("/wiki/"):
        # Get everything after /wiki/ — handle titles with slashes
        title = unquote(parsed.path[len("/wiki/"):])
    elif parsed.path == "/w/index.php":
        title = parse_qs(parsed.query).get("title", [None])[0]
        title = unquote(title) if title else None
    else:
        return None

    if not title:
        return None

    title = title.split("#", 1)[0].strip().replace("_", " ")
    if not title or title.startswith("Special:"):
        return None
    return title

def wikipedia_url_from_title(seed_url: str, title: str) -> str | None:
    """Build a full Wikipedia article URL from a seed URL (for host/scheme) and page title."""
    parsed = urlparse(seed_url)
    host = normalize_wikipedia_host(parsed.netloc)
    if not host or not title.strip():
        return None
    scheme = parsed.scheme or "https"
    path = "/wiki/" + quote(title.strip().replace(" ", "_"), safe="/")
    return f"{scheme}://{host}{path}"


def fetch_wikipedia_outlinks(url: str, limit: int = EXPANSION_LINK_LIMIT) -> list[str]:
    """Return up to `limit` main-namespace outlinks from a Wikipedia article (full URLs)."""
    endpoint = wikipedia_api_endpoint(url)
    title = wikipedia_title_from_url(url)
    if endpoint is None or title is None:
        return []

    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "links",
            "titles": title,
            "plnamespace": MW_NAMESPACE_MAIN,
            "pllimit": limit,
        }
    )
    request = Request(
        f"{endpoint}?{query}",
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECS) as response:
            payload = json.load(response)
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return []
        links = pages[0].get("links") or []
        out = []
        for link in links[:limit]:
            t = (link.get("title") or "").strip()
            if not t:
                continue
            full_url = wikipedia_url_from_title(url, t)
            if full_url:
                out.append(full_url)
        return out
    except Exception:
        return []


def fetch_wikipedia_text(url: str, max_retries: int = 3) -> str | None:
    """Fetch article text from the MediaWiki API for a Wikipedia URL, with retries."""
    endpoint = wikipedia_api_endpoint(url)
    title = wikipedia_title_from_url(url)
    if endpoint is None or title is None:
        return None

    query = urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        }
    )

    for attempt in range(max_retries):
        try:
            request = Request(
                f"{endpoint}?{query}",
                headers={
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            with urlopen(request, timeout=HTTP_TIMEOUT_SECS) as response:
                payload = json.load(response)
            pages = payload.get("query", {}).get("pages", [])
            if not pages:
                return None
            text = (pages[0].get("extract") or "").strip()
            return text or None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))  # exponential backoff
            else:
                return None
    return None

def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > chunk_words and current:
            chunks.append(" ".join(current))
            # keep last `overlap` words for continuity
            overlap_words = " ".join(current).split()[-overlap:]
            current = [" ".join(overlap_words)]
            current_len = len(overlap_words)
        current.append(sent)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch cited Wikipedia article text (with Option C expansion) and chunk into passages"
    )
    parser.add_argument(
        "--notes",
        default=DEFAULT_RAW_TSV,
        help=f"Path to reduced notes TSV (default: {DEFAULT_RAW_TSV})",
    )
    args = parser.parse_args()

    # 1. Load and filter notes
    notes = load_and_filter_notes(args.notes)
    notes.to_parquet(NOTES_OUT, index=False)
    print(f"Saved filtered notes to {NOTES_OUT}")

    # 2. Build seed URLs (Wikipedia only) and url -> set of note_ids
    url_to_notes: dict[str, set[str]] = {}
    for _, row in notes.iterrows():
        nid = str(row["noteId"])
        for url in extract_urls(row["summary"]):
            if not is_wikipedia_url(url):
                continue
            key = normalize_url_for_dedup(url)
            url_to_notes.setdefault(key, set()).add(nid)
    seed_urls = set(url_to_notes.keys())
    print(f"Found {len(seed_urls)} unique cited Wikipedia URLs (seeds)")

    # 3. Fetch outlinks for each seed URL, but KEEP the seeds too
    all_urls_set = set(url_to_notes.keys())  # start with all seed URLs
    for seed_url in tqdm(sorted(url_to_notes.keys()), desc="Fetching outlinks"):
        outlinks = fetch_wikipedia_outlinks(seed_url, limit=EXPANSION_LINK_LIMIT)
        time.sleep(1.0)
        for link in outlinks:
            if link not in all_urls_set:
                all_urls_set.add(link)
                # Associate the new URL with the same notes as the seed
                url_to_notes[link] = url_to_notes.get(seed_url, set()).copy()
    all_urls = sorted(all_urls_set)
    print(f"Expanded from {len(seed_urls)} to {len(all_urls)} URLs (keeping seeds)")

    # 4. Fetch each URL once (URL-level dedup), chunk, then passage-level dedup
    os.makedirs(PASSAGES_DIR, exist_ok=True)
    passage_id = 0
    seen_chunk_key: set[tuple[str, str]] = set()  # (normalized_url, chunk_text) for passage dedup

    with open(PASSAGES_OUT, "w") as f:
        for url in tqdm(all_urls, desc="Fetching & chunking"):
            text = fetch_wikipedia_text(url)
            time.sleep(3.0)
            if not text or len(text.split()) < MIN_TEXT_WORDS:
                continue
            note_ids_for_url = url_to_notes[url]
            rep_note_id = min(note_ids_for_url) if note_ids_for_url else ""
            for chunk in chunk_text(text):
                chunk_key = (url, chunk)
                if chunk_key in seen_chunk_key:
                    continue
                seen_chunk_key.add(chunk_key)
                record = {
                    "id": str(passage_id),
                    "contents": chunk,
                    "note_id": rep_note_id,
                    "source_url": url,
                }
                f.write(json.dumps(record) + "\n")
                passage_id += 1

    print(f"Saved {passage_id} passages to {PASSAGES_OUT} (URL and passage deduplicated)")


if __name__ == "__main__":
    main()
