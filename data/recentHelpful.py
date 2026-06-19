#!/usr/bin/env python3
"""
Filter notes-large.reduced.tsv to only notes rated CURRENTLY_RATED_HELPFUL.

Reads noteStatusHistory-00000.tsv for currentStatus, keeps rows whose noteId
has currentStatus == CURRENTLY_RATED_HELPFUL, and writes them to
notes-large-helpful.reduced.tsv with the same columns as notes-large.reduced.tsv.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
NOTES_REDUCED = DATA_DIR / "notes-large.reduced.tsv"
STATUS_HISTORY = DATA_DIR / "noteStatusHistory-00000.tsv"
OUTPUT_FILE = DATA_DIR / "notes-large-helpful.reduced.tsv"

TARGET_STATUS = "CURRENTLY_RATED_HELPFUL"

# Columns match notes-large.reduced.tsv (same as reduce_tsv.py output)
OUTPUT_FIELDNAMES = (
    "noteId",
    "tweetId",
    "summary",
    "classification",
    "trustworthySources",
)


def main() -> None:
    if not STATUS_HISTORY.exists():
        raise FileNotFoundError(f"Status history not found: {STATUS_HISTORY}")
    if not NOTES_REDUCED.exists():
        raise FileNotFoundError(f"Notes file not found: {NOTES_REDUCED}")

    # 1. From status history: collect noteIds where currentStatus == CURRENTLY_RATED_HELPFUL
    helpful_note_ids = set()
    with open(STATUS_HISTORY, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "currentStatus" not in (reader.fieldnames or []):
            raise ValueError(
                f"noteStatusHistory missing 'currentStatus'; got {list(reader.fieldnames or [])}"
            )
        for row in reader:
            if (row.get("currentStatus") or "").strip() != TARGET_STATUS:
                continue
            nid = (row.get("noteId") or "").strip()
            if nid:
                helpful_note_ids.add(nid)

    if not helpful_note_ids:
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=OUTPUT_FIELDNAMES, delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
        print(f"No helpful notes found; wrote empty {OUTPUT_FILE}")
        return

    # 2. Filter notes-large.reduced.tsv to those noteIds and write output
    written = 0
    with open(NOTES_REDUCED, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(
                fout,
                fieldnames=OUTPUT_FIELDNAMES,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in reader:
                nid = (row.get("noteId") or "").strip()
                if nid in helpful_note_ids:
                    writer.writerow(row)
                    written += 1

    print(f"Wrote {written} helpful notes to {OUTPUT_FILE} (from {len(helpful_note_ids)} helpful noteIds in status history)")


if __name__ == "__main__":
    main()
