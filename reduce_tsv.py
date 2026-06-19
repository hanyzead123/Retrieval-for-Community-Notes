#!/usr/bin/env python3
"""
Reduce a Community Notes TSV: keep selected columns and only rows
that contain a Wikipedia link in summary or trustworthySources.
"""

import argparse
import sys

import pandas as pd


WIKI_PATTERN = "wikipedia.org"


def has_wikipedia_link(value: str) -> bool:
    """Return True if value contains a Wikipedia link."""
    if pd.isna(value):
        return False
    return WIKI_PATTERN in str(value).lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce TSV to selected columns and rows with Wikipedia links."
    )
    parser.add_argument(
        "filename",
        help="Path to input TSV file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output TSV path (default: input filename with .reduced.tsv suffix)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.filename, sep="\t")

    required_cols = ["noteId", "tweetId", "summary", "classification", "trustworthySources"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = df[required_cols].copy()

    # Keep only rows with a Wikipedia link in summary or trustworthySources
    has_wiki = df["summary"].apply(has_wikipedia_link) | df["trustworthySources"].apply(
        has_wikipedia_link
    )
    df = df[has_wiki].reset_index(drop=True)

    out_path = args.output or args.filename.replace(".tsv", ".reduced.tsv")
    if out_path == args.filename:
        out_path = args.filename + ".reduced.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
