"""debug_nli.py
Script to run sentence-level NLI diagnostics for a claim.

Usage: python debug_nli.py
"""
import json
import os
from nli import get_scorer
from retrieval import split_sentences, select_top_sentences

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PASSAGES_JSONL = os.path.join(DATA_DIR, "passages", "passages.jsonl")

CLAIM = "The Great Pyramid is one of the Seven Wonders of the Ancient World"
N = 10


def main():
    scorer = get_scorer()
    print(f"Claim: {CLAIM}\n")
    print(f"Reading up to {N} passages from {PASSAGES_JSONL}\n")

    # load passages into lookup
    passage_lookup = {}
    with open(PASSAGES_JSONL, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            passage_lookup[doc["id"]] = doc

    # inspect first N passages
    keys = list(passage_lookup.keys())[:N]
    pairs = []
    meta = []
    for i, pid in enumerate(keys):
        doc = passage_lookup[pid]
        text = doc.get("contents", "")
        # select top sentence(s)
        sents = select_top_sentences(text, CLAIM, top_m=2)
        for sent, sscore in sents:
            pairs.append((CLAIM, sent))
            meta.append({"passage_id": pid, "sentence": sent, "parent_score": sscore})

    if not pairs:
        print("No sentence candidates found")
        return

    # batch logits
    logits_list = scorer.score_batch_logits(pairs)

    results = []
    for m, logits in zip(meta, logits_list):
        decision = scorer.decide_label(logits)
        out = {
            "passage_id": m["passage_id"],
            "sentence": m["sentence"],
            "parent_score": m["parent_score"],
            "logits": logits,
            "entail_prob": decision["entail_prob"],
            "contra_prob": decision["contra_prob"],
            "margin": decision["margin"],
            "label": decision["label"],
        }
        results.append(out)

    # print a compact report
    from pprint import pprint
    pprint({"claim": CLAIM, "results": results}, width=120)


if __name__ == "__main__":
    main()
