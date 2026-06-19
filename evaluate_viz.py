# Enhanced `evaluate_viz.py`


import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import (
    load_gold_sources_by_note,
    load_nli_results,
    normalize_url,
    recall_at_k,
    support_score_at_k,
)

# =========================
# Paths
# =========================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
VIZ_DIR = os.path.join(RESULTS_DIR, "viz")
NOTES_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "notes_filtered.parquet"
)

# =========================
# Systems
# =========================
SYSTEMS = {
    "BM25": os.path.join(RESULTS_DIR, "bm25_nli_results.jsonl"),
    "Hybrid": os.path.join(RESULTS_DIR, "hybrid_nli_results.jsonl"),
    "Hybrid+Rerank": os.path.join(
        RESULTS_DIR,
        "hybrid_reranked_nli_results.jsonl"
    ),
}

# =========================
# Plot Styling
# =========================
plt.style.use("ggplot")

COLORS = {
    "BM25": "#2196F3",
    "Hybrid": "#4CAF50",
    "Hybrid+Rerank": "#FF9800",
}

MARKERS = {
    "BM25": "o",
    "Hybrid": "s",
    "Hybrid+Rerank": "^",
}


# =====================================================
# Precision@k
# =====================================================
def precision_at_k(nli_rows, gold_by_note, k=5):
    """Compute Precision@k."""

    by_note = defaultdict(list)

    for r in nli_rows:
        by_note[r["note_id"]].append(r)

    precisions = []

    for nid, rows in by_note.items():
        G = gold_by_note.get(nid, set())

        if not G:
            continue

        seen = set()
        retrieved = []

        for r in sorted(rows, key=lambda x: x.get("rank", 999)):
            src = normalize_url(r.get("source_url", ""))

            if src and src not in seen:
                seen.add(src)
                retrieved.append(src)

            if len(retrieved) >= k:
                break

        hits = sum(1 for x in retrieved if x in G)
        precisions.append(hits / k)

    return float(np.mean(precisions)) if precisions else 0.0


# =====================================================
# Recall Curve
# =====================================================
def compute_recall_curve(nli_rows, gold_by_note, max_k=10):
    """Compute Recall@k for k=1..max_k."""

    ks = list(range(1, max_k + 1))
    recalls = []

    for k in ks:
        recalls.append(recall_at_k(nli_rows, gold_by_note, k))

    return ks, recalls


# =====================================================
# Precision Curve
# =====================================================
def compute_precision_curve(nli_rows, gold_by_note, max_k=10):
    """Compute Precision@k for k=1..max_k."""

    ks = list(range(1, max_k + 1))
    precisions = []

    for k in ks:
        precisions.append(precision_at_k(nli_rows, gold_by_note, k))

    return ks, precisions


# =====================================================
# Recall Curves Plot
# =====================================================
def plot_recall_curves(all_results, gold_by_note, max_k=10):
    """Plot Recall@k curves for all systems."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, rows in all_results.items():
        if rows:
            ks, recalls = compute_recall_curve(
                rows,
                gold_by_note,
                max_k
            )

            ax.plot(
                ks,
                recalls,
                color=COLORS.get(name, "gray"),
                marker=MARKERS.get(name, "o"),
                linewidth=2.5,
                markersize=8,
                label=name,
            )

    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel("Recall@k", fontsize=14)
    ax.set_title(
        "Recall@k Curves",
        fontsize=16,
        fontweight="bold"
    )

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max_k + 1))
    ax.set_ylim(0, 1.05)

    path = os.path.join(VIZ_DIR, "recall_at_k_curves.png")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved -> {path}")


# =====================================================
# Precision Curves Plot
# =====================================================
def plot_precision_curves(all_results, gold_by_note, max_k=10):
    """Plot Precision@k curves."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, rows in all_results.items():
        if rows:
            ks, precisions = compute_precision_curve(
                rows,
                gold_by_note,
                max_k
            )

            ax.plot(
                ks,
                precisions,
                color=COLORS.get(name, "gray"),
                marker=MARKERS.get(name, "o"),
                linewidth=2.5,
                markersize=8,
                label=name,
            )

    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel("Precision@k", fontsize=14)
    ax.set_title(
        "Precision@k Curves",
        fontsize=16,
        fontweight="bold"
    )

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max_k + 1))
    ax.set_ylim(0, 1.05)

    path = os.path.join(VIZ_DIR, "precision_at_k_curves.png")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved -> {path}")


# =====================================================
# Bar Comparison
# =====================================================
def plot_bar_comparison(all_results, gold_by_note, k=5):
    """Bar chart comparing metrics."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = []
    recalls = []
    precisions = []
    supports = []

    for name, rows in all_results.items():
        if rows:
            names.append(name)
            recalls.append(recall_at_k(rows, gold_by_note, k))
            precisions.append(precision_at_k(rows, gold_by_note, k))
            supports.append(support_score_at_k(rows, k))

    colors = [COLORS[n] for n in names]

    metrics = [
        (axes[0], recalls, f"Recall@{k}"),
        (axes[1], precisions, f"Precision@{k}"),
        (axes[2], supports, f"SupportScore@{k}"),
    ]

    for ax, values, title in metrics:
        bars = ax.bar(
            names,
            values,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylim(0, 1.05)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.4f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

    path = os.path.join(VIZ_DIR, f"bar_comparison_k{k}.png")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved -> {path}")


# =====================================================
# Hit/Miss Plot
# =====================================================
def plot_hit_miss(all_results, gold_by_note, k=5):
    """Stacked hit/miss distribution plot."""

    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    hits_list = []
    misses_list = []

    for name, rows in all_results.items():
        if not rows:
            continue

        names.append(name)

        by_note = defaultdict(list)

        for r in rows:
            by_note[r["note_id"]].append(r)

        hits = 0
        total = 0

        for nid, note_rows in by_note.items():
            G = gold_by_note.get(nid, set())

            if not G:
                continue

            total += 1

            seen = set()
            R_k = set()

            for r in sorted(
                note_rows,
                key=lambda x: x.get("rank", 999)
            ):
                src = normalize_url(r.get("source_url", ""))

                if src and src not in seen:
                    seen.add(src)
                    R_k.add(src)

                    if len(R_k) >= k:
                        break

            if G & R_k:
                hits += 1

        hits_list.append(hits)
        misses_list.append(total - hits)

    x = np.arange(len(names))
    width = 0.5

    ax.bar(
        x,
        hits_list,
        width,
        label="Hit",
        color="#4CAF50",
        edgecolor="black",
    )

    ax.bar(
        x,
        misses_list,
        width,
        bottom=hits_list,
        label="Miss",
        color="#F44336",
        edgecolor="black",
    )

    ax.set_ylabel("Number of Notes", fontsize=14)
    ax.set_title(
        f"Hit/Miss Distribution @k={k}",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.legend(fontsize=12)

    path = os.path.join(VIZ_DIR, f"hit_miss_k{k}.png")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved -> {path}")


# =====================================================
# Failure Analysis
# =====================================================
def export_failure_analysis(all_results, gold_by_note, k=5):
    """Export failed retrieval cases."""

    failures = []

    for system_name, rows in all_results.items():
        by_note = defaultdict(list)

        for r in rows:
            by_note[r["note_id"]].append(r)

        for nid, note_rows in by_note.items():
            G = gold_by_note.get(nid, set())

            if not G:
                continue

            seen = set()
            retrieved = []

            for r in sorted(note_rows, key=lambda x: x.get("rank", 999)):
                src = normalize_url(r.get("source_url", ""))

                if src and src not in seen:
                    seen.add(src)
                    retrieved.append(src)

                if len(retrieved) >= k:
                    break

            hit = any(r in G for r in retrieved)

            if not hit:
                failures.append({
                    "system": system_name,
                    "note_id": nid,
                    "gold_urls": list(G),
                    "retrieved_urls": retrieved,
                })

    df = pd.DataFrame(failures)

    path = os.path.join(VIZ_DIR, "failure_analysis.csv")

    df.to_csv(path, index=False)

    print(f"Saved -> {path}")


# =====================================================
# Metrics CSV Export
# =====================================================
def export_metrics_csv(all_results, gold_by_note, k=5):
    """Export summary metrics CSV."""

    summary = []

    for name, rows in all_results.items():
        if rows:
            summary.append({
                "system": name,
                f"recall@{k}": recall_at_k(rows, gold_by_note, k),
                f"precision@{k}": precision_at_k(rows, gold_by_note, k),
                f"support_score@{k}": support_score_at_k(rows, k),
            })

    df = pd.DataFrame(summary)

    path = os.path.join(VIZ_DIR, "metrics_summary.csv")

    df.to_csv(path, index=False)

    print(f"Saved -> {path}")


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced visualization for retrieval systems"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="k for evaluation"
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="max k for curves"
    )

    args = parser.parse_args()

    os.makedirs(VIZ_DIR, exist_ok=True)

    print("Loading gold URLs...")

    gold_by_note = load_gold_sources_by_note(NOTES_PATH)

    print(f"Loaded {len(gold_by_note)} notes.")

    all_results = {}

    for name, path in SYSTEMS.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            all_results[name] = load_nli_results(path)
        else:
            print(f"WARNING: Missing {path}")

    if not all_results:
        print("No result files found.")
        return

    print("\nGenerating Recall curves...")
    plot_recall_curves(all_results, gold_by_note, args.max_k)

    print("Generating Precision curves...")
    plot_precision_curves(all_results, gold_by_note, args.max_k)

    print("Generating comparison bars...")
    plot_bar_comparison(all_results, gold_by_note, args.k)

    print("Generating hit/miss distributions...")
    plot_hit_miss(all_results, gold_by_note, args.k)

    print("Exporting failure analysis...")
    export_failure_analysis(all_results, gold_by_note, args.k)

    print("Exporting metrics CSV...")
    export_metrics_csv(all_results, gold_by_note, args.k)

    print(f"\n| System | Recall@{args.k} | Precision@{args.k} | SupportScore@{args.k} |")
    print("|---|---|---|---|")

    for name, rows in all_results.items():
        if rows:
            r = recall_at_k(rows, gold_by_note, args.k)
            p = precision_at_k(rows, gold_by_note, args.k)
            s = support_score_at_k(rows, args.k)

            print(
                f"| {name} | {r:.4f} | {p:.4f} | {s:.4f} |"
            )

    print(f"\nAll plots and CSV files saved to:\n{VIZ_DIR}")


if __name__ == "__main__":
    main()
