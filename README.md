# 224N Final — Retrieval for Community Notes

Retrieve and score source passages for Community Notes using BM25, hybrid (BM25 + dense), hybrid+rerank, or Gemini parametric retrieval, then evaluate with NLI-based support scoring.

## Setup

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **For Gemini retrieval** — create a `.env` file in the project root:
   ```
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   ```
   Then authenticate: `gcloud auth application-default login`

2. **Java 11+** (required by Pyserini for the BM25 index)
   - **Apple Silicon Mac**: Use an ARM64 JDK (e.g. [Eclipse Temurin](https://adoptium.net/) or `conda install -c conda-forge openjdk=21`). Then:
     ```bash
     export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home
     export PATH="$JAVA_HOME/bin:$PATH"
     ```
   - **Intel Mac / Linux**: Install via your package manager (e.g. `brew install openjdk@21`).

## How to run the pipeline

### Single command (recommended)

Use `run_pipeline.py` to run the full pipeline in one go:

```bash
# BM25 only (default), skip steps whose outputs already exist, k=5
python run_pipeline.py

# Both BM25 and hybrid retrieval
python run_pipeline.py --modes bm25 hybrid

# Hybrid only, force all steps to run, evaluate with k=10
python run_pipeline.py --modes hybrid --no-skip --k 10
```

| Option | Description |
|--------|-------------|
| `--modes` | Retrieval modes to run: `bm25`, `hybrid`, or both (default: `bm25`). Invalid modes are rejected. |
| `--skip` | Skip a step if its output file(s) already exist (default: on). |
| `--no-skip` | Run every step even if outputs exist. |
| `--k` | k for Recall@k and SupportScore@k in evaluation (default: 5). |
| `--notes` | Path to reduced notes TSV (default: `data/notes-small.reduced.tsv`). Use `data/notes-large-helpful.reduced.tsv` to run on the helpful-only subset (from `data/recentHelpful.py`). |

The script exits with an error if the notes file is missing (create with `reduce_tsv.py` or `data/recentHelpful.py`; see Utility below).

### Step-by-step (manual)

Run these steps in order. Inputs and outputs are under `data/` and `results/`.

| Step | Command | What it does |
|------|--------|----------------|
| **1. Preprocess** | `python preprocess.py` | Reads `data/notes-small.reduced.tsv`, fetches cited Wikipedia article text via the MediaWiki API, chunks text → `data/passages/passages.jsonl` and `data/notes_filtered.parquet`. |
| **2. Retrieve** | `python retrieve.py --mode bm25` | Builds BM25 index (if needed), retrieves top-10 passages per note → `results/bm25_results.jsonl`. |
| | `python retrieve.py --mode hybrid` | Same, but BM25 + dense retrieval with RRF merge → `results/hybrid_results.jsonl`. |
| **3. NLI scoring** | `python test.py` | Scores each (query, passage) with RoBERTa-large-MNLI → adds `nli_score` to each row. Default input: `results/bm25_results.jsonl` → output: `results/bm25_nli_results.jsonl`. |
| | `python test.py -i results/hybrid_results.jsonl` | Same for hybrid results → `results/hybrid_nli_results.jsonl`. |
| **4. Evaluate** | `python evaluate.py` | Computes **Recall@k** and **SupportScore@k**. Default: reads `results/bm25_nli_results.jsonl`, uses `--k 5`. Prints the two scores. |
| | `python evaluate.py -i results/hybrid_nli_results.jsonl --k 10` | Evaluate hybrid NLI results with k=10. |
| **Gemini baseline** | `python gemini_retrieve.py` | Queries Gemini 2.0 Flash (Vertex AI) for top-5 Wikipedia URLs per note → `results/gemini_results.jsonl`. No passage text; Recall@k only. |
| | `python evaluate.py -i results/gemini_results.jsonl --k 5` | Evaluate Gemini results (SupportScore will be 0 — no NLI scores). |

### Quick copy-paste

**Single command (BM25 only):**
```bash
python run_pipeline.py
```

**Single command (both modes):**
```bash
python run_pipeline.py --modes bm25 hybrid
```

**Large helpful notes (recommended for full runs):**
```bash
python run_pipeline.py --notes data/notes-large-helpful.reduced.tsv --modes bm25 hybrid
```

**Manual — BM25 only:**
```bash
python preprocess.py
python retrieve.py --mode bm25
python test.py
python evaluate.py
```

**Manual — Hybrid (BM25 + dense):**
```bash
python preprocess.py
python retrieve.py --mode hybrid
python test.py -i results/hybrid_results.jsonl
python evaluate.py -i results/hybrid_nli_results.jsonl
```

### Evaluation metrics

- **Recall@k**: Fraction of notes where at least one of the top-k *sources* returned is a cited (gold) URL. Gold URLs come from the note’s `summary`; `evaluate.py` reads `data/notes_filtered.parquet` to get them.
- **SupportScore@k**: Mean NLI entailment score over the top-k passages per note (how well excerpts support the claim), using the `nli_score` values already in the NLI results file.

### Benchmark results

BM25, Hybrid, and Hybrid+Rerank are evaluated on the 100-note test set. Gemini is a side experiment evaluated on a 1,000-note subset (full 14k-note run is infeasible at ~1.2s/note via Vertex AI).

| System | Notes | Recall@5 | SupportScore@5 |
|--------|-------|----------|----------------|
| BM25 | 100 | 0.8586 | 0.6233 |
| Hybrid (BM25 + dense RRF) | 100 | 0.8900 | 0.6238 |
| Hybrid + Rerank | 100 | 0.8700 | 0.6190 |
| Gemini 2.0 Flash (parametric) | 1,000 | 0.3980 | 0.0000 |

Gemini SupportScore is 0 — it returns URLs only (no passage text for NLI scoring).

Optional args: `evaluate.py --input <path> --k <int> --notes <parquet>`.

## Data and results layout

| Path | Description |
|------|-------------|
| `data/notes-small.reduced.tsv` | Default input notes (small test set, ~101 rows). |
| `data/notes-large-helpful.reduced.tsv` | Helpful-only subset (from `recentHelpful.py`); use with `--notes` for full runs. |
| `data/notes_filtered.parquet` | Notes used by retrieval + evaluation (from preprocess). |
| `data/passages/passages.jsonl` | Chunked passages from MediaWiki API article text. |
| `data/bm25_index/` | Lucene index (built by retrieve). |
| `results/bm25_results.jsonl` | BM25 retrieval output (passage-level). |
| `results/hybrid_results.jsonl` | Hybrid retrieval output. |
| `results/*_nli_results.jsonl` | Retrieval results + `nli_score` (from test.py). |
| `results/gemini_results.jsonl` | Gemini parametric retrieval output. Fields: `note_id`, `query`, `rank`, `source_url`. |

## Utility

Reduce a raw Community Notes TSV to rows with Wikipedia links and key columns:
```bash
python reduce_tsv.py data/notes.tsv -o data/notes.reduced.tsv
```

To use only notes currently rated helpful (for the large set):
```bash
python data/recentHelpful.py   # reads notes-large.reduced.tsv + noteStatusHistory, writes notes-large-helpful.reduced.tsv
python run_pipeline.py --notes data/notes-large-helpful.reduced.tsv
```

## Troubleshooting

- **SIGBUS / Java crash on Apple Silicon**: If `retrieve.py` crashes with a JVM CodeHeap error, try Eclipse Temurin (ARM64) and ensure `JAVA_HOME` and `PATH` point to it. If it still fails, see in-repo notes on running under Rosetta or building the index on another machine.
