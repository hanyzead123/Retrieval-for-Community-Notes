<!-- Pull Request template for retrieval + NLI improvements -->
## Summary
- Short description of changes (one line):

## Changes
- Files modified:

- Key behavior changes:

## Checklist
- [ ] `nli.py` updated to return logits and calibrated probabilities
- [ ] `retrieval.py` added with BM25+FAISS fusion and sentence selection
- [ ] `debug_nli.py` added/updated for sentence-level diagnostics
- [ ] Unit/manual checks performed (describe below)

## Testing
- How to reproduce locally:
```bash
python debug_nli.py
```

## Notes
- Any data or calibration required (e.g., temperature fitting):
