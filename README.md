# Amazon ML Challenge 2026 — v1 (public LB 0.956)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches.
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features (embedding cosine/rank, fuzzy name/address scores, house-number overlap, candidate context).
5. Train a LightGBM matcher with 2-fold CV.
6. Assign each Source 2/3 record to its best Source 1 match if score >= 0.60 (tuned for macro F0.5).

Run order: see `src/RUN.md`.
