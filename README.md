# Amazon ML Challenge 2026 — v2 (public LB 0.982)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches.
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features and train a stage-1 LightGBM matcher.
5. Keep the top-3 candidates per record and score them with a fine-tuned cross-encoder.
6. Add house-number, extra-name-word and legal-form features; train a stage-2 LightGBM.
7. Assign each Source 2/3 record to its best Source 1 match if score >= 0.40 (tuned for macro F0.5).

Run order: see `src/RUN.md`.
