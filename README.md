# Amazon ML Challenge 2026 — v3a (public LB 0.9838)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches.
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features and train a stage-1 LightGBM matcher.
5. Keep the top-3 candidates per record and score them with fine-tuned cross-encoders.
6. Add house-number, extra-name-word and legal-form features; train a stage-2 LightGBM.
7. Pseudo-label confident France test pairs from v2 (p >= 0.9 positive, p <= 0.05 negative) and fine-tune the small cross-encoder on them plus train pairs.
8. Rescore top-3 pairs with the France-tuned cross-encoder, refit stage 2, assign each record to its best match if score >= 0.45.

Run order: see `src/RUN.md`.
