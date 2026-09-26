# Amazon ML Challenge 2026 — v3c (public LB 0.985845; validation 0.9909, lockbox 0.9908)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches.
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features and train a stage-1 LightGBM matcher.
5. Keep the top-3 candidates per record and score them with fine-tuned cross-encoders.
6. Add house-number, extra-name-word and legal-form features; train a stage-2 LightGBM.
7. Pseudo-label confident France test pairs and fine-tune the small cross-encoder (as in v3a).
8. Train a larger multilingual-e5-base cross-encoder and fine-tune it on the same France pseudo-labels.
9. Remove the density-dependent competitor-count features (s1_top1, s1_top1_be) from stages 1 and 2: they were computed on a query subset in training and test has ~1.9x more records per business.
10. Retrain stage 1 without them, refit stage 2 on the same pairs and cross-encoder scores (small + base, France-tuned); assign each record to its best match if score >= 0.45 (chosen on the tune half of validation).

Run order: see `src/RUN.md`.
