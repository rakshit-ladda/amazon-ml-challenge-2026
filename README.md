# Amazon ML Challenge 2026 — v3d (public LB 0.985657)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches (fold A).
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Stage-1 LightGBM on pair features without density-dependent competitor counts (v3c fix); keep top-3 per record.
5. Score top-3 pairs with two fine-tuned cross-encoders (multilingual-e5-small, multilingual-e5-base), both France pseudo-label tuned.
6. Stage-2 LightGBM with house-number, extra-name-word and legal-form features.
7. US/India pairs: v3c stage-2 model. France pairs: a separate stage-2 model without the 19 features that shift on France (adversarial AUC 0.9995), own threshold 0.40.
8. Assign each record to its best match if score >= threshold.

Notes: v3d | v3c for US/India; France pairs scored by a France-safe stage-2 model without the 19 features that shift on France (adversarial AUC 0.9995): p1/r1/p1_gap, extra-token weights+unknown counts, address-number counts, legal flags, runner-up cosine; own threshold t=0.40 | US/IN lockbox: France-safe model 0.9906 vs v3c 0.9908 | ~10k France links changed (+4874/-5255) | LB=0.985657 (public, below v3c 0.985845 -> France-safe model dropped)

Run order: see `src/RUN.md`.
