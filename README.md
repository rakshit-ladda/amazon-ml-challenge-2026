# Amazon ML Challenge 2026 — v4 (public LB 0.9852)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches.
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features and train a stage-1 LightGBM matcher.
5. Keep the top-3 candidates per record and score them with fine-tuned cross-encoders.
6. Add house-number, extra-name-word and legal-form features; train a stage-2 LightGBM.
7. Pseudo-label France and train small + base cross-encoders; stage 2 uses both (as in v3b).
8. Second self-training round: new France pseudo-labels from v3b; fine-tune three cross-encoders in parallel (small, base, base France-heavy).
9. Stage-3 cascade: only uncertain queries (best stage-2 score between 0.01 and 0.99, ~5%) are rescored with the new cross-encoders and re-decided by a small LightGBM.
10. Assign each record to its best match if score >= 0.45; all other queries keep the stage-2 decision.

Run order: see `src/RUN.md`.
