# Amazon ML Challenge 2026 — v3b (public LB 0.9852)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches.
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features and train a stage-1 LightGBM matcher.
5. Keep the top-3 candidates per record and score them with fine-tuned cross-encoders.
6. Add house-number, extra-name-word and legal-form features; train a stage-2 LightGBM.
7. Pseudo-label confident France test pairs and fine-tune the small cross-encoder (as in v3a).
8. Train a larger multilingual-e5-base cross-encoder and fine-tune it on the same France pseudo-labels.
9. Stage 2 uses both cross-encoders; assign each record to its best match if score >= 0.50.

Run order: see `src/RUN.md`.
