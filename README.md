# Amazon ML Challenge 2026 — v3q (public LB 0.986853, best so far)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches (fold A).
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Stage-1 LightGBM on pair features without density-dependent competitor counts (v3c fix); keep top-3 per record.
5. Score top-3 pairs with two fine-tuned cross-encoders (multilingual-e5-small, multilingual-e5-base), both France pseudo-label tuned.
6. Stage-2 LightGBM with house-number, extra-name-word and legal-form features.
7. Qwen2.5-1.5B LoRA cross-encoder (Apache-2.0; fold-A pairs + France pseudo-labels) scores the uncertain v3c pairs (best v3c score 0.05-0.95).
8. Add the Qwen score as an extra stage-2 feature; refit; assign best match if score >= 0.50.

Notes: v3q | v3c + Qwen2.5-1.5B LoRA cross-encoder (our e5 candidates, fold A, ~300k pairs incl 60k France pseudo) scoring v3c uncertain band as extra stage-2 feature | t=0.50 | val 0.9912, lockbox 0.9911 (v3c 0.9909/0.9908) | LB=0.986853 (public, +0.0010 over v3c; lockbox only showed +0.0003)

Run order: see `src/RUN.md`.
