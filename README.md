# Amazon ML Challenge 2026 — v4f / candidate C (public LB 0.98678)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches (fold A).
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Stage-1 LightGBM on pair features without density-dependent competitor counts (v3c fix); keep top-3 per record.
5. Score top-3 pairs with two fine-tuned cross-encoders (multilingual-e5-small, multilingual-e5-base), both France pseudo-label tuned.
6. Stage-2 LightGBM with house-number, extra-name-word and legal-form features.
7. Retrain both e5 cross-encoders on ~85% of the businesses (all except the held-out set E) and rescore all stage-2 pairs.
8. Add Qwen2.5-1.5B (uncertain band 0.01-0.99) and Qwen2.5-7B LoRA (band 0.05-0.95) scores as stage-2 features; refit; threshold 0.45.

Notes: v4f (C) | v3c stage 2 + e5-small/e5-base CEs retrained on 85% (all S1 except E) + Qwen2.5-1.5B LoRA (wide band) + Qwen2.5-7B LoRA (narrow band) | t=0.45 | val 0.9915, lockbox 0.9915 (v3q 0.9912/0.9911) | LB=0.98678 (public, below v3q 0.986853 despite +0.0004 lockbox)

Run order: see `src/RUN.md`.
