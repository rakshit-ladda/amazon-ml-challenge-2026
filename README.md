# Amazon ML Challenge 2026 — v4d / candidate D (public LB 0.986648; lockbox 0.9913)

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a multilingual-e5-small bi-encoder on training matches (fold A).
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Stage-1 LightGBM on pair features without density-dependent competitor counts (v3c fix); keep top-3 per record.
5. Score top-3 pairs with two fine-tuned cross-encoders (multilingual-e5-small, multilingual-e5-base), both France pseudo-label tuned.
6. Stage-2 LightGBM with house-number, extra-name-word and legal-form features.
7. Keep v3c's cross-encoders (v3q base). Add Qwen2.5-1.5B LoRA scores on the wide uncertain band (0.01-0.99) and Qwen2.5-7B LoRA scores (0.05-0.95) as stage-2 features.
8. Refit stage 2; assign best match if score >= 0.45.

Notes: v4d (D) | v3c stage 2 (original e5 CEs, = v3q base) + Qwen2.5-1.5B LoRA (wide band 0.01-0.99) + Qwen2.5-7B LoRA (narrow band 0.05-0.95) | t=0.45 | val 0.9913, lockbox 0.9913 (v3q 0.9912/0.9911) | LB=0.986648 (public, below v3q 0.986853)

Run order: see `src/RUN.md`.
