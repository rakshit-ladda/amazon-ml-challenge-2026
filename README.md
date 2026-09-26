# Amazon ML Challenge 2026 — sv1 (experimental: larger models)

v2 pipeline with the retrieval and reranking models scaled up. Not yet validated or submitted.

## Methodology

1. Normalise names and addresses (transliterate to ASCII, lowercase, expand abbreviations, split legal forms).
2. Fine-tune a BGE-M3 bi-encoder (568M, MIT) on fold-A training matches (batch 512, in-batch negatives).
3. Retrieve top-10 Source 1 candidates per Source 2/3 record by embedding similarity (plus TF-IDF char-3gram top-3).
4. Compute pair features and train a stage-1 LightGBM matcher.
5. Keep the top-3 candidates per record and score them with a fine-tuned Qwen2.5-1.5B cross-encoder (Apache-2.0, sequence classification head).
6. Experimental: Qwen2.5-7B-Instruct cross-encoder with LoRA adapters (r=8, q/v projections) in `src/crossenc_v3.py`.
7. Add house-number, extra-name-word and legal-form features; train a stage-2 LightGBM.
8. Assign each Source 2/3 record to its best Source 1 match if score >= threshold (tuned for macro F0.5).

Run order: see `src/RUN.md`.
