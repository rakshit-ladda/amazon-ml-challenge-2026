# Onboarding — Amazon ML Challenge 2026 (Business Entity Resolution)

Read this first. It covers the task, the rules, where we stand, and how the code is laid out.

## 1. The problem in one paragraph

We get business records from 3 sources. **Source 1 (S1)** is a clean, deduplicated reference list.
**S2 and S3** are noisy copies (typos, abbreviations, reordered addresses, names in Hindi/Bengali script,
missing addresses) plus **decoys**: near-copies of a real business with one real change
(house number 159 → 160, an extra word like "Southside", Pvt Ltd → Limited).
For every S1 business we must list which S2/S3 records are the same business (zero, one or many).

- Train: US + India, with ground truth (2.2M S1, 10.3M S2+S3 records).
- Test: US + India + **France** (unseen in training), 1.73M S1, 10M S2+S3, no labels.
- Each S2/S3 record belongs to **at most one** S1 business. ~26% of train S2/S3 records are decoys (~40% in test).

## 2. Scoring and rules

- **Macro F0.5 per S1 business**, averaged. Precision counts 2× recall. A business with no true match scores 1.0
  only if we predict nothing for it; any false match on it scores 0.
- Output 1: `matching_results.tsv` (one row per test S1, comma-separated matched IDs). This is what's scored.
- Output 2: `candidate_pairs.tsv` — the exact candidate set our final model scores.
  **New rule:** a smaller candidate set per S1 business ranks higher in the final evaluation.
- Final ranking uses the **private** leaderboard (rest of the test set), not the public one.
- Models must be **MIT or Apache-2.0** and **≤ 8B parameters**.
- **No external data or lookups** (no geocoding, registries, internet data) — instant disqualification.
- Always run `utils/validate_submission.py` before uploading.

## 3. Where we stand

| Version | What it adds | Public LB |
|---|---|---|
| v1 | bi-encoder retrieval + LightGBM | 0.956 |
| v2 | first cross-encoder (e5-small) | 0.9818 |
| v3a | France self-training (pseudo-labels) | 0.9839 |
| v3b | bigger cross-encoder (e5-base) | 0.9852 |
| v3c | removed train/test-inconsistent count features | 0.9858 |
| **v3q** | **+ Qwen2.5-1.5B LoRA cross-encoder as an extra feature** | **0.986853 (best)** |
| C / D / v4b7 | 85% e5 retrain, wider band, Qwen 7B | 0.98678 / 0.98665 / 0.98665 (worse) |

Top-50 cutoff is ~0.9877–0.9882. Local validation ("lockbox") can no longer rank the top candidates reliably —
the leaderboard disagreed with it twice. Treat lockbox gains ≤ +0.0002 as noise.

## 4. The pipeline (v3q)

1. **Normalise** (`normalize.py`, `prep.py`): transliterate to ASCII, lowercase, expand abbreviations, split legal forms.
2. **Candidates** (`biencoder.py`, `block.py`): fine-tuned multilingual-e5-small bi-encoder, top-10 nearest S1 per record,
   plus TF-IDF char-3gram top-3. Recall@10 = 99.1%.
3. **Stage 1** (`features.py`, `train.py`, `v3c.py`): LightGBM on ~40 pair features; keep each record's top-3 candidates.
4. **Cross-encoders** (`crossenc.py`, `crossenc2.py`): e5-small and e5-base read both records together (France pseudo-label tuned).
5. **Qwen** (`crossenc_llm.py`): Qwen2.5-1.5B LoRA scores the uncertain pairs.
6. **Stage 2** (`stage2.py`, `v3q.py` / `v3f.py`): LightGBM with cross-encoder scores, house-number, extra-word, legal-form features.
7. **Decision**: each S2/S3 record goes to its best S1 if score ≥ threshold (tuned on macro F0.5).

Validation discipline: neural models train on fold A (or everything except the held-out set E); stage 2 is
validated on E, split into a tune half and a **lockbox** half that is never used for choices.

## 5. What worked vs. what didn't

- **Worked:** stronger text readers added on top (cross-encoders, Qwen), fixing train/test mismatches.
- **Didn't:** bigger bi-encoder (BGE-M3, same recall), GBDT blends, sibling features, exact-name blocking,
  département rewrite, stage-3 cascades of e5 models, Qwen 7B, replacing the base cross-encoders.
- **Biggest remaining error:** records with **no address** (62% of false merges, ~70–90% of misses).

## 6. In progress right now

- Candidate pruning (stage-1 prob ≥ 0.005): candidates per S1 **17.3 → 5.1** with no validation loss — for the new rule.
- Name-frequency features for name-only records (`nkf.py`).
- Final submission package (zip: outputs, code, README, requirements, filled documentation template).

## 7. Branches

`v1`, `v2`, `v3a`, `v3b`, `v3c`, `v3d`, `v3q`, `v4`, `v4a`, `v4b7`, `v4d`, `v4f`: exact code of each version.
`sv1`: BGE-M3 + Qwen experiments. `qwen-ce-kit`: scoring kit. **`onboarding` (this branch): latest code of everything.**

Compute: the college Blackwell node (8 × 24 GB MIG slices, shared disk — check `df -h` before writing big files).
Ask the team for access; never share passwords or keys in chat.
