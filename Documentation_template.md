# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Ctrl Alt Del  
**Team Members:** Abdul Malik, Rakshit Ladda, Samarth Bhandegaonkar, Ashwin Rathore — Indian Institute of Information Technology (IIIT), Nagpur  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We retrieve candidates with a fine-tuned multilingual bi-encoder, score them with a two-stage LightGBM matcher fed by cross-encoders that read both records jointly (multilingual-e5-small/base and a LoRA-tuned Qwen2.5-1.5B), and assign every Source 2/3 record to at most one Source 1 entity under a threshold tuned directly on macro F0.5. The key ideas were (i) cross-encoders trained on hard negatives to separate noisy true copies from deliberate near-copy decoys, (ii) self-training on France (test-only country) with high-confidence pseudo-labels, and (iii) removing features whose meaning shifts between train and test. A final pruning step cuts the candidate set to **5.14 candidates per Source 1 entity** with no loss in validation accuracy. Public leaderboard: **0.987462**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train 2.2M S1 / 10.3M S2+S3 records; test 1.73M S1 / 10.0M S2+S3.
- **Structure:** every S2/S3 record belongs to at most one S1 entity (7.64M matched train IDs, all unique); an S1 has 3.5 matches on average, 5.6% have none.
- **Decoys:** ~26% of train S2/S3 records match nothing (~40% in test, about 1.9× more per S1). They are near-copies of a real business with one *real* change: house number ±1 (159 → 160), an added real word ("Aligarh Leasing **Group**", "Cultural Fund **Southside**"), or a different legal form (Pvt Ltd → Limited).
- **Noise on true copies:** typos and garbled numbers (11345 → 91345), filler words (Partners, Center, Services), reordered address components, invented replacement names, empty addresses, and names in Devanagari/Bengali/Kannada script.
- **France:** appears only in test (15% of S1). Adversarial validation shows it is fully out-of-distribution (AUC 0.9995 vs 0.69 for US/India): département vs région (Nord vs Hauts-de-France), French filler words, no postal codes.
- **Train/test density shift:** test has ~5.75 S2/S3 records per S1 vs 4.68 in train, so any feature that counts competitors behaves differently on test.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (hybrid: dense retrieval, gradient boosting, transformer cross-encoders).  
**Core Innovation:** transformer cross-encoders trained on the retrieval's own hard negatives (including decoys) as features for a gradient-boosted matcher; France self-training; systematic removal of train/test-inconsistent features; per-record assignment respecting the "each S2/S3 record belongs to ≤ 1 S1" structure.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. Fine-tuned **multilingual-e5-small bi-encoder** (MIT, 118M), trained contrastively (MultipleNegativesRankingLoss, batch 512, same-country in-batch negatives) on fold-A matches; each S2/S3 record takes its top-10 nearest S1 records within its country (GPU exact search).
  2. **TF-IDF character 3-grams** on name, address and name+address (sparse top-k with frequent 3-grams pruned, `max_df=0.01`); top-3 per field.
  3. **Stage-1 LightGBM** on ~40 cheap pair features keeps each record's top-3 candidates.
  4. **Final filter:** only candidates with stage-1 probability ≥ 0.005 go to the final model.
- **Candidate pairs generated (test):** 8,906,044 = **5.14 per Source 1 entity** (down from 17.3 before the final filter). All possible within-country pairs ≈ 6.7 × 10¹² → reduction ratio ≈ 99.9999%.
- **How we ensured true matches were not lost:** recall measured on held-out S1 entities the retrieval never trained on: bi-encoder recall@1 97.4%, @10 99.1%; bi-encoder ∪ TF-IDF 99.2%; after stage-1 top-3 and the final filter 98.97% of true links remain. Perfect decisions on this candidate set would score macro F0.5 0.9968, so the remaining loss is in decisions, not retrieval. Most missed links are records with **no address and a name shared by several S1 entities**, which no retriever can disambiguate. A 568M BGE-M3 bi-encoder gave identical recall (99.07% vs 99.09%).

---

## 4. Matching Model

**Features used:**
- **Name features:** fuzzy ratio / token-set / token-sort / partial ratio and Jaro-Winkler on the normalised core name (legal forms removed) and on the raw name; legal-form agreement flags; **extra-token weights**: log-odds, learned on training folds, that a word present in only one name is noise (invented filler words) rather than a real difference (Southside, Midtown, transliterated "svits"); bi-encoder cosine and rank.
- **Address features:** token-set / ratio / partial similarity; digit-run Jaccard; house-number relations (edit distance, numeric difference, off-by-one flag, first/any number equal); empty-address flags; TF-IDF cosines.
- **Other:** three cross-encoder scores (see below) and their per-query gap and rank; stage-1 probability, its gap to the query's best candidate and its rank; number of candidates. Competitor-count features ("how many records point at this S1") were **removed** because they depend on corpus density and shift between train and test (removal: validation +0.0009, leaderboard +0.0006).

**Model type:**
- **Stage 1:** LightGBM (127 leaves, 300 rounds) on all blocked pairs.
- **Cross-encoders** (read both records jointly, trained on the bi-encoder's top-3 per record, i.e. true matches plus hard negatives and decoys):
  - multilingual-e5-small (MIT) and multilingual-e5-base (MIT), 1 epoch on 1.2M–1.5M queries, each then fine-tuned on France pseudo-labels (high-confidence test predictions, p ≥ 0.9 positive / p ≤ 0.05 negative, two rounds);
  - **Qwen2.5-1.5B** (Apache-2.0) as a sequence classifier with **LoRA** (rank 16, α 32, all attention and MLP projections; 18.5M trainable parameters), ~300k pairs, bf16 base with fp32 adapters; scores the uncertain pairs (best stage-2 probability 0.05–0.95). Four runs on different query samples (seeds 0, 101, 102, 103) are averaged into one feature, which reduces run-to-run noise (+0.0002 on the leaderboard).
- **Stage 2:** LightGBM (63 leaves, 500 rounds) on stage-1 features + cross-encoder scores + hand features, trained on held-out S1 entities that none of the neural models saw.
- **Assignment:** each S2/S3 record goes to its single most probable S1 candidate if p ≥ threshold, else to nobody (an S1 can receive many records).

**Threshold selection method:** macro F0.5 (with the empty-list convention) maximised on the tune half of the held-out S1 set; reported on the untouched lockbox half. Final threshold fixed at 0.50 (the ensemble's tune half preferred 0.45, but lower thresholds had cost us on the leaderboard before); France, which has no labels and cannot be validated locally, uses 0.70: we moved only the France threshold in public-leaderboard probes (0.5 → 0.6 → 0.7: 0.987183 → 0.987378 → 0.987462), which shows the model is over-confident on the unseen country; separate thresholds for no-address or common-name records gave no gain.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation 0.9914, lockbox 0.9913 (US + India held-out S1 entities); **public leaderboard 0.987462**.

| Version | Change | Public LB |
|---|---|---|
| v1 | bi-encoder + TF-IDF blocking, LightGBM | 0.956 |
| v2 | + e5-small cross-encoder | 0.9818 |
| v3a | + France pseudo-label fine-tuning | 0.9839 |
| v3b | + e5-base cross-encoder | 0.9852 |
| v3c | − density-dependent competitor counts | 0.9858 |
| v3q | + Qwen2.5-1.5B LoRA cross-encoder | 0.9869 |
| pruned | + candidate pruning before the final model (17.3 → 5.14 candidates / S1) | 0.986975 |
| qens | + 4 Qwen LoRA runs averaged into one feature | 0.987183 |
| **final** | + France-only threshold 0.7 (0.6: 0.987378) | **0.987462** |

- **Common false positives (wrong merges):** 62% involve a query with **no address** whose name matches several S1 entities (e.g. "Optimal Ventures Co." with no address); the rest are decoys with small but real changes (house numbers differing entirely, letter-suffix units like 1083A/1083B).
- **Common false negatives (missed matches):** 69% of misses with a candidate present and 91% of never-retrieved misses are **no-address** records (e.g. "THE DENT PUB CENTER" vs "The Dent Pub Center" with competing same-name S1 entities); others are heavily garbled names or numbers.

---

## 6. Conclusion

Joint text readers (cross-encoders) trained on the retriever's hard negatives, stacked into a gradient-boosted matcher, gave most of the accuracy (+0.026 on the leaderboard over string features alone); self-training on the unseen country and removing train/test-inconsistent features gave the rest. Our main lesson: at this level local validation cannot see distribution shift (new country, 1.9× decoy density), so changes that keep train and test consistent, or that only reduce variance (averaging runs), transfer, and the largest remaining loss is France: raising only its threshold gained +0.0003, so a French-aware normaliser and a validation set with France are the next step, while larger models (Qwen 7B, a 568M bi-encoder) and extra stacking layers did not.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `run_final.sh` — single entry point: data → normalisation → blocking → stage 1 → cross-encoders → France self-training → v3c stage 2 → Qwen LoRA → candidate pruning → final stage 2 → `output_final/{matching_results,candidate_pairs}.tsv`, then checks (matches ⊆ candidates, no S2/S3 record assigned twice, official validator) and a manifest (threshold, counts, hashes).
- `src/` — all source (file-by-file table in `README.md`); `requirements.txt` — pinned versions (torch 2.11 cu128, transformers, peft, lightgbm, polars, …).
- Seeds fixed (`src/seeds.py`), LightGBM deterministic. `output/` contains the exact files we uploaded.

### B. Additional Results

| Experiment (not in final) | Local lockbox | Public LB | Outcome |
|---|---|---|---|
| e5 cross-encoders retrained on 85% of data + Qwen + Qwen 7B | 0.9915 | 0.98678 | worse on LB |
| Qwen 1.5B + Qwen2.5-7B LoRA | 0.9913 | 0.98665 | worse on LB |
| France-only feature-safe stage 2 | — | 0.98566 | worse on LB |
| Stage-3 cascade of extra e5 cross-encoders on uncertain queries | 0.9901 | 0.98524 | no gain |
| BGE-M3 568M bi-encoder | recall 99.07% vs 99.09% | — | no gain |
| LightGBM + XGBoost + CatBoost blend | 0.9908 | — | no gain |
| Sibling-consistency, name-frequency, house-number/joined-name features | ≤ +0.0001 | — | noise |
| Harder pruning (p ≥ 0.01, 4.81 cand/S1) | 0.9912 | 0.98655 | lower LB |
| France self-training round 3 (e5-small on pseudo-labels from the ensemble) | 0.99127 | — | 99.7% same France links; not used |

Blocking recall on held-out S1 entities: bi-encoder recall@1 / @5 / @10 / @20 = 97.4% / 98.8% / 99.1% / 99.3%; TF-IDF alone 93.8%.
