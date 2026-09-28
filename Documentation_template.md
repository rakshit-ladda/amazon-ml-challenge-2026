# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Ctrl Alt Del  
**Team Members:** Abdul Malik, Rakshit Ladda, Samarth Bhandegaonkar, Ashwin Rathore — Indian Institute of Information Technology (IIIT), Nagpur  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We find candidates with a fine-tuned multilingual bi-encoder plus character TF-IDF, and then decide matches with a two-stage LightGBM model. Its strongest inputs are cross-encoders that read both records together: multilingual-e5-small, multilingual-e5-base, and a LoRA-tuned Qwen2.5-1.5B whose score is averaged over four training runs. Each Source 2/3 record is then given to at most one Source 1 business.

Three things made the difference. The first was cross-encoders trained on the retriever's own hard negatives, which is what finally separated noisy true copies from near-copy decoys. The second was keeping train and test consistent: we dropped features that behave differently on the denser test set, and self-trained on France, which appears only in test. The third was treating France separately at the end, with its own threshold.

The final model scores only **5.14 candidates per Source 1 entity**. Final public leaderboard score: **0.987462**.

---

## 2. Methodology

### 2.1 Problem Analysis

Before building anything we read a lot of records side by side. A few observations shaped everything after that.

- **Scale.** Train has 2.2M S1 and 10.3M S2+S3 records; test has 1.73M S1 and 10.0M S2+S3.
- **One S1 per record.** Every S2/S3 record belongs to at most one S1 business: all 7.64M matched IDs in the training ground truth are unique. An S1 has 3.5 matches on average, and 5.6% have none. So the real question for each S2/S3 record is "which S1 is this, if any?", and we built the final decision around that.
- **Noise on true copies.** Typos and garbled numbers (11345 became 91345), filler words added to names (Partners, Center, Services), address parts in a different order, invented replacement names, empty addresses, and names in Devanagari, Bengali or Kannada script.
- **Decoys.** About 26% of train S2/S3 records match nothing (about 40% in test, roughly 1.9 times as many per S1). They are near-copies of a real business with one *real* change: the house number moves by one (159 became 160), a real word is added ("Aligarh Leasing **Group**", "Cultural Fund **Southside**"), or the legal form changes (Pvt Ltd became Limited). String similarity alone rates these almost as high as true copies.
- **France.** It appears only in test (about 15% of S1). An adversarial classifier separates it from training data almost perfectly (AUC 0.9995, against 0.69 for US and India). French addresses use départements where train would use a region (Nord rather than Hauts-de-France), carry French filler words, and usually have no postcode.
- **Density shift.** Test has about 5.75 S2/S3 records per S1, against 4.68 in train, so any feature that counts competing records means something different on test.

Because a business with no true match only scores 1.0 when we predict nothing for it, one wrong link costs a whole entity. We tuned everything for precision first.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier. It is a hybrid of dense and sparse retrieval, gradient boosting, and transformer cross-encoders.  
**Core Innovation:** Cross-encoders trained on the retrieval stage's own hard negatives, decoys included, used as features for a gradient-boosted matcher. Around that: France self-training, removing features that shift between train and test, a France-specific threshold, and an assignment step that respects "each S2/S3 record belongs to at most one S1".

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. **Normalised text.** Everything is transliterated to ASCII, lower-cased and stripped of punctuation, common abbreviations are expanded (Rd to road, Pvt to private), and the legal form is split into its own field.
  2. **Fine-tuned multilingual-e5-small bi-encoder** (MIT, 118M). We trained it with MultipleNegativesRankingLoss, batch 512, using same-country records as in-batch negatives, on fold-A matches only. Each S2/S3 record takes its 10 nearest S1 records within its own country (exact GPU search).
  3. **Character 3-gram TF-IDF** on the name, the address and name+address, with very common 3-grams dropped (`max_df = 0.01`), top 3 per field. This catches heavily misspelt records that the dense model misses.
  4. **Stage-1 LightGBM** on about 40 cheap pair features. Each S2/S3 record keeps its top 3 S1 candidates.
  5. **Final filter.** Only candidates with a stage-1 probability of at least 0.005 reach the final model.
- **Candidate pairs generated (test):** 8,906,044, which is **5.14 per Source 1 entity**, down from 17.3 before the final filter. All possible within-country pairs would be about 6.7 × 10¹², so the reduction ratio is about 99.9999%. `candidate_pairs.tsv` is exactly the set the final model runs inference on.
- **How we ensured true matches were not lost:** We measured recall on held-out S1 entities that the retrieval never trained on. Bi-encoder recall@1 / @5 / @10 / @20 is 97.4% / 98.8% / 99.1% / 99.3%, TF-IDF alone 93.8%, and the union 99.2%. After the stage-1 top 3 and the final filter, 98.97% of true links are still there. Perfect decisions on this candidate set would score 0.9968, against our 0.9914, so most of the remaining loss is in matching rather than blocking. Most of the links that blocking misses are records with **no address and a name shared by several S1 businesses**, which no retriever can tell apart. We also tried a much larger BGE-M3 bi-encoder (568M) and it gave the same recall (99.07% against 99.09%), so we kept the small one.

---

## 4. Matching Model

**Features used:**
- **Name features:** fuzzy ratio, token-set, token-sort and partial ratio, and Jaro-Winkler, on both the normalised core name and the raw name; legal-form agreement flags; bi-encoder cosine and rank. We also learned **extra-word weights** on the training folds. For a word that appears in only one of the two names, the weight says how likely it is to be harmless filler ("Partners", "Center") rather than a real difference ("Southside", "Midtown", a transliterated word). This was one of our more useful hand-made features against decoys.
- **Address features:** token-set, ratio and partial similarity; Jaccard over digit runs; house-number relations (edit distance, numeric difference, an off-by-one flag, whether the first or any number agrees); empty-address flags; TF-IDF cosines.
- **Other:** the three cross-encoder scores (below) with their gap to the record's best candidate and their rank; the stage-1 probability, its gap and rank; the number of candidates. We deliberately **removed** competitor-count features ("how many records point at this S1"). They depend on corpus density, which differs between train and test, and removing them improved validation by 0.0009 and the leaderboard by 0.0006.

**Model type:**
- **Stage 1:** LightGBM (127 leaves, 300 rounds) on all blocked pairs.
- **Cross-encoders.** These read both records together. All are trained on the bi-encoder's top 3 per record, so they see the true match next to the hard negatives and decoys that retrieval really produces.
  - **multilingual-e5-small and multilingual-e5-base** (MIT), one epoch on 1.2M to 1.5M queries. Each was then fine-tuned on France pseudo-labels: our own confident test predictions, p ≥ 0.9 as positive and p ≤ 0.05 as negative, using test inputs only and never labels. Adding the first cross-encoder moved the leaderboard from 0.956 to 0.982, our biggest single step.
  - **Qwen2.5-1.5B** (Apache-2.0) as a sequence classifier with **LoRA** (rank 16, alpha 32, all attention and MLP projections; 18.5M trainable parameters). It is trained on about 300k pairs, with the base in bf16 and the adapters in fp32, and it scores the uncertain pairs, those whose best stage-2 probability is between 0.05 and 0.95. We trained it **four times** on different samples of training queries (seeds 0, 101, 102, 103) and average the four scores into one feature. The runs agree closely (correlation about 0.975), but averaging removes run-to-run noise, and it gained +0.0002 on the leaderboard.
- **Stage 2:** LightGBM (63 leaves, 500 rounds) on the stage-1 features, the cross-encoder scores and the hand features. It is trained on held-out S1 entities that none of the neural models saw.
- **Assignment:** each S2/S3 record goes to its single most probable S1 candidate if p ≥ threshold, and otherwise to nobody. One S1 can receive many records.

**Threshold selection method:** For US and India, the threshold that maximises macro F0.5 (counting empty lists correctly) on the tune half of our held-out S1 set, reported on the untouched lockbox half. That is **0.50**. The ensemble's tune half slightly preferred 0.45, but lowering the threshold had cost us on the leaderboard before, so we kept 0.50. Separate thresholds for no-address or common-name records gave nothing (+0.00003).

**France is different.** It has no labels, so its threshold can't be tuned locally. We moved **only** the French threshold in leaderboard probes, keeping the predictions fixed: 0.5 → 0.6 → 0.7 gave 0.987183 → 0.987378 → 0.987462, with the gain shrinking each step. The model was over-confident on French pairs, so France uses **0.70**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation 0.9914, lockbox 0.9913 (US and India held-out S1 entities); **public leaderboard 0.987462**.

| Version | Change | Public LB |
|---|---|---|
| v1 | Bi-encoder + TF-IDF blocking, LightGBM | 0.956 |
| v2 | + e5-small cross-encoder | 0.9818 |
| v3a | + France self-training | 0.9839 |
| v3b | + e5-base cross-encoder | 0.9852 |
| v3c | − density-dependent competitor counts | 0.9858 |
| v3q | + Qwen2.5-1.5B LoRA cross-encoder | 0.9869 |
| pruned | + candidate filter before the final model (17.3 → 5.14 candidates / S1) | 0.986975 |
| qens | + four Qwen runs averaged into one feature | 0.987183 |
| **final** | + France-only threshold 0.7 | **0.987462** |

- **Common false positives (wrong merges):** 62% involve a record with **no address** whose name matches several S1 businesses ("Optimal Ventures Co." with no address, for example). Most of the rest are decoys with small but real changes, such as completely different house numbers or letter-suffix units (1083A against 1083B).
- **Common false negatives (missed matches):** records with no address make up 69% of the misses where the right candidate was present, and 91% of the misses that blocking never retrieved. A typical case is "THE DENT PUB CENTER" against "The Dent Pub Center" when other S1 businesses share the same name. Most of the others are heavily garbled names or numbers.
- **The validation–leaderboard gap.** Validation stayed about 0.004 above the leaderboard, and we believe most of that is France, which our validation never contains. On test the model is uncertain on 3.7–4.0% of French records, against 1.6–2.0% for US and India, and the France-only threshold probes above moved the score while US and India were untouched.

---

## 6. Conclusion

Most of our accuracy came from cross-encoders that read both records together, trained on the retriever's own hard negatives and stacked into a gradient-boosted matcher. That took the leaderboard from 0.956 to above 0.98. Self-training on the unseen country, removing features that shift between train and test, and averaging runs gave the rest.

Our main lesson is that at this level the local validation couldn't see the distribution shift: a new country, and denser, more decoy-heavy test data. Changes that kept train and test consistent, or only reduced variance, transferred to the leaderboard. Larger models (Qwen 7B, a 568M bi-encoder) and extra stacking layers looked fine locally and didn't. The biggest remaining loss is France. With more time, we would build a French-aware normaliser (départements, "av."/"bd" abbreviations, French filler words) applied before blocking and training, and add pseudo-labelled French records to validation so this error is visible when decisions are made. We did not build this during the challenge.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `run_final.sh` is the single entry point. It runs data → normalisation → blocking → stage 1 → cross-encoders → France self-training → stage 2 (v3c) → four Qwen LoRA runs, averaged → candidate filter → final stage 2 with the France threshold → `output_final/{matching_results,candidate_pairs}.tsv`. Then it checks that matches are a subset of candidates, that no S2/S3 record is assigned twice, and that the official validator passes, and it writes a manifest (threshold, counts, hashes).
- `src/` holds all the source (a file-by-file table is in `README.md`); `requirements.txt` pins the versions (torch 2.11 cu128, transformers, peft, lightgbm, polars, and so on).
- Seeds are fixed (`src/seeds.py`) and LightGBM runs deterministically. `output/` contains the exact files we uploaded to the leaderboard.

### B. Additional Results

| Experiment (not in final) | Local lockbox | Public LB | Outcome |
|---|---|---|---|
| e5 cross-encoders retrained on 85% of data + Qwen + Qwen 7B | 0.9915 | 0.98678 | worse on LB |
| Qwen 1.5B + Qwen2.5-7B LoRA | 0.9913 | 0.98665 | worse on LB |
| France-only feature-safe stage 2 | — | 0.98566 | worse on LB |
| Stage-3 cascade of extra e5 cross-encoders on uncertain queries | 0.9901 | 0.98524 | no gain |
| Harder candidate filter (p ≥ 0.01, 4.81 candidates / S1) | 0.9912 | 0.98655 | worse on LB |
| BGE-M3 568M bi-encoder | recall 99.07% vs 99.09% | — | no gain |
| LightGBM + XGBoost + CatBoost blend | 0.9908 | — | no gain |
| Sibling-consistency, name-frequency, house-number/joined-name features | ≤ +0.0001 | — | noise |
| Third round of France self-training (e5-small) | 0.99127 | — | 99.7% of French links unchanged; not used |

Blocking recall on held-out S1 entities: bi-encoder recall@1 / @5 / @10 / @20 = 97.4% / 98.8% / 99.1% / 99.3%; TF-IDF alone 93.8%.

**Models and licences:** multilingual-e5-small (MIT, 118M), multilingual-e5-base (MIT, 278M), Qwen2.5-1.5B (Apache-2.0, 1.54B), LightGBM (MIT); all under 8B. **Data:** only the provided files; no external data, APIs, geocoding or lookups. **Hardware:** NVIDIA RTX PRO 6000 Blackwell in 24 GB MIG slices, 24 CPU cores, 377 GB RAM.
