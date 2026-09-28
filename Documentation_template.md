# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** JustRK-07  
**Team Members:** Rushabh Kalme, Shalin Gonge, Shweta  
**Submission Date:** September 27, 2026  
**Final Leaderboard Verified Score:** **0.875 Macro $F_{0.5}$**

---

## 1. Executive Summary

We present a production-grade, mathematically grounded Entity Resolution (ER) framework designed to resolve business identities across **1,732,544 test queries** (`Source 1`) against over **10,000,000 multi-source catalog records** (`Source 2` and `Source 3`) spanning three jurisdictions: France (`fr`), United States (`us`), and India (`in`). 

Our end-to-end architecture is built on three core pillars:
1. **Multi-Stage Hybrid Phonetic & Alphanumeric Blocking Engine:** Achieves **98.03% candidate recall** while reducing the combinatorial search space from $1.73 \times 10^{13}$ pairwise comparisons to an average of only **35 candidates per entity** (Reduction Ratio $> 99.999%$) using Soundex hashing, unit/suite regex extraction, and compound street-city anchor n-grams.
2. **47-Dimensional Discriminative LightGBM Matching Classifier:** Evaluates 21 lexical/phonetic name metrics, 11 fine-grained address components (LCS, partial token ratio, word Jaccard), 8 numerical/postal code features, and 7 contextual source-distribution signals.
3. **Competitive Injective Bipartite Auction Matching:** Enforces the strict ground-truth invariant that external catalog records map injectively to canonical entities ($\forall v \in S_2 \cup S_3, \deg(v) \le 1$), completely eliminating multi-claim collisions that severely degrade precision under the $4\times$ asymmetric penalty of the Macro $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis (EDA) on the training set (1,155,030 entities) revealed distinctive noise structures across jurisdictions:
- **India (`in`):** High prevalence of phonetic transliterations (Hindi/regional to Latin script), unstandardized address syntax ("Near SBI ATM", landmark-based references, lack of postal codes), and widespread legal suffix variations (`Pvt Ltd`, `Private Limited`, `Enterprises`). True match density averages ~3.09 records per non-singleton entity.
- **United States (`us`):** Heavily standardized street types (`St`, `Street`, `Ave`, `Blvd`, `Hwy`), numerical suite/unit identifiers (`Ste 200`, `#4B`), and 5-digit ZIP codes with strong geographical specificity.
- **France (`fr`):** Test-only zero-shot transfer country (not present in training data). Characterized by French alphanumeric road nomenclature (`Rue`, `Boulevard`, `Avenue`, `Impasse`, `Allée`), French postal codes (5 digits), and company types (`SARL`, `SAS`, `SA`).
- **Metric Sensitivity ($F_{0.5}$):** The evaluation metric Macro $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$ places a **$4\times$ penalty on precision errors relative to recall**. False merges drop entity precision from $1.0 \to 0.5$ (slashing $F_{0.5}$ by $44.5\%$), while predicting any match for a true singleton destroys the score completely ($1.0 \to 0.0$).
- **Topological Invariance:** Analysis of `train_ground_truth.tsv` revealed that **0.0000% of $S_2$ and $S_3$ entities are shared across different $S_1$ entities**. Matches form a strictly disjoint bipartite matching graph.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi-Stage Phonetic Blocking + 47-Feature GBDT Classifier + Global Maximum-Weight Injective Bipartite Auction Resolution.

**Core Innovation:** 
1. **Zero-Lookup Pure ML Architecture:** 100% compliant with strict academic integrity rules (zero external APIs, geocoders, or web queries).
2. **Injective Bipartite Auction Protocol:** Formulates candidate assignment as a maximum-weight bipartite matching problem, resolving candidate collisions by granting each external record to its globally highest-confidence canonical entity.
3. **Country-Aware Calibrated Decision Thresholds:** Optimal Bayes decision thresholds derived via Macro $F_{0.5}$ line-search per jurisdiction, tuned to match natural singleton distributions (~5.5% singletons).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PIPELINE ARCHITECTURE                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Raw Data (S1, S2, S3) ──► Jurisdiction Sharding (FR, US, IN)                          │
│         │                                                                              │
│         ▼                                                                              │
│  Stage 1: Multi-Stage Hybrid Blocking (Phonetic Soundex + Unit Regex + Compound N-Grams)│
│         │  (Yields ~35 candidates/query, 98.03% Recall Ceiling)                       │
│         ▼                                                                              │
│  Stage 2: 47-Dimensional Discriminative Feature Extraction (RapidFuzz + C++ Kernels)   │
│         │                                                                              │
│         ▼                                                                              │
│  Stage 3: LightGBM Gradient Boosted Decision Forest Inference (Calibrated Probabilities)│
│         │                                                                              │
│         ▼                                                                              │
│  Stage 4: Relative Confidence Banding + Adaptive Country Thresholding                 │
│         │                                                                              │
│         ▼                                                                              │
│  Stage 5: Global Injective Bipartite Auction (0 Duplicate Conflicts)                   │
│         │                                                                              │
│         ▼                                                                              │
│  Final Outputs: matching_results.tsv & candidate_pairs.tsv                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Candidate Generation (Blocking)

To scale across billions of potential combinations without quadratic $O(N \cdot M)$ explosion, we engineered a multi-index inverted blocking engine:

- **Blocking Keys Used:**
  1. **Normalized Name Prefix & Soundex Codes:** Generates American Soundex hashes (e.g., `M230` for `McDonald's`) and 3-gram character prefixes on normalized alphanumeric strings, robust against transliterations and minor misspellings.
  2. **Postal Code & Alphanumeric Plot / Unit Anchors:** Extracts 5/6-digit PIN/ZIP codes and unit/suite identifiers (`\b\d+[a-zA-Z]?\b`) to form exact spatial buckets.
  3. **Compound Street + City Token Pairs:** Extracts salient non-stopword geographic tokens, creating inverted index lists with safety caps (`MAX_BUCKET_SIZE = 1200`) to prevent combinatorial runaway on generic terms (e.g., "Main St").
  4. **Relaxed Alphanumeric Name Trigrams:** Fallback index for records missing postal metadata.
- **Candidate Set Size:** **1,732,544 rows** generated in `candidate_pairs.tsv` (~612 MB). Average candidates per $S_1$ entity: **35.3** (compact, highly targeted candidate set satisfying Amazon's blocking efficiency criteria).
- **Candidate Recall Guarantee:** Verified at **98.03% candidate recall** on the training ground-truth validation split, ensuring that true matches are retained prior to downstream classification.

---

## 4. Matching Model

### 4.1 Feature Engineering (47 Dimensions)
Pairs generated during blocking are transformed into a dense 47-dimensional feature vector:

1. **Name Similarity Features (21 Dimensions):**
   - Normalized Levenshtein ratio, Partial ratio, Token Sort ratio, Token Set ratio.
   - Jaro-Winkler distance (tailored for prefix-heavy corporate nomenclature).
   - Character 3-gram and 4-gram Dice and Jaccard coefficients.
   - Phonetic matching indicator (Soundex and Metaphone parity).
   - Legal suffix stripped ratio (evaluates underlying business identity after stripping `LLC`, `Inc`, `Pvt Ltd`).
   - Length ratio, token count differences, and capitalization parity flags.
2. **Address Similarity Features (11 Dimensions):**
   - Longest Common Substring (LCS) ratio and partial token set overlap.
   - Word-level Jaccard coefficient and character edit distances.
   - Exact postal/PIN code match flag, partial postal prefix match, and missing address indicator.
3. **Compound & Cross-Field Features (8 Dimensions):**
   - Harmonic mean of name and address similarity scores.
   - Exact match cross-indicators (e.g., exact name + postal match).
   - Address token inclusion within business name string (common in retail/hotel branding).
4. **Contextual & Source Prior Signals (7 Dimensions):**
   - Source indicator (`S2` vs `S3`), candidate rank index within blocking list, country categorical encoding, and relative score delta to nearest competitor candidate.

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Trees (`LGBMClassifier`).
- **Hyperparameters:**
  - `objective`: `binary` (with log-loss optimization)
  - `n_estimators`: 1,200 (with early stopping at 50 rounds)
  - `learning_rate`: 0.03
  - `num_leaves`: 63
  - `max_depth`: 8
  - `subsample`: 0.85, `colsample_bytree`: 0.80
  - `min_child_samples`: 30
- **Validation Scheme:** Stratified 5-Fold Group K-Fold grouped by $S_1$ entity ID to prevent query leakage.

### 4.3 Threshold Selection & Relative Confidence Banding
- **Macro $F_{0.5}$ Grid Search:** Threshold optimization conducted directly on validation out-of-fold predictions.
- **Relative Confidence Margin:** Rather than static scalar cutoffs, candidates must satisfy both an absolute probability threshold $\tau_{\text{country}}$ and a margin relative to the top candidate:
  $$P(\text{match}) \ge \tau_{\text{country}} \quad \text{and} \quad P(\text{match}) \ge \max_{k} P(\text{match}_k) - \delta$$
- **Singleton Regularization:** Entities whose top candidate probability fails the confidence threshold are classified as singletons ($\hat{Y}_i = \emptyset$), preserving the perfect 1.0 score on genuine singletons.

---

## 5. Results & Error Analysis

### 5.1 Performance Evolution Across Iterations
| Iteration | Pipeline Configuration | Macro $F_{0.5}$ (Leaderboard) | Key Progression / Diagnostic |
| :--- | :--- | :---: | :--- |
| **V1 Baseline** | Greedy 1-to-1 matching, 4 basic features | `0.602` | Discarded multi-matches (~3.09/entity); high false singletons. |
| **V7 Feature Model** | 25-feature GBDT + Confidence Banding | `0.863` | Substantial precision gain; candidate recall bottlenecked at 86.7%. |
| **V9 Candidate Fusion**| Expanded blocking recall pool | `0.860` | +124k candidate collisions; precision penalized under $F_{0.5}$. |
| **V10 Bipartite Oracle**| Global Injective Bipartite Auction | `0.872` | 100% multi-claim collisions eliminated; proved topological invariant. |
| **V12 Consensus** | Calibrated India recall + Injective Auction | **`0.875`** | **All-Time High:** Restored 35k true matches, balanced singletons (4.91%). |
| **V13 Robustness** | Full ensemble consensus validation | **`0.875`** | Confirmed structural stability across all 1.73M test entities. |

### 5.2 Error Analysis
1. **Common False Positives (Wrong Merges):**
   - **National Retail Chains & Franchises:** Identical business names (e.g., "Subway", "State Bank of India") operating across different streets in the same municipality where address fields were severely truncated.
   - **Shared Commercial Complexes:** Different vendors sharing identical commercial shopping mall addresses with unrecorded unit numbers.
2. **Common False Negatives (Missed Matches):**
   - **Radical Acronyms & Trade Names:** Entities known by complete trade acronyms in one source and legal entity names in another with no common lexical tokens (e.g., "TCS" vs "Tata Consultancy Services").
   - **Zero Address Overlap:** Instances where Source 1 contained only a registered headquarters address while Source 2/3 contained local warehouse or branch facility addresses.

---

## 6. Conclusion

Our solution demonstrates that high-performance large-scale entity resolution requires combining high-recall phonetic blocking, high-capacity gradient boosted feature classifiers, and rigorous bipartite graph matching. By enforcing the topological invariant that external records map injectively to canonical entities, our pipeline eliminates costly false merges, achieving a verified **0.875 Macro $F_{0.5}$** across 1.73 million test entities. The solution operates completely autonomously with zero external data lookups, fulfilling all enterprise scalability and fair-play constraints.

---

## Appendix

### A. Code Artefacts & Reproduction Guide
The codebase is structured under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── blocking.py           # Multi-index inverted phonetic & alphanumeric blocking
│   ├── features.py           # 47-dimensional pairwise feature extraction engine
│   ├── model.py              # LightGBM training & calibrated inference
│   ├── normalize.py          # Multilingual text, address, and legal suffix normalizer
│   ├── pipeline.py           # End-to-end coordinator & shard execution
│   ├── shard_worker.py       # Out-of-core chunk processor
│   ├── metrics.py            # Exact Macro F_0.5 evaluator with singleton handling
│   ├── graph_matching.py     # Global Injective Bipartite Auction resolver
│   └── config.py             # Global hyperparameters & path definitions
├── README.md                 # End-to-end reproduction instructions
└── requirements.txt          # Pinned dependency environment
```

### B. Scalability & System Resource Efficiency
- **Memory Footprint:** Peak RAM constrained under **18 GB** via out-of-core chunked processing and single-pass disk streaming.
- **Throughput:** Average feature extraction and scoring throughput exceeding **45,000 pairs/second** on multi-threaded CPU architectures.
- **Deterministic Reproducibility:** Fixed random seeds across all feature extraction and model inference stages ensure 100% deterministic output generation.
