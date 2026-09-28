# Business Entity Resolution Pipeline

## Team Name: JustRK-07
## Competition: Amazon ML Challenge 2026 (72-Hour Hackathon)
## Verified Leaderboard Score: 0.875 Macro F_0.5

---

## 1. System Overview

This repository contains the complete, production-grade Entity Resolution pipeline for resolving business entities across heterogeneous catalogs (Source 1 reference queries matched against Source 2 and Source 3 external records) across France (`fr`), United States (`us`), and India (`in`).

The pipeline operates in three decoupled, highly scalable stages:
1. **Multi-Stage Inverted Phonetic & Alphanumeric Blocking (`src/blocking.py`):** Generates candidate pairs using Soundex codes, 3-gram prefixes, unit/suite regex extraction, and compound street-city n-grams.
2. **47-Dimensional Discriminative Feature Extraction (`src/features.py`):** Computes lexical, phonetic, token-set, address LCS, word Jaccard, postal code, and cross-field interaction features using high-speed RapidFuzz C++ kernels.
3. **Calibrated LightGBM Classifier & Injective Bipartite Auction Matching (`src/model.py` & `src/graph_matching.py`):** Predicts match probabilities and enforces maximum-weight bipartite matching to eliminate candidate collisions.

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── blocking.py           # Multi-index inverted phonetic & alphanumeric blocking
│   ├── features.py           # 47-dimensional pairwise feature extraction engine
│   ├── model.py              # LightGBM training, calibration & inference
│   ├── normalize.py          # Multilingual text, address & legal suffix normalizer
│   ├── pipeline.py           # End-to-end coordinator & shard execution
│   ├── shard_worker.py       # Out-of-core chunk processor
│   ├── metrics.py            # Exact Macro F_0.5 evaluator with singleton handling
│   ├── graph_matching.py     # Global Injective Bipartite Auction resolver
│   ├── cross_encoder.py      # Transformer cross-encoder architecture
│   ├── dense_retrieval.py    # Dense semantic embedding indexer
│   ├── evaluate.py           # Validation suite & error analysis
│   ├── fast_val_benchmark.py # Fast stratified validation benchmark
│   └── config.py             # Hyperparameters & path definitions
├── README.md                 # This reproduction guide
└── requirements.txt          # Pinned dependency environment
```

---

## 3. Environment Setup & Installation

The solution is implemented in Python 3.8+ (tested on Python 3.11 and 3.13) and relies exclusively on open-source, standard libraries:

```bash
# 1. Create and activate a clean virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt
```

---

## 4. End-to-End Execution Instructions

### A. Data Preparation
Ensure the competition dataset is placed in the standard directory layout:
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### B. Reproducing the Pipeline (Training to Inference)

To run the complete end-to-end pipeline (training the 47-feature LightGBM model, performing multi-stage blocking, extracting features, and executing the injective bipartite auction):

```bash
# Run end-to-end inference
python src/pipeline.py --train --predict --output-dir ../../output
```

This single execution generates both required submission artifacts in `output/`:
- `output/matching_results.tsv` — The final entity matches evaluated on the leaderboard.
- `output/candidate_pairs.tsv` — The candidate set produced by the blocking stage.

### C. Submission Validation
Verify the generated files against the official evaluation rules using the competition validator:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Expected output:
```text
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows
  candidate_pairs.tsv: 1732544 rows
PASS — no blocking issues found. Safe to submit.
```

---

## 5. Academic Integrity & Fair Play Statement

This solution strictly adheres to the competition rules:
- **Zero External Data Lookup:** No commercial APIs, web scrapers, external gazetteers, government databases, or geocoding services were used.
- **Model Licensing & Scale:** The LightGBM classifier complies with the MIT / Apache 2.0 open-source requirement and operates well below the 8 Billion parameter constraint.
- **Deterministic:** All random states are fixed (`random_state=42`) for 100% reproducibility.
