import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import math
import argparse
import gc
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from src.config import (
    ROOT_DIR,
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_CANDIDATES, OUTPUT_MATCHING,
    MAX_CANDIDATES_PER_ENTITY, DEFAULT_THRESHOLD, SINGLETON_THRESHOLD
)
from src.normalize import normalize_business_name, normalize_address
from src.blocking import BlockingEngine, COMMON_GEO_WORDS
from src.features import extract_pair_features, extract_zips
from src.model import EntityResolutionModel

MODEL_PATH = os.path.join(ROOT_DIR, "models", "lgbm_er.pkl")
VALIDATOR_SCRIPT = os.path.join(ROOT_DIR, "utils", "validate_submission.py")


def format_time(seconds: float) -> str:
    """Formats seconds into human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"


def train_model(sample_size: int = 75000) -> EntityResolutionModel:
    """Trains V5 Champion LightGBM model with 47 deep features and clean rank-calibrated negative sampling."""
    print(f"\n=== Starting V5 Champion Model Training (Sample Size: {sample_size:,} S1 entities) ===", flush=True)
    t_start = time.time()

    print("Loading training S1 entities...", flush=True)
    s1_df = pd.read_csv(TRAIN_S1, sep="\t", nrows=sample_size)
    s1_ids = set(s1_df["entity_id"])

    # Load Ground Truth for this sample & gather all true external match IDs
    gt_map = {}
    known_match_ids = set()
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in s1_ids:
                if len(parts) > 1 and parts[1]:
                    matches = parts[1].split(",")
                    gt_map[parts[0]] = set(matches)
                    known_match_ids.update(matches)
                else:
                    gt_map[parts[0]] = set()

    print(f"Loaded ground truth: {len(gt_map):,} entities ({len(known_match_ids):,} known true positive external IDs).", flush=True)

    # Positive Guarantee Mining: Load external records ensuring ALL known matches are included
    print("Loading external records with Positive Guarantee Mining...", flush=True)
    ext_records = []

    # Read S2
    s2_count = 0
    for chunk in pd.read_csv(TRAIN_S2, sep="\t", chunksize=300000):
        mask = chunk["entity_id"].isin(known_match_ids)
        if s2_count < sample_size * 2:
            ext_records.append(chunk)
            s2_count += len(chunk)
        elif mask.any():
            ext_records.append(chunk[mask])

    # Read S3
    s3_count = 0
    for chunk in pd.read_csv(TRAIN_S3, sep="\t", chunksize=300000):
        mask = chunk["entity_id"].isin(known_match_ids)
        if s3_count < sample_size * 2:
            ext_records.append(chunk)
            s3_count += len(chunk)
        elif mask.any():
            ext_records.append(chunk[mask])

    ext_df = pd.concat(ext_records, ignore_index=True).drop_duplicates(subset=["entity_id"])
    del ext_records
    gc.collect()

    print(f"Fitting Compound Geo BlockingEngine on {len(ext_df):,} external records (max_candidates=50)...", flush=True)
    blocking = BlockingEngine(max_candidates=50)
    blocking.fit(ext_df)

    print("Normalizing external records cache (including postal codes & phone/nums)...", flush=True)
    ext_cache = {}
    for row in ext_df.itertuples(index=False):
        eid = row.entity_id
        raw_name = getattr(row, "business_name", "")
        raw_addr = getattr(row, "business_address", "")
        c_name, suff, _ = normalize_business_name(raw_name)
        c_addr, nums, _ = normalize_address(raw_addr)
        zips = extract_zips(raw_addr)
        ext_cache[eid] = (c_name, suff, c_addr, nums, zips)

    del ext_df
    gc.collect()

    print(f"Extracting 47-dimensional pairwise training features for {len(s1_df):,} entities...", flush=True)
    X_train = []
    y_train = []
    t_feat_start = time.time()
    total_s1 = len(s1_df)

    for idx, row in enumerate(s1_df.itertuples(index=False), start=1):
        s1_id = row.entity_id
        raw_name = getattr(row, "business_name", "")
        raw_addr = getattr(row, "business_address", "")
        s1_name, s1_suff, _ = normalize_business_name(raw_name)
        s1_addr, s1_nums, _ = normalize_address(raw_addr)
        s1_zips = extract_zips(raw_addr)

        candidates = blocking.query(raw_name, raw_addr)
        true_matches = gt_map.get(s1_id, set())

        for rank, cand_id in enumerate(candidates):
            if cand_id not in ext_cache:
                continue
            ext_name, ext_suff, ext_addr, ext_nums, ext_zips = ext_cache[cand_id]
            feat = extract_pair_features(
                s1_name, s1_suff, s1_addr, s1_nums, s1_zips,
                cand_id, ext_name, ext_suff, ext_addr, ext_nums, ext_zips, rank
            )
            is_match = 1 if cand_id in true_matches else 0
            X_train.append(feat)
            y_train.append(is_match)

        if idx % 5000 == 0 or idx == total_s1:
            el = time.time() - t_feat_start
            speed = idx / max(1e-3, el)
            rem = (total_s1 - idx) / speed if speed > 0 else 0
            pct = (idx / total_s1) * 100.0
            pos_pct = (np.mean(y_train) * 100.0) if y_train else 0.0
            print(
                f"  [Features] {idx:,} / {total_s1:,} ({pct:.1f}%) | "
                f"Pairs: {len(X_train):,} ({pos_pct:.2f}% pos) | "
                f"Speed: {speed:,.0f} ent/s | "
                f"Elapsed: {format_time(el)} | "
                f"ETA: {format_time(rem)}",
                flush=True
            )

    del s1_df, ext_cache, blocking
    gc.collect()

    X = np.array(X_train, dtype=np.float32)
    y = np.array(y_train, dtype=np.int32)
    print(f"Training dataset ready: {len(X):,} pairs ({np.mean(y)*100:.2f}% positives)", flush=True)

    model = EntityResolutionModel(
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=127,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_samples=35,
        match_threshold=0.80,
        singleton_threshold=0.980,
        delta=0.025,
    )
    print("Fitting V5 LightGBM ER Model (700 trees, 127 leaves, lr=0.04)...", flush=True)
    t_fit = time.time()
    model.fit(X, y)
    print(f"Fit completed in {time.time() - t_fit:.2f}s", flush=True)

    v5_path = os.path.join(ROOT_DIR, "models", "lgbm_er_v5.pkl")
    model.save(v5_path)
    model.save(MODEL_PATH)
    print(f"V5 Champion Model saved to {v5_path} and {MODEL_PATH} in {time.time() - t_start:.2f}s", flush=True)
    return model


def align_outputs_to_test_sequence():
    """Re-aligns matching_results.tsv and candidate_pairs.tsv to follow the exact line-by-line sequence of test_source1.tsv with strict Linux LF line endings."""
    print("\n=== Aligning Output Files to Exact test_source1.tsv Sequence (Strict Linux LF) ===", flush=True)

    ordered_ids = []
    with open(TEST_S1, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                ordered_ids.append(parts[0].strip())

    print(f"  Loaded {len(ordered_ids):,} Source 1 test IDs in exact order.", flush=True)

    # Re-align matching_results.tsv
    if os.path.exists(OUTPUT_MATCHING):
        match_map = {}
        with open(OUTPUT_MATCHING, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if parts:
                    match_map[parts[0]] = parts[1] if len(parts) > 1 else ""

        temp_match = OUTPUT_MATCHING + ".tmp"
        with open(temp_match, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for eid in ordered_ids:
                matches = match_map.get(eid, "")
                f.write(f"{eid}\t{matches}\n")

        os.replace(temp_match, OUTPUT_MATCHING)
        print(f"  [OK] matching_results.tsv aligned to exact test order with Linux LF.", flush=True)

    # Re-align candidate_pairs.tsv
    if os.path.exists(OUTPUT_CANDIDATES):
        cand_map = {}
        with open(OUTPUT_CANDIDATES, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t", 1)
                if parts:
                    cand_map[parts[0]] = parts[1] if len(parts) > 1 else ""

        temp_cand = OUTPUT_CANDIDATES + ".tmp"
        with open(temp_cand, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for eid in ordered_ids:
                cands = cand_map.get(eid, "")
                f.write(f"{eid}\t{cands}\n")

        os.replace(temp_cand, OUTPUT_CANDIDATES)
        print(f"  [OK] candidate_pairs.tsv aligned to exact test order with Linux LF.", flush=True)


def load_processed_ids(filepath: str) -> Set[str]:
    """Reads already processed S1 entity IDs to enable instant resumption."""
    processed = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.split("\t", 1)
                if parts and parts[0].strip():
                    processed.add(parts[0].strip())
    return processed


def run_test_inference(model: EntityResolutionModel, chunk_size: int = 50000, fresh: bool = False):
    """Generates matching_results.tsv and candidate_pairs.tsv streaming by country with live progress and auto-resumption."""
    print("\n=== Starting V7 Champion Test Inference by Country ===", flush=True)
    t_start = time.time()

    if fresh:
        print("  >>> FRESH RUN: Clearing output directory to generate brand-new V7 Champion predictions! <<<", flush=True)
        already_done = set()
        with open(OUTPUT_CANDIDATES, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(OUTPUT_MATCHING, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
    else:
        processed_candidates = load_processed_ids(OUTPUT_CANDIDATES)
        processed_matches = load_processed_ids(OUTPUT_MATCHING)
        already_done = processed_candidates & processed_matches

        if already_done:
            print(f"  >>> RESUMING: Found {len(already_done):,} entities already processed and saved! <<<", flush=True)
        else:
            with open(OUTPUT_CANDIDATES, "w", encoding="utf-8") as f_cand:
                f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            with open(OUTPUT_MATCHING, "w", encoding="utf-8") as f_match:
                f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # Calibrate V7 Champion parameters on model
    model.delta = 0.025
    model.match_threshold = 0.80
    model.singleton_threshold = 0.980
    model.country_thresholds = {
        "US": {"match": 0.80, "singleton": 0.980, "delta": 0.025},
        "France": {"match": 0.80, "singleton": 0.980, "delta": 0.025},
        "India": {"match": 0.80, "singleton": 0.980, "delta": 0.020},
    }

    print("Loading test_source1.tsv into memory for high-speed country streaming...", flush=True)
    t_load_s1 = time.time()
    s1_full = pd.read_csv(TEST_S1, sep="\t")
    total_expected_s1 = len(s1_full)
    country_totals = s1_full["country"].value_counts().to_dict()
    # Process smaller countries first for rapid streaming verification
    countries = sorted(country_totals.keys(), key=lambda c: country_totals[c])
    print(f"Loaded {total_expected_s1:,} test entities in {time.time() - t_load_s1:.2f}s across countries: {countries}", flush=True)

    total_processed = len(already_done)

    for country in countries:
        country_expected = country_totals.get(country, 0)
        print(f"\n==========================================", flush=True)
        print(f"Processing Country: {country} ({country_expected:,} entities)", flush=True)
        print(f"==========================================", flush=True)
        t_country = time.time()

        print(f"  Filtering test_source2 & test_source3 for '{country}'...", flush=True)
        s2_chunks = []
        for c in pd.read_csv(TEST_S2, sep="\t", chunksize=500000):
            sub = c[c["country"] == country]
            if not sub.empty:
                s2_chunks.append(sub)
        s2_count = sum(len(x) for x in s2_chunks)
        print(f"    Source 2 filtered: {s2_count:,} records", flush=True)
        s2_c = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()
        del s2_chunks

        s3_chunks = []
        for c in pd.read_csv(TEST_S3, sep="\t", chunksize=500000):
            sub = c[c["country"] == country]
            if not sub.empty:
                s3_chunks.append(sub)
        s3_count = sum(len(x) for x in s3_chunks)
        print(f"    Source 3 filtered: {s3_count:,} records", flush=True)
        s3_c = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()
        del s3_chunks

        ext_country = pd.concat([s2_c, s3_c], ignore_index=True).drop_duplicates(subset=["entity_id"])
        del s2_c, s3_c
        gc.collect()

        print(f"  Single-pass indexing & caching for {country} ({len(ext_country):,} external records, max_candidates=20)...", flush=True)
        t_fit = time.time()
        blocking = BlockingEngine(max_candidates=20)
        blocking.fit(ext_country)
        ext_cache = blocking.ext_cache
        del ext_country
        gc.collect()
        print(f"  Indexed & cached {len(ext_cache):,} records in {time.time() - t_fit:.2f}s", flush=True)

        # Get country test entities from in-memory DataFrame
        s1_country = s1_full[s1_full["country"] == country]
        if already_done:
            s1_country = s1_country[~s1_country["entity_id"].isin(already_done)]

        country_s1_count = 0
        t_stream = time.time()
        batch_size = 5000

        all_country_tuples = list(s1_country[["entity_id", "business_name", "business_address"]].fillna("").itertuples(index=False, name=None))

        def process_single_s1(row_tuple):
            s1_id, raw_name, raw_addr = row_tuple
            if s1_id in already_done:
                return s1_id, [], [], "", []

            s1_name, s1_suff, _ = normalize_business_name(raw_name)
            s1_addr, s1_nums, _ = normalize_address(raw_addr)
            s1_zips = extract_zips(raw_addr)

            candidates = blocking.query(raw_name, raw_addr)

            cand_ids_valid = []
            cand_suffixes_valid = []
            feats_valid = []

            for rank, cid in enumerate(candidates):
                if cid in ext_cache:
                    ext_name, ext_suff, ext_addr, ext_nums, ext_zips = ext_cache[cid]
                    f = extract_pair_features(
                        s1_name, s1_suff, s1_addr, s1_nums, s1_zips,
                        cid, ext_name, ext_suff, ext_addr, ext_nums, ext_zips, rank
                    )
                    cand_ids_valid.append(cid)
                    cand_suffixes_valid.append(ext_suff)
                    feats_valid.append(f)

            return s1_id, cand_ids_valid, cand_suffixes_valid, s1_suff, feats_valid

        with open(OUTPUT_CANDIDATES, "a", encoding="utf-8") as f_cand, open(OUTPUT_MATCHING, "a", encoding="utf-8") as f_match:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for chunk_start in range(0, len(all_country_tuples), batch_size):
                    batch_tuples = all_country_tuples[chunk_start : chunk_start + batch_size]
                    if not batch_tuples:
                        continue

                    results = list(executor.map(process_single_s1, batch_tuples))

                    chunk_items = []
                    chunk_all_feats = []
                    feat_idx = 0

                    for s1_id, cand_ids, cand_suffixes, s1_suff, feats in results:
                        if s1_id in already_done:
                            continue
                        start_pos = feat_idx
                        end_pos = feat_idx + len(feats)
                        chunk_all_feats.extend(feats)
                        feat_idx = end_pos
                        chunk_items.append((s1_id, cand_ids, cand_suffixes, s1_suff, start_pos, end_pos))

                    if not chunk_items:
                        continue

                    # Batch score all candidates simultaneously with OpenMP C++ across all threads
                    if chunk_all_feats:
                        chunk_probs = model.clf.predict_proba(np.array(chunk_all_feats, dtype=np.float32))[:, 1]
                    else:
                        chunk_probs = np.array([])

                    cand_lines = []
                    match_lines = []

                    for s1_id, cand_ids, cand_suffixes, s1_suff, start_pos, end_pos in chunk_items:
                        if start_pos < end_pos and len(chunk_probs) > 0:
                            probs = chunk_probs[start_pos:end_pos]
                            feats = chunk_all_feats[start_pos:end_pos]
                            matches = model.filter_candidate_matches(
                                cand_ids, probs, feats, country=country, s1_suffix=s1_suff, cand_suffixes=cand_suffixes
                            )
                        else:
                            matches = set()

                        cand_str = ",".join(cand_ids)
                        match_str = ",".join(sorted(matches))

                        cand_lines.append(f"{s1_id}\t{cand_str}\n")
                        match_lines.append(f"{s1_id}\t{match_str}\n")
                        country_s1_count += 1
                        total_processed += 1
                        already_done.add(s1_id)

                    f_cand.writelines(cand_lines)
                    f_match.writelines(match_lines)
                    f_cand.flush()
                    f_match.flush()

                    # Dynamic progress metrics
                    elapsed_country_stream = time.time() - t_stream
                    stream_speed = country_s1_count / max(1e-3, elapsed_country_stream)
                    country_remaining = max(0, country_expected - country_s1_count)
                    country_eta_s = country_remaining / max(1e-3, stream_speed)
                    overall_remaining = max(0, total_expected_s1 - total_processed)
                    overall_eta_s = overall_remaining / max(1e-3, stream_speed)
                    pct = (total_processed / total_expected_s1) * 100.0

                    print(
                        f"  [{country}] {country_s1_count:,} / {country_expected:,} ({country_s1_count/country_expected*100:.1f}%) | "
                        f"Speed: {stream_speed:,.0f} ent/s | Country ETA: {format_time(country_eta_s)} | "
                        f"Total: {total_processed:,}/{total_expected_s1:,} ({pct:.1f}%) | Est Total ETA: {format_time(overall_eta_s)}",
                        flush=True
                    )

        print(f"  Done {country}: finished in {time.time() - t_country:.2f}s", flush=True)
        del ext_cache, blocking
        gc.collect()

    del s1_full
    gc.collect()

    print(f"\nAll countries finished in {time.time() - t_start:.2f}s", flush=True)

    # Re-align outputs to match the exact row sequence of test_source1.tsv with Linux LF
    align_outputs_to_test_sequence()

    print(f"Outputs generated & strictly aligned:", flush=True)
    print(f"  - {OUTPUT_MATCHING}", flush=True)
    print(f"  - {OUTPUT_CANDIDATES}", flush=True)

    print("\n=== Running Official Submission Validator ===", flush=True)
    res = subprocess.run([
        sys.executable,
        VALIDATOR_SCRIPT,
        "--matching", OUTPUT_MATCHING,
        "--candidate", OUTPUT_CANDIDATES,
        "--test-dir", os.path.join(ROOT_DIR, "dataset", "test")
    ])
    if res.returncode == 0:
        print("\n>>> VALIDATION RESULT: PASS (Exit code 0). Ready for portal upload! <<<", flush=True)
    else:
        print(f"\n>>> VALIDATION RESULT: FAILED (Exit code {res.returncode}). Check output above. <<<", flush=True)


def main():
    parser = argparse.ArgumentParser(description="End-to-end Entity Resolution Pipeline")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--predict", action="store_true", help="Run test inference")
    parser.add_argument("--all", action="store_true", help="Train and predict")
    parser.add_argument("--fresh", action="store_true", help="Generate fresh predictions from scratch (overwrites output/)")
    parser.add_argument("--sample", type=int, default=50000, help="Training sample size")
    args = parser.parse_args()

    if args.train or args.all:
        model = train_model(sample_size=args.sample)
    elif os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...", flush=True)
        model = EntityResolutionModel.load(MODEL_PATH)
    else:
        print("No trained model found. Running training first...", flush=True)
        model = train_model(sample_size=args.sample)

    if args.predict or args.all:
        run_test_inference(model, fresh=args.fresh)


if __name__ == "__main__":
    main()
