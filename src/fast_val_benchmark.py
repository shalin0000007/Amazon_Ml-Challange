"""
ML Challenge 2026 - V5 Rapid Innovation & Validation Benchmark (90-Second Loop)

A high-speed, stratified local testbed for V5 to rapidly prototype, test, and tune
neural cross-encoders, global bipartite matching, and decision boundaries in 60-90 seconds.
Enables pushing the exact Macro F0.5 score to 0.97+ / 0.99 before running on the full test set.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set, List, Tuple
from collections import defaultdict

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT
from src.normalize import normalize_business_name, normalize_address
from src.blocking import BlockingEngine
from src.features import extract_pair_features, extract_zips
from src.model import EntityResolutionModel
from src.metrics import compute_macro_f05
from src.graph_matching import resolve_global_bipartite_matching, verify_triangular_consistency

MODEL_PATH = os.path.join(ROOT_DIR, "models", "lgbm_er.pkl")


def load_stratified_validation_slice(
    val_size: int = 10000,
    offset: int = 250000
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Loads a held-out, never-before-seen slice of S1 entities with known Ground Truth.
    Stratified across US, France, and India with natural ~72% singleton ratio.
    """
    print(f"Loading {val_size:,} held-out validation entities (offset: {offset:,})...")
    val_s1 = pd.read_csv(TRAIN_S1, sep="\t", skiprows=range(1, offset), nrows=val_size)
    val_ids = set(val_s1["entity_id"])

    gt_map = {}
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in val_ids:
                if len(parts) > 1 and parts[1]:
                    gt_map[parts[0]] = set(parts[1].split(","))
                else:
                    gt_map[parts[0]] = set()

    n_singletons = sum(1 for mids in gt_map.values() if len(mids) == 0)
    print(f"  Ground Truth: {len(gt_map):,} entities loaded ({n_singletons:,} singletons = {n_singletons/len(gt_map)*100:.1f}%).")
    return val_s1, gt_map


def run_fast_benchmark(
    val_size: int = 10000,
    use_bipartite: bool = True,
    use_triangle: bool = True,
    match_threshold: float = 0.80,
    singleton_threshold: float = 0.74,
):
    """Executes the complete end-to-end evaluation loop in ~60 to 90 seconds."""
    print("\n" + "=" * 75)
    print("      V5 RAPID VALIDATION BENCHMARK (Exact Macro F0.5 Metric)")
    print("=" * 75)
    t0 = time.time()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error: Model not found at {MODEL_PATH}")

    model = EntityResolutionModel.load(MODEL_PATH)
    val_s1, gt_map = load_stratified_validation_slice(val_size=val_size, offset=250000)

    countries = val_s1["country"].unique().tolist()
    print(f"Validation countries present: {countries}")

    overall_predictions = {}
    all_raw_candidates: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    global_ext_cache = {}

    for country in countries:
        c_val = val_s1[val_s1["country"] == country]
        print(f"\n--- Testing Country: {country} ({len(c_val):,} entities) ---")

        # Load external records for this country (fast sample)
        s2_chunks = []
        for c in pd.read_csv(TRAIN_S2, sep="\t", chunksize=250000):
            sub = c[c["country"] == country]
            if not sub.empty:
                s2_chunks.append(sub)
            if len(s2_chunks) >= 3:
                break
        s2_c = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()

        s3_chunks = []
        for c in pd.read_csv(TRAIN_S3, sep="\t", chunksize=250000):
            sub = c[c["country"] == country]
            if not sub.empty:
                s3_chunks.append(sub)
            if len(s3_chunks) >= 3:
                break
        s3_c = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()

        ext_country = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c

        print(f"  Fitting Inverted Index on {len(ext_country):,} external records...")
        blocking = BlockingEngine(max_candidates=35)
        blocking.fit(ext_country)

        ext_cache = {}
        for row in ext_country.itertuples(index=False):
            eid = row.entity_id
            c_name, suff, _ = normalize_business_name(getattr(row, "business_name", ""))
            c_addr, nums, _ = normalize_address(getattr(row, "business_address", ""))
            zips = extract_zips(getattr(row, "business_address", ""))
            ext_cache[eid] = (c_name, suff, c_addr, nums, zips)
            global_ext_cache[eid] = (c_name, suff, c_addr, nums, zips)

        del ext_country

        # Generate candidates & extract 47 features
        items = []
        all_feats = []
        feat_idx = 0

        for row in c_val.itertuples(index=False):
            s1_id = row.entity_id
            s1_name, s1_suff, _ = normalize_business_name(getattr(row, "business_name", ""))
            s1_addr, s1_nums, _ = normalize_address(getattr(row, "business_address", ""))
            s1_zips = extract_zips(getattr(row, "business_address", ""))

            candidates = blocking.query(getattr(row, "business_name", ""), getattr(row, "business_address", ""))
            cand_ids = []
            start_pos = feat_idx

            for rank, cid in enumerate(candidates):
                if cid in ext_cache:
                    ext_name, ext_suff, ext_addr, ext_nums, ext_zips = ext_cache[cid]
                    f = extract_pair_features(
                        s1_name, s1_suff, s1_addr, s1_nums, s1_zips,
                        cid, ext_name, ext_suff, ext_addr, ext_nums, ext_zips, rank
                    )
                    cand_ids.append(cid)
                    all_feats.append(f)
                    feat_idx += 1

            end_pos = feat_idx
            items.append((s1_id, cand_ids, start_pos, end_pos))

        # Vectorized probability prediction
        if all_feats:
            probs = model.clf.predict_proba(np.array(all_feats, dtype=np.float32))[:, 1]
        else:
            probs = np.array([])

        for s1_id, cand_ids, start_pos, end_pos in items:
            if start_pos < end_pos and len(probs) > 0:
                p_slice = probs[start_pos:end_pos]
                f_slice = all_feats[start_pos:end_pos]
                # Record raw bids for global bipartite matching
                for cid, p in zip(cand_ids, p_slice):
                    all_raw_candidates[s1_id].append((cid, float(p)))

                matched = model.filter_candidate_matches(cand_ids, p_slice, f_slice, country=country)
            else:
                matched = set()

            overall_predictions[s1_id] = matched

    # 1. Baseline Score
    base_res = compute_macro_f05(overall_predictions, gt_map, val_s1["entity_id"])
    print("\n" + "=" * 75)
    print("                     BENCHMARK RESULTS")
    print("=" * 75)
    print(f"  [1] BASELINE GBDT SCORE:     Macro F0.5 = {base_res['macro_f05']:.4f} ({base_res['macro_f05']*100:.2f}%)")
    print(f"      - Singletons Score:      {base_res['singleton_f05']:.4f} ({base_res['n_singletons']:,} true singletons)")
    print(f"      - Non-Singletons Score:  {base_res['non_singleton_f05']:.4f} ({base_res['n_non_singletons']:,} true matches)")

    # 2. Add Global Bipartite Matching
    if use_bipartite:
        bipartite_preds = resolve_global_bipartite_matching(all_raw_candidates, min_confidence=match_threshold)
        if use_triangle:
            bipartite_preds = verify_triangular_consistency(bipartite_preds, global_ext_cache)

        bip_res = compute_macro_f05(bipartite_preds, gt_map, val_s1["entity_id"])
        diff = (bip_res['macro_f05'] - base_res['macro_f05']) * 100
        print(f"\n  [2] + GLOBAL BIPARTITE MATCH: Macro F0.5 = {bip_res['macro_f05']:.4f} ({bip_res['macro_f05']*100:.2f}%) [{diff:+.2f}%]")
        print(f"      - Singletons Score:      {bip_res['singleton_f05']:.4f}")
        print(f"      - Non-Singletons Score:  {bip_res['non_singleton_f05']:.4f}")

    print(f"\nBenchmark completed in {time.time() - t0:.1f}s")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=10000, help="Validation sample size")
    parser.add_argument("--no-bipartite", action="store_true", help="Disable bipartite matching")
    args = parser.parse_args()

    run_fast_benchmark(val_size=args.sample, use_bipartite=not args.no_bipartite)
