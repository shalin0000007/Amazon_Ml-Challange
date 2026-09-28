import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set

from src.config import (
    ROOT_DIR, TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    MAX_CANDIDATES_PER_ENTITY
)
from src.normalize import normalize_business_name, normalize_address
from src.blocking import BlockingEngine
from src.features import extract_pair_features, extract_zips
from src.model import EntityResolutionModel
from src.metrics import compute_macro_f05

MODEL_PATH = os.path.join(ROOT_DIR, "models", "lgbm_er.pkl")


def evaluate_model(val_sample_size: int = 10000, start_offset: int = 100000):
    """Evaluates the trained model on a held-out slice of the training data using the exact leaderboard metric."""
    print("=" * 60)
    print("      LOCAL MODEL PERFORMANCE EVALUATION (Macro F0.5)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error: Trained model not found at {MODEL_PATH}. Run training first.")

    print(f"Loading trained model from {MODEL_PATH}...")
    model = EntityResolutionModel.load(MODEL_PATH)
    print(f"  Match Threshold: {model.match_threshold:.2f} | Singleton Threshold: {model.singleton_threshold:.2f}")

    print(f"\nLoading {val_sample_size:,} held-out validation entities (offset: {start_offset:,})...")
    val_s1 = pd.read_csv(TRAIN_S1, sep="\t", skiprows=range(1, start_offset), nrows=val_sample_size)
    val_s1_ids = set(val_s1["entity_id"])

    # Load Ground Truth
    print("Loading Ground Truth labels...")
    gt_map = {}
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in val_s1_ids:
                if len(parts) > 1 and parts[1]:
                    gt_map[parts[0]] = set(parts[1].split(","))
                else:
                    gt_map[parts[0]] = set()

    # Determine countries present in validation sample
    countries = val_s1["country"].unique().tolist()
    print(f"Validation countries: {countries}")

    overall_predictions = {}
    t_start = time.time()

    for country in countries:
        c_val = val_s1[val_s1["country"] == country]
        print(f"\n--- Evaluating Country: {country} ({len(c_val):,} entities) ---")
        t_c = time.time()

        # Load external records for this country
        print(f"  Loading training external records for {country}...")
        s2_chunks = []
        for c in pd.read_csv(TRAIN_S2, sep="\t", chunksize=500000):
            sub = c[c["country"] == country]
            if not sub.empty:
                s2_chunks.append(sub)
        s2_c = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()

        s3_chunks = []
        for c in pd.read_csv(TRAIN_S3, sep="\t", chunksize=500000):
            sub = c[c["country"] == country]
            if not sub.empty:
                s3_chunks.append(sub)
        s3_c = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()

        ext_country = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c

        print(f"  Indexing {len(ext_country):,} external records ({country})...")
        blocking = BlockingEngine(max_candidates=35)
        blocking.fit(ext_country)

        print(f"  Caching normalized records for {country}...")
        ext_cache = {}
        for row in ext_country.itertuples(index=False):
            eid = row.entity_id
            raw_name = getattr(row, "business_name", "")
            raw_addr = getattr(row, "business_address", "")
            c_name, suff, _ = normalize_business_name(raw_name)
            c_addr, nums, _ = normalize_address(raw_addr)
            zips = extract_zips(raw_addr)
            ext_cache[eid] = (c_name, suff, c_addr, nums, zips)

        del ext_country

        # Generate candidates & features
        print(f"  Scoring candidates for {len(c_val):,} entities...")
        items = []
        all_feats = []
        feat_idx = 0

        for row in c_val.itertuples(index=False):
            s1_id = row.entity_id
            raw_name = getattr(row, "business_name", "")
            raw_addr = getattr(row, "business_address", "")
            s1_name, s1_suff, _ = normalize_business_name(raw_name)
            s1_addr, s1_nums, _ = normalize_address(raw_addr)
            s1_zips = extract_zips(raw_addr)

            candidates = blocking.query(raw_name, raw_addr)
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

        # Vectorized batch prediction
        if all_feats:
            probs = model.clf.predict_proba(np.array(all_feats, dtype=np.float32))[:, 1]
        else:
            probs = np.array([])

        country_preds = {}
        for s1_id, cand_ids, start_pos, end_pos in items:
            if start_pos < end_pos and len(probs) > 0:
                p_slice = probs[start_pos:end_pos]
                f_slice = all_feats[start_pos:end_pos]
                matched = model.filter_candidate_matches(cand_ids, p_slice, f_slice, country=country)
            else:
                matched = set()

            country_preds[s1_id] = matched
            overall_predictions[s1_id] = matched

        # Country-level score
        c_res = compute_macro_f05(country_preds, gt_map, c_val["entity_id"])
        print(f"  Results for {country} ({len(c_val):,} entities):")
        print(f"    Macro F_0.5 Score: {c_res['macro_f05']:.4f}")
        print(f"    Singleton Score:   {c_res['singleton_f05']:.4f} ({c_res['n_singletons']:,} singletons)")
        print(f"    Non-Singleton:     {c_res['non_singleton_f05']:.4f} ({c_res['n_non_singletons']:,} entities)")
        print(f"    Elapsed:           {time.time() - t_c:.1f}s")

    # Overall Results
    print("\n" + "=" * 60)
    print("                FINAL EVALUATION REPORT")
    print("=" * 60)
    overall_res = compute_macro_f05(overall_predictions, gt_map, val_s1["entity_id"])
    print(f"Total Validation Entities: {overall_res['n_entities']:,}")
    print(f"OVERALL MACRO F_0.5 SCORE: {overall_res['macro_f05']:.4f} ({overall_res['macro_f05']*100:.2f}%)")
    print(f"Singleton Accuracy:        {overall_res['singleton_f05']:.4f} ({overall_res['singleton_f05']*100:.2f}%)")
    print(f"Non-Singleton F_0.5:       {overall_res['non_singleton_f05']:.4f} ({overall_res['non_singleton_f05']*100:.2f}%)")
    print(f"Total Evaluation Time:     {time.time() - t_start:.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on held-out validation set")
    parser.add_argument("--sample", type=int, default=10000, help="Validation sample size")
    parser.add_argument("--offset", type=int, default=100000, help="Row offset in train_source1.tsv")
    args = parser.parse_args()

    evaluate_model(val_sample_size=args.sample, start_offset=args.offset)


if __name__ == "__main__":
    main()
