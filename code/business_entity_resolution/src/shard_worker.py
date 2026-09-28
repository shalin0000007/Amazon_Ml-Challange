"""
High-Speed Shard Worker for Entity Resolution Inference
======================================================
Runs an independent, isolated process on 1 dedicated CPU core.
Reads pre-filtered country catalog, builds local index in ~45s,
and processes its interleaved shard at ~45-50 entities/sec.
"""
import os
import sys
import time
import argparse
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ROOT_DIR, TEST_S1
from src.normalize import normalize_business_name, normalize_address
from src.blocking import BlockingEngine
from src.features import extract_pair_features, extract_zips
from src.model import EntityResolutionModel

MODEL_PATH = os.path.join(ROOT_DIR, "models", "lgbm_er.pkl")

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m:02d}m"

def run_worker(country: str, shard_id: int, total_shards: int, ext_file: str, out_match: str, out_cand: str, already_done_file: str = ""):
    prefix = f"[{country}-Worker {shard_id}/{total_shards}]"
    print(f"{prefix} Starting on dedicated core...", flush=True)

    # 1. Load trained model
    model = EntityResolutionModel.load(MODEL_PATH)
    model.delta = 0.030
    model.match_threshold = 0.80
    model.singleton_threshold = 0.950
    model.country_thresholds = {
        "US": {"match": 0.80, "singleton": 0.970, "delta": 0.030},
        "France": {"match": 0.80, "singleton": 0.970, "delta": 0.030},
        "India": {"match": 0.80, "singleton": 0.980, "delta": 0.030},
    }

    # 2. Load pre-filtered external catalog for this country (fast local read)
    t0 = time.time()
    ext_df = pd.read_csv(ext_file, sep="\t")
    print(f"{prefix} Loaded {len(ext_df):,} external records in {time.time()-t0:.1f}s", flush=True)

    # 3. Fit blocking engine
    t_fit = time.time()
    blocking = BlockingEngine(max_candidates=35)
    blocking.fit(ext_df)
    ext_cache = blocking.ext_cache
    del ext_df
    print(f"{prefix} Built index in {time.time()-t_fit:.1f}s", flush=True)

    # 4. Load already processed IDs to skip (if any)
    already_done = set()
    if already_done_file and os.path.exists(already_done_file):
        with open(already_done_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split("\t", 1)
                if parts and parts[0].strip() and parts[0].strip() != "source1_entity_id":
                    already_done.add(parts[0].strip())
        if already_done:
            print(f"{prefix} Found {len(already_done):,} already completed entities to skip", flush=True)

    # 5. Load test entities for this country
    s1_full = pd.read_csv(TEST_S1, sep="\t")
    s1_country = s1_full[s1_full["country"] == country]
    if already_done:
        s1_country = s1_country[~s1_country["entity_id"].isin(already_done)]
    del s1_full

    # 6. Interleaved shard assignment (perfectly balanced distribution)
    tuples = list(s1_country[["entity_id", "business_name", "business_address"]].fillna("").itertuples(index=False, name=None))
    shard_tuples = tuples[shard_id::total_shards]
    total_shard_entities = len(shard_tuples)
    del tuples, s1_country

    print(f"{prefix} Assigned {total_shard_entities:,} entities. Beginning inference...", flush=True)

    batch_size = 2500
    processed_count = 0
    t_stream = time.time()

    with open(out_cand, "w", encoding="utf-8") as f_cand, open(out_match, "w", encoding="utf-8") as f_match:
        for chunk_start in range(0, total_shard_entities, batch_size):
            batch = shard_tuples[chunk_start : chunk_start + batch_size]
            if not batch:
                continue

            chunk_items = []
            chunk_all_feats = []
            feat_idx = 0

            for s1_id, raw_name, raw_addr in batch:
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

                start_pos = feat_idx
                end_pos = feat_idx + len(feats_valid)
                chunk_all_feats.extend(feats_valid)
                feat_idx = end_pos
                chunk_items.append((s1_id, cand_ids_valid, cand_suffixes_valid, s1_suff, start_pos, end_pos))

            # Batch score with OpenMP C++
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
                processed_count += 1

            f_cand.writelines(cand_lines)
            f_match.writelines(match_lines)
            f_cand.flush()
            f_match.flush()

            del chunk_all_feats, chunk_items, chunk_probs, cand_lines, match_lines
            gc.collect()

            elapsed = time.time() - t_stream
            speed = processed_count / max(1e-3, elapsed)
            remaining = max(0, total_shard_entities - processed_count)
            eta_s = remaining / max(1e-3, speed)
            pct = (processed_count / max(1, total_shard_entities)) * 100.0

            print(
                f"{prefix} {processed_count:,} / {total_shard_entities:,} ({pct:.1f}%) | "
                f"Speed: {speed:.1f} ent/s | ETA: {format_time(eta_s)}",
                flush=True
            )

    print(f"{prefix} FINISHED in {time.time()-t_stream:.1f}s! Wrote: {out_match}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--total-shards", type=int, default=6)
    parser.add_argument("--ext-file", type=str, required=True)
    parser.add_argument("--out-match", type=str, required=True)
    parser.add_argument("--out-cand", type=str, required=True)
    parser.add_argument("--already-done", type=str, default="")
    args = parser.parse_args()

    run_worker(
        country=args.country,
        shard_id=args.shard_id,
        total_shards=args.total_shards,
        ext_file=args.ext_file,
        out_match=args.out_match,
        out_cand=args.out_cand,
        already_done_file=args.already_done
    )
