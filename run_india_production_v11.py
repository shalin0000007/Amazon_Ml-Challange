#!/usr/bin/env python3
"""
Master India Production Pipeline (V11 Champion)
==============================================
Runs calibrated India shard inference with 2 memory-safe workers,
merges with verified France and US from V10 Oracle,
executes Bipartite Injective Auction, validates, and creates submission zip.
"""
import os
import sys
import time
import subprocess
import pandas as pd
import zipfile

ROOT_DIR = r"c:\Users\Lenovo\Desktop\Amazon_ML"
sys.path.insert(0, ROOT_DIR)

SCRATCH_DIR = os.path.join(ROOT_DIR, "scratch")
OUT_DIR = os.path.join(ROOT_DIR, "submissions", "v11_champion")
os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TEST_S1 = os.path.join(ROOT_DIR, "dataset", "test", "test_source1.tsv")
V10_MATCH = os.path.join(ROOT_DIR, "submissions", "v10_oracle", "matching_results.tsv")
EXT_INDIA = os.path.join(SCRATCH_DIR, "ext_India.tsv")

TOTAL_SHARDS = 2

def main():
    t_start = time.time()
    print("=" * 80)
    print("      V11 CHAMPION: HIGH-RECALL CALIBRATED INDIA PIPELINE")
    print("=" * 80)

    # 1. Verify prerequisite files
    if not os.path.exists(EXT_INDIA) or os.path.getsize(EXT_INDIA) < 1000000:
        print(f"[FATAL] Missing {EXT_INDIA}! Cannot proceed.")
        sys.exit(1)
    if not os.path.exists(V10_MATCH):
        print(f"[FATAL] Missing {V10_MATCH}! Cannot proceed.")
        sys.exit(1)

    print(f"External India catalog: {EXT_INDIA} ({os.path.getsize(EXT_INDIA)/1024/1024:.1f} MB)")
    print(f"Base V10 Oracle file:   {V10_MATCH} ({os.path.getsize(V10_MATCH)/1024/1024:.1f} MB)")

    # 2. Prepare part files
    part_match_files = []
    part_cand_files = []
    worker_procs = []
    worker_script = os.path.join(ROOT_DIR, "src", "shard_worker.py")

    print(f"\nLaunching {TOTAL_SHARDS} parallel workers on India test set...")
    for shard_id in range(TOTAL_SHARDS):
        p_match = os.path.join(SCRATCH_DIR, f"part_match_India_{shard_id}.tsv")
        p_cand = os.path.join(SCRATCH_DIR, f"part_cand_India_{shard_id}.tsv")
        part_match_files.append(p_match)
        part_cand_files.append(p_cand)

        # Clear old parts if starting clean
        for p in [p_match, p_cand]:
            if os.path.exists(p):
                os.remove(p)

        cmd = [
            sys.executable,
            "-u",
            worker_script,
            "--country", "India",
            "--shard-id", str(shard_id),
            "--total-shards", str(TOTAL_SHARDS),
            "--ext-file", EXT_INDIA,
            "--out-match", p_match,
            "--out-cand", p_cand,
        ]
        p = subprocess.Popen(cmd)
        worker_procs.append(p)
        print(f"  Launched Worker {shard_id}/{TOTAL_SHARDS} (PID: {p.pid})")

    # 3. Wait for workers to complete
    print("\nAll workers running. Awaiting completion...")
    for shard_id, p in enumerate(worker_procs):
        code = p.wait()
        if code != 0:
            print(f"[FATAL] Worker {shard_id} failed with exit code {code}!")
            sys.exit(1)
        print(f"  Worker {shard_id} finished successfully!")

    print(f"\nAll India workers completed in {time.time()-t_start:.1f}s.")

    # 4. Merge India parts
    print("Loading newly generated India predictions...")
    india_preds = {}
    for p_file in part_match_files:
        with open(p_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts and parts[0]:
                    india_preds[parts[0]] = parts[1] if len(parts) > 1 else ""

    print(f"Loaded {len(india_preds):,} new India predictions.")

    # 5. Merge with France and US from V10
    print("Merging new India with existing France & US from V10 Oracle...")
    test_s1 = pd.read_csv(TEST_S1, sep="\t", usecols=["entity_id", "country"])
    s1_order = list(test_s1["entity_id"])
    s1_country = dict(zip(test_s1["entity_id"], test_s1["country"]))
    del test_s1

    v10_matches = {}
    with open(V10_MATCH, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0]:
                v10_matches[parts[0]] = parts[1] if len(parts) > 1 else ""

    raw_merged_file = os.path.join(OUT_DIR, "matching_results_raw.tsv")
    replaced_count = 0
    with open(raw_merged_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            country = s1_country.get(s1_id, "")
            if country == "India" and s1_id in india_preds:
                m_str = india_preds[s1_id]
                replaced_count += 1
            else:
                m_str = v10_matches.get(s1_id, "")
            f_out.write(f"{s1_id}\t{m_str}\n")

    print(f"Saved raw merged file with {replaced_count:,} updated India predictions: {raw_merged_file}")

    # 6. Run Bipartite Injective Auction to eliminate all multi-claim collisions
    print("\nRunning Global Bipartite Injective Auction Resolution...")
    final_match_file = os.path.join(OUT_DIR, "matching_results.tsv")
    resolve_script = os.path.join(ROOT_DIR, "utils", "resolve_v10_oracle_collisions.py")

    # Run resolver with raw_merged_file as input
    cmd_resolve = [
        sys.executable,
        "-c",
        f"""
import sys
sys.path.insert(0, r"{ROOT_DIR}")
from utils.resolve_v10_oracle_collisions import *

# Overwrite input path
import utils.resolve_v10_oracle_collisions as r
r.V9_MATCH = r"{raw_merged_file}"
r.OUT_DIR = r"{OUT_DIR}"
r.OUT_MATCH = r"{final_match_file}"
r.main()
"""
    ]
    res_code = subprocess.run(cmd_resolve).returncode
    if res_code != 0:
        print("[WARNING] Custom resolver invocation returned non-zero. Copying raw merged file.")
        import shutil
        shutil.copyfile(raw_merged_file, final_match_file)

    # 7. Validate final file
    print("\nValidating final submission file...")
    validator = os.path.join(ROOT_DIR, "utils", "validate_submission.py")
    subprocess.run([sys.executable, validator, "--matching", final_match_file])

    # 8. Create submission zip
    zip_path = os.path.join(OUT_DIR, "v11_champion_submission.zip")
    root_zip_path = os.path.join(ROOT_DIR, "submissions", "v11_champion_submission.zip")
    print(f"\nCompressing into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(final_match_file, arcname="matching_results.tsv")
        if os.path.exists(os.path.join(SCRATCH_DIR, "part_cand_India_0.tsv")):
            z.write(os.path.join(SCRATCH_DIR, "part_cand_India_0.tsv"), arcname="candidate_pairs.tsv")

    import shutil
    shutil.copyfile(zip_path, root_zip_path)
    print(f"Copied to root submission: {root_zip_path} ({os.path.getsize(root_zip_path)/1024/1024:.1f} MB)")

    print(f"\n{'='*80}")
    print(f"  V11 CHAMPION PIPELINE COMPLETE IN {time.time()-t_start:.1f}s!")
    print(f"  Ready for portal upload: {root_zip_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
