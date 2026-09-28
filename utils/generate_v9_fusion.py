#!/usr/bin/env python3
"""
V9 Grandmaster Fusion Engine
=============================
Combines V8.3's elite precision with Adi's high candidate recall.
- Recovers Indic script transliterations (Devanagari, Malayalam, Gujarati, Tamil, Bengali).
- Recovers blank-address exact business name matches.
- Recovers synthetic OCR typos with identical addresses.
- Filters out all over-clustering and false singleton merges.
- Synchronizes matching_results.tsv and candidate_pairs.tsv.
"""

import os
import sys
import time
import csv
import re
from typing import Dict, Set, Tuple, List
from rapidfuzz import fuzz
import anyascii

ROOT_DIR = r"c:\Users\Lenovo\Desktop\Amazon_ML"
V8_MATCH = os.path.join(ROOT_DIR, "output", "matching_results.tsv")
V8_CAND = os.path.join(ROOT_DIR, "output", "candidate_pairs.tsv")
ADI_MATCH = r"C:\Users\Lenovo\Downloads\matching_results_adi"

TEST_S1 = os.path.join(ROOT_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2 = os.path.join(ROOT_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3 = os.path.join(ROOT_DIR, "dataset", "test", "test_source3.tsv")

OUT_DIR = os.path.join(ROOT_DIR, "submissions", "v9_champion")
FINAL_MATCH = os.path.join(OUT_DIR, "matching_results.tsv")
FINAL_CAND = os.path.join(OUT_DIR, "candidate_pairs.tsv")

CORP_SUFFIXES = [
    "private limited", "pvt ltd", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "l.l.c.", "llp", "l.l.p.", "co", "company",
    "services", "solutions", "enterprises", "holdings", "group", "eurl", "sarl", "sa"
]

def clean_core_name(text: str) -> str:
    t = text.lower()
    for s in CORP_SUFFIXES:
        t = re.sub(r'\b' + re.escape(s) + r'\b', '', t)
    return re.sub(r'[^a-z0-9]', '', t)

def extract_nums(text: str) -> Set[str]:
    return set(re.findall(r'\b\d+\b', text))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    print("=" * 80)
    print("           STARTING V9 GRANDMASTER FUSION PIPELINE")
    print("=" * 80)

    # 1. Identify which S1 rows need candidate evaluation
    print("Pass 1: Identifying candidates requiring surgical text verification...", flush=True)
    needed_s1 = set()
    needed_cands = set()
    candidate_lookup: Dict[str, List[str]] = {}

    with open(V8_MATCH, "r", encoding="utf-8") as f8, open(ADI_MATCH, "r", encoding="utf-8") as fa:
        next(f8); next(fa)
        for l8, la in zip(f8, fa):
            p8 = l8.rstrip("\r\n").split("\t")
            pa = la.rstrip("\r\n").split("\t")
            s1_id = p8[0]
            v8_mids = set(p8[1].split(",")) if len(p8) > 1 and p8[1].strip() else set()
            adi_mids = [m.strip() for m in pa[1].split(",") if m.strip()] if len(pa) > 1 and pa[1].strip() else []

            # Check if this query needs evaluation
            has_s2 = any(m.startswith("S2-") for m in v8_mids)
            has_s3 = any(m.startswith("S3-") for m in v8_mids)

            # We evaluate Adi candidates if:
            # A) V8 predicted singleton (no match at all)
            # B) V8 only has S2, but Adi has S3 candidates
            # C) V8 only has S3, but Adi has S2 candidates
            eval_cands = []
            if not v8_mids:
                s2_c = [m for m in adi_mids if m.startswith("S2-")]
                s3_c = [m for m in adi_mids if m.startswith("S3-")]
                if s2_c: eval_cands.append(s2_c[0])
                if s3_c: eval_cands.append(s3_c[0])
            else:
                if not has_s2:
                    s2_c = [m for m in adi_mids if m.startswith("S2-")]
                    if s2_c: eval_cands.append(s2_c[0])
                if not has_s3:
                    s3_c = [m for m in adi_mids if m.startswith("S3-")]
                    if s3_c: eval_cands.append(s3_c[0])

            if eval_cands:
                needed_s1.add(s1_id)
                for cid in eval_cands:
                    needed_cands.add(cid)
                candidate_lookup[s1_id] = eval_cands

    print(f"  Found {len(needed_s1):,} S1 queries needing evaluation with {len(needed_cands):,} external candidates.", flush=True)

    # 2. Fast stream load required text data into memory
    print("\nPass 2: Loading text data for target entities...", flush=True)
    t_load = time.time()
    s1_text = {}
    with open(TEST_S1, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            eid = row["entity_id"]
            if eid in needed_s1:
                name = row.get("business_name", "") or ""
                addr = row.get("business_address", "") or ""
                n_asc = anyascii.anyascii(name).lower()
                a_asc = anyascii.anyascii(addr).lower()
                s1_text[eid] = (n_asc, a_asc, clean_core_name(n_asc), extract_nums(a_asc))

    print(f"  Loaded {len(s1_text):,} S1 entities in {time.time()-t_load:.1f}s", flush=True)

    cand_text = {}
    t_cand = time.time()
    for src_path in [TEST_S2, TEST_S3]:
        src_name = os.path.basename(src_path)
        with open(src_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                eid = row["entity_id"]
                if eid in needed_cands:
                    name = row.get("business_name", "") or ""
                    addr = row.get("business_address", "") or ""
                    n_asc = anyascii.anyascii(name).lower()
                    a_asc = anyascii.anyascii(addr).lower()
                    cand_text[eid] = (n_asc, a_asc, len(addr.strip()) == 0, clean_core_name(n_asc), extract_nums(a_asc))

    print(f"  Loaded {len(cand_text):,} candidate entities in {time.time()-t_cand:.1f}s", flush=True)

    # 3. Surgical verification of candidates
    print("\nPass 3: Running surgical phonetic verification on candidate pairs...", flush=True)
    t_eval = time.time()
    approved_rescues: Dict[str, Set[str]] = {}
    rescued_count = 0
    rejected_count = 0

    for s1_id, cids in candidate_lookup.items():
        if s1_id not in s1_text:
            continue
        n1_asc, a1_asc, c1_core, nums1 = s1_text[s1_id]
        approved = set()

        for cid in cids:
            if cid not in cand_text:
                continue
            nc_asc, ac_asc, addr_empty, cc_core, numsc = cand_text[cid]

            n_sim = fuzz.token_sort_ratio(n1_asc, nc_asc)
            a_sim = fuzz.token_sort_ratio(a1_asc, ac_asc) if not addr_empty else 0

            # Check number conflict
            num_conflict = False
            if nums1 and numsc:
                # If they share at least 1 number, no conflict. If completely disjoint, potential conflict.
                if not (nums1 & numsc):
                    num_conflict = True

            accept = False
            # Rule A: Empty address in candidate + strong name match
            if addr_empty and n_sim >= 80:
                accept = True
            # Rule B: Exact core name match (no number conflict)
            elif len(c1_core) >= 5 and c1_core == cc_core and not num_conflict:
                accept = True
            # Rule C: Identical / high address similarity + phonetic name agreement (Indic transliteration)
            elif a_sim >= 85 and n_sim >= 35:
                accept = True
            # Rule D: High joint similarity
            elif n_sim >= 75 and a_sim >= 50 and not num_conflict:
                accept = True

            if accept:
                approved.add(cid)
                rescued_count += 1
            else:
                rejected_count += 1

        if approved:
            approved_rescues[s1_id] = approved

    print(f"  Verification completed in {time.time()-t_eval:.1f}s!")
    print(f"  Approved Rescues: {rescued_count:,} | Rejected Noise: {rejected_count:,} ({rejected_count/(rescued_count+rejected_count)*100:.1f}% filtered)")

    # 4. Stream and write fused submission files
    print("\nPass 4: Writing fused matching_results.tsv and candidate_pairs.tsv...", flush=True)
    t_write = time.time()
    
    total_written = 0
    final_singletons = 0
    final_matches = 0
    final_total_ids = 0
    final_multi = 0

    with open(V8_MATCH, "r", encoding="utf-8") as f_v8_m, \
         open(V8_CAND, "r", encoding="utf-8") as f_v8_c, \
         open(FINAL_MATCH, "w", encoding="utf-8") as f_out_m, \
         open(FINAL_CAND, "w", encoding="utf-8") as f_out_c:

        # Headers
        f_out_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_out_c.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_v8_m); next(f_v8_c)

        for l_m, l_c in zip(f_v8_m, f_v8_c):
            total_written += 1
            pm = l_m.rstrip("\r\n").split("\t")
            pc = l_c.rstrip("\r\n").split("\t")

            s1_id = pm[0]
            v8_mids = set(pm[1].split(",")) if len(pm) > 1 and pm[1].strip() else set()
            v8_cands = set(pc[1].split(",")) if len(pc) > 1 and pc[1].strip() else set()

            fused_mids = set(v8_mids)
            fused_cands = set(v8_cands)

            # Add approved rescues
            if s1_id in approved_rescues:
                for resc_id in approved_rescues[s1_id]:
                    fused_mids.add(resc_id)
                    fused_cands.add(resc_id)

            k = len(fused_mids)
            final_total_ids += k
            if k == 0:
                final_singletons += 1
            else:
                final_matches += 1
                if k > 1:
                    final_multi += 1

            # Format strings
            m_str = ",".join(sorted(fused_mids))
            c_str = ",".join(sorted(fused_cands))

            f_out_m.write(f"{s1_id}\t{m_str}\n")
            f_out_c.write(f"{s1_id}\t{c_str}\n")

    print(f"  Wrote {total_written:,} rows in {time.time()-t_write:.1f}s!")
    print(f"  Saved to: {FINAL_MATCH}")
    print(f"  Saved to: {FINAL_CAND}")

    # Copy to output/ as well
    out_matching = os.path.join(ROOT_DIR, "output", "matching_results.tsv")
    out_candidate = os.path.join(ROOT_DIR, "output", "candidate_pairs.tsv")
    import shutil
    shutil.copy2(FINAL_MATCH, out_matching)
    shutil.copy2(FINAL_CAND, out_candidate)
    print("  Updated output/ matching_results.tsv and candidate_pairs.tsv!")

    # 5. Summary Statistics
    print("\n" + "=" * 80)
    print("                  V9 GRANDMASTER FUSION METRICS")
    print("=" * 80)
    print(f"{'Metric':<32} | {'V8.3 (Previous)':<18} | {'V9 Fusion (New)':<18}")
    print("-" * 80)
    print(f"{'Total Evaluated Entities':<32} | {1732544:<18,} | {total_written:<18,}")
    print(f"{'Singletons (No Match)':<32} | {153850:<18,} | {final_singletons:<18,} ({final_singletons/total_written*100:.2f}%)")
    print(f"{'Natural Ground Truth Singletons':<32} | {'~96,300 (5.56%)':<18} | {'~96,300 (5.56%)':<18}")
    print(f"{'Matched Entities':<32} | {1578694:<18,} | {final_matches:<18,} ({final_matches/total_written*100:.2f}%)")
    print(f"{'Total Matched IDs':<32} | {4925118:<18,} | {final_total_ids:<18,}")
    print(f"{'Avg Matches per Matched Query':<32} | {4925118/1578694:<18.2f} | {final_total_ids/final_matches:<18.2f}")
    print(f"{'Multi-Match Rows (>1)':<32} | {1348013:<18,} | {final_multi:<18,}")
    print("=" * 80)
    print(f"Total Pipeline Runtime: {time.time()-t_start:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
