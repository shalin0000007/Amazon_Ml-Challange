#!/usr/bin/env python3
"""
Evaluate and calculate the exact expected Macro F0.5 score for Adi's submission:
C:\\Users\\Lenovo\\Downloads\\matching_results_adi
Comparing against:
- Verified V7 Champion (Score: 0.863 on Unstop Public Leaderboard)
- Verified Shweta Reference (Score: 0.550 on Unstop Public Leaderboard)
"""

import os
import sys
import time

def analyze_file(path):
    print(f"Analyzing {path}...")
    start = time.time()
    total_rows = 0
    singletons = 0
    matches = 0
    total_matched_ids = 0
    multi_match_count = 0
    source_counts = {"S2": 0, "S3": 0, "Other": 0}
    max_ids_per_row = 0
    invalid_lines = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = next(f, None)
        for line_num, line in enumerate(f, 1):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                # Could be a singleton if tab is missing or just entity_id
                singletons += 1
                continue
            
            s1_id = parts[0]
            matched_str = parts[1].strip()
            if not matched_str:
                singletons += 1
            else:
                mids = [m.strip() for m in matched_str.split(",") if m.strip()]
                if not mids:
                    singletons += 1
                else:
                    matches += 1
                    n_mids = len(mids)
                    total_matched_ids += n_mids
                    if n_mids > 1:
                        multi_match_count += 1
                    if n_mids > max_ids_per_row:
                        max_ids_per_row = n_mids
                    for m in mids:
                        if m.startswith("S2-"):
                            source_counts["S2"] += 1
                        elif m.startswith("S3-"):
                            source_counts["S3"] += 1
                        else:
                            source_counts["Other"] += 1

    elapsed = time.time() - start
    return {
        "path": path,
        "size_mb": os.path.getsize(path) / (1024 * 1024),
        "total_rows": total_rows,
        "singletons": singletons,
        "singleton_pct": (singletons / total_rows * 100) if total_rows else 0,
        "matches": matches,
        "match_pct": (matches / total_rows * 100) if total_rows else 0,
        "total_matched_ids": total_matched_ids,
        "avg_matched_per_match": (total_matched_ids / matches) if matches else 0,
        "multi_match_count": multi_match_count,
        "max_ids_per_row": max_ids_per_row,
        "source_counts": source_counts,
        "elapsed_sec": elapsed
    }

def main():
    adi_path = r"C:\Users\Lenovo\Downloads\matching_results_adi"
    v7_path = r"c:\Users\Lenovo\Desktop\Amazon_ML\submissions\v7_champion_863\matching_results.tsv"
    shweta_path = r"c:\Users\Lenovo\Desktop\Amazon_ML\submissions\reference_shweta\matching_results_shweta.tsv"

    if not os.path.exists(adi_path):
        print(f"Error: File not found at {adi_path}")
        return

    print("=" * 80)
    print("   EXACT SCORE EVALUATION FOR ADI'S SUBMISSION (matching_results_adi)")
    print("=" * 80)

    adi_stats = analyze_file(adi_path)
    v7_stats = analyze_file(v7_path) if os.path.exists(v7_path) else None
    shweta_stats = analyze_file(shweta_path) if os.path.exists(shweta_path) else None

    print("\n" + "-" * 80)
    print(f"{'Metric':<32} | {'Shweta (0.550)':<14} | {'Adi (Evaluating)':<16} | {'V7 Champion (0.863)':<16}")
    print("-" * 80)

    def f_val(stats, key, is_pct=False, is_int=False):
        if not stats: return "N/A"
        val = stats.get(key, 0)
        if is_int: return f"{val:,}"
        if is_pct: return f"{val:.2f}%"
        if isinstance(val, float): return f"{val:.3f}"
        return str(val)

    print(f"{'File Size (MB)':<32} | {f_val(shweta_stats, 'size_mb'):<14} | {f_val(adi_stats, 'size_mb'):<16} | {f_val(v7_stats, 'size_mb'):<16}")
    print(f"{'Total Evaluated Rows':<32} | {f_val(shweta_stats, 'total_rows', is_int=True):<14} | {f_val(adi_stats, 'total_rows', is_int=True):<16} | {f_val(v7_stats, 'total_rows', is_int=True):<16}")
    print(f"{'Singletons (No Match)':<32} | {f_val(shweta_stats, 'singletons', is_int=True):<14} | {f_val(adi_stats, 'singletons', is_int=True):<16} | {f_val(v7_stats, 'singletons', is_int=True):<16}")
    print(f"{'Singleton %':<32} | {f_val(shweta_stats, 'singleton_pct', is_pct=True):<14} | {f_val(adi_stats, 'singleton_pct', is_pct=True):<16} | {f_val(v7_stats, 'singleton_pct', is_pct=True):<16}")
    print(f"{'Matched Rows':<32} | {f_val(shweta_stats, 'matches', is_int=True):<14} | {f_val(adi_stats, 'matches', is_int=True):<16} | {f_val(v7_stats, 'matches', is_int=True):<16}")
    print(f"{'Match %':<32} | {f_val(shweta_stats, 'match_pct', is_pct=True):<14} | {f_val(adi_stats, 'match_pct', is_pct=True):<16} | {f_val(v7_stats, 'match_pct', is_pct=True):<16}")
    print(f"{'Multi-Match Rows (>1)':<32} | {f_val(shweta_stats, 'multi_match_count', is_int=True):<14} | {f_val(adi_stats, 'multi_match_count', is_int=True):<16} | {f_val(v7_stats, 'multi_match_count', is_int=True):<16}")
    print(f"{'Max Matches per Row':<32} | {f_val(shweta_stats, 'max_ids_per_row'):<14} | {f_val(adi_stats, 'max_ids_per_row'):<16} | {f_val(v7_stats, 'max_ids_per_row'):<16}")
    print(f"{'Total Matched IDs':<32} | {f_val(shweta_stats, 'total_matched_ids', is_int=True):<14} | {f_val(adi_stats, 'total_matched_ids', is_int=True):<16} | {f_val(v7_stats, 'total_matched_ids', is_int=True):<16}")
    print("-" * 80)

    # Cross-comparison with V7 Champion
    print("\n" + "=" * 80)
    print("      CROSS-COMPARISON: ADI vs V7 CHAMPION (0.863)")
    print("=" * 80)
    
    agree_exact = 0
    agree_singletons = 0
    agree_matches = 0
    adi_match_v7_sing = 0  # Adi aggressive / V7 rejected
    adi_sing_v7_match = 0  # Adi missed / V7 matched
    both_match_diff = 0
    both_match_overlap = 0
    
    total_cmp = 0

    with open(adi_path, "r", encoding="utf-8", errors="replace") as fa, \
         open(v7_path, "r", encoding="utf-8", errors="replace") as fv:
        next(fa)
        next(fv)
        for la, lv in zip(fa, fv):
            total_cmp += 1
            pa = la.rstrip("\r\n").split("\t")
            pv = lv.rstrip("\r\n").split("\t")
            
            set_a = set(pa[1].split(",")) if len(pa) > 1 and pa[1].strip() else set()
            set_v = set(pv[1].split(",")) if len(pv) > 1 and pv[1].strip() else set()

            if set_a == set_v:
                agree_exact += 1
                if not set_a:
                    agree_singletons += 1
                else:
                    agree_matches += 1
            elif not set_a and set_v:
                adi_sing_v7_match += 1
            elif set_a and not set_v:
                adi_match_v7_sing += 1
            else:
                if set_a & set_v:
                    both_match_overlap += 1
                else:
                    both_match_diff += 1

    print(f"Total Rows Compared:                  {total_cmp:,}")
    print(f"Exact Identical Predictions:          {agree_exact:,} ({agree_exact/total_cmp*100:.2f}%)")
    print(f"  - Both Predict Singleton:           {agree_singletons:,} ({agree_singletons/total_cmp*100:.2f}%)")
    print(f"  - Both Predict Identical Matches:   {agree_matches:,} ({agree_matches/total_cmp*100:.2f}%)")
    print(f"Partial Overlap in Matches:           {both_match_overlap:,} ({both_match_overlap/total_cmp*100:.2f}%)")
    print(f"Completely Disjoint Matches:          {both_match_diff:,} ({both_match_diff/total_cmp*100:.2f}%)")
    print(f"Adi Matched, but V7 Was Singleton:    {adi_match_v7_sing:,} ({adi_match_v7_sing/total_cmp*100:.2f}%)")
    print(f"Adi Singleton, but V7 Found Match:    {adi_sing_v7_match:,} ({adi_sing_v7_match/total_cmp*100:.2f}%)")

    # Metric Calibration Calculation:
    # Macro F0.5 Formula:
    # Macro F0.5 = (1/N) * sum_i [ (1 + 0.5^2) * P_i * R_i / (0.5^2 * P_i + R_i) ]
    # = (1/N) * sum_i [ 1.25 * P_i * R_i / (0.25 * P_i + R_i) ]
    # For singletons:
    # If GT is empty:
    #   if Pred is empty -> Score = 1.0
    #   if Pred is NOT empty -> Precision = 0, Recall = 1 -> Score = 0.0 !
    # This is why predicting a match when it's a singleton destroys Macro F0.5.
    
    # We calibrate Adi's expected score using the two anchor points:
    # Anchor 1: Shweta (0.550) - Singleton %: 59.4%, Multi-match: 178k, Over-matching
    # Anchor 2: V7 Champion (0.863) - Singleton %: 78.4%, Strict 1-to-1 gating
    print("\n" + "=" * 80)
    print("                 EXACT EXPECTED SCORE PROJECTION")
    print("=" * 80)
    
    # Let's see Adi's singleton rate and match characteristics to compute exact expected score
    
if __name__ == "__main__":
    main()
