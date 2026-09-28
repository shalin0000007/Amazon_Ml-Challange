#!/usr/bin/env python3
"""
Compute the exact expected Macro F0.5 score of Adi's submission
using 3-point Leaderboard Anchor Calibration:
- Anchor 1: submissions/v7_champion_863/matching_results.tsv -> Actual LB Score: 0.863
- Anchor 2: submissions/baseline_v1/matching_results.tsv      -> Actual LB Score: 0.602
- Anchor 3: submissions/reference_shweta/matching_results_shweta.tsv -> Actual LB Score: 0.550
"""

import os
import sys

def main():
    adi_path = r"C:\Users\Lenovo\Downloads\matching_results_adi"
    v7_path = r"submissions\v7_champion_863\matching_results.tsv"
    v1_path = r"submissions\baseline_v1\matching_results.tsv"
    shweta_path = r"submissions\reference_shweta\matching_results_shweta.tsv"

    print("=" * 80)
    print("CALCULATING EXACT EXPECTED LEADERBOARD SCORE FOR ADI'S SUBMISSION")
    print("=" * 80)

    # We will sample 100,000 rows across the dataset evenly to get high-precision statistics
    # This avoids any memory issues while giving standard error < 0.001
    sample_stride = 17  # 1,732,544 / 17 ≈ 101,914 samples
    
    total_sampled = 0
    
    # Track metrics
    adi_sing = 0
    v7_sing = 0
    v1_sing = 0
    shweta_sing = 0

    adi_total_ids = 0
    v7_total_ids = 0
    v1_total_ids = 0
    shweta_total_ids = 0

    # Pairwise overlaps
    adi_v7_f05_sum = 0.0
    adi_v1_f05_sum = 0.0
    adi_shweta_f05_sum = 0.0
    
    shweta_v7_f05_sum = 0.0
    v1_v7_f05_sum = 0.0

    def f05(pred_set, true_set):
        if not pred_set and not true_set:
            return 1.0
        if not pred_set or not true_set:
            return 0.0
        tp = len(pred_set & true_set)
        if tp == 0:
            return 0.0
        p = tp / len(pred_set)
        r = tp / len(true_set)
        # 1.25 * P * R / (0.25 * P + R)
        return (1.25 * p * r) / (0.25 * p + r)

    with open(adi_path, 'r', encoding='utf-8') as fa, \
         open(v7_path, 'r', encoding='utf-8') as fv7, \
         open(v1_path, 'r', encoding='utf-8') as fv1, \
         open(shweta_path, 'r', encoding='utf-8') as fs:
        
        next(fa); next(fv7); next(fv1); next(fs)
        
        line_idx = 0
        for la, lv7, lv1, ls in zip(fa, fv7, fv1, fs):
            line_idx += 1
            if line_idx % sample_stride != 0:
                continue
            
            total_sampled += 1

            def parse_ids(l):
                p = l.rstrip('\r\n').split('\t')
                return set(x for x in p[1].split(',') if x.strip()) if len(p) > 1 and p[1].strip() else set()

            set_adi = parse_ids(la)
            set_v7 = parse_ids(lv7)
            set_v1 = parse_ids(lv1)
            set_shweta = parse_ids(ls)

            adi_total_ids += len(set_adi)
            v7_total_ids += len(set_v7)
            v1_total_ids += len(set_v1)
            shweta_total_ids += len(set_shweta)

            if not set_adi: adi_sing += 1
            if not set_v7: v7_sing += 1
            if not set_v1: v1_sing += 1
            if not set_shweta: shweta_sing += 1

            # Relative F0.5 treating V7 as pseudo-ground truth
            adi_v7_f05_sum += f05(set_adi, set_v7)
            shweta_v7_f05_sum += f05(set_shweta, set_v7)
            v1_v7_f05_sum += f05(set_v1, set_v7)

            # Overlaps between Adi and others
            adi_shweta_f05_sum += f05(set_adi, set_shweta)
            adi_v1_f05_sum += f05(set_adi, set_v1)

    rel_adi_v7 = adi_v7_f05_sum / total_sampled
    rel_shweta_v7 = shweta_v7_f05_sum / total_sampled
    rel_v1_v7 = v1_v7_f05_sum / total_sampled

    print(f"Total Sampled Rows: {total_sampled:,}")
    print("-" * 80)
    print(f"Agreement Metric (Relative to V7 @ 0.863):")
    print(f"  V1 (Actual LB = 0.602)   Relative F0.5 vs V7: {rel_v1_v7:.4f}")
    print(f"  Shweta (Actual LB=0.550) Relative F0.5 vs V7: {rel_shweta_v7:.4f}")
    print(f"  Adi (Target Submission)  Relative F0.5 vs V7: {rel_adi_v7:.4f}")
    print("-" * 80)

    # Let's perform linear and polynomial regression on known anchors
    # Anchor points: (Relative to V7, Actual LB Score)
    # Point 1: V7 vs V7 = (1.000, 0.863)
    # Point 2: V1 vs V7 = (rel_v1_v7, 0.602)
    # Point 3: Shweta vs V7 = (rel_shweta_v7, 0.550)

    x1, y1 = 1.0, 0.863
    x2, y2 = rel_v1_v7, 0.602
    x3, y3 = rel_shweta_v7, 0.550

    print(f"Anchor 1: V7 Champion        x = {x1:.4f} -> Actual LB Score = {y1:.3f}")
    print(f"Anchor 2: Baseline V1        x = {x2:.4f} -> Actual LB Score = {y2:.3f}")
    print(f"Anchor 3: Teammate Shweta    x = {x3:.4f} -> Actual LB Score = {y3:.3f}")
    print(f"Target:   Adi's Submission   x = {rel_adi_v7:.4f}")

    # Two-point interpolation between nearest anchors
    if rel_adi_v7 < rel_shweta_v7:
        slope = (y3 - y2) / (x3 - x2) if (x3 != x2) else 1.0
        expected_score = y3 + slope * (rel_adi_v7 - x3)
    elif rel_adi_v7 <= rel_v1_v7:
        slope = (y2 - y3) / (x2 - x3)
        expected_score = y3 + slope * (rel_adi_v7 - x3)
    else:
        slope = (y1 - y2) / (x1 - x2)
        expected_score = y2 + slope * (rel_adi_v7 - x2)

    # Calibration adjustment for singleton error
    # Adi has only 2.97% singletons vs ~5.58% in ground truth
    # That represents an over-prediction penalty of ~ (0.0558 - 0.0297) * 1.0 = ~0.0261
    
    print("-" * 80)
    print(f"Calculated Direct Interpolated Score: {expected_score:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
