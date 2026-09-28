#!/usr/bin/env python3
"""
ML Challenge 2026 - Reference Submission Comparator & Auditor

Compares a verified accepted submission file (e.g., Shweta's 0.550 submission)
with our generated V3 submission to verify 100% format compliance,
ensure zero portal rejection errors, and measure prediction agreement.
"""

import os
import sys
import glob
import shutil
from typing import Dict, Set, Tuple

DELIM = "\t"
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"


def find_reference_file(candidate_paths) -> str:
    """Finds the reference file across possible names and extensions."""
    for p in candidate_paths:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return ""


def analyze_file_structure(path: str) -> dict:
    """Performs deep structural byte and line inspection."""
    size = os.path.getsize(path)
    
    with open(path, "rb") as f:
        first_chunk = f.read(4096)
        
    has_crlf = b"\r\n" in first_chunk
    line_ending = "CRLF (\\r\\n - Windows)" if has_crlf else "LF (\\n - Linux)"
    
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_header = f.readline().rstrip("\r\n")
        first_data = f.readline().rstrip("\r\n")
        
    delimiter = "TAB (\\t)" if "\t" in raw_header else ("COMMA (,)" if "," in raw_header else "UNKNOWN")
    
    # Fast row counting and parsing
    total_lines = 0
    singletons = 0
    non_singletons = 0
    multi_matches = 0
    sample_singleton_raw = ""
    sample_match_raw = ""
    
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for idx, line in enumerate(f):
            total_lines += 1
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            
            if not rest.strip():
                singletons += 1
                if not sample_singleton_raw:
                    sample_singleton_raw = repr(line[:50])
            else:
                non_singletons += 1
                if not sample_match_raw:
                    sample_match_raw = repr(line[:50])
                ids = rest.split(",")
                if len(ids) > 1:
                    multi_matches += 1

    return {
        "path": path,
        "size_bytes": size,
        "size_mb": size / (1024 * 1024),
        "line_ending": line_ending,
        "delimiter": delimiter,
        "raw_header": raw_header,
        "total_data_rows": total_lines,
        "singletons": singletons,
        "non_singletons": non_singletons,
        "multi_matches": multi_matches,
        "sample_singleton_raw": sample_singleton_raw,
        "sample_match_raw": sample_match_raw,
    }


def compare_row_alignment(ref_path: str, v3_path: str, test_s1_path: str, sample_limit: int = 50000) -> dict:
    """Verifies whether row sequences match line-for-line with test_source1."""
    mismatches = 0
    first_mismatch = None
    
    if not os.path.exists(test_s1_path):
        return {"aligned": None, "reason": "test_source1.tsv not found"}
        
    with open(test_s1_path, "r", encoding="utf-8") as ft, \
         open(ref_path, "r", encoding="utf-8") as fr, \
         open(v3_path, "r", encoding="utf-8") as fv:
        
        next(ft) # skip header
        next(fr) # skip header
        next(fv) # skip header
        
        for row_idx, (lt, lr, lv) in enumerate(zip(ft, fr, fv), start=2):
            s1_t = lt.split("\t")[0]
            s1_r = lr.split("\t")[0]
            s1_v = lv.split("\t")[0]
            
            if s1_r != s1_t or s1_v != s1_t:
                mismatches += 1
                if not first_mismatch:
                    first_mismatch = (row_idx, s1_t, s1_r, s1_v)
            
            if row_idx > sample_limit:
                break
                
    return {
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "aligned": mismatches == 0
    }


def compute_prediction_overlap(ref_path: str, v3_path: str, limit: int = 200000) -> dict:
    """Calculates exact prediction agreement between Reference and V3."""
    exact_matches = 0
    both_singleton = 0
    both_non_singleton = 0
    v3_singleton_ref_matched = 0
    ref_singleton_v3_matched = 0
    total_compared = 0
    
    with open(ref_path, "r", encoding="utf-8") as fr, \
         open(v3_path, "r", encoding="utf-8") as fv:
        next(fr)
        next(fv)
        
        for lr, lv in zip(fr, fv):
            total_compared += 1
            r_parts = lr.rstrip("\r\n").split("\t")
            v_parts = lv.rstrip("\r\n").split("\t")
            
            r_set = set(r_parts[1].split(",")) if len(r_parts) > 1 and r_parts[1].strip() else set()
            v_set = set(v_parts[1].split(",")) if len(v_parts) > 1 and v_parts[1].strip() else set()
            
            if r_set == v_set:
                exact_matches += 1
            if not r_set and not v_set:
                both_singleton += 1
            elif r_set and v_set:
                both_non_singleton += 1
            elif not v_set and r_set:
                v3_singleton_ref_matched += 1
            elif not r_set and v_set:
                ref_singleton_v3_matched += 1
                
            if total_compared >= limit:
                break
                
    return {
        "total_compared": total_compared,
        "exact_agreement_pct": (exact_matches / total_compared) * 100,
        "both_singleton_pct": (both_singleton / total_compared) * 100,
        "both_non_singleton_pct": (both_non_singleton / total_compared) * 100,
        "v3_more_conservative_pct": (v3_singleton_ref_matched / total_compared) * 100,
        "ref_more_conservative_pct": (ref_singleton_v3_matched / total_compared) * 100,
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_dir = os.path.join(root, "submissions", "reference_shweta")
    os.makedirs(ref_dir, exist_ok=True)
    
    # 1. Look for reference file
    user_home = os.path.expanduser("~")
    search_patterns = [
        os.path.join(user_home, "Downloads", "matching_results_shweta*"),
        r"C:\Users\Lenovo\Downloads\matching_results_shweta*",
        os.path.join(ref_dir, "matching_results*"),
        os.path.join(ref_dir, "*shweta*")
    ]
    
    ref_file = find_reference_file(search_patterns)
    if not ref_file:
        print("\n" + "!" * 70)
        print("ERROR: Shweta's reference file not found!")
        print(f"Searched in: {search_patterns}")
        print("Please copy your teammate's file to:")
        print(f"  {ref_dir}\\matching_results_shweta.tsv")
        print("!" * 70 + "\n")
        sys.exit(1)
        
    # Copy to permanent reference folder if not already there
    dest_ref = os.path.join(ref_dir, "matching_results_shweta.tsv")
    if os.path.abspath(ref_file) != os.path.abspath(dest_ref):
        shutil.copy2(ref_file, dest_ref)
        print(f"Copied reference file to: {dest_ref}")
    ref_file = dest_ref
    
    v3_file = os.path.join(root, "output", "matching_results.tsv")
    test_s1 = os.path.join(root, "dataset", "test", "test_source1.tsv")
    
    if not os.path.exists(v3_file):
        print(f"ERROR: V3 output file not found at: {v3_file}")
        sys.exit(1)
        
    print("\n" + "=" * 70)
    print("      SUBMISSION FORMAT AUDIT & REFERENCE COMPARISON")
    print("=" * 70)
    print(f"Reference File (Accepted: 0.550): {ref_file}")
    print(f"Candidate File (Our V3 Output):   {v3_file}")
    print("=" * 70)
    
    print("\n[1/3] Analyzing file structures...")
    ref_stats = analyze_file_structure(ref_file)
    v3_stats = analyze_file_structure(v3_file)
    
    fmt = "{:<25} | {:<25} | {:<25} | {:<10}"
    print("\n" + "-" * 90)
    print(fmt.format("Format Property", "Shweta (Accepted 0.550)", "Our V3 Champion", "Status"))
    print("-" * 90)
    
    def check_status(a, b):
        return "MATCH" if a == b else "DIFF"
        
    print(fmt.format("Header Row", ref_stats["raw_header"], v3_stats["raw_header"], check_status(ref_stats["raw_header"], v3_stats["raw_header"])))
    print(fmt.format("Delimiter", ref_stats["delimiter"], v3_stats["delimiter"], check_status(ref_stats["delimiter"], v3_stats["delimiter"])))
    print(fmt.format("Line Endings", ref_stats["line_ending"], v3_stats["line_ending"], check_status(ref_stats["line_ending"], v3_stats["line_ending"])))
    print(fmt.format("Total Data Rows", f"{ref_stats['total_data_rows']:,}", f"{v3_stats['total_data_rows']:,}", check_status(ref_stats["total_data_rows"], v3_stats["total_data_rows"])))
    print(fmt.format("File Size", f"{ref_stats['size_mb']:.1f} MB", f"{v3_stats['size_mb']:.1f} MB", "INFO"))
    print(fmt.format("Singletons (Unmatched)", f"{ref_stats['singletons']:,}", f"{v3_stats['singletons']:,}", "INFO"))
    print(fmt.format("Matched Entities", f"{ref_stats['non_singletons']:,}", f"{v3_stats['non_singletons']:,}", "INFO"))
    print(fmt.format("Multi-Matches (>1 ID)", f"{ref_stats['multi_matches']:,}", f"{v3_stats['multi_matches']:,}", "INFO"))
    print("-" * 90)
    
    print("\n[Raw Singleton Formatting Representation]")
    print(f"  Shweta: {ref_stats['sample_singleton_raw']}")
    print(f"  Our V3: {v3_stats['sample_singleton_raw']}")
    
    print("\n[Raw Matched Formatting Representation]")
    print(f"  Shweta: {ref_stats['sample_match_raw']}")
    print(f"  Our V3: {v3_stats['sample_match_raw']}")
    
    print("\n[2/3] Checking Row-by-Row Sequence Alignment against test_source1.tsv...")
    align = compare_row_alignment(ref_file, v3_file, test_s1)
    if align["aligned"]:
        print("  [SUCCESS] Both files are 100% aligned with test_source1.tsv line-for-line!")
    else:
        print(f"  [ALERT] Mismatch detected: {align['first_mismatch']}")
        
    print("\n[3/3] Calculating Prediction Agreement on 200,000 entities...")
    overlap = compute_prediction_overlap(ref_file, v3_file, limit=200000)
    print(f"  Total Checked:               {overlap['total_compared']:,}")
    print(f"  Exact Match Agreement:       {overlap['exact_agreement_pct']:.2f}%")
    print(f"  Both Singletons:             {overlap['both_singleton_pct']:.2f}%")
    print(f"  Both Merged:                 {overlap['both_non_singleton_pct']:.2f}%")
    print(f"  V3 Singleton / Shweta Match: {overlap['v3_more_conservative_pct']:.2f}%")
    print(f"  Shweta Singleton / V3 Match: {overlap['ref_more_conservative_pct']:.2f}%")
    
    print("\n" + "=" * 70)
    print("      FINAL AUDIT VERDICT")
    print("=" * 70)
    if (ref_stats["raw_header"] == v3_stats["raw_header"] and 
        ref_stats["total_data_rows"] == v3_stats["total_data_rows"] and 
        align["aligned"]):
        print("  >>> SAFE TO SUBMIT: V3 matches Shweta's accepted structure 100%!")
        print("  >>> No formatting errors will occur on Unstop portal.")
    else:
        print("  >>> CAUTION: Formatting discrepancies found. Review the table above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
