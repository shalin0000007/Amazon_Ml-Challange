#!/usr/bin/env python3
"""
V12 Consensus Pipeline:
Merges V10 Oracle (optimal 5.5% singleton rate) with V11 Champion (50k+ new multi-matches),
eliminating singleton starvation and pruning all collisions.
"""
import os
import sys
import zipfile
import shutil

ROOT_DIR = r"c:\Users\Lenovo\Desktop\Amazon_ML"
sys.path.insert(0, ROOT_DIR)

v10_file = os.path.join(ROOT_DIR, "submissions", "v10_oracle", "matching_results.tsv")
v11_file = os.path.join(ROOT_DIR, "submissions", "v11_champion", "matching_results.tsv")

out_dir = os.path.join(ROOT_DIR, "submissions", "v12_consensus")
os.makedirs(out_dir, exist_ok=True)
raw_consensus = os.path.join(out_dir, "matching_results_raw.tsv")
final_consensus = os.path.join(out_dir, "matching_results.tsv")
out_zip = os.path.join(ROOT_DIR, "submissions", "v12_consensus_submission.zip")

print("Merging V10 and V11 predictions...")
restored = 0
total = 0

with open(v10_file, "r", encoding="utf-8") as f10, \
     open(v11_file, "r", encoding="utf-8") as f11, \
     open(raw_consensus, "w", encoding="utf-8") as fout:
    fout.write(f10.readline())  # Header
    next(f11)

    for l10, l11 in zip(f10, f11):
        total += 1
        p10 = l10.rstrip("\r\n").split("\t")
        p11 = l11.rstrip("\r\n").split("\t")
        s1_id = p10[0]
        m10 = p10[1] if len(p10) > 1 else ""
        m11 = p11[1] if len(p11) > 1 else ""

        # If V11 made it empty (due to 0.98 threshold), but V10 had a high-confidence match, keep V10
        if not m11 and m10:
            final_m = m10
            restored += 1
        else:
            final_m = m11

        fout.write(f"{s1_id}\t{final_m}\n")

print(f"Restored {restored:,} entities from singleton starvation! Total: {total:,}")

# Run collision resolver on raw_consensus -> final_consensus
import utils.resolve_v10_oracle_collisions as r
r.V9_MATCH = raw_consensus
r.OUT_DIR = out_dir
r.OUT_MATCH = final_consensus
r.main()

print(f"Validating final consensus file: {final_consensus}")
validator = os.path.join(ROOT_DIR, "utils", "validate_submission.py")
import subprocess
subprocess.run([sys.executable, validator, "--matching", final_consensus])

print(f"Packaging {out_zip}...")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(final_consensus, arcname="matching_results.tsv")

# Also copy to v12_consensus directory
shutil.copyfile(out_zip, os.path.join(out_dir, "v12_consensus_submission.zip"))
print(f"SUCCESS! Created: {out_zip} ({os.path.getsize(out_zip)/1024/1024:.2f} MB)")
