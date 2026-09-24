#!/usr/bin/env python3
"""
Reassemble the split dataset parts in this folder back into the original
`.tsv` files under ``dataset/``.

This is the inverse of the split step used to fit the data inside GitHub's
per-file limits. Run from the repository root::

    python data/join_data.py

For each ``<name>.tsv.gz.partNN`` group it:
  1. concatenates the parts in order into a ``.tsv.gz``,
  2. decompresses it (gzip),
  3. writes the result to ``dataset/train/<name>.tsv`` or
     ``dataset/test/<name>.tsv`` based on the ``train_``/``test_`` prefix.

Stdlib only (Python 3.8+). The original ``.gz`` is removed afterwards and the
parts are kept untouched, so you can re-run safely.
"""

import gzip
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET_DIR = os.path.join(ROOT, "dataset")

PART_RE = re.compile(r"^(.*\.tsv)\.gz\.part(\d+)$")

# Folder under dataset/ that each file belongs to, from its name prefix.
SUBDIR_OF = {"test_": "test", "train_": "train"}


def group_parts():
    """Return {base_name: [(part_number, part_path), ...]} for data/*.part*."""
    groups = {}
    for entry in os.listdir(HERE):
        m = PART_RE.match(entry)
        if not m:
            continue
        base, num = m.group(1), int(m.group(2))
        groups.setdefault(base, []).append((num, os.path.join(HERE, entry)))
    for base, parts in groups.items():
        parts.sort()
        expected = list(range(1, len(parts) + 1))
        if [n for n, _ in parts] != expected:
            sys.exit(f"ERROR: {base} has gaps in part numbers: {[n for n, _ in parts]}")
    return groups


def reassemble(base, parts):
    """Concatenate parts into base + '.gz' and return its path."""
    gz_path = os.path.join(HERE, base + ".gz")
    if os.path.exists(gz_path):
        os.remove(gz_path)
    with open(gz_path, "wb") as out:
        for _, part in parts:
            with open(part, "rb") as f:
                while chunk := f.read(1 << 20):
                    out.write(chunk)
    return gz_path


def decompress(gz_path, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)


def main():
    groups = group_parts()
    if not groups:
        sys.exit("No *.tsv.gz.partNN files found in data/. Nothing to do.")
    print(f"Found {len(groups)} dataset file(s) to restore.")
    for base in sorted(groups):
        parts = groups[base]
        prefix = base.split("_", 1)[0] + "_"
        subdir = SUBDIR_OF.get(prefix)
        if subdir is None:
            sys.exit(f"ERROR: cannot infer dataset subdir from name: {base!r}")
        out_path = os.path.join(DATASET_DIR, subdir, base)
        gz_path = reassemble(base, parts)
        decompress(gz_path, out_path)
        os.remove(gz_path)
        mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  restored dataset/{subdir}/{base} ({mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()