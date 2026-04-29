"""
Safely combine multiple trace `.npz` files even when feature widths differ.
Usage:
  python3.11 scripts/combine_trace_npz.py \
    --input output/trainticket_trace_encoded/*.normal.trace.npz \
    --output output/trainticket_trace_encoded/trainticket_historical_all.trace.0.0.npz

This script pads 2D arrays (e.g. `data`, `masks`, `root_causes`) along axis=1 to the
maximum width found, using sensible fill values (-1 for floats/ints, False for bools,
empty string for object/string arrays). 1D arrays (`labels`, `trace_ids`) are concatenated.
"""
from __future__ import annotations
import argparse
import glob
import numpy as np
import sys
from pathlib import Path


def choose_fill_value(dtype):
    if np.issubdtype(dtype, np.floating):
        return -1.0
    if np.issubdtype(dtype, np.integer):
        return -1
    if np.issubdtype(dtype, np.bool_):
        return False
    # object / string
    return ""


def pad_2d(arr: np.ndarray, target_cols: int, fill):
    if arr.shape[1] == target_cols:
        return arr
    out = np.full((arr.shape[0], target_cols), fill, dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, nargs="+",
                   help="Input npz files or glob (quote the glob)")
    p.add_argument("--output", "-o", required=True,
                   help="Output npz file path")
    args = p.parse_args(argv)

    # expand globs
    files = []
    for pat in args.input:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        print("No input files found", file=sys.stderr)
        sys.exit(2)

    arrays = [np.load(f, allow_pickle=True) for f in files]
    keys = arrays[0].files

    # determine per-key max columns for 2D arrays
    max_cols = {}
    for key in keys:
        max_c = 0
        for a in arrays:
            arr = a[key]
            if arr.ndim == 2:
                max_c = max(max_c, arr.shape[1])
        if max_c > 0:
            max_cols[key] = max_c

    combined = {}
    for key in keys:
        parts = [a[key] for a in arrays]
        # 2D arrays: pad to max_cols[key]
        if key in max_cols:
            tgt = max_cols[key]
            padded = []
            # choose fill value from first part dtype
            for part in parts:
                fill = choose_fill_value(part.dtype)
                if part.ndim != 2:
                    # try to coerce 1D -> 2D if needed
                    part = np.atleast_2d(part)
                padded.append(pad_2d(part, tgt, fill))
            combined[key] = np.concatenate(padded, axis=0)
        else:
            # 1D or other; concatenate along axis 0
            # preserve dtype; for object arrays, ensure dtype=object
            to_concat = []
            for part in parts:
                to_concat.append(np.asarray(part))
            combined[key] = np.concatenate(to_concat, axis=0)

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outpath, **combined)
    print(f"Wrote {outpath} with keys: {', '.join(sorted(combined.keys()))}")


if __name__ == '__main__':
    main()
