"""
preprocess_re2ob.py — Convert RE2-OB (Online Boutique) trace data to TraceRCA pkl format.

Usage (single case directory):
    python preprocess_re2ob.py \
        --input-dir ~/Downloads/RE2-OB/checkoutservice_cpu/1 \
        --output-dir data/onlineboutique_re2ob

Usage (all cases under a parent directory):
    python preprocess_re2ob.py \
        --input-dir ~/Downloads/RE2-OB \
        --output-dir data/onlineboutique_re2ob

Output structure:
    <output-dir>/test/<case_name>.pkl    — anomalous traces  (label=1)
    <output-dir>/normal/<case_name>.pkl  — normal traces     (label=0)

Differences from RE2-TT handled here:
    1. statusCode is a float gRPC code ('0.0', '2.0', etc.) — parsed with int(float(...))
    2. INVOLVED_SERVICES uses Online Boutique service names, not Train-Ticket names
    3. Service names in traces.csv are already bare (no ts-/service wrapping)
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data.trainticket.download import simple_name
from onlineboutique_config import INVOLVED_SERVICES
from preprocess_re2tt import (
    SUPPORTED_FAULT_TYPES,
    build_span_lookup,
    build_trace_dict,
    find_case_dirs,
    load_traces,
    parse_case_dir,
    read_inject_time,
)

INVOLVED_SET = set(INVOLVED_SERVICES)


# ---------------------------------------------------------------------------
# OB-specific helpers
# ---------------------------------------------------------------------------

def reconstruct_invocations(trace_spans, span_lookup):
    """
    Same logic as preprocess_re2tt, with two OB-specific changes:
      - INVOLVED_SET filters on Online Boutique service names
      - statusCode is a float gRPC code, so parsed as int(float(...))
    """
    invocations = []
    for span in trace_spans:
        parent_id = span.parentSpanID
        if not parent_id:
            continue
        parent = span_lookup.get(parent_id)
        if parent is None:
            continue

        src = simple_name(parent.serviceName)
        tgt = simple_name(span.serviceName)
        if src == tgt:
            continue
        if src not in INVOLVED_SET or tgt not in INVOLVED_SET:
            continue

        start_us = int(span.startTime)
        dur_us = int(span.duration)
        end_us = start_us + dur_us

        try:
            raw_status = span.statusCode.strip()
            status = int(float(raw_status)) if raw_status else 0
        except (ValueError, AttributeError):
            status = 0

        invocations.append((src, tgt, start_us, end_us, dur_us, status))

    return invocations


def process_case(case_dir: Path, output_dir: Path, normal_ratio: float, warmup_seconds: int = 0):
    """
    Convert one RE2-OB case directory to TraceRCA pkl files.
    Writes:
        <output_dir>/test/<case_name>.pkl
        <output_dir>/normal/<case_name>.pkl
    """
    try:
        fault_type, root_cause = parse_case_dir(case_dir)
    except ValueError as exc:
        print(f"  Skipping {case_dir}: {exc}")
        return

    if case_dir.name.isdigit():
        case_name = f"{case_dir.parent.name}_{case_dir.name}"
    else:
        case_name = case_dir.name
    print(f"Processing {case_name} (fault={fault_type}, root_cause={root_cause}) ...")

    inject_ts_sec = read_inject_time(case_dir)
    inject_ts_us = inject_ts_sec * 1_000_000
    warmup_us = warmup_seconds * 1_000_000

    print(f"  Loading traces.csv ...")
    df = load_traces(case_dir)
    print(f"  {len(df):,} spans loaded.")

    span_lookup = build_span_lookup(df)

    grouped = df.groupby('traceID', sort=False)

    normal_traces = []
    anomalous_traces = []

    for trace_id, group in grouped:
        spans = list(group.itertuples(index=False))
        min_start_us = min(s.startTime for s in spans)

        if min_start_us < inject_ts_us:
            label = 0
        elif min_start_us < inject_ts_us + warmup_us:
            label = 0
        else:
            label = 1

        invocations = reconstruct_invocations(spans, span_lookup)
        if not invocations:
            continue

        trace_dict = build_trace_dict(
            trace_id, invocations, label, fault_type, root_cause
        )

        if label == 1:
            anomalous_traces.append(trace_dict)
        else:
            normal_traces.append(trace_dict)

    if normal_ratio < 1.0 and normal_traces:
        k = max(1, int(len(normal_traces) * normal_ratio))
        normal_traces = random.sample(normal_traces, k)

    test_dir = output_dir / 'test'
    norm_dir = output_dir / 'normal'
    test_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    test_path = test_dir / f'{case_name}.pkl'
    norm_path = norm_dir / f'{case_name}.pkl'

    with open(test_path, 'wb') as f:
        pickle.dump(anomalous_traces, f)
    with open(norm_path, 'wb') as f:
        pickle.dump(normal_traces, f)

    print(
        f"  Done. anomalous={len(anomalous_traces)}, "
        f"normal={len(normal_traces)} traces."
    )
    print(f"  test  → {test_path}")
    print(f"  normal→ {norm_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert RE2-OB (Online Boutique) traces to TraceRCA pkl format.'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help=(
            'Path to a single RE2-OB case dir (e.g. RE2-OB/checkoutservice_cpu/1) '
            'OR to the parent RE2-OB/ directory to process all cases.'
        ),
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Directory where test/ and normal/ subdirs will be created.',
    )
    parser.add_argument(
        '--normal-ratio', type=float, default=0.5,
        help='Fraction of pre-injection (normal) traces to keep (default: 0.5).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for normal-trace sub-sampling (default: 42).',
    )
    parser.add_argument(
        '--warmup-seconds', type=int, default=0,
        help='Seconds after injection to treat as warm-up (label=0). '
             'Recommended: 60 for cpu/mem faults (default: 0).',
    )
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    case_dirs = find_case_dirs(input_dir)
    if not case_dirs:
        print(f"No case directories found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(case_dirs)} case(s) to process.")
    for case_dir in case_dirs:
        process_case(case_dir, output_dir, args.normal_ratio, args.warmup_seconds)

    print("\nAll done.")


if __name__ == '__main__':
    main()
