"""
preprocess_re2tt.py — Convert RE2-TT trace data to TraceRCA pkl format.

Usage (single case directory):
    python preprocess_re2tt.py \
        --input-dir ~/Downloads/RE2-TT/ts-auth-service_cpu/1 \
        --output-dir data/trainticket_re2tt

Usage (all cases under a parent directory):
    python preprocess_re2tt.py \
        --input-dir ~/Downloads/RE2-TT \
        --output-dir data/trainticket_re2tt

Output structure:
    <output-dir>/test/<case_name>.pkl    — anomalous traces  (label=1)
    <output-dir>/normal/<case_name>.pkl  — normal traces     (label=0)

Each pkl file is a pickled list of trace dicts matching TraceRCA's expected schema:
    {
        'trace_id':   str,
        'label':      int,        # 0=normal, 1=anomalous
        'fault_type': str,        # 'cpu' | 'delay' | '' for normal
        'root_cause': list[str],  # simplified service names, e.g. ['auth']
        's_t':        list[tuple[str, str]],  # (source, target) per invocation
        'timestamp':  list[int],  # invocation start in microseconds
        'endtime':    list[int],  # invocation end   in microseconds
        'latency':    list[int],  # invocation duration in microseconds
        'http_status':list[int],  # raw HTTP status code (0 if absent)
    }
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

# TraceRCA supported fault types (others are skipped)
SUPPORTED_FAULT_TYPES = {'cpu', 'delay', 'disk', 'loss', 'mem', 'socket'}

# Bring in the config and simple_name from the TraceRCA package
sys.path.insert(0, str(Path(__file__).parent))
from data.trainticket.download import simple_name
from trainticket_config import INVOLVED_SERVICES

INVOLVED_SET = set(INVOLVED_SERVICES)

TRACES_COLS = [
    'traceID', 'spanID', 'serviceName',
    'startTime', 'duration', 'statusCode', 'parentSpanID',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_case_dir(case_dir: Path):
    """
    Extract fault_type and root_cause from a case directory name.

    Expects the *parent* of the numbered subdirectory to be named like:
        ts-auth-service_cpu   → fault_type='cpu',   root_cause=['auth']
        ts-route-plan-service_delay → fault_type='delay', root_cause=['route-plan']

    Returns (fault_type, root_cause) or raises ValueError for unsupported types.
    """
    # case_dir may be  .../ts-auth-service_cpu/1  or  .../ts-auth-service_cpu
    name = case_dir.name
    if name.isdigit():
        name = case_dir.parent.name

    parts = name.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse case name: {name!r}")

    full_svc, fault_type = parts
    if fault_type not in SUPPORTED_FAULT_TYPES:
        raise ValueError(
            f"Unsupported fault type {fault_type!r} in {name!r} "
            f"(supported: {SUPPORTED_FAULT_TYPES})"
        )

    root_cause = [simple_name(full_svc)]
    return fault_type, root_cause


def read_inject_time(case_dir: Path) -> int:
    """Return the fault injection timestamp in Unix seconds."""
    inject_file = case_dir / 'inject_time.txt'
    return int(inject_file.read_text().strip())


def load_traces(case_dir: Path) -> pd.DataFrame:
    """
    Load traces.csv, keeping only the columns we need.

    Column units in RE2-TT:
        startTime  — microseconds
        duration   — microseconds
    """
    csv_path = case_dir / 'traces.csv'
    df = pd.read_csv(
        csv_path,
        usecols=TRACES_COLS,
        dtype={
            'traceID':     str,
            'spanID':      str,
            'serviceName': str,
            'startTime':   'int64',
            'duration':    'int64',
            'statusCode':  str,       # may be empty
            'parentSpanID': str,
        },
        low_memory=False,
    )
    # Normalise missing values
    df['parentSpanID'] = df['parentSpanID'].fillna('')
    df['statusCode'] = df['statusCode'].fillna('0')
    return df


def build_span_lookup(df: pd.DataFrame) -> dict:
    """Build spanID → (serviceName, startTime, duration, statusCode) dict."""
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.spanID] = row
    return lookup


def reconstruct_invocations(trace_spans, span_lookup):
    """
    For each span with a cross-service parent, yield one invocation tuple:
        (src, tgt, start_us, end_us, latency_us, http_status)
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
            status = int(raw_status) if raw_status else 0
        except (ValueError, AttributeError):
            status = 0

        invocations.append((src, tgt, start_us, end_us, dur_us, status))

    return invocations


def build_trace_dict(trace_id, invocations, label, fault_type, root_cause):
    s_t = [(inv[0], inv[1]) for inv in invocations]
    return {
        'trace_id':   trace_id,
        'label':      label,
        'fault_type': fault_type if label == 1 else '',
        'root_cause': root_cause if label == 1 else [],
        's_t':        s_t,
        'timestamp':  [inv[2] for inv in invocations],
        'endtime':    [inv[3] for inv in invocations],
        'latency':    [inv[4] for inv in invocations],
        'http_status':[inv[5] for inv in invocations],
    }


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(case_dir: Path, output_dir: Path, normal_ratio: float, warmup_seconds: int = 0):
    """
    Convert one RE2-TT case directory to TraceRCA pkl files.
    Writes:
        <output_dir>/test/<case_name>.pkl
        <output_dir>/normal/<case_name>.pkl
    """
    try:
        fault_type, root_cause = parse_case_dir(case_dir)
    except ValueError as exc:
        print(f"  Skipping {case_dir}: {exc}")
        return

    # Case name used for output file (e.g. 'ts-auth-service_cpu_1')
    if case_dir.name.isdigit():
        case_name = f"{case_dir.parent.name}_{case_dir.name}"
    else:
        case_name = case_dir.name
    print(f"Processing {case_name} (fault={fault_type}, root_cause={root_cause}) ...")

    inject_ts_sec = read_inject_time(case_dir)
    inject_ts_us = inject_ts_sec * 1_000_000  # microseconds
    warmup_us = warmup_seconds * 1_000_000

    print(f"  Loading traces.csv ...")
    df = load_traces(case_dir)
    print(f"  {len(df):,} spans loaded.")

    span_lookup = build_span_lookup(df)

    # Group by traceID
    grouped = df.groupby('traceID', sort=False)

    normal_traces = []
    anomalous_traces = []

    for trace_id, group in grouped:
        spans = list(group.itertuples(index=False))
        min_start_us = min(s.startTime for s in spans)

        if min_start_us < inject_ts_us:
            label = 0
        elif min_start_us < inject_ts_us + warmup_us:
            label = 0  # warm-up window: fault injected but not yet manifesting
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

    # Sub-sample normal traces
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

def find_case_dirs(input_dir: Path):
    """
    Return a list of leaf case directories to process.

    Handles two layouts:
      1. Direct case dir:  input_dir = .../ts-auth-service_cpu/1
      2. Parent dir:       input_dir = .../RE2-TT  (contains ts-*_*/1 subdirs)
    """
    inject_file = input_dir / 'inject_time.txt'
    if inject_file.exists():
        return [input_dir]

    # Look for <service>_<fault>/<rep> pattern — enumerate all numbered reps
    dirs = sorted(
        rep_dir
        for p in input_dir.iterdir()
        if p.is_dir() and '_' in p.name
        for rep_dir in sorted(p.iterdir())
        if rep_dir.name.isdigit() and (rep_dir / 'inject_time.txt').exists()
    )
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description='Convert RE2-TT traces to TraceRCA pkl format.'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help=(
            'Path to a single RE2-TT case dir (e.g. RE2-TT/ts-auth-service_cpu/1) '
            'OR to the parent RE2-TT/ directory to process all cases.'
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
             'Recommended: 60 for cpu/mem faults (default: 0 = no change).',
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
