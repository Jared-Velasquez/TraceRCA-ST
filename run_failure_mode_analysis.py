"""
Post-hoc failure-mode classifier for baseline TraceRCA on RCAEval RE2-TT.

Reads the artifacts produced by docs/COMMANDS.md and, for each case where the
true cause is not Top-1, classifies the failure as supporting Jaccard dilution 
(Candidate A), caller accumulation (Candidate B), both, or neither.

Pre-validation step #1 from CLAUDE.md: if (A + B + ambiguous) / failures < ~30-40%
the project's premise is not supported on this dataset.
"""

import pickle
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# data/trainticket/download.py is missing from the repo, so reimplement the
# minimal mapping needed: ts-<simple>-service -> <simple>. Matches the
# convention in trainticket_config.INVOLVED_SERVICES and
# preprocess_re2tt.parse_case_dir.
_TS_SERVICE_RE = re.compile(r"^ts-(.+?)-service$")


def simple_name(svc: str) -> str:
    m = _TS_SERVICE_RE.match(svc)
    return m.group(1) if m else svc


def parse_case(case_filename: str):
    """`ts-auth-service_cpu_1` or `ts-auth-service_cpu` -> ('auth', 'cpu')."""
    parts = case_filename.rsplit("_", 1)
    # Strip trailing rep digit if present (per-rep naming: <svc>_<fault>_<rep>)
    if len(parts) == 2 and parts[-1].isdigit():
        case_filename = parts[0]
        parts = case_filename.rsplit("_", 1)
    if len(parts) != 2:
        return None, None
    full_svc, fault_type = parts
    return simple_name(full_svc), fault_type


def load_ranked_list(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    ranked = d.get("Ours-noise=0")
    if ranked is None:
        return None
    out = []
    for item in ranked:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (list, tuple, frozenset, set)):
            out.append(next(iter(item)) if len(item) == 1 else tuple(item))
        else:
            out.append(item)
    return out


def get_rank(target: str, ranked: list) -> int:
    for idx, item in enumerate(ranked):
        if item == target or (isinstance(item, (tuple, list)) and target in item):
            return idx + 1
    return len(ranked) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--localization-dir", default="output/trainticket_root_cause_localization")
@click.option("--invo-dir", default="output/trainticket_anomaly_detection.test")
@click.option("--historical-normal", default="output/trainticket_invo_encoded/trainticket_historical_normal.invo.pkl")
@click.option("--sigma", default="1")
@click.option("--fisher", default="3")
@click.option("--support", default="0.1")
@click.option("--k", "topk", default="100")
@click.option("--base-rate-percentile", default=75.0, type=float)
@click.option("--base-rate-gap", default=0.10, type=float)
@click.option("--caller-depth", default=1, type=int)
@click.option("--fault-types", default="cpu,delay")
@click.option("-o", "--output-csv", default="output/failure_mode_analysis.csv")
def main(localization_dir, invo_dir, historical_normal, sigma, fisher, support, topk,
         base_rate_percentile, base_rate_gap, caller_depth, fault_types, output_csv):
    localization_dir = Path(localization_dir)
    invo_dir = Path(invo_dir)
    historical_normal = Path(historical_normal)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fault_set = set(fault_types.split(","))

    # --- 1. Cache base_rate and fan_in from historical normal traces ----------
    logger.info(f"loading historical normal: {historical_normal}")
    with open(historical_normal, "rb") as f:
        hist = pickle.load(f)
    hist_n_traces = hist["trace_id"].nunique()
    base_rate = (
        hist.groupby("target")["trace_id"].nunique() / hist_n_traces
    ).to_dict()
    fan_in = hist.groupby("target")["source"].nunique().to_dict()
    high_base_rate_threshold = float(np.percentile(list(base_rate.values()), base_rate_percentile))
    logger.info(
        f"historical normal: {hist_n_traces} traces, "
        f"{len(base_rate)} target services, "
        f"p{base_rate_percentile:.0f}(base_rate)={high_base_rate_threshold:.4f}"
    )

    # --- 2. Iterate over per-case localization pickles -----------------------
    suffix = f".association_rule_mining.result.pkl.{support}.{topk}"
    pkls = sorted(localization_dir.glob(f"*{suffix}"))
    if not pkls:
        raise SystemExit(f"no localization pickles matched {localization_dir}/*{suffix}")

    rows = []
    for pkl in pkls:
        case = pkl.name[: -len(suffix)]
        true_cause, fault_type = parse_case(case)
        if true_cause is None or fault_type not in fault_set:
            continue

        ranked = load_ranked_list(pkl)
        if not ranked:
            rows.append({"case": case, "label": "Empty ranking", "true_cause": true_cause,
                         "fault_type": fault_type})
            continue

        top1 = ranked[0]
        # Treat tuple top-1 (multi-service itemset) by taking first element for headline match
        top1_str = top1 if isinstance(top1, str) else (top1[0] if top1 else None)
        true_rank = get_rank(true_cause, ranked)

        if top1_str == true_cause or true_rank == 1:
            continue  # success, not a failure

        invo_pkl = invo_dir / f"{case}.invo.result.pkl.{sigma}.{fisher}"
        if not invo_pkl.exists():
            rows.append({"case": case, "label": "Missing invo result", "true_cause": true_cause,
                         "top1": top1_str, "fault_type": fault_type})
            continue
        with open(invo_pkl, "rb") as f:
            invo = pickle.load(f)
        # Upstream stores source/target as both index levels and columns
        # (set_index(..., drop=False)), which makes groupby ambiguous.
        invo = invo.reset_index(drop=True)

        # (a) Stage 1 sanity: any anomalous edge touching the true cause?
        truth_edges = invo[(invo["source"] == true_cause) | (invo["target"] == true_cause)]
        anomalous_truth_edges = int(truth_edges.get("predict", pd.Series(dtype=int)).sum())

        if anomalous_truth_edges == 0:
            label = "Neither - Stage 1"
            supports_a = supports_b = False
        else:
            # (b) Candidate A: top1 is high base rate, and meaningfully more common than truth
            br_top1 = base_rate.get(top1_str, 0.0)
            br_truth = base_rate.get(true_cause, 0.0)
            supports_a = (
                br_top1 >= high_base_rate_threshold
                and (br_top1 - br_truth) >= base_rate_gap
            )

            # (c) Candidate B: true_cause reachable from top1 within caller_depth hops
            edge_pairs = invo[["source", "target"]].drop_duplicates()
            adj = edge_pairs.groupby("source")["target"].apply(set).to_dict()
            reachable, frontier = set(), {top1_str}
            for _ in range(caller_depth):
                next_frontier = set()
                for s in frontier:
                    next_frontier |= adj.get(s, set())
                next_frontier -= reachable
                reachable |= next_frontier
                frontier = next_frontier
                if not frontier:
                    break
            supports_b = true_cause in reachable

            if supports_a and supports_b:
                label = "Ambiguous (A+B)"
            elif supports_a:
                label = "Supports A"
            elif supports_b:
                label = "Supports B"
            else:
                label = "Neither - other"

        rows.append({
            "case": case,
            "fault_type": fault_type,
            "true_cause": true_cause,
            "top1": top1_str,
            "top2": ranked[1] if len(ranked) > 1 else None,
            "top3": ranked[2] if len(ranked) > 2 else None,
            "true_cause_rank": true_rank,
            "label": label,
            "base_rate_top1": base_rate.get(top1_str, 0.0),
            "base_rate_true_cause": base_rate.get(true_cause, 0.0),
            "fan_in_top1": fan_in.get(top1_str, 0),
            "top1_calls_truth_within_depth": bool(supports_b),
            "stage1_anomalous_edges_on_truth": anomalous_truth_edges,
        })

    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"wrote {len(df)} failure rows to {output_csv}")

    # --- 3. Aggregate -------------------------------------------------------
    n = len(df)
    if n == 0:
        logger.warning("no failures found")
        return
    counts = df["label"].value_counts().to_dict()

    def pct(k):
        return f"{100 * counts.get(k, 0) / n:5.1f}%"

    print(f"\nFailures: {n}")
    for k in ["Supports A", "Supports B", "Ambiguous (A+B)",
              "Neither - Stage 1", "Neither - other",
              "Empty ranking", "Missing invo result"]:
        print(f"  {k:25s} {counts.get(k, 0):4d}  ({pct(k)})")
    a_or_b = sum(counts.get(k, 0) for k in ("Supports A", "Supports B", "Ambiguous (A+B)"))
    print(f"\nA + B + Ambiguous = {a_or_b}/{n} = {100 * a_or_b / n:.1f}%")
    print("Acceptance gate (CLAUDE.md pre-validation #1): >= 30-40%")

    # Per-fault-type breakdown
    print("\nBy fault type:")
    for ft, sub in df.groupby("fault_type"):
        sub_n = len(sub)
        sub_ab = sub["label"].isin(["Supports A", "Supports B", "Ambiguous (A+B)"]).sum()
        print(f"  {ft:8s}  n={sub_n:4d}  A+B+ambig={sub_ab:4d} ({100 * sub_ab / sub_n:.1f}%)")


if __name__ == "__main__":
    main()
