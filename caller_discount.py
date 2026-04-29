"""Caller-Discount Re-ranking (Stage 3 post-processor).

LOC budget tracking (across both modifications, target ≤300):
    caller_discount.py       — this file
    runner edits             — run_localization_association_rule_mining_20210516.py
    sweep driver             — scripts/run_caller_discount_sweep.py

Formula:  f'(S) = f(S) − α · max(f(C) for C in static_callees(S) ∩ items)
α = 0 reduces to baseline exactly.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


def build_static_call_graph(invo_df: pd.DataFrame) -> dict[str, set[str]]:
    """Static call graph from per-invocation rows with `source` (caller) and
    `target` (callee) columns. Self-loops excluded.

    The invo result pkl already contains the union of (caller, callee) edges
    observed across all traces for a case, so we don't need to walk
    parent_span_id ourselves.
    """
    edges = invo_df.reset_index(drop=True)[["source", "target"]].drop_duplicates()
    graph: dict[str, set[str]] = {}
    for src, tgt in edges.itertuples(index=False, name=None):
        if src == tgt:
            continue
        graph.setdefault(src, set()).add(tgt)
    return graph


def apply_caller_discount(
    item_scores: Mapping[str, float],
    pattern_len: Mapping[str, int],
    static_callees: Mapping[str, set[str]],
    alpha: float,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Return (re-ranked items, modified scores, max-callee-score per item).

    Re-ranks using the same three-key tuple as baseline:
      (-modified_score, len(pattern), name)
    so the alphabetical tertiary key remains the deterministic tie-breaker.

    When alpha == 0.0, modified scores equal item_scores exactly and the
    returned list equals the baseline sort.
    """
    items = list(item_scores.keys())
    item_set = set(items)
    max_callee: dict[str, float] = {}
    modified: dict[str, float] = {}
    for s in items:
        callees = static_callees.get(s, set()) & item_set
        max_callee[s] = max((item_scores[c] for c in callees), default=0.0)
        modified[s] = item_scores[s] - alpha * max_callee[s]

    if alpha == 0.0:
        # Off-switch invariant: bit-for-bit equality on scores.
        assert modified == dict(item_scores)

    ranked = sorted(
        items,
        key=lambda s: (-modified[s], pattern_len.get(s, 0), s),
    )
    return ranked, modified, max_callee


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _infer_true_cause(case_id: str) -> str | None:
    """Convention: `<service>_<faulttype>_<rep>`. Service may be hyphenated."""
    parts = case_id.rsplit("_", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0]
    return None


def write_mechanism_log(
    output_path: Path,
    case_id: str,
    alpha: float,
    item_scores: Mapping[str, float],
    pattern_len: Mapping[str, int],
    static_callees: Mapping[str, set[str]],
    baseline_ranked: list[str],
    modified_ranked: list[str],
    modified_scores: Mapping[str, float],
    max_callee: Mapping[str, float],
    config: Mapping[str, object],
) -> None:
    true_cause = _infer_true_cause(case_id)
    baseline_rank = (
        baseline_ranked.index(true_cause) + 1 if true_cause in baseline_ranked else None
    )
    modified_rank = (
        modified_ranked.index(true_cause) + 1 if true_cause in modified_ranked else None
    )
    delta_rank = (
        baseline_rank - modified_rank
        if baseline_rank is not None and modified_rank is not None
        else None
    )

    per_service = {
        s: {
            "f": float(item_scores[s]),
            "max_callee_score": float(max_callee.get(s, 0.0)),
            "f_prime": float(modified_scores[s]),
            "static_callees": sorted(static_callees.get(s, set())),
        }
        for s in item_scores
    }
    record = {
        "case_id": case_id,
        "alpha": alpha,
        "true_cause": true_cause,
        "baseline_rank": baseline_rank,
        "modified_rank": modified_rank,
        "delta_rank": delta_rank,
        "top5_baseline": [
            {"service": s, "score": float(item_scores[s])} for s in baseline_ranked[:5]
        ],
        "top5_modified": [
            {"service": s, "score": float(modified_scores[s])} for s in modified_ranked[:5]
        ],
        "per_service": per_service,
        "manifest": {
            "commit_sha": _git_commit_sha(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": dict(config),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
