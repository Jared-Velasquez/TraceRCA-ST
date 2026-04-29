"""α sweep driver for Caller-Discount Re-Ranking.

Runs run_localization_association_rule_mining_20210516.py over both datasets
for α ∈ {0.0, 0.1, 0.3, 0.5, 0.8} and aggregates rank metrics into a single
CSV. Does NOT regenerate Stage 1 / Stage 2 inputs — those must already exist
under output/<dataset>_anomaly_detection.test/.

Usage:
    python scripts/run_caller_discount_sweep.py [--alphas 0,0.1,0.3,0.5,0.8]
                                                 [--datasets ob,tt]
                                                 [--out output/sweep_summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_localization_association_rule_mining_20210516.py"

DATASETS = {
    "tt": {
        "support": 0.05,
        "invo_dir": ROOT / "output/trainticket_anomaly_detection.test",
        "out_dir": ROOT / "output/trainticket_root_cause_localization",
        "invo_suffix": ".invo.result.pkl.1.1",
    },
    "ob": {
        "support": 0.01,
        "invo_dir": ROOT / "output/onlineboutique_anomaly_detection.test",
        "out_dir": ROOT / "output/onlineboutique_root_cause_localization",
        "invo_suffix": ".invo.result.pkl.1.1",
    },
}


def infer_true_cause(case_id: str) -> str | None:
    parts = case_id.rsplit("_", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return parts[0]
    return None


def rank_of(ranking: list[str], service: str) -> int | None:
    try:
        return ranking.index(service) + 1
    except ValueError:
        return None


def run_one(invo_path: Path, out_path: Path, support: float, alpha: float) -> None:
    cmd = [
        sys.executable, str(RUNNER),
        "-i", str(invo_path),
        "-o", str(out_path),
        "--min-support-rate", str(support),
        "--caller-discount-alpha", str(alpha),
        "-q",
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alphas", default="0.0,0.1,0.3,0.5,0.8")
    p.add_argument("--datasets", default="tt,ob")
    p.add_argument("--out", default=str(ROOT / "output/sweep_summary.csv"))
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    dnames = [d.strip() for d in args.datasets.split(",")]

    rows = []
    for dname in dnames:
        d = DATASETS[dname]
        invo_files = sorted(d["invo_dir"].glob(f"*{d['invo_suffix']}"))
        for alpha in alphas:
            for invo in invo_files:
                case_id = invo.name[: -len(d["invo_suffix"])]
                out_path = d["out_dir"] / (
                    f"{case_id}.association_rule_mining.result.pkl."
                    f"{d['support']}.100.alpha{alpha}"
                )
                run_one(invo, out_path, d["support"], alpha)
                with open(out_path, "rb") as f:
                    payload = pickle.load(f)
                ranking = payload.get("Ours-noise=0", [])
                true_cause = infer_true_cause(case_id)
                tc_rank = rank_of(ranking, true_cause) if true_cause else None
                rows.append({
                    "dataset": dname,
                    "case": case_id,
                    "alpha": alpha,
                    "true_cause": true_cause,
                    "rank": tc_rank,
                    "top1": int(tc_rank == 1) if tc_rank else 0,
                    "top3": int(tc_rank is not None and tc_rank <= 3),
                })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "case", "alpha",
                                          "true_cause", "rank", "top1", "top3"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
