"""Aggregate K-fold + multi-seed results and compute Transfer Degradation (TD).

Consumes the per-run JSON files written by train.py (each has keys "config" and
"best" where best holds acc/mf1/kappa/per_class_f1). Produces mean/std tables
across folds and seeds, and TD = MF1_within - MF1_transfer (PLAN.md section 0).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ("acc", "mf1", "kappa")


def load_result_jsons(results_dir):
    """Load all *.json run files from a directory into a list of dicts."""
    out = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = os.path.basename(f)
        out.append(d)
    return out


def aggregate_runs(runs, group_keys=("ablation",)):
    """Aggregate runs into mean/std per group.

    Args:
        runs: list of dicts with run["best"] metrics and run["config"] fields.
              Each run should also carry the grouping fields either at top level
              or inside config (e.g. "ablation").
        group_keys: config/top-level keys to group by.

    Returns:
        dict group_tuple -> {metric -> {"mean","std","n","values"}}
    """
    groups = defaultdict(list)
    for r in runs:
        key = tuple(_get_field(r, k) for k in group_keys)
        groups[key].append(r["best"])
    agg = {}
    for key, bests in groups.items():
        stats = {}
        for m in METRICS:
            vals = [b[m] for b in bests if m in b]
            if vals:
                stats[m] = {"mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "n": len(vals), "values": vals}
        agg[key] = stats
    return agg


def _get_field(run, key):
    if key in run:
        return run[key]
    return run.get("config", {}).get(key, None)


def transfer_degradation(within_mf1, transfer_mf1):
    """TD = MF1_within - MF1_transfer (PLAN.md). Lower TD = more robust."""
    return float(within_mf1 - transfer_mf1)


def transfer_degradation_from_agg(agg, within_key, transfer_key):
    """Compute TD from an aggregated dict given two group keys."""
    w = agg[within_key]["mf1"]["mean"]
    t = agg[transfer_key]["mf1"]["mean"]
    return transfer_degradation(w, t)


def format_table(agg, group_keys=("ablation",)):
    """Render an aggregated dict as a fixed-width text table."""
    lines = []
    header = "  ".join([k for k in group_keys] +
                       [f"{m}(mean±std)" for m in METRICS])
    lines.append(header)
    lines.append("-" * len(header))
    for key in sorted(agg, key=lambda k: tuple(str(x) for x in k)):
        cells = [str(x) for x in key]
        for m in METRICS:
            if m in agg[key]:
                s = agg[key][m]
                cells.append(f"{s['mean']:.3f}±{s['std']:.3f}(n={s['n']})")
            else:
                cells.append("-")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--group", nargs="+", default=["ablation"])
    args = p.parse_args()
    runs = load_result_jsons(args.results_dir)
    # try to recover ablation from filename if not in record
    for r in runs:
        if _get_field(r, "ablation") is None:
            r["ablation"] = r["_file"].split("_")[0]
    agg = aggregate_runs(runs, tuple(args.group))
    print(format_table(agg, tuple(args.group)))


if __name__ == "__main__":
    main()
