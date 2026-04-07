#!/usr/bin/env python
"""Collect W&B experiment results into a CSV for paper tables.

Usage:
    python scripts/collect_results.py --project astra-cifar --out results/cifar_results.csv
    python scripts/collect_results.py --project sparsimony   --out results/rigl_results.csv

The script fetches all finished runs from the W&B project, filters for the
relevant metrics, and writes a tidy CSV with one row per run containing:
    method, dataset, sparsity, seed, best_val_acc, final_sparsity
"""

import argparse
import pandas as pd
import wandb


METRIC_MAP = {
    # W&B summary key -> output column name
    "best_val_acc": "best_val_acc",
    "val/acc":      "final_val_acc",
    "sparsity":     "final_sparsity",
}

CONFIG_KEYS = ["dataset/name", "sparsity", "training/seed"]


def fetch_runs(project: str, entity: str | None) -> pd.DataFrame:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": "finished"})

    rows = []
    for run in runs:
        row = {
            "run_id":   run.id,
            "name":     run.name,
            "group":    run.group or "",
        }
        # Config values
        for key in CONFIG_KEYS:
            parts = key.split("/")
            val = run.config
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else None
            row[key.replace("/", ".")] = val

        # Summary metrics
        for src, dst in METRIC_MAP.items():
            row[dst] = run.summary.get(src)

        # Infer method from run name / group
        name_lower = (run.name or "").lower()
        if "srigl" in name_lower:
            row["method"] = "SRigL"
        elif "rigl" in name_lower:
            row["method"] = "RigL"
        elif "astra" in name_lower:
            row["method"] = "ASTRA"
        else:
            row["method"] = run.group.split("_")[0] if run.group else "unknown"

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds, grouped by method/dataset/sparsity."""
    group_cols = ["method", "dataset.name", "sparsity"]
    agg = (
        df.groupby(group_cols)["best_val_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = group_cols + ["mean_acc", "std_acc", "n_seeds"]
    agg = agg.sort_values(group_cols)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity/team")
    parser.add_argument("--out", default="results/results.csv", help="Output CSV path")
    parser.add_argument("--summary", action="store_true", help="Print mean±std table")
    args = parser.parse_args()

    print(f"Fetching runs from {args.entity or ''}/{args.project} ...")
    df = fetch_runs(args.project, args.entity)
    print(f"  {len(df)} finished runs")

    df.to_csv(args.out, index=False)
    print(f"Saved raw results to {args.out}")

    if args.summary or True:
        summary = summarize(df)
        print("\n=== Summary (mean ± std over seeds) ===")
        print(summary.to_string(index=False))
        summary_path = args.out.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
