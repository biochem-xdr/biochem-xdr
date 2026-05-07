import pandas as pd
import numpy as np


def assign_tier_aggregate(score):
    if score >= 0.70:
        return "T1"
    elif score >= 0.40:
        return "T2"
    elif score >= 0.10:
        return "T3"
    else:
        return "T4"


def calibrate_tiers(df, output_folder):
    """
    Computes per-item aggregate accuracy across all models
    and assigns difficulty tiers following Easy2Hard-Bench methodology.
    Saves tier_aggregate_calibrated.csv.
    """
    item_scores = df.groupby("item_id")["is_correct"].mean()

    item_tiers = item_scores.apply(assign_tier_aggregate)

    print("Aggregate-calibrated tier distribution:")
    for tier, count in item_tiers.value_counts().sort_index().items():
        pct = count / len(item_tiers) * 100
        print(f"  {tier}: {count:,} items ({pct:.1f}%)")

    out = item_tiers.reset_index()
    out.columns = ["item_id", "tier_aggregate"]
    path = f"{output_folder}/tier_aggregate_calibrated.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")

    return item_tiers


if __name__ == "__main__":
    import glob
    import json
    import sys
    from pathlib import Path

    results_folder = sys.argv[1]
    output_folder  = sys.argv[2] if len(sys.argv) > 2 else "."

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for f in glob.glob(f"{results_folder}/results_*.jsonl"):
        with open(f) as fh:
            for line in fh:
                try:
                    all_rows.append(json.loads(line.strip()))
                except Exception:
                    pass

    df = pd.DataFrame(all_rows)
    df["is_correct"] = df["is_correct"].astype(bool)

    calibrate_tiers(df, output_folder)