import json
import glob
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def wilson_ci(correct, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 0.0
    p      = correct / total
    denom  = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    ci     = z * np.sqrt((p * (1 - p) / total) + (z ** 2 / (4 * total ** 2))) / denom
    return p * 100, max(0, centre * 100 - ci * 100), min(100, centre * 100 + ci * 100)


def load_gold(gold_file):
    with open(gold_file) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["id"] = df["id"].astype(int)
    tier_col = "difficulty_tier" if "difficulty_tier" in df.columns else "tier"
    return df, set(df["id"].tolist()), dict(zip(df["id"], df[tier_col]))


def load_mcq_results(results_folder, gold_ids, tier_map):
    exclude = {"deepseek-reasoner", "o1", "qwq-32b", "r1-distill-qwen-32b"}
    rows    = []
    for rf in glob.glob(f"{results_folder}/results_*.jsonl"):
        if any(ex in rf for ex in exclude):
            continue
        with open(rf) as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    df["item_id"]    = df["item_id"].astype(int)
    df["is_correct"] = df["is_correct"].astype(bool)
    df["tier"]       = df["item_id"].map(tier_map)
    return df[df["item_id"].isin(gold_ids)].copy()


def load_reasoning_results(reasoning_files, gold_ids, tier_map):
    rows = []
    for model_id, filepath in reasoning_files.items():
        if not os.path.exists(filepath):
            print(f"NOT FOUND: {filepath}")
            continue
        with open(filepath) as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                    row["model"] = model_id
                    rows.append(row)
                except Exception:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["item_id"]    = df["item_id"].astype(int)
    df["is_correct"] = df["is_correct"].astype(bool)
    df["tier"]       = df["item_id"].map(tier_map)
    return df[df["item_id"].isin(gold_ids)].copy()


def _get_row(mdf, display_name, category, tiers):
    if mdf.empty:
        return None
    acc, lo, hi = wilson_ci(mdf["is_correct"].sum(), len(mdf))
    row = {
        "Model":       display_name,
        "Category":    category,
        "Overall":     round(acc, 1),
        "Overall_fmt": f"{acc:.1f} ({lo:.1f}–{hi:.1f})",
    }
    for tier in tiers:
        tdf = mdf[mdf["tier"] == tier]
        row[tier] = round(tdf["is_correct"].mean() * 100, 1) if len(tdf) > 0 else float("nan")
    return row


def build_gold_table(mcq_gold, reasoning_gold, gold_df, output_folder):
    tiers    = ["T1", "T2", "T3", "T4"]
    tier_col = "difficulty_tier" if "difficulty_tier" in gold_df.columns else "tier"

    tier_ns = {t: (gold_df[tier_col] == t).sum() for t in tiers}

    REASONING_DISPLAY = {
        "o3":                  "OpenAI o3",
        "deepseek-v4-pro":     "DeepSeek V4 Pro",
        "r1-distill-qwen-32b": "R1-Distill-Qwen-32B",
        "qwq-32b":             "QwQ-32B",
    }
    FRONTIER_DISPLAY = {
        "claude-sonnet-4.6": "Claude Sonnet 4.6",
        "gpt-4o":            "GPT-4o",
        "deepseek-v3":       "DeepSeek-V3",
        "qwen3-235b":        "Qwen3-235B",
    }

    all_rows = []

    print(f"\nGold subset: {len(gold_df)} items")
    for tier in tiers:
        print(f"  {tier}: {tier_ns[tier]}")

    print("\n[Reasoning Models]")
    for model_id, display_name in REASONING_DISPLAY.items():
        mdf = reasoning_gold[reasoning_gold["model"] == model_id]
        row = _get_row(mdf, display_name, "Reasoning", tiers)
        if row is None:
            print(f"  {display_name}: NO DATA")
            continue
        vals = " → ".join(f"{row[t]:.1f}%" for t in tiers if not np.isnan(row[t]))
        print(f"  {display_name}: {row['Overall_fmt']}  [{vals}]")
        all_rows.append(row)

    print("\n[Frontier MCQ]")
    for model_id, display_name in FRONTIER_DISPLAY.items():
        mdf = mcq_gold[mcq_gold["model"] == model_id]
        row = _get_row(mdf, display_name, "Frontier MCQ", tiers)
        if row is None:
            print(f"  {display_name}: NO DATA")
            continue
        vals = " → ".join(f"{row[t]:.1f}%" for t in tiers if not np.isnan(row[t]))
        print(f"  {display_name}: {row['Overall_fmt']}  [{vals}]")
        all_rows.append(row)

    out = pd.DataFrame(all_rows)
    path = f"{output_folder}/reasoning_vs_frontier_gold.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved: reasoning_vs_frontier_gold.csv")
    return out


if __name__ == "__main__":
    gold_file      = sys.argv[1]
    results_folder = sys.argv[2]
    output_folder  = sys.argv[3] if len(sys.argv) > 3 else "analysis_outputs"

    reasoning_files = {
        "o3":                  f"{results_folder}/gold/results_o3.jsonl",
        "deepseek-v4-pro":     f"{results_folder}/gold/results_deepseek-v4-pro.jsonl",
        "r1-distill-qwen-32b": f"{results_folder}/gold_qwen/results_r1-distill-qwen-32b.jsonl",
        "qwq-32b":             f"{results_folder}/gold_qwen/results_qwq-32b.jsonl",
    }

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    gold_df, gold_ids, tier_map = load_gold(gold_file)
    mcq_gold       = load_mcq_results(results_folder, gold_ids, tier_map)
    reasoning_gold = load_reasoning_results(reasoning_files, gold_ids, tier_map)

    build_gold_table(mcq_gold, reasoning_gold, gold_df, output_folder)