import json
import glob
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def load_mcq(results_folder, tier_map):
    exclude = {"deepseek-reasoner", "o1", "qwq-32b", "r1-distill-qwen-32b"}
    rows    = []
    for rf in glob.glob(f"{results_folder}/results_*.jsonl"):
        if any(ex in rf for ex in exclude) or len(open(rf).read()) < 100000:
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
    return df


def load_oe(cooc_folder, tier_map):
    models = ["gpt-4o", "claude-sonnet-4.6",
              "llama-3.3-70b", "llama-3-8b-instruct"]
    rows = []
    for model_id in models:
        fpath = f"{cooc_folder}/cooc_judged_{model_id}.jsonl"
        with open(fpath) as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    df["item_id"] = df["item_id"].astype(int)
    if "tier" not in df.columns:
        df["tier"] = df["item_id"].map(tier_map)
    return df


def table_b6_cooccurrence_by_tier(mcq_df, oe_df, output_folder):
    COOC_MODELS = {
        "gpt-4o":              "GPT-4o",
        "claude-sonnet-4.6":   "Claude Sonnet 4.6",
        "llama-3.3-70b":       "Llama-3.3-70B",
        "llama-3-8b-instruct": "Llama-3-8B",
    }
    tiers = ["T1", "T2", "T3", "T4"]
    rows  = []

    for model_id, model_name in COOC_MODELS.items():
        mcq_m = mcq_df[mcq_df["model"] == model_id]
        oe_m  = oe_df[oe_df["model"]   == model_id]
        for tier in tiers:
            mcq_t   = mcq_m[mcq_m["tier"] == tier]
            oe_t    = oe_m[oe_m["tier"]   == tier]
            mcq_acc = mcq_t["is_correct"].mean() * 100 if len(mcq_t) > 0 else float("nan")
            oe_acc  = oe_t["is_correct_openended"].astype(float).mean() * 100 if len(oe_t) > 0 else float("nan")
            rows.append({
                "Model":   model_name,
                "Tier":    tier,
                "MCQ_%":   round(mcq_acc, 1),
                "OE_%":    round(oe_acc,  1),
                "Gap_pp":  round(oe_acc - mcq_acc, 1),
            })

    out = pd.DataFrame(rows)
    out.to_csv(f"{output_folder}/table_b6_cooccurrence_by_tier.csv", index=False)
    print("Saved table_b6_cooccurrence_by_tier.csv")
    return out


def table_e4_answer_distribution(mcq_df, output_folder):
    MODEL_ORDER = [
        "claude-sonnet-4.6", "gemini-2.5-flash", "qwen3-235b",
        "gpt-4o", "deepseek-v3", "qwen3-32b", "llama-3.3-70b",
        "mistral-7b-instruct", "gemma-3-27b", "llama-3-8b-instruct",
        "openbiollm-8b", "biomistral-7b", "medgemma-27b",
    ]
    rows = []
    for model in MODEL_ORDER:
        mdf = mcq_df[mcq_df["model"] == model]
        if mdf.empty:
            continue
        answers = mdf["model_answer_clean"].astype(str).str.strip().str.upper()
        total   = len(answers)
        rows.append({
            "Model":   model,
            "A_%":     round((answers == "A").sum() / total * 100, 1),
            "B_%":     round((answers == "B").sum() / total * 100, 1),
            "C_%":     round((answers == "C").sum() / total * 100, 1),
            "D_%":     round((answers == "D").sum() / total * 100, 1),
            "Other_%": round((~answers.isin(["A","B","C","D"])).sum() / total * 100, 1),
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"{output_folder}/table_e4_answer_distribution.csv", index=False)
    print("Saved table_e4_answer_distribution.csv")
    return out


def table_gold_by_tier(mcq_df, gold_file, output_folder):
    with open(gold_file) as f:
        gold_data = json.load(f)
    gold_df  = pd.DataFrame(gold_data)
    gold_ids = set(gold_df["id"].astype(int).tolist())

    mcq_gold = mcq_df[mcq_df["item_id"].isin(gold_ids)].copy()

    MODEL_DISPLAY = {
        "claude-sonnet-4.6":   "Claude Sonnet 4.6",
        "gemini-2.5-flash":    "Gemini 2.5 Flash",
        "qwen3-235b":          "Qwen3-235B",
        "gpt-4o":              "GPT-4o",
        "deepseek-v3":         "DeepSeek-V3",
        "qwen3-32b":           "Qwen3-32B",
        "llama-3.3-70b":       "Llama-3.3-70B",
        "mistral-7b-instruct": "Mistral-7B",
        "gemma-3-27b":         "Gemma-3-27B",
        "llama-3-8b-instruct": "Llama-3-8B",
        "openbiollm-8b":       "OpenBioLLM-8B",
        "biomistral-7b":       "BioMistral-7B",
        "medgemma-27b":        "MedGemma-27B",
    }

    tiers = ["T1", "T2", "T3", "T4"]
    rows  = []
    for model, display_name in MODEL_DISPLAY.items():
        mdf = mcq_gold[mcq_gold["model"] == model]
        if mdf.empty:
            continue
        row = {"Model": display_name, "Overall": round(mdf["is_correct"].mean() * 100, 1)}
        for tier in tiers:
            tdf = mdf[mdf["tier"] == tier]
            row[tier] = round(tdf["is_correct"].mean() * 100, 1) if len(tdf) > 0 else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(f"{output_folder}/gold_subset_results_by_tier.csv", index=False)
    print("Saved gold_subset_results_by_tier.csv")
    return out


if __name__ == "__main__":
    results_folder = sys.argv[1]
    cooc_folder    = sys.argv[2]
    gold_file      = sys.argv[3]
    tier_file      = sys.argv[4]
    output_folder  = sys.argv[5] if len(sys.argv) > 5 else "analysis_outputs"

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    tier_df = pd.read_csv(tier_file)
    tier_df["item_id"] = tier_df["item_id"].astype(int)
    tier_map = dict(zip(tier_df["item_id"], tier_df["tier_aggregate"]))

    mcq_df = load_mcq(results_folder, tier_map)
    oe_df  = load_oe(cooc_folder, tier_map)

    table_b6_cooccurrence_by_tier(mcq_df, oe_df, output_folder)
    table_e4_answer_distribution(mcq_df, output_folder)
    table_gold_by_tier(mcq_df, gold_file, output_folder)