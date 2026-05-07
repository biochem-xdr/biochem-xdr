import json
import re
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


TIER_TARGETS = {"T1": 150, "T2": 150, "T3": 120, "T4": 80}


def normalize_q(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_datasets(hf_file, gold_file, verified_500_file):
    with open(hf_file) as f:
        hf_data = json.load(f)
    hf_df = pd.DataFrame(hf_data)
    hf_df["id"] = hf_df["id"].astype(int)
    hf_df["q_norm"] = hf_df["question"].apply(normalize_q)

    hf_q_lookup = {}
    for _, row in hf_df.iterrows():
        q = row["q_norm"]
        if q not in hf_q_lookup:
            hf_q_lookup[q] = int(row["id"])

    with open(gold_file) as f:
        gold_350 = json.load(f)
    gold_350_df = pd.DataFrame(gold_350)
    gold_350_df["id"] = gold_350_df["id"].astype(int)
    gold_350_ids = set(gold_350_df["id"].tolist())

    with open(verified_500_file) as f:
        verified_500 = json.load(f)
    verified_500_df = pd.DataFrame(verified_500)
    verified_500_df["id"] = verified_500_df["id"].astype(int)
    verified_500_ids = set(verified_500_df["id"].tolist())

    return hf_df, hf_data, hf_q_lookup, gold_350_ids, verified_500_ids


def match_questions(sheet_items, q_col, hf_q_lookup):
    matched = []
    for _, row in sheet_items.iterrows():
        q_norm = normalize_q(str(row[q_col]))
        hf_id  = hf_q_lookup.get(q_norm)
        if hf_id is None:
            prefix = q_norm[:200]
            for hf_q, hf_i in hf_q_lookup.items():
                if hf_q[:200] == prefix:
                    hf_id = hf_i
                    break
        if hf_id is not None:
            matched.append(hf_id)
    return set(matched)


def compute_multi_signal_scores(hf_df, hf_data, gold_350_ids, tier_map,
                                mcq_item, oe_item):
    non_gold_df = hf_df[~hf_df["id"].isin(gold_350_ids)].copy()
    non_gold_df["tier"] = non_gold_df["id"].map(tier_map)

    scores = []
    for _, row in non_gold_df.iterrows():
        item_id  = int(row["id"])
        tier     = tier_map.get(item_id, "T2")
        crossing = int(row.get("crossing_count", 3))

        mcq_row = mcq_item[mcq_item["item_id"] == item_id]
        if len(mcq_row) > 0:
            frac      = mcq_row["mcq_frac"].values[0]
            n_correct = mcq_row["mcq_correct_count"].values[0]
            if tier == "T1":   s2 = 1 if frac >= 0.70 else 0
            elif tier == "T2": s2 = 1 if frac >= 0.50 else 0
            elif tier == "T3": s2 = 1 if frac >= 0.30 else 0
            else:              s2 = 1 if (frac <= 0.20 and n_correct >= 1) else 0
        else:
            frac = float("nan"); n_correct = 0; s2 = 0

        oe_row = oe_item[oe_item["item_id"] == item_id]
        if len(oe_row) > 0:
            oe_correct = oe_row["oe_correct_count"].values[0]
            s3 = 1 if (tier in ["T1", "T2"] and oe_correct >= 3) or \
                      (tier in ["T3", "T4"] and oe_correct >= 1) else 0
        else:
            oe_correct = 0; s3 = 0

        scores.append({
            "item_id":        item_id,
            "tier":           tier,
            "crossing_count": crossing,
            "mcq_frac":       round(frac, 3) if not np.isnan(frac) else 0,
            "mcq_n_correct":  n_correct,
            "oe_n_correct":   oe_correct,
            "signal_kg":      1,
            "signal_mcq":     s2,
            "signal_oe":      s3,
            "signal_clean":   1 if 2 <= crossing <= 8 else 0,
            "total_score":    1 + s2 + s3 + (1 if 2 <= crossing <= 8 else 0),
        })

    return pd.DataFrame(scores)


def select_verified(scores_df, threshold=3):
    qualified = scores_df[scores_df["total_score"] >= threshold].copy()
    parts = []
    for tier, target in TIER_TARGETS.items():
        tier_q = qualified[qualified["tier"] == tier].copy()
        if tier == "T4":
            tier_q = tier_q.sort_values(
                ["total_score", "oe_n_correct", "mcq_frac"], ascending=False)
        else:
            tier_q = tier_q.sort_values(
                ["total_score", "mcq_frac", "oe_n_correct"], ascending=False)
        parts.append(tier_q.head(min(target, len(tier_q))))

    return pd.concat(parts).reset_index(drop=True)


def assemble_verified(gold_350_ids, gold_extra_ids, verified_500_ids,
                      keep_ids, remove_ids):
    c1 = gold_350_ids
    c2 = gold_extra_ids - gold_350_ids
    c3 = verified_500_ids - gold_350_ids - c2
    c4 = keep_ids - gold_350_ids - c2 - c3

    raw_union       = c1 | c2 | c3 | c4
    remove_safe     = remove_ids - gold_350_ids
    final_ids       = raw_union - remove_safe

    print(f"\nVerified assembly:")
    print(f"  Gold 350:          {len(c1):,}")
    print(f"  Gold extras:       {len(c2):,}")
    print(f"  Verified 500:      {len(c3):,}")
    print(f"  Sheet KEEP:        {len(c4):,}")
    print(f"  Removed:           {len(raw_union & remove_safe):,}")
    print(f"  Final:             {len(final_ids):,}")

    return final_ids, c1, c2, c3, c4


def save_verified(hf_df, final_ids, components, output_folder):
    c1, c2, c3, c4 = components
    tier_col = "difficulty_tier" if "difficulty_tier" in hf_df.columns else "tier"

    out_df = hf_df[hf_df["id"].isin(final_ids)].copy()

    def get_source(item_id):
        s = []
        if item_id in c1: s.append("gold_350")
        if item_id in c2: s.append("gold_extra")
        if item_id in c3: s.append("verified_500")
        if item_id in c4: s.append("sheet_keep")
        return "+".join(s) if s else "unknown"

    out_df["source"] = out_df["id"].apply(get_source)

    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"{output_folder}/biochem_xdr_VERIFIED_{ts}.json"
    records = out_df.drop(columns=["q_norm", "source"], errors="ignore").to_dict(orient="records")
    with open(path, "w") as f:
        import json
        json.dump(records, f, indent=2)

    print(f"\nSaved: biochem_xdr_VERIFIED_{ts}.json ({len(out_df):,} items)")
    print(f"Gold 350 included: {'YES' if c1.issubset(set(out_df['id'].tolist())) else 'CHECK'}")
    return path


if __name__ == "__main__":
    hf_file         = sys.argv[1]
    gold_file       = sys.argv[2]
    verified_500    = sys.argv[3]
    results_folder  = sys.argv[4]
    cooc_folder     = sys.argv[5]
    tier_file       = sys.argv[6]
    output_folder   = sys.argv[7] if len(sys.argv) > 7 else "analysis_outputs"

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    import json as _json
    import glob as _glob

    hf_df, hf_data, hf_q_lookup, gold_350_ids, verified_500_ids = load_datasets(
        hf_file, gold_file, verified_500
    )

    tier_df = pd.read_csv(tier_file)
    tier_df["item_id"] = tier_df["item_id"].astype(int)
    tier_map = dict(zip(tier_df["item_id"], tier_df["tier_aggregate"]))

    # Load MCQ and OE results for signal scoring
    mcq_rows = []
    for rf in _glob.glob(f"{results_folder}/results_*.jsonl"):
        if any(ex in rf for ex in ["deepseek-reasoner", "o1", "qwq-32b", "r1-distill"]):
            continue
        with open(rf) as f:
            for line in f:
                try: mcq_rows.append(_json.loads(line.strip()))
                except Exception: pass

    mcq_df = pd.DataFrame(mcq_rows)
    mcq_df["item_id"]    = mcq_df["item_id"].astype(int)
    mcq_df["is_correct"] = mcq_df["is_correct"].astype(bool)
    non_gold_ids = set(hf_df[~hf_df["id"].isin(gold_350_ids)]["id"].tolist())
    mcq_df = mcq_df[mcq_df["item_id"].isin(non_gold_ids)]
    mcq_item = mcq_df.groupby("item_id")["is_correct"].agg(["sum", "count"]).reset_index()
    mcq_item.columns = ["item_id", "mcq_correct_count", "mcq_total_models"]
    mcq_item["mcq_frac"] = mcq_item["mcq_correct_count"] / mcq_item["mcq_total_models"]

    oe_rows = []
    for model_id in ["gpt-4o", "claude-sonnet-4.6", "llama-3.3-70b", "llama-3-8b-instruct"]:
        fpath = f"{cooc_folder}/cooc_judged_{model_id}.jsonl"
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                try: oe_rows.append(_json.loads(line.strip()))
                except Exception: pass

    oe_df = pd.DataFrame(oe_rows) if oe_rows else pd.DataFrame()
    if not oe_df.empty:
        oe_df["item_id"] = oe_df["item_id"].astype(int)
        oe_df = oe_df[oe_df["item_id"].isin(non_gold_ids)]
        oe_item = oe_df.groupby("item_id")["is_correct_openended"].agg(["sum", "count"]).reset_index()
        oe_item.columns = ["item_id", "oe_correct_count", "oe_total_models"]
    else:
        oe_item = pd.DataFrame(columns=["item_id", "oe_correct_count", "oe_total_models"])

    scores_df    = compute_multi_signal_scores(hf_df, hf_data, gold_350_ids, tier_map, mcq_item, oe_item)
    verified_500_selected = select_verified(scores_df)

    import os
    save_verified(hf_df, set(verified_500_selected["item_id"].tolist()),
                  (gold_350_ids, set(), set(), set()), output_folder)