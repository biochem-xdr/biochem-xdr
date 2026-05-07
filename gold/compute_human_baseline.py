import glob
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def load_expert_sheets(expert_folder):
    files = sorted(glob.glob(f"{expert_folder}/Expert_*.xlsx"))
    return {
        os.path.basename(f).replace("_BioChem_XDR_Evaluation.xlsx", "")
                            .replace(".xlsx", ""): f
        for f in files
    }


def load_answer_key(answer_key_file):
    df = pd.read_excel(answer_key_file)
    df.columns = df.columns.str.strip()
    return df


def load_expert_data(expert_files):
    expert_data = {}
    for expert_name, filepath in expert_files.items():
        df = pd.read_excel(filepath, sheet_name="Questions", header=2)
        df.columns = df.columns.str.strip()

        df = df[df["Question"].notna() &
                (df["Question"].astype(str).str.len() > 10)].copy()
        df["Q_No"] = pd.to_numeric(df["#"], errors="coerce")
        df = df.dropna(subset=["Q_No"])
        df["Q_No"] = df["Q_No"].astype(int)

        df["expert_answer"] = (df["Your Answer"]
                               .astype(str).str.strip().str.upper().str[:1])
        df["expert_answer"] = df["expert_answer"].replace({"N": "", " ": ""})
        df["is_answered"]   = df["expert_answer"].isin(["A", "B", "C", "D"])

        expert_data[expert_name] = df
    return expert_data


def grade(expert_data, answer_key_df):
    all_results    = []
    expert_summary = {}

    for expert_name, df in expert_data.items():
        expert_key = answer_key_df[answer_key_df["Expert"] == expert_name].copy()
        expert_key["Q_No"] = expert_key["Q_No"].astype(int)

        merged = df[df["is_answered"]].merge(
            expert_key[["Q_No", "Item_ID", "Tier", "Correct_Letter", "Correct_Answer"]],
            on="Q_No", how="inner"
        )

        if len(merged) == 0:
            print(f"{expert_name}: no rows merged — check Q_No alignment")
            continue

        merged["is_correct"] = (
            merged["expert_answer"].str.upper() ==
            merged["Correct_Letter"].str.strip().str.upper()
        )

        total    = len(merged)
        correct  = merged["is_correct"].sum()
        accuracy = correct / total * 100

        print(f"\n{expert_name}: {correct}/{total} ({accuracy:.1f}%)")
        for tier in ["T1", "T2", "T3", "T4"]:
            tdf = merged[merged["Tier"] == tier]
            if len(tdf) > 0:
                print(f"  {tier}: {tdf['is_correct'].sum()}/{len(tdf)} "
                      f"({tdf['is_correct'].mean()*100:.1f}%)")

        expert_summary[expert_name] = {
            "answered": total,
            "correct":  correct,
            "accuracy": accuracy,
        }

        merged["Expert"] = expert_name
        all_results.append(merged[[
            "Expert", "Q_No", "Item_ID", "Tier",
            "expert_answer", "Correct_Letter", "Correct_Answer", "is_correct"
        ]])

    return all_results, expert_summary


def compute_kappa(combined):
    overlap = combined.groupby("Item_ID").filter(
        lambda x: x["Expert"].nunique() > 1
    )
    if len(overlap) == 0:
        return None, None, 0

    pivot = overlap.pivot_table(
        index="Item_ID", columns="Expert",
        values="expert_answer", aggfunc="first"
    )

    agreements = []
    for i, e1 in enumerate(pivot.columns):
        for e2 in pivot.columns[i + 1:]:
            shared = pivot[[e1, e2]].dropna()
            if len(shared) > 0:
                agree = (shared[e1] == shared[e2]).mean() * 100
                agreements.append(agree)
                print(f"  {e1} vs {e2}: {agree:.1f}% ({len(shared)} items)")

    if not agreements:
        return None, None, overlap["Item_ID"].nunique()

    avg_agree = np.mean(agreements)
    kappa     = (avg_agree / 100 - 0.25) / (1 - 0.25)
    interp    = ("almost perfect" if kappa >= 0.8 else
                 "substantial"    if kappa >= 0.6 else
                 "moderate"       if kappa >= 0.4 else "fair")
    print(f"\n  Avg agreement: {avg_agree:.1f}%")
    print(f"  Cohen's κ:     {kappa:.3f} ({interp})")
    return avg_agree, kappa, overlap["Item_ID"].nunique()


def run(expert_folder, answer_key_file, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    expert_files = load_expert_sheets(expert_folder)
    print(f"Experts found: {list(expert_files.keys())}")

    answer_key_df = load_answer_key(answer_key_file)
    expert_data   = load_expert_data(expert_files)

    print("\n" + "=" * 70)
    print("GRADING RESULTS")
    print("=" * 70)

    all_results, expert_summary = grade(expert_data, answer_key_df)

    if not all_results:
        print("No results — check file paths and column names")
        return

    combined      = pd.concat(all_results, ignore_index=True)
    total_correct = combined["is_correct"].sum()
    total_items   = len(combined)
    overall_acc   = total_correct / total_items * 100

    print(f"\n{'='*70}")
    print(f"OVERALL: {total_correct}/{total_items} ({overall_acc:.1f}%)")

    print(f"\n{'='*70}")
    print("INTER-ANNOTATOR AGREEMENT")
    print(f"{'='*70}")
    avg_agree, kappa, n_overlap = compute_kappa(combined)

    print(f"\n{'='*70}")
    print("NUMBERS FOR PAPER")
    print(f"{'='*70}")
    print(f"Human accuracy:  {overall_acc:.1f}%")
    print(f"Items:           {total_items:,}")
    print(f"Experts:         {len(expert_summary)}")
    if kappa is not None:
        print(f"Agreement:       {avg_agree:.1f}%")
        print(f"Kappa:           κ = {kappa:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"{output_folder}/human_expert_results_{ts}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: human_expert_results_{ts}.csv")


if __name__ == "__main__":
    expert_folder   = sys.argv[1]
    answer_key_file = sys.argv[2]
    output_folder   = sys.argv[3] if len(sys.argv) > 3 else "analysis_outputs"
    run(expert_folder, answer_key_file, output_folder)