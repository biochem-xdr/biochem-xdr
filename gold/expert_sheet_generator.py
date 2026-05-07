import pickle
import random
import pandas as pd
import sys


def generate_sheets(dataset_pkl, project_folder, n_subsets=12, seed=42):
    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    print(f"Total items: {len(dataset)}")

    rng      = random.Random(seed)
    shuffled = dataset.copy()
    rng.shuffle(shuffled)

    base_size = len(shuffled) // n_subsets
    remainder = len(shuffled) % n_subsets

    subsets = []
    start   = 0
    for i in range(n_subsets):
        size = base_size + (1 if i < remainder else 0)
        subsets.append(shuffled[start:start + size])
        start += size

    labels = list("ABCDEFGHIJKL")[:n_subsets]

    for subset, label in zip(subsets, labels):
        annotation_rows = []
        for j, qa in enumerate(subset):
            annotation_rows.append({
                "Item":             j + 1,
                "Question":         qa.get("question", ""),
                "Correct_Answer":   qa.get("correct_answer", ""),
                "Distractor_1":     qa.get("distractor_1", ""),
                "Distractor_2":     qa.get("distractor_2", ""),
                "Distractor_3":     qa.get("distractor_3", ""),
                "Answer_Accurate":  "",
                "Distractors_OK":   "",
                "Verdict":          "",
                "Notes":            "",
            })

        metadata_rows = []
        for j, qa in enumerate(subset):
            metadata_rows.append({
                "Item":         j + 1,
                "Tier":         qa.get("tier", ""),
                "Crossings":    qa.get("crossing_count", ""),
                "KG_Path":      qa.get("path_text", ""),
                "Start_Domain": qa.get("start_domain", ""),
                "End_Domain":   qa.get("end_domain", ""),
            })

        xlsx_path = f"{project_folder}/annotation_subset_{label}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(annotation_rows).to_excel(
                writer, sheet_name="Annotation", index=False
            )
            pd.DataFrame(metadata_rows).to_excel(
                writer, sheet_name="Metadata_DoNotEdit", index=False
            )

        print(f"Subset {label}: {len(annotation_rows)} items → {xlsx_path}")

    print()
    print("Upload each .xlsx to Google Sheets")
    print("Hide the Metadata tab before sharing with annotator")


if __name__ == "__main__":
    dataset_pkl    = sys.argv[1]
    project_folder = sys.argv[2] if len(sys.argv) > 2 else "."
    generate_sheets(dataset_pkl, project_folder)