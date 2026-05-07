import pickle
import json
import sys
from datetime import datetime


def clean_item(qa, idx):
    return {
        "id":                   idx,
        "question":             qa.get("question", ""),
        "short_correct_answer": qa.get("short_correct_answer", ""),
        "short_distractor_1":   qa.get("short_distractor_1", ""),
        "short_distractor_2":   qa.get("short_distractor_2", ""),
        "short_distractor_3":   qa.get("short_distractor_3", ""),
        "crossing_count":       qa.get("crossing_count", 0),
        "answer_format":        qa.get("answer_format", "short_entity"),
        "path_text":            qa.get("path_text", ""),
        "difficulty_tier":      qa.get("new_tier", qa.get("tier", "")),
    }


def export(dataset_pkl, output_folder):
    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    print(f"Loaded: {len(dataset)} items")

    cleaned = [clean_item(qa, i) for i, qa in enumerate(dataset)]

    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"{output_folder}/biochem_xdr_HUGGINGFACE_{ts}.json"
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)

    print(f"Saved: {path}")
    print(f"Fields: {list(cleaned[0].keys())}")


if __name__ == "__main__":
    dataset_pkl   = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "."
    export(dataset_pkl, output_folder)