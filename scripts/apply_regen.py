import pickle
import json
import sys
from datetime import datetime


def apply_regen(dataset_pkl, regen_jsonl, output_folder):
    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    regen = {}
    with open(regen_jsonl) as f:
        for line in f:
            row = json.loads(line)
            if row["status"] == "success":
                regen[row["item_id"]] = row

    print(f"Dataset     : {len(dataset)}")
    print(f"Regen rows  : {len(regen)}")

    applied = skipped = kept_original = 0

    for item in dataset:
        item_id = item.get("id", item.get("original_index"))
        row     = regen.get(item_id)

        if row is None:
            kept_original += 1
            continue

        if row["status"] == "success":
            item["short_distractor_1"] = row["new_distractor_1"]
            item["short_distractor_2"] = row["new_distractor_2"]
            item["short_distractor_3"] = row["new_distractor_3"]
            applied += 1
        else:
            kept_original += 1

    print(f"Applied     : {applied}")
    print(f"Kept orig   : {kept_original}")

    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"{output_folder}/biochem_xdr_final_{ts}.pkl"
    with open(out, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    dataset_pkl   = sys.argv[1]
    regen_jsonl   = sys.argv[2]
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "."
    apply_regen(dataset_pkl, regen_jsonl, output_folder)