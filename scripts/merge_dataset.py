import pickle
import sys
from datetime import datetime


def merge(retiered_pkl, short_entity_pkl, filtered_pkl, output_folder):
    with open(retiered_pkl, "rb") as f:
        retiered = pickle.load(f)
    with open(short_entity_pkl, "rb") as f:
        short_entity = pickle.load(f)
    with open(filtered_pkl, "rb") as f:
        filtered = pickle.load(f)

    print(f"Retiered    : {len(retiered)}")
    print(f"Short entity: {len(short_entity)}")
    print(f"Filtered    : {len(filtered)}")

    short_index = {item["question"]: item for item in short_entity}
    filtered_qs = {item["question"] for item in filtered}

    merged        = []
    matched       = 0
    missing_short = 0
    not_in_filter = 0

    for item in retiered:
        q = item["question"]

        if q not in filtered_qs:
            not_in_filter += 1
            continue

        se_item = short_index.get(q)
        if se_item:
            item["short_correct_answer"] = se_item.get("short_correct_answer", "")
            item["short_distractor_1"]   = se_item.get("short_distractor_1", "")
            item["short_distractor_2"]   = se_item.get("short_distractor_2", "")
            item["short_distractor_3"]   = se_item.get("short_distractor_3", "")
            item["answer_format"]        = "short_entity"
            matched += 1
        else:
            missing_short += 1

        merged.append(item)

    print(f"\nNot in filter  : {not_in_filter}")
    print(f"Short matched  : {matched}")
    print(f"Short missing  : {missing_short}")
    print(f"Final total    : {len(merged)}")

    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = f"{output_folder}/biochem_xdr_merged_{ts}.pkl"
    with open(out, "wb") as f:
        pickle.dump(merged, f)
    print(f"\nSaved: {out}")
    return out


if __name__ == "__main__":
    retiered_pkl    = sys.argv[1]
    short_entity_pkl = sys.argv[2]
    filtered_pkl    = sys.argv[3]
    output_folder   = sys.argv[4] if len(sys.argv) > 4 else "."
    merge(retiered_pkl, short_entity_pkl, filtered_pkl, output_folder)