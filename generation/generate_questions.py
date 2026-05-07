import os
import sys
import pickle
import json
import time
import anthropic
from datetime import datetime

paths_pkl      = sys.argv[1]
project_folder = sys.argv[2] if len(sys.argv) > 2 else "."

CHECKPOINT_DIR = project_folder
RESUME_PATH    = f"{CHECKPOINT_DIR}/biochem_xdr_qa_resume_checkpoint.pkl"
ERROR_LOG_PATH = f"{CHECKPOINT_DIR}/biochem_xdr_gapfill_errors.json"

with open(
    os.path.join(os.path.dirname(__file__), "prompts/generation_prompt.txt")
) as f:
    GENERATION_PROMPT = f.read()


def generate_qa_pair(path_text, client):
    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": GENERATION_PROMPT.format(path_text=path_text),
            }],
        )
        text = message.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"API error: {e}")
        return None


def validate_qa_item(qa):
    issues = []
    if not qa.get("question", "").endswith("?"):
        issues.append("question_not_question")
    if len(qa.get("correct_answer", "").split()) < 20:
        issues.append("answer_too_short")
    for d in ["distractor_1", "distractor_2", "distractor_3"]:
        if not qa.get(d, "").strip():
            issues.append(f"missing_{d}")
    cross        = qa.get("cross_domain_connection", "")
    domain_terms = ["enzyme", "disease", "pathway",
                    "metabolic", "kinetics", "mechanism"]
    if sum(1 for t in domain_terms if t.lower() in cross.lower()) < 2:
        issues.append("weak_cross_domain_field")
    return len(issues) == 0, issues


def _add_metadata(qa, path_info):
    qa["tier"]              = path_info["tier"]
    qa["crossing_count"]    = path_info["crossing_count"]
    qa["path_text"]         = path_info["path_text"]
    qa["domains_traversed"] = path_info["domains_traversed"]
    qa["start_domain"]      = path_info.get("start_domain", "")
    qa["end_domain"]        = path_info.get("end_domain", "")
    return qa


def generate_dataset(paths_by_tier, client):
    """
    Main generation loop with resume support.
    Checkpoints every 500 paths to CHECKPOINT_DIR.
    """
    full_dataset = []
    start_index  = 0
    stats        = {
        "generated":        0,
        "generation_failed": 0,
        "failed_validation": 0,
        "by_tier":          {"T1": 0, "T2": 0, "T3": 0, "T4": 0},
    }

    if os.path.exists(RESUME_PATH):
        print("Found resume checkpoint — loading...")
        with open(RESUME_PATH, "rb") as f:
            resume = pickle.load(f)
        full_dataset = resume["dataset"]
        start_index  = resume["next_index"]
        stats        = resume["stats"]
        print(f"Resuming from index {start_index:,} "
              f"({len(full_dataset):,} pairs already saved)")

    all_paths = []
    for tier, paths in paths_by_tier.items():
        for p in paths:
            p_copy         = dict(p)
            p_copy["tier"] = tier
            all_paths.append(p_copy)

    print(f"Total paths : {len(all_paths):,}")
    print(f"Remaining   : {len(all_paths) - start_index:,}")

    start_time = datetime.now()

    for i, path_info in enumerate(
        all_paths[start_index:], start=start_index
    ):
        try:
            qa = generate_qa_pair(path_info["path_text"], client)
        except Exception:
            stats["generation_failed"] += 1
            continue

        if qa is None:
            stats["generation_failed"] += 1
            continue

        valid, _ = validate_qa_item(qa)
        if not valid:
            stats["failed_validation"] += 1
            continue

        full_dataset.append(_add_metadata(qa, path_info))
        stats["generated"]           += 1
        stats["by_tier"][path_info["tier"]] += 1

        time.sleep(0.3)

        if (i + 1) % 100 == 0:
            elapsed  = (datetime.now() - start_time).seconds / 60
            session  = (i + 1) - start_index
            rate     = session / elapsed if elapsed > 0 else 0
            eta      = (len(all_paths) - i - 1) / rate if rate > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M')}] "
                  f"{i+1:,}/{len(all_paths):,} | "
                  f"Generated: {stats['generated']:,} | "
                  f"Failed: {stats['generation_failed'] + stats['failed_validation']:,} | "
                  f"Rate: {rate:.0f}/min | ETA: {eta:.0f}min")

        if (i + 1) % 500 == 0:
            ckpt = f"{CHECKPOINT_DIR}/checkpoint_qa_{i+1}.pkl"
            with open(ckpt, "wb") as f:
                pickle.dump(full_dataset, f)
            with open(RESUME_PATH, "wb") as f:
                pickle.dump({
                    "dataset":    full_dataset,
                    "next_index": i + 1,
                    "stats":      stats,
                }, f)
            print(f"  >>> Checkpoint {i+1} | "
                  f"{stats['generated']} pairs | resume updated")

    # Clean resume checkpoint on clean finish
    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)

    return full_dataset, stats


def gapfill(all_paths, full_dataset, client):
    """
    Retry paths that failed in the main generation run.
    Identifies failures by comparing path_text against completed set.
    """
    completed  = {qa["path_text"] for qa in full_dataset}
    failed     = [p for p in all_paths if p["path_text"] not in completed]
    error_log  = []
    start_index = 0
    gap_stats  = {
        "generated":         0,
        "failed":            0,
        "api_errors":        0,
        "json_errors":       0,
        "validation_errors": 0,
        "other_errors":      0,
    }

    gapfill_resume = f"{CHECKPOINT_DIR}/biochem_xdr_qa_gapfill_resume.pkl"
    if os.path.exists(gapfill_resume):
        with open(gapfill_resume, "rb") as f:
            saved = pickle.load(f)
        start_index = saved["next_index"]
        gap_stats   = saved["stats"]
        print(f"Resuming gap-fill from index {start_index} "
              f"({gap_stats['generated']} already recovered)")

    print(f"Failed paths to retry: {len(failed)}")

    for i, path_info in enumerate(
        failed[start_index:], start=start_index
    ):
        try:
            qa = generate_qa_pair(path_info["path_text"], client)
        except Exception as e:
            err  = str(e)
            etype = "api_error" if "Error code:" in err else "other_error"
            gap_stats["failed"]       += 1
            gap_stats[f"{etype}s"]    += 1
            error_log.append({
                "index":      i,
                "tier":       path_info.get("tier"),
                "error_type": etype,
                "message":    err,
                "path_preview": path_info.get("path_text", "")[:200],
            })
            continue

        if qa is None:
            gap_stats["failed"]      += 1
            gap_stats["json_errors"] += 1
            continue

        valid, issues = validate_qa_item(qa)
        if not valid:
            gap_stats["failed"]            += 1
            gap_stats["validation_errors"] += 1
            continue

        full_dataset.append(_add_metadata(qa, path_info))
        gap_stats["generated"] += 1

        time.sleep(0.3)

        if (i + 1) % 100 == 0:
            print(f"[{datetime.now().strftime('%H:%M')}] "
                  f"{i+1}/{len(failed)} | "
                  f"Recovered: {gap_stats['generated']} | "
                  f"API: {gap_stats['api_errors']} | "
                  f"JSON: {gap_stats['json_errors']} | "
                  f"Validation: {gap_stats['validation_errors']}")

            with open(f"{CHECKPOINT_DIR}/gapfill_checkpoint_{i+1}.pkl", "wb") as f:
                pickle.dump(full_dataset, f)
            with open(gapfill_resume, "wb") as f:
                pickle.dump({"next_index": i + 1, "stats": gap_stats}, f)
            with open(ERROR_LOG_PATH, "w") as f:
                json.dump(error_log, f, indent=2)

    if os.path.exists(gapfill_resume):
        os.remove(gapfill_resume)

    return full_dataset, gap_stats, error_log


def save_dataset(full_dataset, label="raw"):
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    pkl  = f"{CHECKPOINT_DIR}/biochem_xdr_qa_{label}_{ts}.pkl"
    js   = f"{CHECKPOINT_DIR}/biochem_xdr_qa_{label}_{ts}.json"
    with open(pkl, "wb") as f:
        pickle.dump(full_dataset, f)
    with open(js, "w") as f:
        json.dump(full_dataset, f, indent=2)
    print(f"Saved pkl  : {pkl}")
    print(f"Saved json : {js}")
    return pkl, js


if __name__ == "__main__":
    import sys

    paths_pkl = sys.argv[1]
    api_key   = os.environ.get("ANTHROPIC_API_KEY")

    with open(paths_pkl, "rb") as f:
        paths_by_tier = pickle.load(f)

    client = anthropic.Anthropic(api_key=api_key)

    full_dataset, stats = generate_dataset(paths_by_tier, client)
    print(f"\nGenerated: {stats['generated']:,}")
    print(f"Failed:    {stats['generation_failed'] + stats['failed_validation']:,}")
    for tier, count in stats["by_tier"].items():
        print(f"  {tier}: {count:,}")

    # Gap-fill failures
    all_paths = []
    for tier, paths in paths_by_tier.items():
        for p in paths:
            p_copy         = dict(p)
            p_copy["tier"] = tier
            all_paths.append(p_copy)

    full_dataset, gap_stats, _ = gapfill(all_paths, full_dataset, client)
    print(f"\nGap-fill recovered: {gap_stats['generated']:,}")
    print(f"Permanently failed: {gap_stats['failed']:,}")

    save_dataset(full_dataset, label="gapfilled")