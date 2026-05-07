import pickle
import json
import re
from datetime import datetime


def filter_dataset(full_dataset):
    passed       = []
    failed       = []
    issue_counts = {}
    tier_results = {
        t: {"passed": 0, "failed": 0}
        for t in ["T1", "T2", "T3", "T4"]
    }

    start_time = datetime.now()
    total      = len(full_dataset)

    for i, qa in enumerate(full_dataset):
        passed_item, issues = check(qa)
        tier = qa.get("tier", "unknown")

        if passed_item:
            passed.append(qa)
            if tier in tier_results:
                tier_results[tier]["passed"] += 1
        else:
            failed.append({"qa": qa, "issues": issues})
            if tier in tier_results:
                tier_results[tier]["failed"] += 1
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        if (i + 1) % 500 == 0:
            elapsed = (datetime.now() - start_time).seconds / 60
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (total - i - 1) / rate if rate > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M')}] "
                  f"{i+1:,}/{total:,} | "
                  f"Passed: {len(passed):,} | "
                  f"Failed: {len(failed):,} | "
                  f"Yield: {len(passed)/(i+1):.1%} | "
                  f"ETA: {eta:.0f}min")

    return passed, failed, issue_counts, tier_results


def check(qa):
    issues = []

    q      = qa.get("question", "")
    a      = qa.get("correct_answer", "")
    d1     = qa.get("distractor_1", "")
    d2     = qa.get("distractor_2", "")
    d3     = qa.get("distractor_3", "")
    cross  = qa.get("cross_domain_connection", "")
    path   = qa.get("path_text", "")
    tier   = qa.get("tier", "")
    crossings = qa.get("crossing_count", 0)

    if not q.strip().endswith("?"):
        issues.append("no_question_mark")

    if len(q.split()) < 15:
        issues.append("question_too_short")

    if len(a.split()) < 20:
        issues.append("answer_too_short")

    for name, d in [("d1", d1), ("d2", d2), ("d3", d3)]:
        if not d.strip():
            issues.append(f"missing_{name}")
        elif len(d.split()) < 10:
            issues.append(f"{name}_too_short")

    a_lower = a.lower().strip()
    for name, d in [("d1", d1), ("d2", d2), ("d3", d3)]:
        if d.lower().strip() == a_lower:
            issues.append(f"{name}_identical_to_answer")

    if d1.lower().strip() == d2.lower().strip():
        issues.append("d1_d2_identical")
    if d1.lower().strip() == d3.lower().strip():
        issues.append("d1_d3_identical")
    if d2.lower().strip() == d3.lower().strip():
        issues.append("d2_d3_identical")

    domain_terms = [
        "enzyme", "disease", "pathway", "metabolic",
        "kinetics", "mechanism", "reaction", "compound",
    ]
    if sum(1 for t in domain_terms if t.lower() in cross.lower()) < 2:
        issues.append("weak_cross_domain_field")

    if crossings >= 2 and len(cross.split()) < 20:
        issues.append("cross_domain_field_too_brief")

    path_entities = []
    for line in path.split("\n"):
        parts = re.findall(r"^(.+?)\s*\[", line)
        if parts:
            first = parts[0].split(",")[0].strip()
            if len(first) > 3:
                path_entities.append(first.lower())

    if path_entities and not any(e in a.lower() for e in path_entities):
        issues.append("answer_missing_path_entities")

    if tier == "T1" and crossings > 2:
        issues.append("wrong_tier_assignment")
    elif tier == "T2" and crossings != 3:
        issues.append("wrong_tier_assignment")
    elif tier == "T3" and crossings != 4:
        issues.append("wrong_tier_assignment")
    elif tier == "T4" and crossings < 5:
        issues.append("wrong_tier_assignment")

    return len(issues) == 0, issues


def save_results(passed, issue_counts, tier_results, project_folder):
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    pkl  = f"{project_folder}/biochem_xdr_qa_filtered_{ts}.pkl"
    js   = f"{project_folder}/biochem_xdr_qa_filtered_{ts}.json"
    analysis = f"{project_folder}/biochem_xdr_filter_analysis_{ts}.json"

    with open(pkl, "wb") as f:
        pickle.dump(passed, f)
    with open(js, "w") as f:
        json.dump(passed, f, indent=2)

    total_input = sum(
        v["passed"] + v["failed"] for v in tier_results.values()
    )
    with open(analysis, "w") as f:
        json.dump({
            "timestamp":       ts,
            "input_count":     total_input,
            "passed_count":    len(passed),
            "failed_count":    total_input - len(passed),
            "yield_pct":       len(passed) / total_input if total_input else 0,
            "by_tier":         tier_results,
            "failure_reasons": issue_counts,
        }, f, indent=2)

    print(f"Saved: {pkl}")
    print(f"Saved: {js}")
    print(f"Saved: {analysis}")
    return pkl, js, analysis


if __name__ == "__main__":
    import sys

    dataset_pkl    = sys.argv[1]
    project_folder = sys.argv[2] if len(sys.argv) > 2 else "."

    with open(dataset_pkl, "rb") as f:
        full_dataset = pickle.load(f)

    print(f"Input: {len(full_dataset):,} QA pairs")

    passed, failed, issue_counts, tier_results = filter_dataset(full_dataset)

    print(f"\nPassed  : {len(passed):,} ({len(passed)/len(full_dataset):.1%})")
    print(f"Failed  : {len(failed):,}")
    print(f"\nBy tier:")
    for tier in ["T1", "T2", "T3", "T4"]:
        r = tier_results[tier]
        n = r["passed"] + r["failed"]
        print(f"  {tier}: {r['passed']}/{n} ({r['passed']/n:.1%})" if n else f"  {tier}: 0")

    print(f"\nTop failure reasons:")
    for issue, count in sorted(
        issue_counts.items(), key=lambda x: -x[1]
    )[:10]:
        print(f"  {issue:<40}: {count} ({count/len(full_dataset):.1%})")

    save_results(passed, issue_counts, tier_results, project_folder)