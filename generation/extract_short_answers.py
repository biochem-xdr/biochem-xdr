import os
import pickle
import json
import time
import anthropic
from datetime import datetime


RESUME_PATH = "entity_extraction_resume.pkl"

with open(
    os.path.join(os.path.dirname(__file__), "prompts/short_entity_prompt.txt")
) as f:
    ENTITY_PROMPT = f.read()

with open(
    os.path.join(os.path.dirname(__file__), "prompts/short_distractor_prompt.txt")
) as f:
    DISTRACTOR_PROMPT = f.read()


def extract_entity(qa, client):
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=60,
            temperature=0,
            messages=[{
                "role": "user",
                "content": ENTITY_PROMPT.format(
                    question=qa.get("question", ""),
                    correct_answer=qa.get("correct_answer", ""),
                    path_text=qa.get("path_text", "")[-400:],
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def generate_distractors(qa, correct_entity, client):
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": DISTRACTOR_PROMPT.format(
                    correct_entity=correct_entity,
                    question=qa.get("question", "")[:400],
                ),
            }],
        )
        text = msg.content[0].text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) >= 3:
            return parsed[:3]
        return None
    except Exception:
        return None


def run_extraction(full_dataset, client, project_folder):
    converted = []
    start_idx = 0
    stats     = {"success": 0, "failed_entity": 0, "failed_distractor": 0}

    if os.path.exists(RESUME_PATH):
        print("Resume checkpoint found — loading...")
        with open(RESUME_PATH, "rb") as f:
            resume = pickle.load(f)
        converted = resume["converted"]
        start_idx = resume["next_index"]
        stats     = resume["stats"]
        print(f"Already converted: {len(converted)} | Resuming from {start_idx + 1}")

    start_time = datetime.now()

    for i, qa in enumerate(full_dataset[start_idx:], start=start_idx):
        correct_entity = extract_entity(qa, client)
        time.sleep(0.3)

        if not correct_entity:
            correct_entity = " ".join(qa.get("correct_answer", "").split()[:6])
            stats["failed_entity"] += 1

        wrong_entities = generate_distractors(qa, correct_entity, client)
        time.sleep(0.3)

        if not wrong_entities:
            wrong_entities = [
                " ".join(qa.get("distractor_1", "").split()[:6]),
                " ".join(qa.get("distractor_2", "").split()[:6]),
                " ".join(qa.get("distractor_3", "").split()[:6]),
            ]
            stats["failed_distractor"] += 1
        else:
            stats["success"] += 1

        updated = dict(qa)
        updated["short_correct_answer"] = correct_entity
        updated["short_distractor_1"]   = wrong_entities[0]
        updated["short_distractor_2"]   = wrong_entities[1]
        updated["short_distractor_3"]   = wrong_entities[2]
        updated["answer_format"]        = "short_entity"
        converted.append(updated)

        if (i + 1) % 200 == 0:
            elapsed   = (datetime.now() - start_time).seconds / 60
            rate      = (i - start_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(full_dataset) - i - 1) / rate if rate > 0 else 0
            print(f"[{datetime.now().strftime('%H:%M')}] "
                  f"{i+1}/{len(full_dataset)} | "
                  f"Success: {stats['success']} | "
                  f"ETA: {remaining:.0f}min")

        if (i + 1) % 500 == 0:
            with open(RESUME_PATH, "wb") as f:
                pickle.dump({
                    "converted":  converted,
                    "next_index": i + 1,
                    "stats":      stats,
                }, f)
            print(f"  >>> Checkpoint: {i+1} items")

    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)

    return converted, stats


def save_output(converted, project_folder):
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    pkl = f"{project_folder}/biochem_xdr_short_entities_{ts}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(converted, f)
    print(f"Saved: {pkl}")
    return pkl


if __name__ == "__main__":
    import sys

    dataset_pkl    = sys.argv[1]
    project_folder = sys.argv[2] if len(sys.argv) > 2 else "."
    api_key        = os.environ.get("ANTHROPIC_API_KEY")

    with open(dataset_pkl, "rb") as f:
        full_dataset = pickle.load(f)

    print(f"Dataset: {len(full_dataset)} items")

    client             = anthropic.Anthropic(api_key=api_key)
    converted, stats   = run_extraction(full_dataset, client, project_folder)

    print(f"\nTotal       : {len(converted)}")
    print(f"Full success: {stats['success']}")
    print(f"Entity fallback    : {stats['failed_entity']}")
    print(f"Distractor fallback: {stats['failed_distractor']}")

    save_output(converted, project_folder)