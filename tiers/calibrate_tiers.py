import os
import pickle
import json
import random
import time
from datetime import datetime
import openai


RESUME_PATH = "difficulty_scoring_resume.pkl"

SYSTEM_PROMPT = (
    "You are an expert biochemist. "
    "Answer the following multiple-choice question by selecting "
    "the single best answer. Respond with only the letter "
    "(A, B, C, or D) corresponding to your choice."
)


def shuffle_choices(qa, seed=None):
    rng = random.Random(seed)
    choices = [
        qa["correct_answer"],
        qa["distractor_1"],
        qa["distractor_2"],
        qa["distractor_3"],
    ]
    rng.shuffle(choices)
    correct_letter = "ABCD"[choices.index(qa["correct_answer"])]
    return choices, correct_letter


def score_item(qa, client, item_id):
    choices, correct_letter = shuffle_choices(qa, seed=item_id)
    user_msg = (
        f"Question:\n{qa['question']}\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=5,
            temperature=0,
        )
        letter = response.choices[0].message.content.strip().upper()[:1]
        return letter == correct_letter
    except Exception as e:
        print(f"  API error item {item_id}: {e}")
        return None


def run_scoring(dataset, client, project_folder):
    scored      = []
    start_index = 0

    if os.path.exists(RESUME_PATH):
        print("Found resume checkpoint — loading...")
        with open(RESUME_PATH, "rb") as f:
            resume = pickle.load(f)
        scored      = resume["scored"]
        start_index = resume["next_index"]
        print(f"Resuming from {start_index:,} ({len(scored):,} already scored)")

    start_time = datetime.now()

    for i, qa in enumerate(dataset[start_index:], start=start_index):
        result = score_item(qa, client, item_id=i)
        if result is None:
            result = False

        record = dict(qa)
        record["gpt4o_correct"] = result
        scored.append(record)

        time.sleep(0.2)

        if (i + 1) % 100 == 0:
            elapsed  = (datetime.now() - start_time).seconds / 60
            session  = (i + 1) - start_index
            rate     = session / elapsed if elapsed > 0 else 0
            eta      = (len(dataset) - i - 1) / rate if rate > 0 else 0
            accuracy = sum(s["gpt4o_correct"] for s in scored) / len(scored)
            print(f"[{datetime.now().strftime('%H:%M')}] "
                  f"{i+1:,}/{len(dataset):,} | "
                  f"GPT-4o accuracy: {accuracy:.1%} | "
                  f"Rate: {rate:.0f}/min | ETA: {eta:.0f}min")

        if (i + 1) % 500 == 0:
            with open(RESUME_PATH, "wb") as f:
                pickle.dump({"scored": scored, "next_index": i + 1}, f)
            ts   = datetime.now().strftime("%Y%m%d_%H%M")
            ckpt = f"{project_folder}/difficulty_scoring_ckpt_{i+1}.pkl"
            with open(ckpt, "wb") as f:
                pickle.dump(scored, f)
            print(f"  >>> Checkpoint {i+1}")

    if os.path.exists(RESUME_PATH):
        os.remove(RESUME_PATH)

    return scored


def save_scored(scored, project_folder):
    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    pkl = f"{project_folder}/biochem_xdr_difficulty_scored_{ts}.pkl"
    js  = f"{project_folder}/biochem_xdr_difficulty_scored_{ts}.json"
    with open(pkl, "wb") as f:
        pickle.dump(scored, f)
    with open(js, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"Saved: {pkl}")
    print(f"Saved: {js}")

    correct = sum(s["gpt4o_correct"] for s in scored)
    print(f"\nGPT-4o accuracy: {correct/len(scored):.1%} ({correct}/{len(scored)})")
    return pkl


if __name__ == "__main__":
    import sys

    dataset_pkl    = sys.argv[1]
    project_folder = sys.argv[2] if len(sys.argv) > 2 else "."
    api_key        = os.environ.get("OPENAI_API_KEY")

    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    print(f"Dataset: {len(dataset):,} items")

    client = openai.OpenAI(api_key=api_key)
    scored = run_scoring(dataset, client, project_folder)
    save_scored(scored, project_folder)