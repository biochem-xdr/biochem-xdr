import os
import json
import time
import logging
from models.model_configs import COOC_MODELS, JUDGE_MODEL

OUTPUT_FOLDER = "cooccurrence_results"
MAX_RETRIES   = 5
RETRY_DELAY   = 10
BATCH_SAVE    = 50

log = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(__file__),
          "prompts/judge_prompt.txt")) as f:
    JUDGE_TEMPLATE = f.read()


def call_judge(question, correct_answer, model_response, api_key):
    import openai
    client  = openai.OpenAI(api_key=api_key)
    prompt  = JUDGE_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        model_response=model_response,
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip().upper()
    if "YES" in raw:
        return True
    if "NO" in raw:
        return False
    log.warning(f'Ambiguous judge response: "{raw}" — defaulting to False')
    return False


def run_judging(model_name, api_key):
    raw_file    = f"{OUTPUT_FOLDER}/cooc_raw_{model_name}.jsonl"
    judged_file = f"{OUTPUT_FOLDER}/cooc_judged_{model_name}.jsonl"

    if not os.path.exists(raw_file):
        log.warning(f"No raw file for {model_name} — run inference first")
        return

    judged_ids = set()
    if os.path.exists(judged_file):
        with open(judged_file) as f:
            for line in f:
                try:
                    judged_ids.add(json.loads(line)["item_id"])
                except Exception:
                    pass

    raw_rows = []
    with open(raw_file) as f:
        for line in f:
            try:
                row = json.loads(line)
                if row["item_id"] not in judged_ids and row.get("openended_response"):
                    raw_rows.append(row)
            except Exception:
                pass

    log.info(f"[{model_name}] Judging: {len(judged_ids):,} done, {len(raw_rows):,} remaining")

    if not raw_rows:
        return

    buffer  = []
    correct = total = errors = 0

    for row in raw_rows:
        is_correct = None
        delay      = RETRY_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                is_correct = call_judge(
                    question=row["question"],
                    correct_answer=row["correct_answer"],
                    model_response=row["openended_response"],
                    api_key=api_key,
                )
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"Judge error item {row['item_id']}: {e}. Waiting {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    log.error(f"Judge FAILED item {row['item_id']}: {e}")
                    is_correct = False
                    errors += 1

        row["is_correct_openended"] = is_correct
        buffer.append(row)
        total += 1
        if is_correct:
            correct += 1

        if len(buffer) >= BATCH_SAVE:
            with open(judged_file, "a") as f:
                for r in buffer:
                    f.write(json.dumps(r) + "\n")
            buffer = []

    if buffer:
        with open(judged_file, "a") as f:
            for r in buffer:
                f.write(json.dumps(r) + "\n")

    acc = correct / total * 100 if total > 0 else 0
    log.info(f"[{model_name}] Done. Open-ended accuracy: {acc:.2f}% ({correct}/{total}). Errors: {errors}")


if __name__ == "__main__":
    import sys

    api_key = os.environ["OPENAI_API_KEY"]

    models = sys.argv[1:] if len(sys.argv) > 1 else [m["name"] for m in COOC_MODELS]
    for model_name in models:
        run_judging(model_name, api_key)