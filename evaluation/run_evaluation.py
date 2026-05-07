import os
import json
import random
import time
import re
import logging
import sys
from datetime import datetime
from pathlib import Path

from models.model_configs import MCQ_MODELS, REASONING_MODELS

RANDOM_SEED          = 42
MAX_RETRIES          = 5
RETRY_DELAY          = 10
BATCH_SAVE           = 50
SLEEP_BETWEEN_CALLS  = 0.1

log = logging.getLogger(__name__)


def build_prompt(item, seed_offset=0):
    choices_pool = [
        {"text": item["short_correct_answer"], "is_correct": True},
        {"text": item["short_distractor_1"],   "is_correct": False},
        {"text": item["short_distractor_2"],   "is_correct": False},
        {"text": item["short_distractor_3"],   "is_correct": False},
    ]

    rng = random.Random(item["id"] + seed_offset + RANDOM_SEED)
    rng.shuffle(choices_pool)

    labels         = ["A", "B", "C", "D"]
    correct_letter = None
    choices_text   = ""

    for i, choice in enumerate(choices_pool):
        choices_text += f"{labels[i]}. {choice['text']}\n"
        if choice["is_correct"]:
            correct_letter = labels[i]

    prompt = (
        "You are a biochemistry expert. Answer the following multiple choice "
        "question by selecting the single best answer.\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Answer choices:\n{choices_text.strip()}\n\n"
        "Reply with ONLY the letter of your answer (A, B, C, or D). Nothing else."
    )
    return prompt, correct_letter


def parse_answer(raw_response):
    if not raw_response:
        return None

    text = raw_response.strip()

    if text.upper() in ["A", "B", "C", "D"]:
        return text.upper()

    m = re.match(r"^([ABCD])[.)]\s", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"answer\s*(?:is\s*)?[:\-]?\s*([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Thinking models put final answer at the end — scan last 200 chars
    tail = text[-200:]
    m = re.search(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", tail, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def call_openai(model_id, prompt, api_key):
    import openai
    client   = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def call_anthropic(model_id, prompt, api_key):
    import anthropic
    client   = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_id,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def call_google(model_id, prompt, api_key):
    from google import genai
    from google.genai import types
    client   = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=3000,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


def call_together(model_id, prompt, api_key):
    import openai
    client   = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=50,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def call_deepseek(model_id, prompt, api_key):
    import openai
    client   = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


DISPATCH = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
    "together":  call_together,
    "deepseek":  call_deepseek,
}


def evaluate_model(model_config, dataset, output_folder, api_keys,
                   rerun_errors=False):
    model_name   = model_config["name"]
    model_id     = model_config["model_id"]
    provider     = model_config["provider"]
    category     = model_config["category"]
    results_file = f"{output_folder}/results_{model_name}.jsonl"

    completed_ids = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if rerun_errors and row.get("model_answer_clean") is None:
                        continue
                    completed_ids.add(row["item_id"])
                except Exception:
                    pass

    remaining = [item for item in dataset if item["id"] not in completed_ids]
    log.info(f"[{model_name}] {len(completed_ids):,} done, {len(remaining):,} remaining")

    if not remaining:
        return

    call_fn = DISPATCH[provider]
    api_key = api_keys[provider]
    buffer  = []
    correct = total = errors = 0

    for item in remaining:
        prompt, correct_letter = build_prompt(item)
        raw_response = error_message = None
        delay        = RETRY_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                raw_response  = call_fn(model_id, prompt, api_key)
                error_message = None
                break
            except Exception as e:
                error_message = str(e)
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"[{model_name}] item {item['id']} attempt {attempt+1}: {e}. Waiting {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    log.error(f"[{model_name}] FAILED item {item['id']}: {e}")
                    errors += 1

        clean_answer = parse_answer(raw_response) if raw_response else None
        is_correct   = (clean_answer == correct_letter) if clean_answer is not None else None

        if raw_response is not None:
            total += 1
            if is_correct:
                correct += 1

        buffer.append({
            "item_id":            item["id"],
            "tier":               item.get("difficulty_tier", ""),
            "crossing_count":     item.get("crossing_count", 0),
            "model":              model_name,
            "category":           category,
            "correct_answer":     item["short_correct_answer"],
            "correct_letter":     correct_letter,
            "model_answer_raw":   raw_response,
            "model_answer_clean": clean_answer,
            "is_correct":         is_correct,
            "error":              error_message,
            "timestamp":          datetime.now().isoformat(),
        })

        if len(buffer) >= BATCH_SAVE:
            with open(results_file, "a") as f:
                for r in buffer:
                    f.write(json.dumps(r) + "\n")
            buffer = []

        time.sleep(SLEEP_BETWEEN_CALLS)

    if buffer:
        with open(results_file, "a") as f:
            for r in buffer:
                f.write(json.dumps(r) + "\n")

    acc = correct / total * 100 if total > 0 else 0
    log.info(f"[{model_name}] Done. Accuracy: {acc:.2f}% ({correct}/{total}) Errors: {errors}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output",  default="evaluation_results")
    parser.add_argument("--model",   default=None)
    parser.add_argument("--gold",    action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"{args.output}/evaluation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    with open(args.dataset) as f:
        dataset = json.load(f)

    api_keys = {
        "openai":    os.environ["OPENAI_API_KEY"],
        "anthropic": os.environ["ANTHROPIC_API_KEY"],
        "google":    os.environ["GOOGLE_API_KEY"],
        "together":  os.environ["TOGETHER_API_KEY"],
        "deepseek":  os.environ["DEEPSEEK_API_KEY"],
    }

    Path(args.output).mkdir(parents=True, exist_ok=True)

    model_registry = REASONING_MODELS if args.gold else MCQ_MODELS

    if args.model:
        configs = {k: v for k, v in model_registry.items() if k == args.model}
    else:
        configs = model_registry

    for key, config in configs.items():
        evaluate_model(config, dataset, args.output, api_keys,
                       rerun_errors=args.rerun_errors)