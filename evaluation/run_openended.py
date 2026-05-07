import os
import json
import time
import logging
from datetime import datetime
from models.model_configs import COOC_MODELS, JUDGE_MODEL

OUTPUT_FOLDER = "cooccurrence_results"
MAX_RETRIES   = 5
RETRY_DELAY   = 10
BATCH_SAVE    = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(__file__),
          "prompts/openended_system_prompt.txt")) as f:
    SYSTEM_PROMPT = f.read().strip()

with open(os.path.join(os.path.dirname(__file__),
          "prompts/openended_user_prompt.txt")) as f:
    USER_PROMPT_TEMPLATE = f.read()


def build_prompt(item):
    return USER_PROMPT_TEMPLATE.format(question=item["question"])


def call_openai(model_id, prompt, api_key):
    import openai
    client   = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
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
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    if not response.content:
        return ""
    return response.content[0].text.strip()


def call_together(model_id, prompt, api_key):
    import openai
    client   = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=50,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


DISPATCH = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "together":  call_together,
}


def run_model(model_config, dataset, api_keys):
    model_name = model_config["name"]
    model_id   = model_config["model_id"]
    provider   = model_config["provider"]
    api_key    = api_keys[provider]
    call_fn    = DISPATCH[provider]
    raw_file   = f"{OUTPUT_FOLDER}/cooc_raw_{model_name}.jsonl"

    completed_ids = set()
    if os.path.exists(raw_file):
        with open(raw_file) as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["item_id"])
                except Exception:
                    pass

    remaining = [item for item in dataset if item["id"] not in completed_ids]
    log.info(f"[{model_name}] {len(completed_ids):,} done, {len(remaining):,} remaining")

    if not remaining:
        return

    buffer = []
    errors = 0

    for item in remaining:
        prompt       = build_prompt(item)
        raw_response = None
        delay        = RETRY_DELAY

        for attempt in range(MAX_RETRIES):
            try:
                raw_response = call_fn(model_id, prompt, api_key)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    log.warning(f"[{model_name}] item {item['id']} attempt {attempt+1}: {e}. Waiting {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    log.error(f"[{model_name}] FAILED item {item['id']}: {e}")
                    errors += 1

        buffer.append({
            "item_id":            item["id"],
            "tier":               item.get("difficulty_tier", ""),
            "crossing_count":     item.get("crossing_count", 0),
            "model":              model_name,
            "category":           model_config["category"],
            "correct_answer":     item["short_correct_answer"],
            "question":           item["question"][:200],
            "openended_response": raw_response,
            "timestamp":          datetime.now().isoformat(),
        })

        if len(buffer) >= BATCH_SAVE:
            with open(raw_file, "a") as f:
                for r in buffer:
                    f.write(json.dumps(r) + "\n")
            buffer = []

    if buffer:
        with open(raw_file, "a") as f:
            for r in buffer:
                f.write(json.dumps(r) + "\n")

    log.info(f"[{model_name}] Done. Errors: {errors}")


if __name__ == "__main__":
    import sys
    import json as _json

    dataset_path = sys.argv[1]
    with open(dataset_path) as f:
        dataset = _json.load(f)

    api_keys = {
        "openai":    os.environ["OPENAI_API_KEY"],
        "anthropic": os.environ["ANTHROPIC_API_KEY"],
        "together":  os.environ["TOGETHER_API_KEY"],
    }

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for model_config in COOC_MODELS:
        run_model(model_config, dataset, api_keys)