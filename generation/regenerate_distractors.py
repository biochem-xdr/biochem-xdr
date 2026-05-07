import os
import re
import json
import time
import pickle
import logging
from datetime import datetime
import anthropic


BATCH_SAVE  = 25
MAX_RETRIES = 5
RETRY_DELAY = 15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

with open(
    os.path.join(os.path.dirname(__file__), "prompts/distractor_regen_prompt.txt")
) as f:
    PROMPT_TEMPLATE = f.read()

ENTITY_TYPE_HINTS = {
    "short_entity": "The correct answer is a specific biological entity name (enzyme, gene, metabolite, disease, or pathway).",
    "pathway":      "The correct answer is a metabolic pathway name.",
    "disease":      "The correct answer is a disease or disorder name.",
    "enzyme":       "The correct answer is an enzyme or gene name.",
}


def looks_like_database_id(text):
    if not text or not isinstance(text, str):
        return True
    text = text.strip()
    if re.match(r"^[CGRHMDKE]\d{5}$", text):
        return True
    if re.match(r"^[a-z]{2,4}\d{4,}$", text):
        return True
    if re.match(r"^\d+$", text):
        return True
    if len(text) < 3:
        return True
    return False


def looks_like_sentence_fragment(text):
    if not text or not isinstance(text, str):
        return True
    text = text.strip()
    if re.match(r"^(The |This |Through |A |An |It |In |By )", text) and len(text) > 20:
        return True
    if re.search(r"\b(connects|terminates|occurs|links|involves)\b", text.lower()):
        return True
    return False


def build_prompt(item):
    hint = ENTITY_TYPE_HINTS.get(item.get("answer_format", ""), "")
    return PROMPT_TEMPLATE.format(
        question=item.get("question", ""),
        correct=item.get("short_correct_answer", ""),
        path_text=item.get("path_text", "")[:600],
        entity_type_hint=hint,
    )


def generate_distractors(item, client):
    prompt  = build_prompt(item)
    correct = item.get("short_correct_answer", "").lower()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()

            parsed = json.loads(raw)
            d1 = parsed["distractor_1"]["answer"].strip()
            d2 = parsed["distractor_2"]["answer"].strip()
            d3 = parsed["distractor_3"]["answer"].strip()
            w1 = parsed["distractor_1"].get("why_wrong", "")
            w2 = parsed["distractor_2"].get("why_wrong", "")
            w3 = parsed["distractor_3"].get("why_wrong", "")

            for d in [d1, d2, d3]:
                if d.lower() == correct:
                    raise ValueError(f"Distractor matches correct: {d}")
                if looks_like_database_id(d):
                    raise ValueError(f"Database ID in distractor: {d}")
                if len(d.strip()) < 3:
                    raise ValueError(f"Distractor too short: {d}")

            return d1, d2, d3, w1, w2, w3

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error item {item.get('id')} attempt {attempt+1}: {e}")
        except ValueError as e:
            log.warning(f"Validation error item {item.get('id')} attempt {attempt+1}: {e}")
        except anthropic.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            log.warning(f"Rate limit item {item.get('id')} — waiting {wait}s")
            time.sleep(wait)
            continue
        except Exception as e:
            log.warning(f"API error item {item.get('id')} attempt {attempt+1}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)

    log.error(f"FAILED after {MAX_RETRIES} attempts item {item.get('id')}")
    return None


def run_regen(dataset, client, output_folder):
    output_file = f"{output_folder}/new_distractors.jsonl"
    os.makedirs(output_folder, exist_ok=True)

    completed_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["item_id"])
                except Exception:
                    pass

    remaining = [item for item in dataset if item.get("id") not in completed_ids]
    log.info(f"{len(completed_ids):,} done, {len(remaining):,} remaining")

    if not remaining:
        log.info("All items complete.")
        return

    buffer  = []
    success = failed = skipped = 0

    for item in remaining:
        item_id = item.get("id", item.get("original_index"))
        correct = item.get("short_correct_answer", "")

        if looks_like_database_id(correct) or looks_like_sentence_fragment(correct):
            row = _make_row(item_id, correct, item, status="skipped_broken_correct")
            skipped += 1
        else:
            result = generate_distractors(item, client)
            if result:
                d1, d2, d3, w1, w2, w3 = result
                row = _make_row(item_id, correct, item,
                                status="success",
                                d1=d1, d2=d2, d3=d3,
                                w1=w1, w2=w2, w3=w3)
                success += 1
            else:
                row = _make_row(item_id, correct, item,
                                status="failed_kept_original",
                                d1=item.get("short_distractor_1"),
                                d2=item.get("short_distractor_2"),
                                d3=item.get("short_distractor_3"))
                failed += 1

        buffer.append(row)

        if len(buffer) >= BATCH_SAVE:
            _flush(buffer, output_file)
            buffer = []

        time.sleep(0.5)

    if buffer:
        _flush(buffer, output_file)

    log.info(f"Complete: {success:,} success | {failed:,} failed | {skipped:,} skipped")
    log.info(f"Results: {output_file}")


def _make_row(item_id, correct, item, status,
              d1=None, d2=None, d3=None,
              w1=None, w2=None, w3=None):
    return {
        "item_id":          item_id,
        "status":           status,
        "correct_answer":   correct,
        "new_distractor_1": d1,
        "new_distractor_2": d2,
        "new_distractor_3": d3,
        "why_wrong_1":      w1,
        "why_wrong_2":      w2,
        "why_wrong_3":      w3,
        "original_d1":      item.get("short_distractor_1"),
        "original_d2":      item.get("short_distractor_2"),
        "original_d3":      item.get("short_distractor_3"),
        "timestamp":        datetime.now().isoformat(),
    }


def _flush(buffer, path):
    with open(path, "a") as f:
        for row in buffer:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    import sys

    dataset_pkl   = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "distractor_regen"
    api_key       = os.environ.get("ANTHROPIC_API_KEY")

    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)

    log.info(f"Dataset: {len(dataset):,} items")

    client = anthropic.Anthropic(api_key=api_key)
    run_regen(dataset, client, output_folder)