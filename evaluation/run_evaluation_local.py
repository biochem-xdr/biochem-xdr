import os
import json
import random
import time
import re
import gc
import torch
from datetime import datetime
from pathlib import Path

from models.model_configs import MCQ_MODELS

RANDOM_SEED           = 42
SLEEP_BETWEEN_BATCHES = 0.1


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

    tail = text[-200:]
    m = re.search(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", tail, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def unload_model(model, tokenizer_or_processor, model_name):
    del model
    del tokenizer_or_processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    gc.collect()
    print(f"Unloaded {model_name}")
    if torch.cuda.is_available():
        print(f"  GPU allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  GPU reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")


def load_causal_model(model_id, hf_token):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
    ).eval()
    return model, tokenizer


def load_gemma_model(model_id, hf_token):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model     = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", token=hf_token
    ).eval()
    return model, processor


def load_medgemma_model(model_id, hf_token):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    model     = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token
    ).eval()
    return model, processor


def load_qwen3_model(model_id, hf_token):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", token=hf_token
    ).eval()
    return model, tokenizer


def run_causal_batch(model, tokenizer, prompts, model_name,
                     use_chat_template=True):
    system = (
        "You are a biochemistry expert. Reply with ONLY a single letter "
        "A, B, C, or D. No reasoning, no explanation, nothing else."
    )
    encoded_inputs = []

    for prompt in prompts:
        if use_chat_template:
            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user",   "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            plain = f"{system}\n\n{prompt}\n\nAnswer:"
            ids   = tokenizer(plain, return_tensors="pt")["input_ids"]
        encoded_inputs.append(ids.squeeze(0))

    max_len  = max(t.shape[0] for t in encoded_inputs)
    padded   = torch.full((len(encoded_inputs), max_len),
                          tokenizer.pad_token_id, dtype=torch.long)
    attn     = torch.zeros(len(encoded_inputs), max_len, dtype=torch.long)

    for i, t in enumerate(encoded_inputs):
        start = max_len - t.shape[0]
        padded[i, start:] = t
        attn[i,   start:] = 1

    device = next(model.parameters()).device
    padded = padded.to(device)
    attn   = attn.to(device)

    eos_ids = [tokenizer.eos_token_id]
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot and eot != tokenizer.unk_token_id:
            eos_ids.append(eot)
    except Exception:
        pass

    max_new = 3000 if model_name == "r1-distill-qwen-32b" else 50

    with torch.inference_mode():
        outputs = model.generate(
            padded,
            attention_mask=attn,
            max_new_tokens=max_new,
            eos_token_id=eos_ids,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [
        tokenizer.decode(output[max_len:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def run_gemma_batch(model, processor, prompts, model_type):
    system = (
        "You are a helpful medical assistant. Reply with ONLY a single letter A, B, C, or D."
        if model_type == "medgemma"
        else "You are a biochemistry expert. Reply with ONLY a single letter A, B, C, or D."
    )
    responses = []
    for prompt in prompts:
        inputs = processor.apply_chat_template(
            [{"role": "system", "content": [{"type": "text", "text": system}]},
             {"role": "user",   "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        responses.append(
            processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()
        )
    return responses


def run_qwen3_batch(model, tokenizer, prompts):
    responses = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a biochemistry expert. Reply with ONLY a single letter A, B, C, or D. No reasoning, no explanation, nothing else."},
             {"role": "user",   "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        responses.append(
            tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        )
    return responses


def run_qwq_batch(model, tokenizer, prompts):
    responses = []
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a biochemistry expert. Think through the problem, then end your response with 'Answer: X' where X is A, B, C, or D."},
             {"role": "user",   "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs, max_new_tokens=5000, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        responses.append(
            tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        )
    return responses


def evaluate_model(model_config, dataset, output_folder, hf_token):
    model_name  = model_config["name"]
    model_id    = model_config["model_id"]
    category    = model_config["category"]
    model_type  = model_config.get("type", "causal")
    batch_size  = model_config.get("batch_size", 32)
    results_file = f"{output_folder}/results_{model_name}.jsonl"

    completed_ids = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("model_answer_clean") is not None:
                        completed_ids.add(row["item_id"])
                except Exception:
                    pass

    remaining = [item for item in dataset if item["id"] not in completed_ids]
    print(f"\n[{model_name}] {len(completed_ids):,} done, {len(remaining):,} remaining")

    if not remaining:
        return

    print(f"[{model_name}] Loading model...")

    loaders = {
        "causal":   lambda: load_causal_model(model_id, hf_token),
        "qwen3":    lambda: load_qwen3_model(model_id, hf_token),
        "qwq":      lambda: load_causal_model(model_id, hf_token),
        "gemma":    lambda: load_gemma_model(model_id, hf_token),
        "medgemma": lambda: load_medgemma_model(model_id, hf_token),
    }
    model, tokenizer_or_processor = loaders[model_type]()

    correct = total = errors = 0
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    for batch in batches:
        prompts         = []
        correct_letters = []

        for item in batch:
            prompt, correct_letter = build_prompt(item)
            prompts.append(prompt)
            correct_letters.append(correct_letter)

        try:
            if model_type == "causal":
                raw_responses = run_causal_batch(
                    model, tokenizer_or_processor, prompts, model_name,
                    use_chat_template=model_config.get("use_chat_template", True),
                )
            elif model_type == "qwen3":
                raw_responses = run_qwen3_batch(model, tokenizer_or_processor, prompts)
            elif model_type == "qwq":
                raw_responses = run_qwq_batch(model, tokenizer_or_processor, prompts)
            else:
                raw_responses = run_gemma_batch(model, tokenizer_or_processor, prompts, model_type)
            error_message = None
        except Exception as e:
            print(f"[{model_name}] Batch error: {e}")
            raw_responses = [None] * len(batch)
            error_message = str(e)
            errors += len(batch)

        rows = []
        for i, item in enumerate(batch):
            raw_response = raw_responses[i]
            clean_answer = parse_answer(raw_response)
            is_correct   = (clean_answer == correct_letters[i]) if clean_answer is not None else None

            if raw_response is not None:
                total += 1
                if is_correct:
                    correct += 1

            rows.append({
                "item_id":            item["id"],
                "tier":               item.get("difficulty_tier", ""),
                "crossing_count":     item.get("crossing_count", 0),
                "model":              model_name,
                "category":           category,
                "correct_answer":     item["short_correct_answer"],
                "correct_letter":     correct_letters[i],
                "model_answer_raw":   raw_response,
                "model_answer_clean": clean_answer,
                "is_correct":         is_correct,
                "error":              error_message,
                "timestamp":          datetime.now().isoformat(),
            })

        with open(results_file, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        acc = correct / total * 100 if total > 0 else 0
        print(f"  [{model_name}] {total}/{len(remaining)} | acc {acc:.1f}% | errors {errors}")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    acc = correct / total * 100 if total > 0 else 0
    print(f"[{model_name}] Done. {acc:.2f}% ({correct}/{total}) Errors: {errors}")

    unload_model(model, tokenizer_or_processor, model_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output",  default="evaluation_results")
    parser.add_argument("--model",   default=None)
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)

    hf_token = os.environ.get("HF_TOKEN")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.model:
        configs = {k: v for k, v in MCQ_MODELS.items()
                   if k == args.model and v.get("local")}
    else:
        configs = {k: v for k, v in MCQ_MODELS.items() if v.get("local")}

    for key, config in configs.items():
        evaluate_model(config, dataset, args.output, hf_token)