import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    os.path.join(os.environ.get("SHARED_MODELS_PATH", "/shared_models"), "Qwen/Qwen3.5-4B"),
)
BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "64"))
MAX_INPUT_TOKENS = int(os.environ.get("QWEN_MAX_INPUT_TOKENS", "1536"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "48"))
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.S)
SPACE_RE = re.compile(r"\s+")
JSON_RE = re.compile(r"\{.*?\}", re.S)


def clean_field(value):
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def build_prompt(template, row):
    prompt = template
    for key in ("category", "name", "description"):
        prompt = prompt.replace("{{" + key + "}}", clean_field(row.get(key, "")))
    return prompt


def parse_answer(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    match = JSON_RE.search(raw)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}

    confirmed = data.get("category_confirmed")
    if not isinstance(confirmed, bool):
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict == "не бан":
            confirmed = True
        elif verdict == "бан":
            confirmed = False
        else:
            # Conservative fallback: malformed model output is rejected.
            confirmed = False

    evidence = clean_field(data.get("evidence", ""))
    evidence = evidence.replace("<", " ").replace(">", " ").strip(" .,:;-\n\t")
    if evidence:
        comment = "Категория подтверждена: " + evidence + "." if confirmed else "Категория не подтверждена: " + evidence + "."
    else:
        comment = (
            "Название и описание подтверждают заявленную категорию товара."
            if confirmed
            else "Название и описание не подтверждают заявленную категорию товара."
        )
    comment = SPACE_RE.sub(" ", comment).strip()
    if len(comment) < 50:
        comment += " Решение принято по данным карточки товара."
    if len(comment) > 300:
        cut = comment.rfind(" ", 0, 301)
        comment = comment[: cut if cut > 0 else 300].rstrip(" .,:;-") + "."
        comment = comment[:300]
    verdict = "не бан" if confirmed else "бан"
    result = f"<комментарий>{comment}<вердикт>{verdict}"
    if OUTPUT_RE.fullmatch(result) is None:
        raise ValueError(f"Internal output formatting failure: {result!r}")
    return result


def generate(model, tokenizer, prompts):
    rendered = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered.append(text)

    encoded = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    input_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, input_width:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_data_path", "--test-data-path", dest="input", required=True)
    parser.add_argument("-o", "--output_path", "--output-path", dest="output", required=True)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "name", "description", "category"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"Input misses columns: {sorted(required - set(rows[0]))}")

    template = (ROOT / "prompt.md").read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    results = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]
        raw_answers = generate(model, tokenizer, [build_prompt(template, row) for row in batch])
        results.extend(parse_answer(raw) for raw in raw_answers)
        print(f"processed={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "result"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "result": result} for row, result in zip(rows, results))
    if len(results) != len(rows):
        raise RuntimeError("Output row count mismatch")
    print(f"saved rows={len(results)} path={output_path}")


if __name__ == "__main__":
    main()
