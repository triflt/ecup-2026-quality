from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


ROOT = Path(__file__).resolve().parent
MODEL_PATH = os.environ.get(
    "QWEN_VLM_MODEL_PATH",
    os.path.join(os.environ.get("SHARED_MODELS_PATH", "/shared_models"), "Qwen/Qwen3.5-4B"),
)
BATCH_SIZE = int(os.environ.get("QWEN_VLM_BATCH_SIZE", "8"))
MAX_PIXELS = int(os.environ.get("QWEN_VLM_MAX_PIXELS", "262144"))
MAX_NEW_TOKENS = int(os.environ.get("QWEN_VLM_MAX_NEW_TOKENS", "96"))
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPACE_RE = re.compile(r"\s+")
JSON_RE = re.compile(r"\{.*?\}", re.S)
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.S)

SYSTEM_PROMPT = """Ты модератор карточек маркетплейса. Проверь, подтверждена ли заявленная категория названием, описанием и ВСЕМИ приложенными изображениями. Внимательно читай надписи на упаковке, состав и комплектацию. Не додумывай отсутствующие сведения.

Верни ровно один JSON без markdown:
{"category_confirmed":true,"verdict":"не бан","evidence":"краткий решающий факт из текста или изображения"}

Связь полей строгая: category_confirmed=true означает verdict="не бан"; false означает verdict="бан".

Правила БАД:
- true ТОЛЬКО при явной буквальной маркировке БАД, «биологически активная добавка» или dietary supplement в тексте либо на изображении упаковки;
- явное отрицание «не является БАД» важнее совпавшего слова;
- отсутствие фразы «не является БАД» ничего не подтверждает; нельзя выводить true из отсутствия отрицания;
- слова «добавка», «пищевая добавка», коллаген, витамины, экстракты, полезный состав и 21 active ingredients без буквальной маркировки БАД недостаточны и дают false;
- спортивное питание, протеин, BCAA, аминокислоты и L-карнитин без буквальной маркировки БАД дают false.

Правила легковоспламеняющихся:
- true для самостоятельного горючего вещества, газа, топлива, спичек, зажигалки, растопки, пиротехники и перезаправляемой мини-горелки с резервуаром;
- true, если горючий товар или баллон явно входит в комплект;
- false для мангала, гриля, плиты или горелки-насадки, надеваемой на внешний баллон, если топливо/баллон не входят в комплект; фотография насадки, установленной на демонстрационный баллон, не означает, что баллон входит в комплект;
- false для пневмохлопушки, товара с явным «не пиротехника», встроенного компонента или горючего материала, который является только частью другого изделия.

Контрольные примеры:
- Газовая горелка с пьезоподжигом, которая устанавливается на отдельный баллон: false, «горелка является насадкой, топливо не входит в комплект».
- Морской коллаген с витаминами без буквальных слов БАД/биологически активная добавка/dietary supplement: false, «обязательная маркировка БАД отсутствует».
- Упаковка с буквальной надписью «биологически активная добавка для взрослых»: true, процитируй эту надпись.

Evidence должно ссылаться на реально видимый или написанный факт."""


def clean(value) -> str:
    return SPACE_RE.sub(" ", "" if value is None else str(value)).strip()


def image_paths(data_path: Path, product_id: str) -> list[Path]:
    folder = data_path.parent / "images" / product_id
    if not folder.is_dir():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]


def resize_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width * height > MAX_PIXELS:
        scale = (MAX_PIXELS / (width * height)) ** 0.5
        image = image.resize((max(28, int(width * scale)), max(28, int(height * scale))), Image.LANCZOS)
    return image


def make_messages(row: dict[str, str], paths: list[Path]) -> list[dict]:
    content = [{"type": "image", "image": resize_image(path)} for path in paths]
    content.append({
        "type": "text",
        "text": (
            f"Заявленная категория: {clean(row.get('category'))}\n"
            f"Название: {clean(row.get('name'))}\n"
            f"Описание: {clean(row.get('description'))}"
        ),
    })
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def parse_answer(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    match = JSON_RE.search(raw)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    confirmed = data.get("category_confirmed")
    if not isinstance(confirmed, bool):
        verdict = clean(data.get("verdict")).lower()
        confirmed = verdict == "не бан" if verdict in {"бан", "не бан"} else False
    evidence = clean(data.get("evidence")).replace("<", " ").replace(">", " ").strip(" .,:;-")
    if evidence:
        comment = ("Категория подтверждена: " if confirmed else "Категория не подтверждена: ") + evidence + "."
    else:
        comment = (
            "Текст и изображения подтверждают заявленную категорию товара."
            if confirmed else "Текст и изображения не подтверждают заявленную категорию товара."
        )
    comment = SPACE_RE.sub(" ", comment).strip()
    if len(comment) < 50:
        comment += " Решение принято по данным карточки и упаковки."
    if len(comment) > 300:
        cut = comment.rfind(" ", 0, 300)
        comment = comment[: cut if cut > 0 else 300].rstrip(" .,:;-") + "."
        comment = comment[:300]
    verdict = "не бан" if confirmed else "бан"
    result = f"<комментарий>{comment}<вердикт>{verdict}"
    if OUTPUT_RE.fullmatch(result) is None:
        raise ValueError(f"invalid formatted output: {result!r}")
    return result


def generate_batch(model, processor, messages_batch: list[list[dict]]) -> list[str]:
    try:
        inputs = processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            enable_thinking=False,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    input_width = inputs["input_ids"].shape[1]
    return processor.batch_decode(outputs[:, input_width:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_data_path", "--test-data-path", dest="input", required=True)
    parser.add_argument("-o", "--output_path", "--output-path", dest="output", required=True)
    args = parser.parse_args()
    data_path = Path(args.input)
    with data_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "name", "description", "category"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"missing columns: {sorted(required - set(rows[0]))}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    results = []
    started = time.monotonic()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        messages = [make_messages(row, image_paths(data_path, clean(row["id"]))) for row in batch]
        try:
            raw = generate_batch(model, processor, messages)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raw = []
            for item in messages:
                raw.extend(generate_batch(model, processor, [item]))
        finally:
            for item in messages:
                for message in item:
                    for part in message["content"]:
                        image = part.get("image") if isinstance(part, dict) else None
                        if isinstance(image, Image.Image):
                            image.close()
        results.extend(parse_answer(answer) for answer in raw)
        done = min(start + len(batch), len(rows))
        print(f"processed={done}/{len(rows)} elapsed_min={(time.monotonic()-started)/60:.2f}", flush=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "result"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "result": result} for row, result in zip(rows, results))
    if len(results) != len(rows):
        raise RuntimeError("output row count mismatch")
    print(f"saved rows={len(results)} path={output_path}", flush=True)


if __name__ == "__main__":
    main()
