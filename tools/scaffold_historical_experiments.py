from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ("000", "text_baseline", "TF-IDF text baseline", "completed", "experiments/000_text_baseline/train.py", "experiments/000_text_baseline/submission", "Сильный word+character TF-IDF должен заметно превзойти официальный baseline.", "Гипотеза подтверждена, но random holdout оказался оптимистичным; grouped Macro F1 0.889576, Public 0.713026."),
    ("010", "qwen35_text_prompt", "Qwen3.5 text prompting", "rejected", "experiments/010_qwen35_text_prompt/submission/run.py", "experiments/010_qwen35_text_prompt/submission", "Большая generative model сможет классифицировать карточку по инструкции без обучения.", "Отклонено: Public 0.579901, существенно ниже supervised text model."),
    ("020", "qwen35_vlm_prompt", "Qwen3.5 direct multimodal prompting", "rejected", "experiments/020_qwen35_vlm_prompt/submission/run.py", "experiments/020_qwen35_vlm_prompt/submission", "Прямой VLM prompt сможет применить правила к тексту и изображениям.", "Отклонено: Public 0.468403; generative verdict нестабилен."),
    ("030", "qwen3vl_embedding", "Qwen3-VL embedding classifier", "completed", "research/qwen-train-code/run_train.py", "experiments/030_qwen3vl_embedding/submission", "Multimodal representation улучшит text baseline.", "Изображения полезны, но random OOF 0.902350 не перенёсся: Public 0.722013."),
    ("040", "late_fusion", "Category-specific late fusion", "accepted", "research/group_cv.py", "experiments/040_late_fusion/submission", "Независимые text и multimodal heads лучше переносятся, чем единая модель.", "Подтверждено: grouped Macro F1 0.899297 и лучший Public 0.806579."),
    ("050", "duplicate_stress", "Duplicate-aware stress validation", "completed", "research/image_duplicate_group_cv.py", None, "Image duplicates объясняют завышение обычной grouped CV.", "Подтверждено: ExtraTrees теряет вес, Macro F1 снижается до 0.900610."),
    ("060", "multiview_fusion", "First-image and multi-view fusion", "completed", "research/tri_fusion_cv.py", None, "Первое изображение содержит упаковку и kit evidence, размытые all-images embedding.", "Небольшой first-image head улучшает fusion; supervised multi-image LoRA не улучшает first-image LoRA."),
    ("070", "cross_modal_interaction", "Cross-modal interaction head", "rejected", "research/cross_modal_consistency_cv.py", None, "Text/image difference и product terms улучшат сложные BAD cases.", "Локально улучшает BAD, но advanced submission проигрывает robust late fusion на Public."),
    ("080", "four_head_ensemble", "Four-head and tree ensemble", "rejected", "research/four_head_fusion_cv.py", None, "Нелинейный tree head улучшит редкий flammable class.", "Обычная grouped CV улучшается, duplicate-stress и Public показывают overfitting."),
    ("090", "advanced_mixed", "Category-mixed advanced submission", "rejected", "research/train_advanced_heads.py", "experiments/090_advanced_mixed/submission", "Разные лучшие локальные architectures для категорий улучшат Public.", "Отклонено: Public 0.785500 против 0.806579 у simpler late fusion."),
    ("100", "paddleocr_hard_cases", "Targeted OCR hard-case correction", "rejected", "research/paddleocr_extract_hard.py", None, "OCR первого изображения исправит uncertainty/disagreement cases.", "Помогает слабому flammable baseline, но не складывается с сильной LoRA fusion."),
    ("110", "qwen3vl_lora", "Qwen3-VL supervised rsLoRA", "accepted", "research/qwen3vl_lora_holdout.py", "experiments/110_qwen3vl_lora/submission", "Domain adaptation первого изображения улучшит representation без полного fine-tuning.", "Nested grouped Macro F1 fusion 0.906395, +0.00710 к robust base."),
    ("120", "qwen3vl_lora_prior", "Qwen3-VL LoRA with product-family prior", "completed", "research/annotator_prior_cv.py", "experiments/120_qwen3vl_lora_prior/submission", "Train product families повторятся в hidden и сохранят ambiguity annotator process.", "70/30 recurrence simulations положительны; grouped leave-one-out metric имеет transductive leakage и не используется как основной."),
    ("130", "qwen35_lora", "Qwen3.5 supervised rsLoRA", "accepted", "research/qwen3vl_lora_holdout.py", None, "Qwen3.5 даст ошибки, отличные от Qwen3-VL.", "Подтверждено: three-head nested Macro F1 0.911843."),
    ("140", "dual_lora_fusion", "Dual-LoRA fusion", "accepted", "research/nested_multimodel_fusion.py", "experiments/140_dual_lora_fusion/submission", "Robust base, Qwen3-VL LoRA и Qwen3.5 LoRA дают устойчивую decision diversity.", "Главная architecture: nested Macro F1 0.911843; scaled runtime укладывается в лимит."),
    ("150", "gemma_lora", "Gemma-4-E4B supervised LoRA", "rejected", "research/qwen3vl_lora_holdout.py", None, "Gemma добавит независимый multimodal head.", "Отклонено после пяти folds: nested fusion 0.902686 и высокая variance flammable."),
    ("160", "multiimage_lora", "Qwen3-VL multi-image LoRA", "rejected", "research/qwen3vl_multiimage_lora_holdout.py", None, "First/second/last images улучшат five-image residual cohort.", "Отклонено на untouched fold: дополнительные gallery images размывают first-image signal."),
    ("170", "annotator_prior", "Exact/name annotator prior", "completed", "research/annotator_prior_random_split.py", None, "Empirical labels repeated product families улучшат hidden recurrence.", "Сильный 70/30 эффект, особенно для flammable; применять как отдельную distributional гипотезу."),
    ("180", "fuzzy_family_prior", "Numeric-variant family prior", "accepted", "research/fuzzy_annotator_prior_cv.py", "experiments/180_fuzzy_family_prior/submission", "Unanimous BAD wording families с разными числами можно безопасно объединять.", "Donor-only BAD +0.000761; 17/20 recurrence wins; flammable отключён."),
    ("190", "shingle_neighbor_prior", "Rare-shingle neighbour prior", "accepted", "research/shingle_neighbor_prior_cv.py", "experiments/190_shingle_neighbor_prior/submission", "Rare five-word shingles найдут почти одинаковые BAD cards без exact match.", "Donor-only BAD +0.000853; 20/20 recurrence wins; основной готовый candidate."),
    ("200", "image_hash_prior", "Image-hash prior", "rejected", "research/image_hash_prior_cv.py", None, "Exact/perceptual image repeats добавят сигнал поверх text family.", "Отклонено: Macro gain только +0.000095, flammable overrides не выбраны."),
    ("210", "neighbor_priors", "Text and ID neighbour priors", "rejected", "research/char_tfidf_neighbor_prior_cv.py", None, "Order, character similarity или rare tokens найдут label-consistent neighbours.", "ID, character-TFIDF и rare-token variants не дали надёжного nested gain."),
    ("220", "soft_annotator_prior", "Beta-smoothed soft family prior", "rejected", "research/soft_annotator_prior_cv.py", None, "Смешивание posterior family probability с model score будет устойчивее hard override.", "Donor-only Macro gain только +0.000330; confirmation random split подготовлен, но не запущен."),
    ("230", "qwen35_second_seed", "Second-seed Qwen3.5 ensemble", "ready_for_aggregation", "research/qwen35_seed_ensemble_cv.py", None, "Независимый rsLoRA seed снизит variance и улучшит ensemble.", "Пять OOF jobs и full-data training завершены; artifacts ещё не агрегированы."),
    ("900", "infrastructure_checks", "Infrastructure and submission checks", "completed", "tools/check_publish_safety.py", None, "Schema, runtime и empty-category smokes обнаружат ошибки до leaderboard.", "Обнаружены и исправлены image-size mismatch, cross-batch OCR contamination и empty-category crash."),
]

DATA_VERSION = "competition_train_v1"
DEFAULT_EVALUATION = "grouped_text_v1"
EVALUATION_VERSION = {
    "050": "image_duplicate_stress_v1",
    "110": "nested_grouped_v1",
    "120": "recurrence_70_30_v1",
    "130": "nested_grouped_v1",
    "140": "nested_grouped_v1",
    "150": "nested_grouped_v1",
    "160": "nested_grouped_v1",
    "170": "recurrence_70_30_v1",
    "180": "recurrence_70_30_v1",
    "190": "recurrence_70_30_v1",
    "200": "recurrence_70_30_v1",
    "210": "nested_grouped_v1",
    "220": "recurrence_70_30_v1",
    "230": "nested_grouped_v1",
}


RUNNER = '''from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ecup_quality.experiments.runner import run_entrypoint

if __name__ == "__main__":
    raise SystemExit(run_entrypoint(Path(__file__).with_name("experiment.toml"), sys.argv[1:]))
'''

PREPROCESS = '''from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    command = [sys.executable, str(ROOT / "tools" / "preprocess_experiment.py"), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
'''

BUILD = '''from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    config = Path(__file__).with_name("experiment.toml")
    command = [sys.executable, str(ROOT / "tools" / "build_experiment_submission.py"), "--config", str(config), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
'''


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    log_rows: dict[str, list[dict[str, str]]] = {}
    with (ROOT / "reports" / "experiment-log.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            log_rows.setdefault(row["experiment_group"], []).append(row)

    for number, slug, title, status, entrypoint, submission, hypothesis, outcome in SPECS:
        key = f"{number}_{slug}"
        evaluation_version = EVALUATION_VERSION.get(number, DEFAULT_EVALUATION)
        directory = ROOT / "experiments" / key
        (directory / "results").mkdir(parents=True, exist_ok=True)
        (directory / "artifacts").mkdir(parents=True, exist_ok=True)
        result_file = f"experiments/{key}/results/metrics.json"
        lines = [
            "[experiment]",
            f"id = {quoted(number)}",
            f"slug = {quoted(slug)}",
            f"title = {quoted(title)}",
            f"status = {quoted(status)}",
            f"entrypoint = {quoted(entrypoint)}",
            f"result_file = {quoted(result_file)}",
        ]
        if submission:
            lines.append(f"submission_source = {quoted(submission)}")
        lines += [
            "",
            "[data]",
            f"version = {quoted(DATA_VERSION)}",
            'registry = "datasets/registry.toml"',
            "",
            "[validation]",
            f"primary_version = {quoted(evaluation_version)}",
            'registry = "validation/registry.toml"',
            'folds = "validation/grouped_text_v1/folds.csv"',
            'primary_metric = "macro_f1"',
            "",
            "[execution]",
            "arguments = []",
            "",
            "[artifacts]",
            'directory = "artifacts"',
            "publish = false",
        ]
        (directory / "experiment.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (directory / "run.py").write_text(RUNNER, encoding="utf-8")
        (directory / "preprocess.py").write_text(PREPROCESS, encoding="utf-8")
        (directory / "build_submission.py").write_text(BUILD, encoding="utf-8")
        (directory / "run.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\npython3 "$(dirname "$0")/run.py" "$@"\n',
            encoding="utf-8",
        )
        (directory / "artifacts" / "README.md").write_text(
            "# Local artifacts\n\nЭтот каталог предназначен для weights, embeddings, OOF predictions и других больших файлов. Содержимое не публикуется. Для каждого локального файла следует записать источник, SHA-256 и команду построения в `results/metrics.json`.\n",
            encoding="utf-8",
        )
        readme = f"""# {number}: {title}

## Гипотеза

{hypothesis}

## Протокол

Dataset: `{DATA_VERSION}`. Основной evaluation protocol: `{evaluation_version}` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/{key}/preprocess.py --data /path/to/data.csv --output-dir /tmp/{key}
python3 experiments/{key}/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `{entrypoint}`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/{key}/.local/compute/`.

## Submission

{f"Source directory: `{submission}`. Локальный ZIP строится через `python3 experiments/{key}/build_submission.py --output /tmp/submission.zip`." if submission else "Для этого эксперимента отдельный submission package не формировался."}

## Результат

{outcome}

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
"""
        (directory / "README.md").write_text(readme, encoding="utf-8")
        metrics = {
            "experiment_id": number,
            "slug": slug,
            "status": status,
            "data_version": DATA_VERSION,
            "evaluation_version": evaluation_version,
            "conclusion": outcome,
            "historical_results": log_rows.get(key, []),
        }
        (directory / "results" / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print({"experiment_packages": len(SPECS)})


if __name__ == "__main__":
    main()
