# Воспроизводимость и execution boundary

Public experiment code не знает о scheduler, tenant, object storage и internal model registry. Он принимает обычные CLI arguments и environment variables:

```text
--data /absolute/path/data.csv
--images /absolute/path/images
--output-dir /absolute/path/output
--model-root /absolute/path/pretrained-models
```

Локальный execution preset отвечает только за:

- compute flavor и time limit;
- container image;
- доставку repository/data/model artifacts;
- передачу CLI arguments;
- выгрузку output directory.

Private presets хранятся в `.local/execution/` либо в другом локальном каталоге и исключены из Git. Overrides с credentials также всегда локальны. Это соответствует модели preset + ignored overrides: reusable command остаётся в experiment package, infrastructure binding — за пределами публикуемого репозитория.

Перед публикацией выполнить:

```bash
python tools/check_publish_safety.py
pytest
git status --short
```
