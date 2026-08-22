# Локальные артефакты 602

Здесь не хранятся веса, ZIP-файлы, датасет, изображения, URL или параметры
внутреннего запуска. Локальный output одного seed/fold обязан содержать:

- `lora_holdout_predictions.csv`;
- `lora_holdout_report.json`;
- `adapter.zip`;
- `protocol_inputs/protocol_input_audit.json`;
- `seed_selection_audit.runtime.json`;
- `seed_output_contract.runtime.json`.

Публикуются только итоговый небольшой report и воспроизводимый код после
прохождения контракта.
