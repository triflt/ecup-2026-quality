# Artifacts

Эксперимент не запускался, поэтому каталог не содержит model artifacts.

Каждая из десяти задач должна создать:

- `protocol_inputs/development_data.csv`;
- `protocol_inputs/development_selector_oof.npz`;
- `protocol_inputs/id_mapping.csv`;
- `protocol_inputs/protocol_input_audit.json`;
- `selection_audit.runtime.json`;
- `adapter.zip` и каталог adapter;
- `lora_holdout_predictions.csv`;
- `lora_holdout_report.json`;
- `output_contract.runtime.json`.

Артефакт нельзя принимать без `GO` во всех трёх audit/contract JSON и нулевого
числа sealed rows.
