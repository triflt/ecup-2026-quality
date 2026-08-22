# Локальные артефакты эксперимента 620

В Git не добавляются development manifest, feature bundles, model files, OOF
предсказания и router outputs. Разрешённая локальная структура после прохождения
зависимостей:

- `evidence/development_evidence_manifest.jsonl`;
- `evidence/evidence_manifest_audit.json`;
- `features/hypothesis_features.npz`;
- `features/feature_contract.json`;
- `null/null_router_output.npz`;
- `null/null_control_audit.json`.

Сейчас ни один из этих артефактов не создавался в рамках карточки. Builder и
evaluator отказываются перезаписывать непустые каталоги.
