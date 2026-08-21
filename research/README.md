# Legacy research source

Здесь сохранены исторические training/evaluation scripts. Большие outputs, raw data, manifests, adapters, embeddings и private execution files исключены из Git.

Новые запуски должны идти через соответствующую папку `experiments/`, которая фиксирует hypothesis, validation protocol, status и result record. Общие новые функции следует добавлять в `src/ecup_quality/`, а не копировать между scripts.

## Правило миграции

При изменении исторического script:

1. заменить absolute infrastructure paths на `ECUP_*` environment variables с backward-compatible default;
2. вынести повторяющуюся normalization/metrics/split logic в `src/ecup_quality/`;
3. сохранить исторический result JSON в experiment package как агрегированные метрики;
4. не коммитить локальные artifacts.
