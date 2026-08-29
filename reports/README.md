# Result tables

## `experiment-log.csv`

Append-only журнал offline experiments и infrastructure checks. Каждая строка содержит дату, experiment group, validation protocol, category F1, Macro F1, status и вывод. Failed, invalid и rejected rows не удаляются.

## `submissions.csv`

Журнал platform submissions и готовых candidates. `Private F1` остаётся пустым до фактического результата. ZIP archives и model artifacts в Git не входят.

## `transfer-ledger.json`

Machine-readable журнал переноса local validation в Public. Для каждого проверенного изменения фиксирует тип вмешательства, frozen end-to-end local evidence, Public delta, решение и состояние привязки загруженного файла к SHA. Standalone/AP improvement без выжившего изменения после fusion, thresholds и priors не считается ship evidence.

## Архивы и аудиты

- старые hypothesis/research-memory журналы перенесены в `docs/research/archive/`;
- аудит комплекта данных находится в `datasets/audits/`;
- аудит целостности validation находится в `validation/audits/`.

В корне `reports/` остаются только таблицы, необходимые для чтения текущего
решения, сравнения экспериментов и привязки Public-результатов к SHA.

## Обновление

После запуска сначала обновляется `experiments/<id>/results/metrics.json`, затем добавляется строка в общий CSV. Для leaderboard result дополнительно обновляются `submissions.csv`, `champion.json` и при необходимости `semantic-v3-leaderboard.csv`.
