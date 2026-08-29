# Agent operating rules

Этот файл обязателен для любого AI-агента, продолжающего работу в репозитории.

1. Мы работаем как будущие чемпионы: quality bar — DS Master / Kaggle Grandmaster и первое место на Private/Final leaderboard, а не косметическое улучшение Public score.
2. Сначала прочитать `CODEX.md`, локальный `CODEX.local.md` (если он существует), `docs/current-status.md`, `docs/validation.md`, `docs/research/foundations/literature-and-competitions.md`, `reports/experiment-log.csv` и README целевого experiment package. `CODEX.local.md` никогда не коммитить.
3. Для новой гипотезы записать релевантный SOTA/champion precedent и проверяемый механизм переноса на наши данные.
4. Не выбирать модель по random split. Основной selection protocol — frozen grouped outer folds; любые random 70/30 проверки помечать как recurrence simulation.
5. Не использовать Public leaderboard для последовательного подбора мелких thresholds. Public — независимая проверка архитектурной гипотезы.
6. Новый эксперимент создаётся копированием `templates/experiment/` и получает уникальный числовой prefix.
7. Каждый эксперимент фиксирует dataset/evaluation versions из реестров; существующие версии не редактируются, новая корзина получает новый version ID.
8. Код должен работать с обычными `--data`, `--images`, `--output-dir`, без зависимости от private scheduler.
9. Private execution presets, URLs, credentials, tenant/project names и internal registry paths не коммитить. Локальная папка preset-файлов — `experiments/<id>/.local/compute/`.
10. Raw data, model weights, embeddings и submission ZIP не коммитить; они лежат локально в `artifacts/`.
11. Результат пригоден только после schema check, leakage audit, category F1, macro F1, fold dispersion и runtime estimate.
12. Любой rejected/failed experiment остаётся в журнале с причиной. Не переписывать историю.
13. После эксперимента обновить `results/metrics.json`, `reports/experiment-log.csv` и при необходимости `reports/submissions.csv`, `reports/champion.json` и локальный leaderboard.
14. До дорогого запуска заполнить в `experiment.toml` механизм гипотезы, единственный изменяемый фактор, ожидаемый прирост, условие опровержения и вычислительный бюджет. Отдельная глобальная доска гипотез больше не используется.
15. Новый лидер фиксируется в `reports/champion.json` только после повторной проверки, чистой сборки, сверки контрольных сумм, проверки времени и отсутствия утечки.
16. Параллельные агенты не редактируют одновременно общие журналы. Исследователь предлагает, критик проверяет новизну и честность сравнения, исполнитель запускает утверждённый опыт, интегратор обновляет общий статус.
17. Перед новой гипотезой проверить `experiments/README.md`, `reports/experiment-log.csv` и архив удалённых веток; не повторять отклонённый опыт без нового проверяемого основания.
18. Корпоративные DLP/EDR/quarantine/provenance-блокировки являются безусловной границей. При `Operation not permitted`, `Permission denied` или признаке блокировки немедленно прекратить чтение и любые преобразования объекта, сохранить файл и метаданные и сообщить пользователю. Запрещено удалять `com.apple.provenance`/quarantine, менять права или флаги ради обхода, копировать/переупаковывать/переименовывать заблокированный файл, отключать защиту либо повторять доступ другим инструментом. Перед локальным скачиванием корпоративного архива с данными, predictions/model outputs или размером более 50 MB запросить явное согласие пользователя; по возможности проверять артефакт внутри одобренной compute-среды. Это правило обязательно включать в поручения субагентам.
