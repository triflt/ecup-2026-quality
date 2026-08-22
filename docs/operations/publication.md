# Безопасная публикация

Текущий local Git history содержит ранний binary model artifact. Удаление файла из index защищает будущие commits, но не удаляет его из уже существующего commit.

Без отдельного решения о history rewrite рекомендуется создать clean public history:

```bash
python3 tools/check_publish_safety.py
python3 tools/export_public_snapshot.py /tmp/ecup-quality-public
cd /tmp/ecup-quality-public
git init
git add .
git commit -m "Initial reproducible E-CUP Quality research release"
```

Перед push повторно проверить staged files:

```bash
git status --short
git diff --cached --stat
git ls-files | rg '\.(zip|joblib|safetensors|npz|npy|pt|pth)$' && exit 1 || true
```

Не следует публиковать current local history до clean export либо осознанного `git filter-repo` rewrite. Это правило действует и для private remote: приватность репозитория не заменяет очистку истории. History rewrite является отдельной destructive operation и не выполняется автоматически.
