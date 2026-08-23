# 642: Qwen3.8-27B, LoRA только для класса

Статус: **prompting gate 640 пройден; технический LoRA-smoke ожидает успешный
4B-preflight полного быстрого backend**.

Первый smoke подтвердил тот же OOM резервной реализации линейного внимания,
что и 4B-контроль. Первый kernel-повтор остановился до обучения из-за
несовместимой версии пакета. Повтор с совместимым hub-kernel снова попал в
PyTorch fallback, потому что отсутствовали FLA и causal-conv1d. Полный backend
сначала проверяется на 4B; входы, objective, seed и effective batch не меняются.

Это большой class-only контроль сетки. Он использует те же модельные поля,
первое изображение, prompt, selection multiset, seed, optimizer, эффективный
batch и frozen zero threshold, что `641`. Из-за памяти меняются только
microbatch `4 → 1` и accumulation `4 → 16`; effective batch остаётся `16`.

## Зачем нужен отдельный контроль

Без `642` нельзя понять, улучшает ли `644` качество за счёт масштаба 27B или за
счёт обучения доказательствам. Главные сравнения: `642−641` — эффект размера,
`644−642` — эффект evidence objective внутри 27B.

Техническая совместимость Qwen3.8-27B предварительно проверяется в `640`, но это
не разрешает training автоматически. Сначала выполняются только folds `0/3`.
Остальные folds допускает только `645`.

```bash
python experiments/642_qwen38_27b_class_only_lora/build_runtime.py --help
python experiments/642_qwen38_27b_class_only_lora/run_fold.py --help
```

Первоисточник модели: <https://huggingface.co/Qwen/Qwen3.8-27B>.
