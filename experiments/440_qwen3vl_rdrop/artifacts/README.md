# Локальные артефакты 440

private compute platform bundles, fold adapters, downloaded images and submission archives are not
published. Before accepting a screen bundle, record its source job, byte size,
ZIP integrity, per-file SHA-256 and the Qwen3-VL registry revision.

`selector_probe/qwen_rdrop_selector.zip` получен из короткой задачи
`redacted-job`. В нём один JSON с точными индексами и порядком строк
для folds 0 и 3 в private compute platform runtime. SHA-256 архива:
`79b5738b72cccf84e7906e1c235ea63b3c9d1643649c76353eb5c9c5f18b69a0`;
ZIP integrity проверена.

Оба screen bundle получены и приняты локально:

- fold 0, job `redacted-job`: внешний ZIP SHA-256
  `75c3be479afdf977d45b87f00556e976d9eb9a59d963afecca30b2d2b6c524d1`,
  адаптер `bbb4fb9fdb6ae4255f30167bfe45b06069af051bfc97809d53139ef0d9e1835e`;
- fold 3, job `redacted-job`: внешний ZIP SHA-256
  `a5be21d9b9c6e43fb0e34741cb38908384e9101fdec928f1530f44ad0e45f6ed`,
  адаптер `fb680ba0a7e8ef11de44430645cbd9f709508a03610eaf69e3224230c5bee113`.

Экран провален, поэтому полный manifest, остальные fold, повтор seed и архив
отправки намеренно не создаются. Адаптеры остаются локальными артефактами
отклонённого опыта и не публикуются в S3.
