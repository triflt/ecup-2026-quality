from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
EXPERIMENT = ROOT / "experiments/661_qwen35_4b_class_only_full_control"


def load(name: str):
    path = EXPERIMENT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp661_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuation_gate_is_exact() -> None:
    module = load("verify_gate")
    gate = module.verify(EXPERIMENT / "results/continuation_gate.json", 2)
    assert gate["allowed_folds"] == [1, 2, 4]
    assert gate["runtime_backend"] == "legacy_eager"
    assert gate["public_used"] is False


def test_private_preset_keeps_url_ephemeral(tmp_path: Path) -> None:
    module = load("build_private_preset")
    base = tmp_path / "base.yml"
    url = tmp_path / "url.txt"
    output = tmp_path / "fold1.yml"
    base.write_text(
        """job:
  time_limit: 8h
  flavor: h100-1x
  region: test
  image: test-image
  preemption: forbidden
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: \"false\"
    PYTORCH_ALLOC_CONF: expandable_segments:True
  input:
    - {type: model_registry, mrid: test/model/revision, dst: /hf_models/}
""",
        encoding="utf-8",
    )
    url.write_text("https://example.invalid/bundle", encoding="utf-8")
    args = module.parser().parse_args(
        [
            "--base-preset",
            str(base),
            "--bundle-url-file",
            str(url),
            "--gate",
            str(EXPERIMENT / "results/continuation_gate.json"),
            "--fold",
            "1",
            "--output",
            str(output),
        ]
    )
    payload = module.build(args)
    assert "https://example.invalid" not in payload
    assert "BUNDLE_URL: ${BUNDLE_URL}" in payload
    assert "--fold 1" in payload
    assert "--micro-batch-size-override 2" in payload

