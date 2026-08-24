from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "experiments"
    / "653_qwen36_27b_lora_runtime_preflight"
    / "build_private_preset.py"
)
SPEC = importlib.util.spec_from_file_location("qwen36_lora_preset_653", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_builder_preserves_model_input_and_uses_four_gpu_flavor(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    four = tmp_path / "four.yml"
    script = Path("experiments/653/technical_smoke.py")
    vendor = Path("experiments/653/.local/peft.zip")
    output = tmp_path / "out.yml"
    base.write_text(
        """job:
  generate_name: old
  time_limit: 1m
  flavor: one-gpu
  region: test-region
  image: test-image
  preemption: false
  work_dir: /work
  env:
    RUNTIME_URL: secret
    TOKENIZERS_PARALLELISM: "false"
    PYTORCH_ALLOC_CONF: expandable_segments:True
  input:
    - {type: files, src: old.py, dst: /work/old.py}
    - {type: model_registry, src: model-secret, dst: /hf_models/}
  output:
    - {type: files, name: old, src: /old}
""",
        encoding="utf-8",
    )
    four.write_text("job:\n  flavor: four-gpu\n", encoding="utf-8")
    payload = MODULE.build(
        SimpleNamespace(
            base_qwen36=base,
            base_four_gpu=four,
            smoke_script=script,
            peft_zip=vendor,
            output=output,
        )
    )
    assert "flavor: four-gpu" in payload
    assert "generate_name: qwen-lora" in payload
    assert "RUNTIME_URL" not in payload
    assert payload.count("type: files") == 3
    assert payload.count("type: model_registry") == 1
    assert "name: qwen36_lora_smoke" in payload
    assert "src: /work/output" in payload


def test_builder_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "out.yml"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        MODULE.build(SimpleNamespace(output=output))


def test_builder_rejects_absolute_file_input(tmp_path: Path) -> None:
    output = tmp_path / "out.yml"
    base = tmp_path / "base.yml"
    four = tmp_path / "four.yml"
    base.write_text("job:\n", encoding="utf-8")
    four.write_text("job:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repository-relative"):
        MODULE.build(
            SimpleNamespace(
                output=output,
                base_qwen36=base,
                base_four_gpu=four,
                smoke_script=tmp_path / "smoke.py",
                peft_zip=Path("vendor.zip"),
            )
        )
