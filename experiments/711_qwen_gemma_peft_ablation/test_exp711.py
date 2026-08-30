from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("exp711_run", ROOT / "run_epoch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base():
    return {"r": 16, "lora_alpha": 32, "use_rslora": True, "lora_dropout": 0.05}


def test_rsdora_changes_only_dora_flag():
    module = load()
    assert module.configure_lora_kwargs("rsdora", base()) == {**base(), "use_dora": True}


def test_standard_dora_uses_standard_rank_scaling():
    module = load()
    assert module.configure_lora_kwargs("dora", base()) == {
        **base(), "use_rslora": False, "use_dora": True
    }


def test_rspissa_changes_only_initialization():
    module = load()
    assert module.configure_lora_kwargs("rspissa", base()) == {
        **base(), "init_lora_weights": "pissa_niter_4"
    }


def test_rsloraplus_leaves_adapter_config_matched():
    module = load()
    assert module.configure_lora_kwargs("rsloraplus", base()) == base()


@pytest.mark.parametrize("field,value", [("r", 32), ("lora_alpha", 64), ("use_rslora", False)])
def test_parent_contract_drift_fails_closed(field, value):
    module = load()
    kwargs = base()
    kwargs[field] = value
    with pytest.raises(ValueError):
        module.configure_lora_kwargs("rsdora", kwargs)
