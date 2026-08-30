from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/673_verified_ocr_coverage_audit/audit_coverage.py"


def _module():
    spec = importlib.util.spec_from_file_location("exp673_audit_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_item_availability_is_fail_closed() -> None:
    module = _module()
    assert module.item_availability(["OCR_AVAILABLE"]) == "all_available"
    assert (
        module.item_availability(["OCR_AVAILABLE", "OCR_UNAVAILABLE"])
        == "partially_available"
    )
    assert module.item_availability(["OCR_UNAVAILABLE"]) == "all_unavailable"


def test_normalize_keeps_policy_tokens_but_removes_layout() -> None:
    module = _module()
    assert module.normalize("  БАД—Dietary\nSupplement! ") == "бад dietary supplement"
