from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation/build_semantic_family_graph_audit.py"
SPEC = importlib.util.spec_from_file_location("semantic_family_graph_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_builder
SPEC.loader.exec_module(audit_builder)


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": item_id,
                "name": name,
                "description": description,
                "category": "БАД" if index % 2 == 0 else "Легковоспламеняющиеся",
                "label": index % 2,
            }
            for index, (item_id, name, description) in enumerate(rows)
        ]
    )


def _fingerprint(
    item_id: str,
    position: int,
    *,
    exact: str,
    visual: str,
    informative: bool = True,
) -> audit_builder.ImageFingerprint:
    return audit_builder.ImageFingerprint(
        item_id=item_id,
        position=position,
        exact_sha1=exact,
        phash=f"phash-{visual}",
        dhash=f"dhash-{visual}",
        aspect_bucket=0,
        contrast=30.0 if informative else 2.0,
        entropy=6.0 if informative else 1.0,
        width=640,
        height=640,
    )


def _reused_fingerprint_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": "a",
                "position": 0,
                "exact_sha1": "a" * 40,
                "phash": "1" * 16,
                "dhash": "2" * 16,
                "aspect_bucket": 0,
                "contrast": 30.5,
                "entropy": 6.25,
                "width": 640,
                "height": 640,
            },
            {
                "item_id": "a",
                "position": 1,
                "exact_sha1": "b" * 40,
                "phash": "3" * 16,
                "dhash": "4" * 16,
                "aspect_bucket": 0,
                "contrast": 31.5,
                "entropy": 6.5,
                "width": 800,
                "height": 800,
            },
            {
                "item_id": "b",
                "position": 0,
                "exact_sha1": "c" * 40,
                "phash": "5" * 16,
                "dhash": "6" * 16,
                "aspect_bucket": 0,
                "contrast": 32.5,
                "entropy": 7.0,
                "width": 900,
                "height": 900,
            },
        ]
    )


def _pairs(graph: audit_builder.GraphAudit, stage: str) -> set[tuple[str, str]]:
    return {(edge["left_id"], edge["right_id"]) for edge in graph.edges if edge["stage"] == stage}


def test_text_edges_require_pair_specific_corroboration() -> None:
    frame = _frame(
        [
            ("full-a", "Крем Botanica Aloe A7", "Полностью одинаковое описание товара"),
            ("full-b", "Крем Botanica Aloe A7", "Полностью одинаковое описание товара"),
            (
                "name-a",
                "Туристический газовый баллон",
                "Баллон для походной горелки, пропан-бутан, безопасный клапан",
            ),
            (
                "name-b",
                "Туристический газовый баллон",
                "Баллон для походной горелки, пропан-бутан, безопасный клапан новый",
            ),
            ("generic-a", "Подарочный набор", "Морские водоросли в капсулах"),
            ("generic-b", "Подарочный набор", "Чехол для садовых инструментов"),
            (
                "mask-a",
                "Газовый баллон туристический 450 мл",
                "Сменный баллон для компактной походной горелки с безопасным клапаном",
            ),
            (
                "mask-b",
                "Газовый баллон туристический 500 мл",
                "Сменный баллон для компактной походной горелки с безопасным клапаном новый",
            ),
        ]
    )
    graph = audit_builder.build_label_blind_graph(frame, [], config=audit_builder.AuditConfig())

    assert ("full-a", "full-b") in _pairs(graph, "exact_full_text")
    assert ("name-a", "name-b") in _pairs(graph, "exact_name_corroborated")
    assert ("mask-a", "mask-b") in _pairs(graph, "masked_name_corroborated")
    assert not any(
        {edge["left_id"], edge["right_id"]} == {"generic-a", "generic-b"} for edge in graph.edges
    )
    assert all(edge["label_blind"] is True for edge in graph.edges)
    assert all(
        {
            "stage",
            "left_id",
            "right_id",
            "key_hash",
            "key_degree",
            "corroboration",
            "informative_guard",
            "degree_guard",
        }
        <= set(edge)
        for edge in graph.edges
    )


def test_product_identity_is_specific_and_quantity_normalized() -> None:
    assert audit_builder.product_identity_signature("БАД для иммунитета") == ""
    assert audit_builder.product_identity_signature(
        "Капсулы для похудения и снижения веса, эффективный жиросжигатель"
    ) == ""
    assert audit_builder.product_identity_signature(
        "Хлорофилл жидкий детокс для похудения для ЖКТ от шлаков"
    ) == ""
    assert audit_builder.product_identity_signature("Топливо для зажигалок и зиппо 100 мл") == ""
    assert audit_builder.product_identity_signature(
        "Газовый баллон туристический 450 мл"
    ) == audit_builder.product_identity_signature("Газовый баллон туристический 500 мл")
    assert audit_builder.product_identity_signature("Фонарь модель F7")


def test_text_degree_caps_quarantine_instead_of_connecting() -> None:
    frame = _frame(
        [
            ("a", "Одинаковое распространенное название", "Почти одинаковое описание один"),
            ("b", "Одинаковое распространенное название", "Почти одинаковое описание два"),
            ("c", "Одинаковое распространенное название", "Почти одинаковое описание три"),
        ]
    )
    config = replace(
        audit_builder.AuditConfig(),
        exact_name_max_degree=2,
        masked_name_max_degree=2,
    )
    graph = audit_builder.build_label_blind_graph(frame, [], config=config)

    assert not graph.edges
    exact_name_audits = [
        row for row in graph.key_degrees if row["stage"] == "exact_name_corroborated"
    ]
    assert len(exact_name_audits) == 1
    assert exact_name_audits[0]["status"] == "quarantined_degree"
    assert exact_name_audits[0]["key_degree"] == 3
    assert any(row["sample_type"] == "quarantined_text_key" for row in graph.manual_samples)


def test_exact_and_perceptual_image_rules_are_fail_closed() -> None:
    frame = _frame(
        [
            ("first-a", "Альфа орхидея модель A1", "Красный предмет для кухни"),
            ("first-b", "Альфа орхидея модель A1", "Зеленый предмет для сада"),
            ("aux-a", "Гамма кедр", "Белая аптечка для машины"),
            ("aux-b", "Дельта сосна", "Черный фонарь для гаража"),
            ("two-a", "Эпсилон береза", "Синий контейнер для воды"),
            ("two-b", "Дзета липа", "Желтая сумка для обуви"),
            (
                "identity-a",
                "Термокружка Арктика модель T5",
                "Стальная кружка с крышкой",
            ),
            (
                "identity-b",
                "Термокружка Арктика модель T5",
                "Другая редакция описания",
            ),
            ("percept-aux-a", "Эта сирень", "Оранжевая коробка для ниток"),
            ("percept-aux-b", "Тета мята", "Фиолетовый чехол для удочки"),
            (
                "percept-first-a",
                "Фонарь туристический Луч F7",
                "Серебряная кружка для чая",
            ),
            (
                "percept-first-b",
                "Фонарь туристический Луч F7",
                "Золотая лейка для цветов",
            ),
            ("flat-a", "Лямбда пихта", "Квадратная салфетка из ткани"),
            ("flat-b", "Лямбда пихта", "Круглая подставка из дерева"),
        ]
    )
    fingerprints = [
        _fingerprint("first-a", 0, exact="exact-first", visual="first"),
        _fingerprint("first-b", 0, exact="exact-first", visual="first"),
        _fingerprint("aux-a", 1, exact="exact-aux", visual="aux"),
        _fingerprint("aux-b", 1, exact="exact-aux", visual="aux"),
        _fingerprint("two-a", 1, exact="exact-two-one", visual="two-one"),
        _fingerprint("two-b", 1, exact="exact-two-one", visual="two-one"),
        _fingerprint("two-a", 2, exact="exact-two-two", visual="two-two"),
        _fingerprint("two-b", 2, exact="exact-two-two", visual="two-two"),
        _fingerprint("identity-a", 1, exact="identity-aux", visual="identity"),
        _fingerprint("identity-b", 1, exact="identity-aux", visual="identity"),
        _fingerprint("percept-aux-a", 1, exact="raw-pa", visual="near-aux"),
        _fingerprint("percept-aux-b", 1, exact="raw-pb", visual="near-aux"),
        _fingerprint("percept-first-a", 0, exact="raw-pfa", visual="near-first"),
        _fingerprint("percept-first-b", 0, exact="raw-pfb", visual="near-first"),
        _fingerprint("flat-a", 0, exact="flat", visual="flat", informative=False),
        _fingerprint("flat-b", 0, exact="flat", visual="flat", informative=False),
    ]
    graph = audit_builder.build_label_blind_graph(
        frame, fingerprints, config=audit_builder.AuditConfig()
    )

    assert ("first-a", "first-b") in _pairs(graph, "exact_first_image")
    assert ("aux-a", "aux-b") not in _pairs(graph, "exact_auxiliary_images")
    assert ("two-a", "two-b") not in _pairs(graph, "exact_auxiliary_images")
    assert ("identity-a", "identity-b") in _pairs(graph, "exact_auxiliary_images")
    assert ("percept-aux-a", "percept-aux-b") not in _pairs(graph, "perceptual_corroborated")
    assert ("percept-first-a", "percept-first-b") in _pairs(graph, "perceptual_corroborated")
    assert not any(
        {edge["left_id"], edge["right_id"]} == {"flat-a", "flat-b"} for edge in graph.edges
    )
    assert any(row["sample_type"] == "rejected_auxiliary_pair" for row in graph.manual_samples)
    assert any(row["sample_type"] == "rejected_perceptual_pair" for row in graph.manual_samples)
    assert any(row["status"] == "rejected_uninformative" for row in graph.key_degrees)


def test_reused_auxiliary_keys_are_quarantined_across_product_identities() -> None:
    frame = _frame(
        [
            ("a", "Plantago Skin Hair Nails", "Комплекс с биотином"),
            ("b", "Olimp Creatine Monohydrate", "Креатин для тренировок"),
        ]
    )
    fingerprints = [
        _fingerprint("a", 1, exact="seller-slide-1", visual="seller-1"),
        _fingerprint("b", 1, exact="seller-slide-1", visual="seller-1"),
        _fingerprint("a", 2, exact="seller-slide-2", visual="seller-2"),
        _fingerprint("b", 2, exact="seller-slide-2", visual="seller-2"),
    ]

    graph = audit_builder.build_label_blind_graph(
        frame, fingerprints, config=audit_builder.AuditConfig()
    )

    assert not graph.edges
    reused = [
        row for row in graph.key_degrees if row["status"] == "quarantined_cross_identity_reuse"
    ]
    assert reused
    assert all(row["accepted_edges"] == 0 for row in reused)
    assert any(row["sample_type"] == "quarantined_reused_image_key" for row in graph.manual_samples)


def test_generic_exact_text_conflicting_first_images_is_quarantined() -> None:
    frame = _frame(
        [
            (
                "generic-a",
                "БАД для иммунитета",
                "БАД из натуральных ингредиентов для укрепления иммунитета.",
            ),
            (
                "generic-b",
                "БАД для иммунитета",
                "БАД из натуральных ингредиентов для укрепления иммунитета.",
            ),
        ]
    )
    fingerprints = [
        _fingerprint("generic-a", 0, exact="product-a", visual="product-a"),
        _fingerprint("generic-b", 0, exact="product-b", visual="product-b"),
    ]

    graph = audit_builder.build_label_blind_graph(
        frame, fingerprints, config=audit_builder.AuditConfig()
    )

    assert not graph.edges
    exact_text = [row for row in graph.key_degrees if row["stage"] == "exact_full_text"]
    assert exact_text[0]["status"] == "quarantined_first_image_conflict"
    assert any(
        row["sample_type"] == "quarantined_generic_text_group" for row in graph.manual_samples
    )


KNOWN_FALSE_REPLAY_CASES = [
    ("100", "8785", "cross_identity_auxiliary"),
    ("10537", "17869", "generic_perceptual_auxiliary"),
    ("17869", "285", "generic_perceptual_auxiliary"),
    ("17869", "781", "generic_perceptual_auxiliary"),
    ("6776", "7213", "generic_exact_text_auxiliary"),
    ("932", "9462", "cross_identity_auxiliary"),
    ("10400", "153", "generic_exact_text_auxiliary"),
    ("153", "7198", "generic_exact_text_auxiliary"),
    ("2163", "3563", "generic_marketing_name"),
    ("11833", "10643", "commodity_masked_name"),
    ("2820", "714", "commodity_exact_text"),
    ("245", "13818", "generic_exact_text"),
]


@pytest.mark.parametrize(("left_id", "right_id", "case"), KNOWN_FALSE_REPLAY_CASES)
def test_all_twelve_known_false_pairs_are_blocked(
    left_id: str, right_id: str, case: str
) -> None:
    generic_weight_name = (
        "Капсулы для похудения и снижения веса, эффективный жиросжигатель"
    )
    generic_weight_description = (
        "Натуральные капсулы помогают снизить вес и уменьшить аппетит."
    )
    if case == "cross_identity_auxiliary":
        rows = [
            (left_id, "Plantago Skin Hair Nails A1", "Комплекс с биотином для волос"),
            (right_id, "Olimp Creatine Monohydrate B2", "Креатин для тренировок"),
        ]
    elif case in {"generic_perceptual_auxiliary", "generic_exact_text_auxiliary"}:
        rows = [
            (left_id, generic_weight_name, generic_weight_description),
            (right_id, generic_weight_name, generic_weight_description),
        ]
    elif case == "generic_marketing_name":
        rows = [
            (left_id, generic_weight_name, "Редуксин Лайт для контроля аппетита"),
            (right_id, generic_weight_name, generic_weight_description),
        ]
    elif case == "commodity_masked_name":
        rows = [
            (
                left_id,
                "Беспламенный нагреватель туристического питания, набор 10 шт",
                "Нагреватель для разогрева реторт пакетов и сухпайков",
            ),
            (
                right_id,
                "Беспламенный нагреватель туристического питания, набор 100 шт",
                "Нагреватель для разогрева реторт пакетов и сухпайков",
            ),
        ]
    elif case == "commodity_exact_text":
        rows = [
            (left_id, "Топливо для зажигалок и зиппо 100 мл", "Очищенное топливо"),
            (right_id, "Топливо для зажигалок и зиппо 100 мл", "Очищенное топливо"),
        ]
    elif case == "generic_exact_text":
        rows = [
            (
                left_id,
                "Хлорофилл жидкий детокс для похудения для ЖКТ от шлаков",
                "Хлорофилл для детокс программ и нормализации обмена веществ",
            ),
            (
                right_id,
                "Хлорофилл жидкий детокс для похудения для ЖКТ от шлаков",
                "Хлорофилл для детокс программ и нормализации обмена веществ",
            ),
        ]
    else:  # pragma: no cover - guards the closed replay vocabulary
        raise AssertionError(case)

    fingerprints = [
        _fingerprint(left_id, 0, exact=f"first-{left_id}", visual=f"first-{left_id}"),
        _fingerprint(right_id, 0, exact=f"first-{right_id}", visual=f"first-{right_id}"),
    ]
    if case in {"cross_identity_auxiliary", "generic_exact_text_auxiliary"}:
        fingerprints.extend(
            [
                _fingerprint(left_id, 1, exact="shared-aux", visual="shared-aux"),
                _fingerprint(right_id, 1, exact="shared-aux", visual="shared-aux"),
            ]
        )
    if case == "cross_identity_auxiliary":
        fingerprints.extend(
            [
                _fingerprint(left_id, 2, exact="shared-aux-2", visual="shared-aux-2"),
                _fingerprint(right_id, 2, exact="shared-aux-2", visual="shared-aux-2"),
            ]
        )
    if case == "generic_perceptual_auxiliary":
        fingerprints.extend(
            [
                _fingerprint(left_id, 1, exact=f"raw-{left_id}", visual="marketing"),
                _fingerprint(right_id, 1, exact=f"raw-{right_id}", visual="marketing"),
            ]
        )

    graph = audit_builder.build_label_blind_graph(
        _frame(rows), fingerprints, config=audit_builder.AuditConfig()
    )

    assert not any(
        {edge["left_id"], edge["right_id"]} == {left_id, right_id}
        for edge in graph.edges
    )
    components = dict(zip([left_id, right_id], graph.component_ids))
    assert components[left_id] != components[right_id]


def test_component_closure_veto_requires_pairwise_strong_anchors() -> None:
    frame = _frame(
        [
            ("a", "Brand Alpha model A1", "Полностью одинаковое описание товара"),
            ("b", "Brand Alpha model A1", "Полностью одинаковое описание товара"),
            ("c", "БАД для иммунитета", "Другое описание"),
        ]
    )
    fingerprints = [
        _fingerprint("b", 0, exact="bridge-first", visual="bridge-first"),
        _fingerprint("c", 0, exact="bridge-first", visual="bridge-first"),
    ]

    graph = audit_builder.build_label_blind_graph(
        frame, fingerprints, config=audit_builder.AuditConfig()
    )

    assert ("a", "b") in _pairs(graph, "exact_full_text")
    assert ("b", "c") not in _pairs(graph, "exact_first_image")
    assert any(
        row["sample_type"] == "quarantined_component_merge" for row in graph.manual_samples
    )


def test_perceptual_auxiliary_edge_needs_text_corroboration() -> None:
    frame = _frame(
        [
            (
                "a",
                "Термокружка дорожная",
                "Стальная термокружка с плотной крышкой для долгой поездки",
            ),
            (
                "b",
                "Термокружка дорожная",
                "Стальная термокружка с плотной крышкой для долгой поездки новая",
            ),
        ]
    )
    fingerprints = [
        _fingerprint("a", 1, exact="raw-a", visual="near"),
        _fingerprint("b", 1, exact="raw-b", visual="near"),
    ]
    graph = audit_builder.build_label_blind_graph(
        frame, fingerprints, config=audit_builder.AuditConfig()
    )

    edges = [edge for edge in graph.edges if edge["stage"] == "perceptual_corroborated"]
    assert len(edges) == 1
    assert edges[0]["corroboration"].startswith("two_hashes_per_match_plus_")


def test_topology_is_label_blind_and_row_order_invariant() -> None:
    frame = _frame(
        [
            ("z", "Крем Botanica Aloe A7", "Полностью одинаковое описание товара"),
            ("a", "Крем Botanica Aloe A7", "Полностью одинаковое описание товара"),
            ("m", "Уникальный фонарь", "Отдельное описание без совпадений"),
        ]
    )
    first = audit_builder.build_label_blind_graph(frame, [], config=audit_builder.AuditConfig())
    changed = frame.iloc[::-1].reset_index(drop=True).copy()
    changed["category"] = ["Легковоспламеняющиеся", "БАД", "Легковоспламеняющиеся"]
    changed["label"] = [1, 0, 1]
    second = audit_builder.build_label_blind_graph(changed, [], config=audit_builder.AuditConfig())

    assert first.edges == second.edges
    assert first.key_degrees == second.key_degrees
    first_components = dict(zip(frame.id, first.component_ids))
    second_components = dict(zip(changed.id, second.component_ids))
    assert first_components == second_components


def test_incremental_audit_reports_each_edge_stage() -> None:
    frame = _frame(
        [
            ("a", "Крем Botanica Aloe A7", "Полностью одинаковое описание товара"),
            ("b", "Крем Botanica Aloe A7", "Полностью одинаковое описание товара"),
            ("c", "Уникальный фонарь", "Отдельное описание без совпадений"),
        ]
    )
    graph = audit_builder.build_label_blind_graph(frame, [], config=audit_builder.AuditConfig())
    incremental = audit_builder.incremental_component_audit(frame, graph.edges)

    assert [row["stage"] for row in incremental] == list(audit_builder.STAGES)
    assert incremental[0]["new_unions"] == 1
    assert incremental[-1]["largest_component"] == 2


def test_output_targets_are_never_overwritten(tmp_path: Path) -> None:
    paths = audit_builder.target_paths(tmp_path)
    paths["edges"].write_text("existing\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_builder.ensure_targets_absent(paths)


def test_file_sha256_uses_strict_macos_fallback_only_after_permission_errors() -> None:
    path = Path("fresh-image-fingerprints.csv.gz")
    digest = "a" * 64
    completed = subprocess.CompletedProcess(
        args=["shasum"],
        returncode=0,
        stdout=f"{digest}  {path}\n",
        stderr="",
    )

    with (
        patch.object(Path, "open", side_effect=PermissionError) as direct_open,
        patch.object(audit_builder.time, "sleep"),
        patch.object(audit_builder.subprocess, "run", return_value=completed) as checksum,
    ):
        assert audit_builder.file_sha256(path) == digest

    assert direct_open.call_count == 4
    checksum.assert_called_once_with(
        ["shasum", "-a", "256", "--", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_file_sha256_rejects_malformed_fallback_output() -> None:
    path = Path("fresh-image-fingerprints.csv.gz")
    completed = subprocess.CompletedProcess(
        args=["shasum"], returncode=0, stdout="not-a-digest\n", stderr=""
    )

    with (
        patch.object(Path, "open", side_effect=PermissionError),
        patch.object(audit_builder.time, "sleep"),
        patch.object(audit_builder.subprocess, "run", return_value=completed),
        pytest.raises(RuntimeError, match="invalid shasum output"),
    ):
        audit_builder.file_sha256(path)


def test_file_sha256_does_not_fallback_for_other_read_errors() -> None:
    path = Path("missing-image-fingerprints.csv.gz")
    with (
        patch.object(Path, "open", side_effect=FileNotFoundError),
        patch.object(audit_builder.subprocess, "run") as checksum,
        pytest.raises(FileNotFoundError),
    ):
        audit_builder.file_sha256(path)

    checksum.assert_not_called()


def test_reused_fingerprints_accept_exact_complete_sorted_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fingerprints.csv.gz"
    _reused_fingerprint_frame().to_csv(path, index=False, compression="gzip")

    fingerprints = audit_builder.read_reused_fingerprints(path, {"a", "b"})

    assert [(row.item_id, row.position) for row in fingerprints] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
    ]
    assert all(isinstance(row.position, int) for row in fingerprints)
    assert all(isinstance(row.contrast, float) for row in fingerprints)


@pytest.mark.parametrize(
    "failure",
    [
        "schema",
        "empty",
        "id_set",
        "duplicate",
        "gap",
        "sha1",
        "phash",
        "metric",
        "dimensions",
        "aspect",
        "order",
    ],
)
def test_reused_fingerprints_reject_invalid_artifacts(tmp_path: Path, failure: str) -> None:
    frame = _reused_fingerprint_frame()
    expected_ids = {"a", "b"}
    if failure == "schema":
        frame = frame.rename(columns={"entropy": "image_entropy"})
    elif failure == "empty":
        frame = frame.iloc[:0]
    elif failure == "id_set":
        expected_ids = {"a", "b", "missing"}
    elif failure == "duplicate":
        frame.loc[1, "position"] = 0
    elif failure == "gap":
        frame.loc[1, "position"] = 2
    elif failure == "sha1":
        frame.loc[0, "exact_sha1"] = "not-a-sha1"
    elif failure == "phash":
        frame.loc[0, "phash"] = "g" * 16
    elif failure == "metric":
        frame.loc[0, "entropy"] = 9.0
    elif failure == "dimensions":
        frame.loc[0, "width"] = 0
    elif failure == "aspect":
        frame.loc[0, "aspect_bucket"] = 1
    elif failure == "order":
        frame = frame.iloc[::-1]
    path = tmp_path / f"{failure}.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError):
        audit_builder.read_reused_fingerprints(path, expected_ids)


def test_reused_fingerprint_reader_has_permission_fallback() -> None:
    path = Path("fresh-image-fingerprints.csv.gz")
    expected = _reused_fingerprint_frame().astype(str)
    completed = subprocess.CompletedProcess(
        args=["gzip"], returncode=0, stdout=b"compressed-output", stderr=b""
    )
    with (
        patch.object(
            audit_builder.pd,
            "read_csv",
            side_effect=[PermissionError, expected],
        ) as reader,
        patch.object(audit_builder.subprocess, "run", return_value=completed) as decompressor,
    ):
        frame = audit_builder._read_fingerprint_frame(path)

    assert frame.equals(expected)
    assert reader.call_count == 2
    decompressor.assert_called_once_with(
        ["gzip", "-dc", "--", str(path)],
        check=True,
        capture_output=True,
    )


def test_cli_image_sources_are_required_and_mutually_exclusive() -> None:
    parser = audit_builder.build_argument_parser()
    common = ["--data", "data.csv", "--output-dir", "output", "--draft-only"]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args(
            common
            + [
                "--images-zip",
                "images.zip",
                "--reuse-fingerprints",
                "fingerprints.csv.gz",
            ]
        )

    parsed = parser.parse_args(common + ["--reuse-fingerprints", "fingerprints.csv.gz"])
    assert parsed.images_zip is None
    assert parsed.reuse_fingerprints == Path("fingerprints.csv.gz")
