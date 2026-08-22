from __future__ import annotations

"""Build frozen donor-only manifests without persisting source text or URLs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from contract import (
    MAX_NEGATIVE_REUSE,
    NEGATIVES_PER_POSITIVE,
    PINNED_SELECTOR_MANIFEST_SHA256,
    SCREEN_FOLDS,
    SELECTOR_BY_FOLD,
    SELECTOR_ROWS,
    SELECTOR_VERSION,
)
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/data.csv"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
SELECTOR_REPORT = (
    ROOT
    / "experiments/300_internvl35_attribute_screen/results/attribute_probe_selector_report.json"
)
CUE_COLUMNS = (
    "cue_fuel_or_gas",
    "cue_empty_equipment",
    "cue_ignition_source",
    "cue_fuel_included",
    "cue_combustible_material",
    "cue_absence_or_negation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_pair_sha256(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{row.positive_id}\t{row.negative_id}" for row in frame.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def boolean_column(values: pd.Series, *, name: str) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(f"invalid boolean values in {name}: {unexpected}")
    return normalized == "true"


def cue_masks(frame: pd.DataFrame) -> pd.Series:
    columns = []
    for name in CUE_COLUMNS:
        if name not in frame:
            raise ValueError(f"selector manifest lacks {name}")
        columns.append(boolean_column(frame[name], name=name).to_numpy())
    matrix = np.column_stack(columns).astype(np.int8)
    return pd.Series(["".join(map(str, row.tolist())) for row in matrix], index=frame.index)


def load_frozen_selector(selector_manifest: Path) -> pd.DataFrame:
    observed_sha = sha256(selector_manifest)
    if observed_sha != PINNED_SELECTOR_MANIFEST_SHA256:
        raise ValueError(
            "frozen selector manifest checksum mismatch: "
            f"observed={observed_sha} expected={PINNED_SELECTOR_MANIFEST_SHA256}"
        )
    selector_report = json.loads(SELECTOR_REPORT.read_text(encoding="utf-8"))
    if selector_report.get("selector_version") != SELECTOR_VERSION:
        raise ValueError("selector report version mismatch")
    if selector_report.get("selection_uses_labels") is not False:
        raise ValueError("selector must remain label-blind")
    if selector_report.get("selected_rows") != SELECTOR_ROWS:
        raise ValueError("selector report row count mismatch")
    reported_by_fold = {
        int(fold): int(count) for fold, count in selector_report.get("selected_by_fold", {}).items()
    }
    if reported_by_fold != SELECTOR_BY_FOLD:
        raise ValueError("selector report fold counts mismatch")

    selector = pd.read_csv(selector_manifest, compression="gzip", dtype={"id": str})
    required = {"id", "fold", "group_hash", "name", "description", *CUE_COLUMNS}
    missing = required - set(selector.columns)
    if missing:
        raise ValueError(f"selector manifest lacks columns: {sorted(missing)}")
    if len(selector) != SELECTOR_ROWS or selector.id.duplicated().any():
        raise ValueError("selector manifest row/identity invariant failed")
    observed_by_fold = selector.fold.astype(int).value_counts().sort_index().to_dict()
    if observed_by_fold != SELECTOR_BY_FOLD:
        raise ValueError("selector manifest fold counts mismatch")

    data = pd.read_csv(DATA, dtype={"id": str})
    guard = pd.read_csv(GUARD, dtype={"id": str})
    selector = selector.merge(
        data[["id", "category", "label"]], on="id", validate="one_to_one"
    ).merge(
        guard[["id", "fold", "connected_component", "safe_for_selection"]],
        on="id",
        suffixes=("", "_guard"),
        validate="one_to_one",
    )
    if not (selector.category.astype(str) == "Легковоспламеняющиеся").all():
        raise ValueError("frozen selector contains another category")
    if not np.array_equal(selector.fold.to_numpy(np.int8), selector.fold_guard.to_numpy(np.int8)):
        raise ValueError("selector/guard fold mismatch")
    selector["safe_for_selection"] = boolean_column(
        selector.safe_for_selection, name="safe_for_selection"
    )
    selector["cue_mask"] = cue_masks(selector)
    selector["text"] = (
        selector.name.fillna("").astype(str) + "\n" + selector.description.fillna("").astype(str)
    )
    return selector


def similarity_order(
    positive_position: int,
    negative_positions: list[int],
    matrix,
    ids: np.ndarray,
) -> list[tuple[float, int]]:
    similarities = (matrix[negative_positions] @ matrix[positive_position].T).toarray()[:, 0]
    return sorted(
        zip(similarities.astype(float), negative_positions),
        key=lambda item: (-item[0], str(ids[item[1]])),
    )


def build_pairs(selector: pd.DataFrame, *, outer_fold: int) -> pd.DataFrame:
    if outer_fold not in SCREEN_FOLDS:
        raise ValueError(f"outer fold must be one of {SCREEN_FOLDS}")
    donor = selector.loc[
        (selector.fold.astype(int) != outer_fold) & selector.safe_for_selection
    ].copy()
    donor = donor.sort_values("id", kind="mergesort").reset_index(drop=True)
    positives = donor.index[donor.label.astype(int) == 1].tolist()
    negatives = donor.index[donor.label.astype(int) == 0].tolist()
    if not positives or len(negatives) < NEGATIVES_PER_POSITIVE:
        raise ValueError("insufficient donor labels for pair construction")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(donor.text)
    ids = donor.id.astype(str).to_numpy()
    masks = donor.cue_mask.astype(str).to_numpy()
    reuse: Counter[int] = Counter()
    records: list[dict[str, object]] = []
    for positive_position in positives:
        same_mask = [
            position
            for position in negatives
            if masks[position] == masks[positive_position] and reuse[position] < MAX_NEGATIVE_REUSE
        ]
        chosen: list[tuple[float, int, str]] = []
        for similarity, negative_position in similarity_order(
            positive_position, same_mask, matrix, ids
        ):
            if len(chosen) == NEGATIVES_PER_POSITIVE:
                break
            chosen.append((similarity, negative_position, "same_cue_mask"))
        if len(chosen) < NEGATIVES_PER_POSITIVE:
            already = {position for _, position, _ in chosen}
            fallback = [
                position
                for position in negatives
                if position not in already and reuse[position] < MAX_NEGATIVE_REUSE
            ]
            for similarity, negative_position in similarity_order(
                positive_position, fallback, matrix, ids
            ):
                if len(chosen) == NEGATIVES_PER_POSITIVE:
                    break
                chosen.append((similarity, negative_position, "global_fallback"))
        if len(chosen) != NEGATIVES_PER_POSITIVE:
            raise ValueError(
                f"negative-reuse cap prevents four pairs for positive {ids[positive_position]}"
            )
        for similarity, negative_position, scope in chosen:
            reuse[negative_position] += 1
            records.append(
                {
                    "pair_index": len(records),
                    "outer_fold": outer_fold,
                    "positive_id": ids[positive_position],
                    "negative_id": ids[negative_position],
                    "positive_cue_mask": masks[positive_position],
                    "negative_cue_mask": masks[negative_position],
                    "selection_scope": scope,
                    "cosine_similarity": round(float(similarity), 8),
                }
            )
    pairs = pd.DataFrame(records)
    positive_counts = pairs.positive_id.value_counts()
    negative_counts = pairs.negative_id.value_counts()
    if not (positive_counts == NEGATIVES_PER_POSITIVE).all():
        raise RuntimeError("positive pair-count invariant failed")
    if int(negative_counts.max()) > MAX_NEGATIVE_REUSE:
        raise RuntimeError("negative reuse invariant failed")
    donor_by_id = donor.set_index("id")
    if not (donor_by_id.loc[pairs.positive_id, "label"].to_numpy() == 1).all():
        raise RuntimeError("positive label invariant failed")
    if not (donor_by_id.loc[pairs.negative_id, "label"].to_numpy() == 0).all():
        raise RuntimeError("negative label invariant failed")
    return pairs


def targets(output_dir: Path) -> dict[str, Path]:
    return {
        "membership": output_dir / "selector_membership.csv",
        **{f"fold_{fold}": output_dir / f"pair_manifest_fold_{fold}.csv" for fold in SCREEN_FOLDS},
        "audit": output_dir / "pair_manifest_audit.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_paths = targets(args.output_dir)
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite pair-manifest outputs: " + ", ".join(map(str, existing))
        )
    selector = load_frozen_selector(args.selector_manifest)
    membership = selector[
        ["id", "fold", "group_hash", "connected_component", "safe_for_selection", "cue_mask"]
    ].copy()
    membership = membership.sort_values("id", kind="mergesort").reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_paths["membership"], index=False)

    fold_audits: dict[str, dict[str, object]] = {}
    for fold in SCREEN_FOLDS:
        pairs = build_pairs(selector, outer_fold=fold)
        path = output_paths[f"fold_{fold}"]
        pairs.to_csv(path, index=False)
        negative_reuse = pairs.negative_id.value_counts()
        fold_audits[str(fold)] = {
            "pairs": len(pairs),
            "positive_rows": int(pairs.positive_id.nunique()),
            "negative_rows_used": int(pairs.negative_id.nunique()),
            "same_cue_mask_pairs": int((pairs.selection_scope == "same_cue_mask").sum()),
            "global_fallback_pairs": int((pairs.selection_scope == "global_fallback").sum()),
            "maximum_negative_reuse": int(negative_reuse.max()),
            "outer_fold_ids_in_pairs": len(
                (set(pairs.positive_id) | set(pairs.negative_id))
                & set(selector.loc[selector.fold.astype(int) == fold, "id"])
            ),
            "unsafe_ids_in_pairs": len(
                (set(pairs.positive_id) | set(pairs.negative_id))
                & set(selector.loc[~selector.safe_for_selection, "id"])
            ),
            "ordered_pair_sha256": ordered_pair_sha256(pairs),
            "file_sha256": sha256(path),
        }
    audit = {
        "version": "internvl_pairwise_flammable_manifests_v1",
        "status": "frozen_not_trained",
        "selector_version": SELECTOR_VERSION,
        "selector_label_blind": True,
        "pair_labels_source": "outer-donor train labels only",
        "selector_rows": len(selector),
        "screen_folds": list(SCREEN_FOLDS),
        "pair_rule": {
            "negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "maximum_negative_reuse": MAX_NEGATIVE_REUSE,
            "similarity": "donor-only char_wb TF-IDF cosine, ngram 3-5",
            "priority": "same six-bit cue mask, then all donor negatives",
            "tie_break": "string item id",
        },
        "folds": fold_audits,
        "input_sha256": {
            "selector_manifest": sha256(args.selector_manifest),
            "selector_report": sha256(SELECTOR_REPORT),
            "data": sha256(DATA),
            "guard": sha256(GUARD),
        },
        "output_sha256": {
            "selector_membership": sha256(output_paths["membership"]),
            **{
                f"pair_manifest_fold_{fold}": sha256(output_paths[f"fold_{fold}"])
                for fold in SCREEN_FOLDS
            },
        },
    }
    if any(
        details["outer_fold_ids_in_pairs"] or details["unsafe_ids_in_pairs"]
        for details in fold_audits.values()
    ):
        raise RuntimeError("donor isolation audit failed")
    output_paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "selector_rows": audit["selector_rows"],
                "folds": audit["folds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
