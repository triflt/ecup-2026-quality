from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PAIR_PATH = ROOT / "evaluate_retrieval_pair_verifier.py"
PREREGISTER_SELF_SHA256 = "0762ff0c76064f0ab2d7e842ad9d324aeaaf83d567f035f35a6c3f030670776b"
SEED = 42
PROJECTION_DIMENSION = 256
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
COMPATIBILITY_MINIMUM = 0.5
NEIGHBORS_MAX = 5
NEIGHBORS_MIN = 2
AGREEMENT_MINIMUM = 0.75
MEAN_COMPATIBILITY_MINIMUM = 0.6
RESIDUAL_SCALE = 0.10
FLAMMABLE_THRESHOLD = 0.953912615776062
QUERY_SCALAR_WIDTH = 11
DONOR_LABEL_SCALAR_INDEX = 49


def load_pair_module():
    spec = importlib.util.spec_from_file_location("exp699_learned_pair_base", PAIR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen pair-verifier module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_deterministic(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class Projection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2048, PROJECTION_DIMENSION, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.linear(values), dim=-1)


class PairHead(nn.Module):
    def __init__(self, scalar_width: int) -> None:
        super().__init__()
        width = 4 * PROJECTION_DIMENSION + scalar_width
        self.layers = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class LearnedPairModel(nn.Module):
    def __init__(self, scalar_width: int, *, frozen_projection: bool = False) -> None:
        super().__init__()
        self.projection = Projection()
        self.head = PairHead(scalar_width)
        if frozen_projection:
            for parameter in self.projection.parameters():
                parameter.requires_grad_(False)

    def pair_input(
        self,
        query: torch.Tensor,
        donor: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        projected_query = self.projection(query)
        projected_donor = self.projection(donor)
        return torch.cat(
            [
                projected_query,
                projected_donor,
                torch.abs(projected_query - projected_donor),
                projected_query * projected_donor,
                scalar,
            ],
            dim=-1,
        )

    def query_input(self, query: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        projected_query = self.projection(query)
        zero = torch.zeros_like(projected_query)
        query_scalar = torch.zeros_like(scalar)
        query_scalar[:, :QUERY_SCALAR_WIDTH] = scalar[:, :QUERY_SCALAR_WIDTH]
        return torch.cat([projected_query, zero, zero, zero, query_scalar], dim=-1)

    def forward_pair(
        self,
        query: torch.Tensor,
        donor: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(self.pair_input(query, donor, scalar))

    def forward_query(self, query: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self.head(self.query_input(query, scalar))


def cloned_models(scalar_width: int, device: torch.device) -> dict[str, LearnedPairModel]:
    set_deterministic(SEED)
    template = LearnedPairModel(scalar_width).to(device)
    initial = template.state_dict()
    models = {
        "learned_pair": LearnedPairModel(scalar_width).to(device),
        "query_only": LearnedPairModel(scalar_width).to(device),
        "fixed_embedding": LearnedPairModel(scalar_width, frozen_projection=True).to(device),
        "permutation": LearnedPairModel(scalar_width).to(device),
    }
    for model in models.values():
        model.load_state_dict(initial)
    return models


def permuted_targets_and_scalar(
    queries: np.ndarray,
    targets: np.ndarray,
    scalar: np.ndarray,
    query_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    permuted_targets = targets.copy()
    permuted_scalar = scalar.copy()
    for query in sorted({int(value) for value in queries}):
        positions = np.flatnonzero(queries == query)
        shuffled = rng.permutation(targets[positions])
        if int(shuffled.sum()) != int(targets[positions].sum()):
            raise ValueError("permutation changed pair balance")
        permuted_targets[positions] = shuffled
        donor_labels = shuffled if int(query_labels[query]) == 1 else 1 - shuffled
        permuted_scalar[positions, DONOR_LABEL_SCALAR_INDEX] = donor_labels
    return permuted_targets, permuted_scalar


def train_branch(
    model: LearnedPairModel,
    *,
    branch: str,
    embeddings: torch.Tensor,
    queries: np.ndarray,
    donors: np.ndarray,
    scalar: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    orders: list[np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scalar_tensor = torch.from_numpy(scalar).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(targets.astype(np.float32)).to(device)
    weight_tensor = torch.from_numpy(weights.astype(np.float32)).to(device)
    losses: list[float] = []
    steps = 0
    model.train()
    for epoch, order in enumerate(orders):
        torch.manual_seed(SEED + epoch)
        torch.cuda.manual_seed_all(SEED + epoch)
        epoch_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            positions = order[start : start + BATCH_SIZE]
            local = torch.from_numpy(positions).to(device=device, dtype=torch.long)
            query_index = torch.from_numpy(queries[positions]).to(device=device)
            donor_index = torch.from_numpy(donors[positions]).to(device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if branch == "query_only":
                    logits = model.forward_query(embeddings[query_index], scalar_tensor[local])
                else:
                    logits = model.forward_pair(
                        embeddings[query_index],
                        embeddings[donor_index],
                        scalar_tensor[local],
                    )
            loss_rows = nn.functional.binary_cross_entropy_with_logits(
                logits.float(), target_tensor[local], reduction="none"
            )
            loss = (loss_rows * weight_tensor[local]).sum() / weight_tensor[local].sum()
            if not torch.isfinite(loss):
                raise ValueError(f"{branch} loss is non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            steps += 1
        losses.append(float(np.mean(epoch_losses)))
    return {
        "steps": steps,
        "epochs": EPOCHS,
        "final_epoch_loss": losses[-1],
        "epoch_losses": losses,
    }


@torch.no_grad()
def predict_branch(
    model: LearnedPairModel,
    *,
    branch: str,
    embeddings: torch.Tensor,
    queries: np.ndarray,
    donors: np.ndarray,
    scalar: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(queries), BATCH_SIZE * 4):
        stop = min(len(queries), start + BATCH_SIZE * 4)
        query_index = torch.from_numpy(queries[start:stop]).to(device=device)
        donor_index = torch.from_numpy(donors[start:stop]).to(device=device)
        local_scalar = torch.from_numpy(scalar[start:stop]).to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if branch == "query_only":
                logits = model.forward_query(embeddings[query_index], local_scalar)
            else:
                logits = model.forward_pair(
                    embeddings[query_index], embeddings[donor_index], local_scalar
                )
        result.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(result)


def residual_from_vote(vote: float, mean_compatibility: float) -> float:
    if vote == 0.5:
        return 0.0
    return (
        RESIDUAL_SCALE * (1.0 if vote > 0.5 else -1.0) * mean_compatibility * abs(2.0 * vote - 1.0)
    )


def aggregate_pairs(
    *,
    heldout_queries: np.ndarray,
    eval_donors: np.ndarray,
    offsets: np.ndarray,
    probabilities: np.ndarray,
    donor_labels: np.ndarray,
    baseline_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, list[dict[str, Any]]]]:
    scores = baseline_scores.copy()
    head_predictions = (baseline_scores >= FLAMMABLE_THRESHOLD).astype(np.int8)
    contributions: dict[int, list[dict[str, Any]]] = {}
    for local_query, query in enumerate(heldout_queries):
        start, stop = int(offsets[local_query]), int(offsets[local_query + 1])
        donors = eval_donors[start:stop]
        local_probability = probabilities[start:stop]
        compatible = np.flatnonzero(local_probability >= COMPATIBILITY_MINIMUM)
        compatible = sorted(
            compatible,
            key=lambda index: (
                -float(local_probability[index]),
                int(donors[index]),
            ),
        )[:NEIGHBORS_MAX]
        if len(compatible) < NEIGHBORS_MIN:
            continue
        selected_probabilities = local_probability[compatible]
        mean_compatibility = float(np.mean(selected_probabilities))
        if mean_compatibility < MEAN_COMPATIBILITY_MINIMUM:
            continue
        selected_donors = donors[compatible]
        selected_labels = donor_labels[selected_donors].astype(np.float64)
        vote = float(np.average(selected_labels, weights=selected_probabilities))
        agreement = max(vote, 1.0 - vote)
        if agreement < AGREEMENT_MINIMUM:
            continue
        residual = residual_from_vote(vote, mean_compatibility)
        scores[query] = float(np.clip(scores[query] + residual, 0.0, 1.0))
        head_predictions[query] = int(vote >= 0.5)
        total_weight = float(selected_probabilities.sum())
        contributions[int(query)] = [
            {
                "donor": int(donor),
                "label": int(label),
                "compatibility": float(probability),
                "normalized_weight": float(probability / total_weight),
                "signed_residual_contribution": float(residual * probability / total_weight),
            }
            for donor, label, probability in zip(
                selected_donors, selected_labels, selected_probabilities
            )
        ]
    predictions = (scores >= FLAMMABLE_THRESHOLD).astype(np.int8)
    return scores, predictions, head_predictions, contributions


def aggregate_query_only(
    *,
    heldout_queries: np.ndarray,
    offsets: np.ndarray,
    probabilities: np.ndarray,
    baseline_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = baseline_scores.copy()
    for local_query, query in enumerate(heldout_queries):
        start, stop = int(offsets[local_query]), int(offsets[local_query + 1])
        probability = float(np.mean(probabilities[start:stop]))
        residual = residual_from_vote(probability, 1.0)
        scores[query] = float(np.clip(scores[query] + residual, 0.0, 1.0))
    return scores, (scores >= FLAMMABLE_THRESHOLD).astype(np.int8)


def apply_prior(
    predictions: np.ndarray, prior_source: np.ndarray, prior_value: np.ndarray
) -> np.ndarray:
    result = predictions.copy()
    mask = prior_source != 0
    result[mask] = prior_value[mask]
    return result


def metric_summary(
    labels: np.ndarray,
    categories: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    pair,
) -> dict[str, Any]:
    return pair.metric_summary(labels, categories, predictions, mask)


def comparison(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    pair,
) -> dict[str, int]:
    return pair.comparison(labels, baseline, candidate, mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--preflight-npz", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to overwrite learned-pair output")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one CUDA GPU is required")
    started = time.monotonic()
    pair = load_pair_module()
    preregister = json.loads(args.preregister.read_text())
    preregister_self = pair.verify_self_hash(preregister)
    if (
        preregister_self != PREREGISTER_SELF_SHA256
        or preregister.get("schema") != "exp699_retrieval_learned_pair_preregister_v1"
    ):
        raise ValueError("learned-pair preregistration mismatch")
    preflight_report = json.loads(args.preflight_report.read_text())
    preflight_self = pair.verify_self_hash(preflight_report)
    if (
        preflight_report.get("schema") != "exp699_learned_pair_preflight_v1"
        or preflight_report.get("decision") != "ACCEPT_LEARNED_PAIR_PREFLIGHT"
        or pair.sha256_file(args.preflight_npz) != preflight_report["packet"]["file_sha256"]
        or preflight_report.get("public_used") is not False
        or preflight_report.get("sealed_rows") != 0
    ):
        raise ValueError("preflight packet mismatch")
    packet = np.load(args.preflight_npz, allow_pickle=False)
    embeddings_np = packet["embeddings"].astype(np.float32)
    if embeddings_np.shape[1] != 2048 or not np.isfinite(embeddings_np).all():
        raise ValueError("preflight embeddings mismatch")
    labels = packet["labels"].astype(np.int8)
    folds = packet["folds"].astype(np.int8)
    categories = packet["categories"].astype(str)
    family_sizes = packet["family_sizes"].astype(np.int32)
    baseline_scores = packet["baseline_scores"].astype(np.float32)
    baseline_before = packet["baseline_before"].astype(np.int8)
    baseline_after = packet["baseline_after"].astype(np.int8)
    prior_source = packet["prior_source"].astype(np.int8)
    prior_value = packet["prior_value"].astype(np.int8)
    harmful_five = {int(value) for value in packet["v1_harmful_five"]}
    screen_mask = np.isin(folds, [0, 3])
    flammable = categories == pair.FLAMMABLE
    baseline_metrics = metric_summary(labels, categories, baseline_after, screen_mask, pair)
    if (
        baseline_metrics["categories"][pair.FLAMMABLE]["fp"] != 4
        or baseline_metrics["categories"][pair.FLAMMABLE]["fn"] != 3
    ):
        raise ValueError("frozen solution140 4FP/3FN screen binding mismatch")

    device = torch.device("cuda")
    embeddings = torch.from_numpy(embeddings_np).to(device=device)
    branch_predictions: dict[str, np.ndarray] = {
        name: baseline_after.copy()
        for name in ("learned_pair", "query_only", "fixed_embedding", "permutation")
    }
    branch_before: dict[str, np.ndarray] = {
        name: baseline_before.copy() for name in branch_predictions
    }
    learned_head = baseline_before.copy()
    training_reports: dict[str, dict[str, Any]] = defaultdict(dict)
    contributions: dict[str, dict[str, Any]] = {}
    steps_by_fold: dict[str, int] = {}
    checkpoint_states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    for outer_fold in (0, 3):
        prefix = f"f{outer_fold}_"
        train_queries = packet[prefix + "train_queries"].astype(np.int64)
        train_donors = packet[prefix + "train_donors"].astype(np.int64)
        train_targets = packet[prefix + "train_targets"].astype(np.int8)
        train_weights = packet[prefix + "train_weights"].astype(np.float32)
        train_scalar = packet[prefix + "train_scalar"].astype(np.float32)
        heldout = packet[prefix + "heldout_queries"].astype(np.int64)
        eval_queries = packet[prefix + "eval_queries"].astype(np.int64)
        eval_donors = packet[prefix + "eval_donors"].astype(np.int64)
        eval_offsets = packet[prefix + "eval_offsets"].astype(np.int64)
        eval_scalar = packet[prefix + "eval_scalar"].astype(np.float32)
        scalar_width = int(train_scalar.shape[1])
        if scalar_width <= DONOR_LABEL_SCALAR_INDEX:
            raise ValueError("scalar feature width mismatch")
        models = cloned_models(scalar_width, device)
        permuted_targets, permuted_scalar = permuted_targets_and_scalar(
            train_queries, train_targets, train_scalar, labels
        )
        rng = np.random.default_rng(SEED)
        orders = [rng.permutation(len(train_queries)).astype(np.int64) for _ in range(EPOCHS)]
        branch_data = {
            "learned_pair": (train_targets, train_scalar),
            "query_only": (labels[train_queries], train_scalar),
            "fixed_embedding": (train_targets, train_scalar),
            "permutation": (permuted_targets, permuted_scalar),
        }
        for name, model in models.items():
            targets, scalar = branch_data[name]
            training_reports[name][str(outer_fold)] = train_branch(
                model,
                branch=name,
                embeddings=embeddings,
                queries=train_queries,
                donors=train_donors,
                scalar=scalar,
                targets=targets,
                weights=train_weights,
                orders=orders,
                device=device,
            )
        step_values = {training_reports[name][str(outer_fold)]["steps"] for name in models}
        if len(step_values) != 1:
            raise ValueError("control optimizer-step parity failed")
        steps_by_fold[str(outer_fold)] = step_values.pop()

        for name, model in models.items():
            probabilities = predict_branch(
                model,
                branch=name,
                embeddings=embeddings,
                queries=eval_queries,
                donors=eval_donors,
                scalar=eval_scalar,
                device=device,
            )
            if name == "query_only":
                _, before = aggregate_query_only(
                    heldout_queries=heldout,
                    offsets=eval_offsets,
                    probabilities=probabilities,
                    baseline_scores=baseline_scores,
                )
                local_contributions: dict[int, list[dict[str, Any]]] = {}
            else:
                _, before, local_head, local_contributions = aggregate_pairs(
                    heldout_queries=heldout,
                    eval_donors=eval_donors,
                    offsets=eval_offsets,
                    probabilities=probabilities,
                    donor_labels=labels,
                    baseline_scores=baseline_scores,
                )
                if name == "learned_pair":
                    learned_head[heldout] = local_head[heldout]
            after = apply_prior(before, prior_source, prior_value)
            branch_before[name][heldout] = before[heldout]
            branch_predictions[name][heldout] = after[heldout]
            if name == "learned_pair":
                contributions[str(outer_fold)] = {
                    str(query): rows for query, rows in local_contributions.items()
                }
        checkpoint_states[str(outer_fold)] = {
            name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for name, model in models.items()
        }

    if not np.array_equal(
        branch_predictions["learned_pair"][~flammable], baseline_after[~flammable]
    ):
        raise ValueError("BAD route changed")
    metrics: dict[str, Any] = {}
    for name, predictions in branch_predictions.items():
        metrics[name] = {
            "summary": metric_summary(labels, categories, predictions, screen_mask, pair),
            "comparison": comparison(labels, baseline_after, predictions, screen_mask, pair),
            "folds": {
                str(fold): {
                    "summary": metric_summary(labels, categories, predictions, folds == fold, pair),
                    "comparison": comparison(
                        labels, baseline_after, predictions, folds == fold, pair
                    ),
                }
                for fold in (0, 3)
            },
        }
    learned = branch_predictions["learned_pair"]
    learned_comparison = metrics["learned_pair"]["comparison"]
    learned_flammable = metrics["learned_pair"]["summary"]["categories"][pair.FLAMMABLE]
    baseline_flammable = baseline_metrics["categories"][pair.FLAMMABLE]
    original_fp = screen_mask & flammable & (labels == 0) & (baseline_after == 1)
    original_fn = screen_mask & flammable & (labels == 1) & (baseline_after == 0)
    fp_corrected = int((original_fp & (learned == 0)).sum())
    fn_corrected = int((original_fn & (learned == 1)).sum())
    learned_macro = metrics["learned_pair"]["summary"]["macro_f1"]
    query_macro = metrics["query_only"]["summary"]["macro_f1"]
    fixed_macro = metrics["fixed_embedding"]["summary"]["macro_f1"]
    permutation_macro = metrics["permutation"]["summary"]["macro_f1"]
    baseline_macro = baseline_metrics["macro_f1"]
    gates = {
        "at_least_one_fp_corrected": fp_corrected >= 1,
        "at_least_one_fn_corrected": fn_corrected >= 1,
        "pooled_net_positive": learned_comparison["net"] > 0,
        "regressions_at_most_one": learned_comparison["regressions"] <= 1,
        "flammable_fn_nonincrease": learned_flammable["fn"] <= baseline_flammable["fn"],
        "beats_query_only": learned_macro > query_macro,
        "beats_fixed_embedding": learned_macro > fixed_macro,
        "permutation_no_positive_gain": permutation_macro <= baseline_macro
        and metrics["permutation"]["comparison"]["net"] <= 0,
        "bad_exact": np.array_equal(learned[~flammable], baseline_after[~flammable]),
    }
    changed_before = screen_mask & (branch_before["learned_pair"] != baseline_before)
    changed_after = screen_mask & (learned != baseline_after)
    family_masks = {
        "singleton": screen_mask & (family_sizes == 1),
        "repeated": screen_mask & (family_sizes > 1),
        "v1_harmful_five": np.asarray(
            [index in harmful_five for index in range(len(labels))], dtype=bool
        ),
    }
    cohorts = {
        name: comparison(labels, baseline_after, learned, mask, pair)
        for name, mask in family_masks.items()
    }
    survival = {
        "head_queries_with_contributions": int(sum(len(rows) for rows in contributions.values())),
        "head_decision_changes": int((screen_mask & (learned_head != baseline_before)).sum()),
        "residual_changes_before_prior": int(changed_before.sum()),
        "survived_after_prior": int((changed_before & changed_after).sum()),
        "suppressed_by_prior": int((changed_before & ~changed_after).sum()),
    }

    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "learned_pair_final_epoch.pt"
    torch.save(
        {
            "schema": "exp699_learned_pair_checkpoint_v1",
            "preregister_self_sha256": preregister_self,
            "models": checkpoint_states,
            "training": training_reports,
            "note": "final-epoch screen checkpoint; package authorization remains false",
        },
        checkpoint_path,
    )
    runtime_seconds = time.monotonic() - started
    if runtime_seconds > 30 * 60:
        gates["wall_clock_within_30_minutes"] = False
    else:
        gates["wall_clock_within_30_minutes"] = True
    decision = "ACCEPT_LEARNED_PAIR_SCREEN" if all(gates.values()) else "REJECT_LEARNED_PAIR_SCREEN"
    report = {
        "schema": "exp699_learned_pair_screen_v1",
        "experiment": 699,
        "decision": decision,
        "public_used": False,
        "sealed_rows": 0,
        "package_authorized": False,
        "ods_submit_authorized": False,
        "runtime_seconds": runtime_seconds,
        "gpu_count": 1,
        "recipe": preregister,
        "training": training_reports,
        "steps_by_fold": steps_by_fold,
        "baseline": baseline_metrics,
        "metrics": metrics,
        "fp_corrected": fp_corrected,
        "fn_corrected": fn_corrected,
        "gates": gates,
        "cohorts": cohorts,
        "survival": survival,
        "donor_contributions": contributions,
        "source_bindings": {
            "preregister_file_sha256": pair.sha256_file(args.preregister),
            "preregister_self_sha256": preregister_self,
            "preflight_file_sha256": pair.sha256_file(args.preflight_npz),
            "preflight_report_file_sha256": pair.sha256_file(args.preflight_report),
            "preflight_report_self_sha256": preflight_self,
            "trainer_code_sha256": pair.sha256_file(Path(__file__).resolve()),
            "checkpoint_file_sha256": pair.sha256_file(checkpoint_path),
        },
    }
    file_sha, self_sha = pair.write_self_hashed(args.output_report, report)
    print(
        json.dumps(
            {
                "decision": decision,
                "file_sha256": file_sha,
                "self_sha256": self_sha,
                "runtime_seconds": runtime_seconds,
                "gates": gates,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
