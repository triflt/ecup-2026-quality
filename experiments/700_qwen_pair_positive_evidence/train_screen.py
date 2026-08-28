from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

SEED = 42
BATCH = 512
EPOCHS = 10
LR = 3e-4
DONOR_LABEL = 49
MIN_PAIR_PROBABILITY = 0.70
MIN_QUERY_MARGIN = 0.10
MIN_POSITIVE_DONORS = 2


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deterministic() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def train(model, embeddings, queries, donors, scalar, target, order, *, query_only=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scalar_t = torch.from_numpy(scalar).cuda().float()
    target_t = torch.from_numpy(target.astype(np.float32)).cuda()
    losses = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = []
        for start in range(0, len(order[epoch]), BATCH):
            pos = order[epoch][start : start + BATCH]
            ix = torch.from_numpy(pos).cuda().long()
            q = torch.from_numpy(queries[pos]).cuda().long()
            d = torch.from_numpy(donors[pos]).cuda().long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = (
                    model.forward_query(embeddings[q], scalar_t[ix])
                    if query_only
                    else model.forward_pair(embeddings[q], embeddings[d], scalar_t[ix])
                )
            loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), target_t[ix])
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_loss)))
    return losses


@torch.no_grad()
def predict(model, embeddings, queries, donors, scalar, *, query_only=False):
    model.eval()
    output = []
    for start in range(0, len(queries), BATCH * 4):
        stop = min(start + BATCH * 4, len(queries))
        q = torch.from_numpy(queries[start:stop]).cuda().long()
        d = torch.from_numpy(donors[start:stop]).cuda().long()
        s = torch.from_numpy(scalar[start:stop]).cuda().float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.forward_query(embeddings[q], s) if query_only else model.forward_pair(embeddings[q], embeddings[d], s)
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def promote(heldout, donors, offsets, scalar, pair_prob, query_prob, baseline, prior_source, prior_value):
    before = baseline.copy()
    evidence = {}
    for local, query in enumerate(heldout):
        if baseline[query] != 0:
            continue
        start, stop = int(offsets[local]), int(offsets[local + 1])
        positive = np.flatnonzero(scalar[start:stop, DONOR_LABEL] == 1)
        if len(positive) < MIN_POSITIVE_DONORS:
            continue
        local_pair = pair_prob[start:stop][positive]
        local_query = query_prob[start:stop][positive]
        rank = sorted(range(len(positive)), key=lambda i: (-float(local_pair[i]), int(donors[start:stop][positive[i]])))[:3]
        pair_mean = float(np.mean(local_pair[rank]))
        query_mean = float(np.mean(local_query[rank]))
        if pair_mean >= MIN_PAIR_PROBABILITY and pair_mean - query_mean >= MIN_QUERY_MARGIN:
            before[query] = 1
            evidence[int(query)] = {"pair_mean": pair_mean, "query_mean": query_mean, "margin": pair_mean-query_mean, "donors": [int(donors[start:stop][positive[i]]) for i in rank]}
    after = before.copy()
    mask = prior_source != 0
    after[mask] = prior_value[mask]
    return before, after, evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--parent-trainer", type=Path, required=True)
    parser.add_argument("--pair-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    started = time.monotonic()
    parent = load_module(args.parent_trainer, "exp700_parent")
    pair = load_module(args.pair_module, "exp700_pair")
    packet = np.load(args.packet, allow_pickle=False)
    labels = packet["labels"].astype(np.int8)
    folds = packet["folds"].astype(np.int8)
    categories = packet["categories"].astype(str)
    baseline = packet["baseline_after"].astype(np.int8)
    prior_source = packet["prior_source"].astype(np.int8)
    prior_value = packet["prior_value"].astype(np.int8)
    embeddings = torch.from_numpy(packet["embeddings"].astype(np.float32)).cuda()
    flammable = categories == pair.FLAMMABLE
    screen = np.isin(folds, [0, 3])
    candidate = baseline.copy()
    all_evidence = {}
    training = {}
    for fold in (0, 3):
        prefix = f"f{fold}_"
        tq = packet[prefix+"train_queries"].astype(np.int64)
        td = packet[prefix+"train_donors"].astype(np.int64)
        ts = packet[prefix+"train_scalar"].astype(np.float32)
        target = labels[tq]
        heldout = packet[prefix+"heldout_queries"].astype(np.int64)
        eq = packet[prefix+"eval_queries"].astype(np.int64)
        ed = packet[prefix+"eval_donors"].astype(np.int64)
        eo = packet[prefix+"eval_offsets"].astype(np.int64)
        es = packet[prefix+"eval_scalar"].astype(np.float32)
        deterministic()
        models = parent.cloned_models(ts.shape[1], torch.device("cuda"))
        learned, query_only, permuted = models["learned_pair"], models["query_only"], models["permutation"]
        rng = np.random.default_rng(SEED)
        orders = [rng.permutation(len(tq)).astype(np.int64) for _ in range(EPOCHS)]
        perm = es.copy()
        for local in range(len(heldout)):
            a,b=int(eo[local]),int(eo[local+1])
            perm[a:b,DONOR_LABEL] = np.random.default_rng(SEED+int(heldout[local])).permutation(perm[a:b,DONOR_LABEL])
        training[str(fold)] = {
            "learned": train(learned, embeddings, tq, td, ts, target, orders),
            "query_only": train(query_only, embeddings, tq, td, ts, target, orders, query_only=True),
            "permutation": train(permuted, embeddings, tq, td, ts, target, orders),
        }
        pp = predict(learned, embeddings, eq, ed, es)
        qp = predict(query_only, embeddings, eq, ed, es, query_only=True)
        _, local_after, evidence = promote(heldout, ed, eo, es, pp, qp, baseline, prior_source, prior_value)
        candidate[heldout] = local_after[heldout]
        all_evidence[str(fold)] = evidence
    base_metrics = pair.metric_summary(labels, categories, baseline, screen)
    metrics = pair.metric_summary(labels, categories, candidate, screen)
    compare = pair.comparison(labels, baseline, candidate, screen)
    original_fn = screen & flammable & (labels == 1) & (baseline == 0)
    fn_corrected = int((original_fn & (candidate == 1)).sum())
    new_fp = int((screen & flammable & (labels == 0) & (baseline == 0) & (candidate == 1)).sum())
    bad_exact = bool(np.array_equal(candidate[~flammable], baseline[~flammable]))
    only_up = bool(np.all(candidate[screen & flammable] >= baseline[screen & flammable]))
    gates = {"fn_corrected": fn_corrected >= 1, "new_fp_at_most_one": new_fp <= 1, "net_positive": compare["net"] > 0, "bad_exact": bad_exact, "only_zero_to_one": only_up}
    report = {"schema":"exp700_positive_evidence_screen_v1","decision":"ACCEPT_SCREEN" if all(gates.values()) else "REJECT_SCREEN","public_used":False,"folds":[0,3],"runtime_seconds":time.monotonic()-started,"thresholds":{"pair":MIN_PAIR_PROBABILITY,"margin":MIN_QUERY_MARGIN,"min_donors":MIN_POSITIVE_DONORS},"baseline":base_metrics,"candidate":metrics,"comparison":compare,"fn_corrected":fn_corrected,"new_fp":new_fp,"gates":gates,"training":training,"evidence":all_evidence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pair.write_self_hashed(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

