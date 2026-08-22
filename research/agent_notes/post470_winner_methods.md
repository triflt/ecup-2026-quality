# Post-470 research: selective checkpoint / adapter transfer after 430, 450, 460

Date: 2026-08-22. This is a read-only research note. No training, inference, deployment, submission, or shared experiment-log mutation was performed.

## Outcome

The best next hypothesis is **Fisher-anchored, blockwise LoRA interpolation between 190 and 260**, using 430 (`alpha=0.5`) as the mandatory baseline. It preserves the existing inference graph, priors, prompt, and output contract; needs no external teacher; and produces one merged checkpoint for one-pass inference.

The new part is not another global `alpha`. For every compatible LoRA target block, estimate how important that block is to:

1. 190's broad positive-recall behavior on a representative anchor set; and
2. 260's flammable behavior on the rare-positive / hard-flammable set.

Use those importance estimates to set a different interpolation coefficient per block. Blocks important to broad recall stay near 190; blocks disproportionately important to flammable move toward 260.

This is the most direct continuation of the local evidence:

- 430 says the 190–260 weight-space path is useful and apparently has no immediate loss barrier at `alpha=0.5`.
- 450 says a keyword / transaction-scope side route does not transfer cleanly.
- 460 says a separate embedding readout does not transfer cleanly.
- Therefore the next experiment should refine the only locally positive mechanism—weight transfer—without adding a rule system, representation head, teacher, or another training distribution.

## Four primary sources and what actually transfers

### 1. Official winner: SIGIR eCom 2020 multimodal product classification

Primary source: [A Multimodal Late Fusion Model for E-Commerce Product Classification](https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_4.pdf).

The `pa_curis` team won first place with macro-F1 0.9144. Their decision-level fusion outperformed feature-level fusion, and the final system exploited diversity from different configurations and checkpoints saved late in training. The relevant lesson is narrow: **different trained states contain complementary product-classification signal**. The paper's 12-model vote is not deployable under our one-pass constraint, and its separate fusion policy resembles the kind of extra readout that 460 already argues against. We should transfer only the checkpoint-complementarity observation, then collapse it in weight space.

### 2. AdapterSoup: adapter averaging without extra training

Primary source: [AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models](https://aclanthology.org/2023.findings-eacl.153/), Findings of EACL 2023.

AdapterSoup averages compatible adapters in weight space. The paper reports improved transfer to new domains without extra training and finds that averaging same-domain adapters from different hyperparameters can retain new-domain performance while preserving strong in-domain performance. This is directly relevant because 190 and 260 are parameter-efficient adaptations, not unrelated full models. It supports using adapter composition rather than a new embedding head or online ensemble.

Important limitation: AdapterSoup does not establish that a uniform average is optimal for rare-positive recall. It supports adapter-space composition, not our proposed blockwise Fisher coefficients.

### 3. Fisher-weighted model merging

Primary source: [Merging Models with Fisher-Weighted Averaging](https://proceedings.neurips.cc/paper_files/paper/2022/hash/70c26937fbf3d4600b69a129031b66ec-Abstract-Conference.html), NeurIPS 2022.

Matena and Raffel replace isotropic parameter averaging with a diagonal-Fisher-weighted average. For parameter coordinate `j`, their closed-form merge is:

```text
theta*(j) = sum_i lambda_i F_i(j) theta_i(j) / sum_i lambda_i F_i(j)
```

They report that Fisher merging improves the trade-off over simple averaging in robust fine-tuning and can approach output-ensemble quality with one merged model. They also explicitly require common architecture and initialization and note that Fisher is local, so distant checkpoints may not merge well.

This is the closest primary mechanism to the proposed next experiment. Our use of separate broad-recall and flammable calibration slices is an adaptation of their method, not a result claimed by the paper.

### 4. LT-Soups: weight merging for head/tail trade-offs

Primary source: [LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html), NeurIPS 2025.

LT-Soups is directly about long-tailed classification. It merges specialists trained under different imbalance ratios into one inference-efficient network. Across the paper's long-tail benchmarks it improves the head/tail trade-off over ordinary soups; its controlled TinyImageNet-LT result reports tail accuracy 75.2 for LT-Soups versus 73.0 for the ordinary model soup, while keeping stronger overall performance. It also reports that merely retraining the final classifier has little or no effect for its PEFT and ordinary-soup baselines.

The transfer to our setting is again selective. The evidence supports combining general and tail-specialized weight states into one network. It does **not** justify repeating 420's balanced continuation or 460's head training. Their full multi-checkpoint training recipe is too expensive and locally contradicted; we should borrow only the head/tail-aware merging principle.

## Proposed mechanism

### Preconditions

Proceed only if 190 and 260 share the same base checkpoint, tokenizer, architecture, LoRA target modules, ranks, scaling convention, and output head. If any coordinate system differs, the Fisher formula is not valid without an explicit alignment step.

Keep the following exactly as in 190/430:

- prompt and deterministic decoding;
- family priors, thresholds, and post-processing;
- inference branches and output schema;
- input image/text preprocessing.

Do not add keyword features, a transaction-scope rule, an embedding classifier, or a teacher-produced label.

### Blockwise empirical Fisher

Use block-level rather than raw per-parameter Fisher weights for the first screen. Per-parameter estimates will be noisy on the small rare-positive slice, while one scalar per LoRA target block is cheap and auditable.

For each target block `l`:

```text
G_190,l = mean squared gradient norm of log p_190(y|x)
          on the broad anchor/calibration set

G_260,l = mean squared gradient norm of log p_260(y|x)
          on the flammable-positive/hard-flammable calibration set

alpha_l(lambda) = lambda * G_260,l /
                  (G_190,l + lambda * G_260,l + eps)

Delta_l = (1 - alpha_l) * Delta_190,l + alpha_l * Delta_260,l
```

Use capped per-family contributions so one duplicated SKU/family cannot dominate a block's importance. Normalize gradient energies within layer type before calculating `alpha_l`; otherwise blocks with different parameter counts or scales are not comparable. Where both energies are near zero, default to 190, following the Fisher-merging paper's privileged-target fallback.

For LoRA, interpolate the **effective updates** `Delta = scaling * B @ A`, not `A` and `B` independently. Independent factor averaging adds cross terms because the factorization is bilinear. The merged effective delta can be baked into the base or represented as a rank-sum adapter; if a fixed-rank artifact is mandatory, SVD compression is a separate ablation and must not be silently mixed into the Fisher test.

### Why this can beat fixed `alpha=0.5`

Global 0.5 assumes every block should split the difference equally. That is unlikely when 260's useful change is concentrated in a few OCR/vision-language or late decision blocks while its harmful changes affect broader semantic blocks. Fisher weighting estimates where each checkpoint's behavior is locally sensitive:

- high `G_190,l`, low `G_260,l` -> retain 190 in block `l`;
- low `G_190,l`, high `G_260,l` -> import more of 260;
- both high -> controlled compromise through `lambda`;
- both low -> default to 190.

This is capability transfer without another optimization trajectory, so it directly avoids the forgetting mode seen after balanced continuation.

## Cheap, honest validation screen

### Data separation

Use three group-disjoint partitions made before looking at results:

1. **Fisher calibration**: train-side records only. Broad anchor contains all positive families plus representative negatives; rare slice contains all available flammable positives and hard near-boundary flammable negatives, with family caps.
2. **Coefficient tuning screen**: a compact group-stratified development set, enriched for flammable positives but retaining the natural-distribution portion needed to measure overall recall and false-positive rate.
3. **Untouched confirmation**: the existing locked group-clean holdout. It is opened once for the chosen candidate.

No validation/test label is used to estimate Fisher.

### Candidate set

Compare:

- 190;
- 260;
- 430 fixed `alpha=0.5`;
- Fisher-block soups with `lambda in {0.25, 0.5, 1, 2}`.

Each candidate gets one deterministic inference pass per item: no TTA, no prediction ensemble, no per-item routing. Use the same frozen priors and thresholds for every candidate. Fisher computation requires backward passes on calibration records but no optimizer step and no new model training.

Select the smallest deviation from 190 that satisfies all predeclared tuning gates:

- flammable recall strictly above 430;
- overall positive recall non-inferior to 430 within a fixed margin;
- no other critical family loses more than the fixed family-level margin;
- false-positive rate remains under the 430 cap;
- fewer broad-positive negative flips than 260.

Then run exactly one confirmation pass with the selected checkpoint. Report paired, group-bootstrap intervals for overall positive recall, flammable recall, false-positive rate, and net positive-to-negative flips versus both 190 and 430. Accept only if the flammable gain survives and the lower confidence bound for overall-recall difference clears the non-inferiority margin.

### Early rejection rules

Keep 430 and stop if any of these occurs:

- blockwise coefficients collapse to approximately 0.5 everywhere, meaning Fisher added no selectivity;
- coefficients are unstable across two calibration resamples;
- the tuning gain disappears when duplicate/family groups are kept intact;
- improved flammable recall is explained only by a broad false-positive increase;
- the final rank/SVD representation erases the pre-compression gain;
- the untouched holdout does not beat 430 under the predeclared gate.

## Bottom line

430 validates a useful shared basin; 450 and 460 argue against adding semantics through rules or a new readout. The next low-cost experiment should therefore ask a tighter question: **can curvature/importance information choose where the already-successful 190→260 transfer should happen?** A Fisher-anchored blockwise LoRA soup is a single-checkpoint, no-teacher, no-training way to test that question honestly.

No exact public winner for this specific regulated-product taxonomy was found in the second search round. The sources above are the closest primary evidence: one official multimodal product-classification winner, one adapter-merging study, one importance-weighted merging paper, and one rare-class model-soup paper. The proposed combination is an evidence-based transfer hypothesis, not a reported competition recipe.
