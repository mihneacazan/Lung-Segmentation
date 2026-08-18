# Experiments

Every run goes through `src/training/train.py` on the same 44 / 9 / 10 patient
split, with the same evaluation code and the same checkpoint selection rule
(soft Dice on validation — [details](README.md#5-training-srctrainingtrainpy)).
Each experiment changes **one variable** against the baseline, so a difference in
score is attributable to that variable and nothing else.

The baseline is a 2D U-Net trained with DiceCE loss on every slice, with
anatomically valid augmentation. It reaches Dice 0.4853 ± 0.0153 over three seeds
and every other configuration is measured against it.

Held constant everywhere: `--batch_size 16`, `--lr 1e-3`, `--min_epochs 15`, AdamW
(`weight_decay=1e-4`), cosine annealing, gradient clipping at norm 1.0.

---

## Final results

3D Dice on the test set, in each patient's original NIfTI geometry. **Every run in
this table uses the same code version**. Intermediate results, obtained on earlier versions,
are in the [journal](#journal) at the end of this document.

The `t` column is a paired test over the 10 test patients against the baseline;
the significance threshold at df=9, p=0.05 is ±2.262.

| Configuration | Dice 3D | Δ vs baseline | t | wins |
|---|---:|---:|---:|---:|
| **baseline** — `all` + `anatomic` + `dice_ce`, 3 seeds | **0.4853 ± 0.0153** | — | — | — |
| 2.5D, `n_adjacent=3` | 0.4259 | −0.059 | −1.52 | 3/10 |
| SegResNet, `lr=3e-4` | 0.4181 | −0.067 | −1.29 | 3/10 |
| DiceFocal | 0.3536 | −0.132 | −3.13 | 1/10 |
| Attention U-Net | 0.3155 | −0.170 | −2.92 | 0/10 |
| `balanced` sampling | 0.3023 | −0.183 | **−4.49** | 0/10 |
| no augmentation | 0.2648 | −0.221 | −2.82 | 4/10 |
| Focal Tversky (β=0.7 / 0.6) | 0.1043 / 0.0864 | −0.39 | — | 0/10 |
| Tversky (β=0.7 / 0.6) | 0.0771 / 0.0885 | −0.40 | — | 0/10 |

Baseline per seed: 0.5069 (42), 0.4751 (43), 0.4737 (44).

### What the data says

**The data pipeline matters more than the architecture.** The two large effects are
augmentation (+0.221) and sampling (+0.183). Architectures trail far behind, and
2.5D and SegResNet are statistically indistinguishable from the baseline (|t|
below threshold).

**`all` beats `balanced` by +0.18.** This contradicts the original reasoning, which justified `balanced` as a way to avoid collapsing to "always predict
negative". Empirically, matching the training distribution to the evaluation
distribution matters more than the class imbalance does. It is also the strongest
single difference in the table — t = −4.49, losing on 10 patients out of 10.

**Extra capacity does not help at 44 training patients.** Attention U-Net carries
22% more parameters than the baseline plus attention gates that must themselves be
learned, and lands 0.170 lower. SegResNet behaves the same way. The additional
parameters do not extract more information from a set this small; they add more
ways to fit noise.

**The pure Tversky family does not fit this problem.** Not because it fails to find
the tumour — sensitivity is 0.39–0.43, comparable to the baseline's 0.434 — but
because it paints seven to eight times more volume than exists. The cause is
structural and is worked out below.

**HD95 is destroyed by distant false positives, not by boundary quality.**
lung_001 reached sensitivity 0.969 — the lesion is found almost completely — yet
carried 7 false-positive components and HD95 of 189 mm. Hence the connected
component filter, now reported alongside the raw metrics: −27.98 mm HD95 and
+0.046 precision, at a cost of −0.0033 sensitivity and zero change in failures.

### Why Tversky cannot be fixed by tuning β

The initial hypothesis was that `alpha=0.3, beta=0.7` overcompensates — a miss
costs 2.33× a false alarm, too aggressive for a target under 1% of the volume.
The test refuted it. At β=0.6 `tversky` rose marginally (0.0771 → 0.0885) while
`focal_tversky` actually **fell** (0.1043 → 0.0864), and the predicted-to-real
volume ratio stayed at 7–8×.

The real cause, measured on a slice with an empty ground truth — that is, 90% of
the training data under `--sampling all`:

| loss | value | max \|grad\| | suppresses background? |
|---|---:|---:|---|
| `dice_ce` | 1.3133 | 6.566e-05 | yes |
| `dice_focal` | 1.0227 | 1.284e-05 | yes |
| `tversky` | 1.0000 | **0.000e+00** | **no — zero gradient** |
| `focal_tversky` | 1.0000 | **0.000e+00** | **no — zero gradient** |

With `smooth_nr=0` the Dice/Tversky term has exactly zero gradient on empty
slices, which was the whole point of the fix. `dice_ce` and `dice_focal` survive
because they carry a **second, pixel-level term** that still penalises false
positives. Pure Tversky carries nothing.

So for the Tversky family, 90% of the training data produces no learning signal at
all, and nothing stops the network from painting everywhere.

---

## The controlled experiments

### 1 — DiceCE + balanced sampling

```bash
python -m src.training.train --exp_name dicece_balanced \
    --loss_type dice_ce --sampling balanced --epochs 100 --lr_t_max 50 --patience 20
```

`dice_ce` combines Dice loss with binary cross-entropy (MONAI's `DiceCELoss`).
`balanced` takes every positive slice in train plus an equal number of negatives,
redrawn each epoch ([dataset.py](src/training/dataset.py)), on the theory that it
prevents the network from learning "always predict negative" from the real ~9%
positive rate. Measured result: 0.3023, well below `all`.

### 2 — DiceCE + every slice (the baseline)

```bash
python -m src.training.train --exp_name baseline \
    --loss_type dice_ce --sampling all --epochs 100 --lr_t_max 50 --patience 20 \
    --seeds 42,43,44
```

Identical to experiment 1 except for `--sampling all`: every epoch sees all slices
of the training patients at the real ~9% positive rate. **Isolates the effect of
class imbalance** — loss, augmentation and architecture are unchanged.

### 3 — DiceFocal

```bash
python -m src.training.train --exp_name dice_focal --loss_type dice_focal \
    --sampling all --epochs 100 --lr_t_max 50 --patience 20
```

Same recipe, different loss: `DiceFocalLoss` replaces the cross-entropy component
with focal loss, which down-weights pixels already classified confidently
(`gamma=2.0`) so training concentrates on the uncertain ones.

### 4 — Tversky and Focal Tversky

```bash
python -m src.training.train --exp_name tversky --loss_type tversky \
    --sampling all --tversky_alpha 0.3 --tversky_beta 0.7 \
    --epochs 100 --lr_t_max 50 --patience 20
```

The Tversky index generalises Dice by weighting false positives (`alpha`) and false
negatives (`beta`) separately: `TI = TP / (TP + α·FP + β·FN)`. With `beta > alpha`
a missed tumour costs more than a hallucinated one, which pushes the network toward
higher sensitivity — relevant here because tumour voxels are under 1% of the volume
and a network minimising plain Dice can score well by under-segmenting. At
`alpha = beta = 0.5` the index reduces to Dice exactly, which bounds the useful
range from below.

```bash
python -m src.training.train --exp_name focal_tversky --loss_type focal_tversky \
    --sampling all --tversky_alpha 0.3 --tversky_beta 0.7 \
    --epochs 100 --lr_t_max 50 --patience 20
```

`focal_tversky` adds an exponent, `FTL = (1 − TI)^0.75`. With gamma below 1 the
gradient grows as the loss shrinks, so training does not stall once the easy slices
are solved. MONAI's `TverskyLoss` does not accept a gamma argument, so
[losses.py](src/training/losses.py) composes it explicitly.

Both were also re-run at `alpha=0.4, beta=0.6` — `alpha + beta` held at 1, as in
the setting above, just less aggressive about false negatives
(`--exp_name tversky_b06` / `--exp_name focal_tversky_b06`). See
[above](#why-tversky-cannot-be-fixed-by-tuning-) for why that changed nothing.

### 5 — No augmentation

```bash
python -m src.training.train --exp_name no_augment --augment none \
    --loss_type dice_ce --sampling all --epochs 100 --lr_t_max 50 --patience 20
```

No geometric or intensity transform on the training images. Paired with the
baseline, this isolates what augmentation contributes: +0.221 Dice, the largest
single effect measured.

### 6 — Anatomically valid augmentation (the baseline's setting)

`anatomic` restricts augmentation to variation that genuinely occurs at
acquisition: ±15° rotation, ±8% translation, ±10% scaling, gamma 0.8–1.25,
Gaussian noise. Nothing that would produce impossible anatomy — which rules out the
horizontal flips and 90° rotations of the `standard` policy, since a mirrored chest
CT is not a chest CT.

### 7 — Hard negative sampling

`hard_negatives` replaces the randomly drawn negatives of `balanced` with negatives
concentrated along Z around the tumour (70% near, 30% far) — lung or other tissue
rather than air, so the network learns a finer boundary instead of merely
separating "tumour vs obvious air". Never completed: the session was interrupted
during training and the run directory holds only checkpoints.

### 8 — Attention U-Net

```bash
python -m src.training.train --exp_name attention_unet \
    --model_type attention_unet --sampling all --epochs 100 --lr_t_max 50 --patience 20
```

Adds attention gates on every skip connection
([models/README.md](src/models/README.md)), which learn to suppress healthy tissue
and air before concatenation with the decoder. Relevant in principle because the
target occupies under 1% of the volume. 1,987,417 parameters against the
baseline's 1,624,844.

### 9 — SegResNet

```bash
python -m src.training.train --exp_name segresnet \
    --model_type segresnet --sampling all --lr 3e-4 \
    --epochs 100 --lr_t_max 50 --patience 20
```

A residual encoder-decoder with group normalisation, deeper than the base U-Net
(`blocks_down=(1,2,2,4)`).

**The only architecture run at a different learning rate, deliberately.** At
`lr=1e-3` SegResNet diverged to NaN twice, at epoch 10 and then at epoch 12, and
gradient clipping did not fix it. At `3e-4` it completed 37 epochs.

Since a lower learning rate could plausibly help or hurt any architecture, the
baseline U-Net was also re-run at `3e-4` as a control, so the SegResNet-vs-U-Net
comparison is not confounded by the learning rate difference:

```bash
python -m src.training.train --exp_name unet_lowlr --loss_type dice_ce \
    --sampling all --lr 3e-4 --epochs 100 --lr_t_max 50 --patience 20
```

The control scored 0.4642, so the like-for-like comparison is SegResNet 0.4181
against U-Net 0.4642 — the gap most attributable to the architecture, not the
learning rate.

### 10 — 2.5D, three consecutive slices

```bash
python -m src.training.train --exp_name unet_25d --n_adjacent 3 \
    --sampling all --epochs 100 --lr_t_max 50 --patience 20
```

The base U-Net with a 3-channel input (the current slice plus one neighbour on
each side along Z) instead of 1; the label still belongs to the centre slice. The
network sees neighbourhood context without the cost of a full 3D model. At volume
edges the missing neighbours are **replicated, not zero-filled**
([test_dataset.py](tests/test_dataset.py)) — zero is a real intensity in this
normalised space and means air, so zero-padding would tell the network there is air
above the apex of the lung. Z spacing is uniform by construction, since resampling
to 1 mm isotropic precedes the stacking.

### 11 — Attention U-Net under the earlier epoch budget

```bash
python -m src.training.train --exp_name attention_unet_50ep_pat10 \
    --model_type attention_unet --sampling all --lr_t_max 50 --patience 10 --epochs 50
```

Not a new variable — a diagnostic. Attention U-Net had scored 0.4615 in an early
benchmark and 0.3155 in the final table, and the two runs differed in four ways at
once: the loss, its `smooth_nr`, the epoch budget and the patience window. This
re-run holds everything at the final-table setting except the epoch budget and
patience, which are set back to what the early benchmark used. It came back
**identical to six decimals** to the final-table run — same best epoch, same
validation Dice, same test Dice — which settled the question: the budget was never
responsible, only the loss. See [Round 5](#round-5--one-code-version-and-two-questions-closed)
for the full account.

### Output layout

Each run writes to `output/experiments/{exp_name}/seed_{seed}/`: `config.json`,
`best_model.pt`, `checkpoint.pt`, `training_history.csv`, `threshold_sweep.json`,
`test_results_per_patient.csv`, `benchmark_report.json`. `config.json`'s `exp_name`
field is rewritten to match its directory when results are archived, so the two
never disagree. With multiple seeds, `multi_seed_summary.json` is added one level
up with mean ± standard deviation across seeds for Dice 3D, HD95, sensitivity and
precision — see `output/experiments/baseline/` for the only run in this project
with more than one seed.

---

## Journal

What was tried, in order, and what each round changed. Numbers from earlier rounds
are **not** comparable with the final table — the code changed between them, which
is exactly why everything was eventually re-run on one version.

### Round 1 — the first full benchmark (12 configurations)

Best result: `attention_unet` + `all` at Dice 0.4615, single seed. `unet` + `all`
gave 0.4380 ± 0.0047 over three seeds. `balanced` configurations landed between
0.0835 and 0.3440.

Five findings came out of it, three of them problems:

**Sampling dominated everything.** `all` beat `balanced` by +0.17 on U-Net and
+0.16 on Attention U-Net. No other factor came close. It also proved six times more
stable across seeds (std 0.0047 against 0.0297).

**A larger epoch budget made things worse** — 0.4177 against 0.4446 on the same
seed. The cause was that `T_max` was tied to `--epochs`: at 120 epochs the learning
rate never dropped below 5.8e-04 before early stopping fired at epoch 54, while the
50-epoch run reached its best epoch at 7.8e-05, deep in the annealed tail. Raising
the budget was changing the schedule rather than extending training. Hence
`--lr_t_max`.

**Two runs diverged to NaN**, both with `--sampling all`, neither with `balanced`.
Hence gradient clipping and the guard that aborts a diverged run instead of
spending the full patience budget on frozen weights.

**HD95 sat between 115 and 184 mm while Dice hovered around 0.44**, which is not
what a poorly delineated boundary looks like. Per patient the cause was visible:
lung_001 had sensitivity 0.969 but 7 false-positive components and HD95 189 mm.
Hence the connected component filter.

**exp6 turned out bit-identical to exp1** — same per-seed values — because
`augment="anatomic"` was already the default. A duplicate run.

### Round 2 — stability fixes, and a result that would not replicate

Gradient clipping, the NaN guard, `lr_t_max` decoupling, the component filter and
an extended threshold sweep grid went in. Notebooks A and C ran.

**The headline result of round 1 did not replicate.** `attention_unet` + `all`
came back at 0.3865 ± 0.0369 over three seeds; the same seed 42 went from 0.4615
to 0.3413. A single seed had been reporting the top of a wide distribution.

**SegResNet diverged again** at epoch 12 despite clipping, but the guard caught it
and its 0.3398 matched the earlier 0.3409 — confirming the architecture is
genuinely weaker rather than merely broken.

**Post-processing validated across all five runs**: +0.0072 Dice, +0.046 precision,
−27.98 mm HD95, at −0.0033 sensitivity and zero change in the failure count.

### Round 3 — the collapse, and the loss that was paying for silence

The Attention U-Net runs showed the same pattern on **all three seeds**: seventeen
epochs climbing to Dice ≈0.39, then at epoch 18 the loss falling from 0.93 to 0.115
while validation Dice went to exactly 0.0000 and stayed there.

It was not training instability. On a slice with an empty target the intersection
is identically zero, so the whole Dice term reduces to `smooth_nr / (|P| + smooth_dr)`,
where `|P|` — the summed sigmoid output — is entirely under the model's control.
With MONAI's default `smooth_nr = smooth_dr = 1e-5` that expression rises to 1 as
the model silences itself. Under `--sampling all` about 90% of slices are empty, so
the degenerate solution scored **0.10 against 0.99** for the network that was
actually learning.

AMP made it arrive sooner: the loss was computed inside `autocast`, and fp16 flushes
any sigmoid below ~6e-8 to exactly zero, so `|P|` reaches 0 at a background logit
near −17 instead of the −22 fp32 requires. The measured discontinuity was 0.87 over
two logit units.

The collapse is irreversible: with the sigmoid saturated its derivative is zero
too, so 90% of the data produces no gradient, and escaping would mean passing back
through loss 0.9.

Three changes followed — `smooth_nr = 0`, the loss moved out of `autocast` into
fp32, and a `CollapseDetector` that stops a run whose validation Dice has read
0.0000 for three consecutive epochs after passing 0.10.

`batch=True` was considered and rejected: it removes the reward equally well but
pools the intersection across the batch, turning a macro average into a micro one.
On a mixed batch that shifted the cost of missing a small lesion against a large
one from 1:8.7 to **1:71**, and small lesions were already the weakest category.

### Round 4 — the fix worked, unevenly

Zero epochs with Dice 0.0000 across all eight runs of notebooks E and F. But the
effect was not uniform:

| | before | after |
|---|---:|---:|
| baseline Dice | 0.4444 | **0.5069** |
| baseline sensitivity | 0.32 | **0.434** |
| baseline precision | 0.750 | 0.677 |
| attention_unet Dice | 0.4615 | **0.3155** |

Exactly the predicted mechanism on the baseline: no longer paid to stay silent, the
model marks more — sensitivity up, precision slightly down, Dice up net. Sensitivity
rose on 7 of the 9 evaluable patients.

Attention U-Net went the other way, losing on **10 patients out of 10** (t = −3.68).
That contradiction was left open at the end of this round.

Two other things settled here. **SegResNet's divergence was the learning rate** —
at 3e-4 it ran 37 epochs cleanly after two NaN failures at 1e-3. And
`lr=1e-3` was confirmed as the better choice: the control at 3e-4 cost the baseline
0.043.

**lung_036 was diagnosed.** All ten models score exactly 0.000 on it. The model's
probability inside the tumour is max 0.0000 in both preprocessed and original
geometry, while reaching 0.9924 elsewhere in the same volume — total blindness, not
a near miss. Preprocessing is clean (round-trip Dice 0.9467, the dataset median).
The tumour sits at the 92nd percentile for mediastinal attachment: 34.2% of its
boundary touches soft tissue, double the next-highest test case at 17.2%. Only 7 of
44 training patients exceed 30%. The predictions land in slices 132–196 while the
ground truth occupies 88–101 — the model is segmenting a different region entirely.

### Round 5 — one code version, and two questions closed

Notebooks G and H completed the required experiments on the final code and resolved
the two open ambiguities.

**The Attention U-Net contradiction was the loss, not the budget.** Four things
had changed between the round-1 run that scored 0.4615 and the round-4 run that
scored 0.3155: the loss, its precision, the epoch budget and the patience window.
Rerunning with the *old* budget (50 epochs, patience 10) left the loss as the only
remaining difference — and returned figures **identical to six decimals** to the
round-4 run: same best at epoch 6 (validation soft Dice 0.389825), same
first-eight trajectory, same test Dice 0.3155. Both saved the same checkpoint;
only early stopping fired at different times, 16 epochs against 26.

So the budget was never the cause. Under the old loss the network peaked at epoch
8 and 0.4058; under the new one it peaks at epoch 6 and 0.3898 — earlier and
lower. As a side effect this is a clean determinism check: two independent
sessions, same seed, bit-identical trajectories.

**The β hypothesis for Tversky was refuted**, and the real cause found. See
[above](#why-tversky-cannot-be-fixed-by-tuning-).

**Seed spread turned out three times larger than believed** — 0.0153 against the
0.0047 measured in round 1. Single-seed conclusions are less safe than that early
figure suggested.

### What remains open

**The test set has ten patients.** Three of the eight configurations are
statistically indistinguishable from the baseline at that size, and one patient
(lung_036) is a guaranteed zero for reasons unrelated to model quality, which caps
macro Dice at 0.9 × the mean of the rest. Five-fold cross-validation would give a
far more stable estimate and would let lung_036 appear in training for four folds
out of five. It was considered and deliberately not run, for GPU budget reasons.

**Sensitivity remains the bottleneck**, at 0.434 against a precision of 0.677. The
model still misses over half the tumour volume. The loss fix moved this in the right
direction but did not solve it.

**Attention U-Net is now measurable but unexplained.** It is genuinely weaker under
the final loss, and the mechanism — plausibly that the attention gates need the
gradient signal on empty slices that `smooth_nr = 0` removes — is a hypothesis, not
a tested result.

---

## Requirements traceability

| requirement | where |
|---|---|
| per-case checks (13 items) | `src/eda/eda_report.py` → `output/eda_report.md`, `preprocessing_qc.csv` |
| stratified 44/9/10 split | `src/preprocessing/create_split.py`; criteria recorded in `patient_split.json` |
| canonical orientation, 1 mm resampling | `src/geometry.py` — `reorient_to_canonical`, `resample_volume` |
| linear interpolation for CT, nearest for masks | `preprocessing.py:226-227` (`order=1` / `order=0`) |
| HU windowing | `preprocessing.py:106` — `[-1000, 400]`, normalised to `[0,1]` |
| reproducible body crop | `geometry.py:240` — `crop_body_3d`, bbox stored in metadata |
| metadata for reconstruction | one JSON per patient under `output/metadata/` |
| alignment verified by overlay | `src/visualization/overlay_check.py` |
| sampling on train only | `dataset.py:291` — val and test are hardcoded to `sampling="all"` |
| error introduced by preprocessing alone | `preprocessing_qc.csv`: `pos_slices_lost_to_resize`, `tumor_volume_change_pct`, `roundtrip_dice` |
| separate evaluation script | `src/evaluation/evaluate.py`, with its own CLI |
| threshold sweep on validation only | `evaluate.py:377` — explicitly refuses `--split test` |
| artificial metric tests | `metrics.py:686` — `run_artificial_metric_tests()`, the 6 scenarios |
| macro and micro averages, stratified | `metrics.py:563`, `metrics.py:609` |
| resume, full checkpoint, config, history | `train.py` — `save_checkpoint` / `load_checkpoint`, `--resume` |
| 2.5D with 3 or 5 slices | `--n_adjacent {1,3,5}`; edges replicate the end slice rather than zero-padding |
| Z spacing for 2.5D | uniform by construction — 1 mm isotropic resampling precedes stacking |
| per-model reporting | `benchmark_report.json`: parameters, time per epoch, GPU memory, inference time per volume |
| regression suite | `tests/` — 95 tests |

Two items are handled differently from their original phrasing, both deliberately.
`DiceMetric` is no longer used at all, so the question of `ignore_empty` does not
arise: metrics are computed directly, with empty-mask handling verified by the
artificial tests. And the abrupt jump in the Dice curve was a thresholding
artifact — early stopping now watches *soft* Dice, which rises smoothly from the
first epoch, with `hard@0.5` still reported alongside for comparison.
