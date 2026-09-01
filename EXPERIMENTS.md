# Experiments


Every run goes through `src/training/train.py` on the same 44 / 9 / 10 patient
split, with the same evaluation code and the same checkpoint selection rule
(soft Dice on validation — [details](README.md#5-training-srctrainingtrainpy)).

The baseline is a 2D U-Net trained with DiceCE loss on every slice, with
anatomically valid augmentation, and every other configuration is measured against
it.

Held constant everywhere: `--batch_size 16`, `--lr 1e-3`, AdamW
(`weight_decay=1e-4`), cosine annealing, gradient clipping at norm 1.0, and a
budget of **49,390 optimiser steps** with the schedule defined in steps rather
than epochs.

---

## Final results

3D Dice on the test set, in each patient's original NIfTI geometry, on the
**all-slice** evaluation — every slice of every patient is predicted, never
zero-filled. Two columns are given for each run: the raw score, and the score
after connected-component post-processing drops components smaller than 10% of
the largest. Post-processing is part of the reported pipeline, so the second
column is the one to quote; the first is kept so the filter's contribution stays
visible.

| Configuration | Dice raw | Dice post | FP raw | FP post |
|---|---:|---:|---:|---:|
| **320 x 320 stretch, 5 channels** | 0.5659 | **0.5834** | 5.30 | **0.80** |
| 320 x 320 pad, 5 channels | 0.5119 | 0.5176 | 5.70 | 1.70 |
| 192 x 192, 5 channels | 0.5014 | 0.5161 | 3.70 | 0.50 |
| 320 x 320 pad, 7 channels | 0.4827 | 0.4910 | 4.80 | 0.80 |
| 320 x 320 pad, 1 channel | 0.4572 | 0.4664 | 9.80 | 2.70 |
| 192 x 192, 3 channels | 0.4616 | 0.4723 | 5.90 | 1.00 |
| 256 x 256 pad, 1 channel | 0.4292 | 0.4386 | 9.60 | 3.10 |
| 192 x 192, 7 channels | 0.4446 | 0.4569 | 4.30 | 0.80 |
| baseline, 192 x 192, 1 channel | 0.4488 | 0.4638 | 7.40 | 1.90 |


The best configuration is **320 x 320 with square-preserving stretch and 5 input
channels: Dice 0.5659 raw, 0.5834 after post-processing**, with false-positive
components down from 5.30 to 0.80 per patient. Its full six-metric report is in
[section 19](#19--resolution-crossed-with-context).

### Earlier results, before the evaluation corrections

The table below was produced before the corrections described in
[Evaluation protocol](#evaluation-protocol). Slices the model never ran on were
filled with zeros, the threshold was chosen on the 192 x 192 grid rather than in
original geometry, and the budget was defined in epochs rather than optimiser
steps. The *rankings* largely survive; the values do not transfer and should not
be compared with the table above.

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
| `hard_negatives` sampling | 0.2659 | −0.219 | **−5.87** | 0/10 |
| tumour crop 96 px, matched inference | 0.3666 | −0.119 | **−2.97** | 1/10 |
| no augmentation | 0.2648 | −0.221 | −2.82 | 4/10 |
| Focal Tversky (β=0.7 / 0.6) | 0.1043 / 0.0864 | −0.39 | — | 0/10 |
| Tversky (β=0.7 / 0.6) | 0.0771 / 0.0885 | −0.40 | — | 0/10 |

Baseline per seed: 0.5069 (42), 0.4751 (43), 0.4737 (44).

Every row is evaluated on full slices except the crop row, which uses sliding-window inference so that its training and inference field of view agree; evaluated on full slices the same checkpoint scores 0.1398, and [section 8](#8--tumour-centred-crop-against-the-full-slice) explains why that number measures the mismatch rather than the crop.

### What the data says

**The data pipeline matters more than the architecture.** The two large effects are
augmentation (+0.221) and sampling (+0.183). Architectures trail far behind, and
2.5D and SegResNet are statistically indistinguishable from the baseline (|t|
below threshold).

**`all` beats `balanced` by +0.18.** This contradicts the original reasoning, which justified `balanced` as a way to avoid collapsing to "always predict
negative". Empirically, matching the training distribution to the evaluation
distribution matters more than the class imbalance does. It is also among the
strongest single differences in the table — t = −4.49, losing on 10 patients out
of 10.

**What matters is how many negatives the model sees, not which ones.** `balanced`
and `hard_negatives` keep the same number of negative slices and differ only in
where they are drawn from: uniformly across the body, or clustered around the
lesion. The clustered version was expected to raise precision, since false
positives appear next to the tumour. It lowered it — 0.3784 to 0.2704, with false
positive components rising from 8.1 to 19.6 — because negatives drawn only from
beside the lesion never teach the model to reject the liver, the shoulders or the
scanner table. The two schemes are statistically indistinguishable on Dice
(t = −1.01), while `all`, which simply shows more of everything, beats both by
roughly 0.2.

**Extra capacity does not help at 44 training patients.** Attention U-Net carries
22% more parameters than the baseline plus attention gates that must themselves be
learned, and lands 0.170 lower. SegResNet behaves the same way. The additional
parameters do not extract more information from a set this small; they add more
ways to fit noise.

**The pure Tversky family does not fit this problem.** Not because it fails to find
the tumour — sensitivity is 0.39–0.43, comparable to the baseline's 0.434 — but
because it paints seven to eight times more volume than exists. The cause is
structural and is worked out [below](#why-tversky-cannot-be-fixed-by-tuning-β).

**Field of view is a variable, not a free choice.** Handing a trained model a
different field of view at inference than it saw during training costs a large
fraction of its score in either direction — the baseline U-Net falls from 0.5069
to 0.1961 on tiled windows, the same checkpoint on the same data. The
normalisation layer explains much of the spread, since InstanceNorm and GroupNorm
compute their statistics from each input's own content while BatchNorm uses fixed
ones. Worked out [below](#field-of-view-is-a-variable-in-its-own-right).

**Cropping to the lesion does not help once that mismatch is removed.** At a
matched field of view it costs the baseline 0.140 (t = −2.95) and leaves 2.5D and
SegResNet unchanged. The exception is Attention U-Net at +0.129, just under
significance, which also produces the best score any configuration has reached on
lung_058 — the smallest tumour in the test set, at 0.518 against the baseline's
0.185.

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

## Evaluation protocol

Seven defects in the evaluation and training loop were found and fixed. They are
listed here because every number above depends on them, and because the older
results in this document were produced before the fixes.

### 1. Slices with no prediction were filled with zeros

A slice the model never ran on became an all-zero slice — a confident claim of
"no tumour here" made without evaluating the model. Under an evaluation
restricted to tumour slices, **2,999 of 3,272** test slices were filled that way,
and the same checkpoint scored 0.4645 under the fill against 0.0540 when every
slice was actually predicted.

`stack_slice_predictions` now refuses by default
([`metrics.py`](src/evaluation/metrics.py)) and names the remedy in the error.
A restricted evaluation is still available — it is a legitimate thing to measure —
but it must be asked for with `require_full_coverage=False`, and
[`train.py`](src/training/train.py) then stamps
`oracle_positive_slices_evaluation: true` into the report so it can never be
mistaken for whole-volume performance. Five tests in
[`test_metrics.py`](tests/test_metrics.py) pin the behaviour, including one that
asserts whole-volume Dice *must* differ between the two protocols — so if the
guard is ever removed, the suite fails.

### 2. The threshold was chosen in preprocessed space

The sweep ran on the 192 x 192 grid while the reported Dice was computed after
resampling back to each patient's own lattice. Interpolation moves probability
mass across the decision boundary, so the optimum before reconstruction is not the
optimum after it. `threshold_sweep_original_geometry` reconstructs every
validation volume to its original geometry first, then sweeps.

### 3. Checkpoint selection ignored the negative slices

For the positives-only experiment the checkpoint was also selected on
positives-only validation, so the model was never charged for what it painted on
the remaining 92% of each volume. `eval_sampling` now defaults to `"all"` and the
validation loader is built from it, so early stopping, checkpoint selection and
the threshold sweep all read the full validation set.

Both selection rules were then run at the same budget to measure what the wrong
one costs:

| selection rule | headline reported | scored on all slices | oracle protocol |
|---|---:|---:|---:|
| all validation slices | 0.3045 | **0.3045** | 0.5357 |
| positives only | 0.5515 | **0.2067** | 0.5515 |

Compared headline to headline the defective rule looks **0.2470 better**. Compared
on the same slices it **costs 0.0978**. Scored under the oracle protocol the two
models are nearly identical — the entire apparent advantage was the protocol, not
the model.

### 4. Threshold and slice set were varied together

Two evaluations had been quoted at two different thresholds, so neither
attributed its difference to anything.
[`protocol_matrix.py`](src/evaluation/protocol_matrix.py) computes both protocols
at seven thresholds on one grid, which separates the two effects. The matrices are
saved under `output/protocol_matrices/`.

The requested cells, for the run trained on positive slices with the checkpoint
selected on all validation slices:

| | protocol `all` | protocol `positives` | delta |
|---|---:|---:|---:|
| **threshold 0.25** | | | |
| Dice 2D, tumour slices | 0.397675 | 0.397658 | −1.69×10⁻⁵ |
| Dice 3D, whole volume | 0.3229 | 0.5423 | +0.2194 |
| **threshold 0.40** | | | |
| Dice 2D, tumour slices | 0.380367 | 0.378986 | −1.38×10⁻³ |
| Dice 3D, whole volume | 0.3251 | 0.5206 | +0.1956 |

and for the same training with the checkpoint selected on positives only:

| | protocol `all` | protocol `positives` | delta |
|---|---:|---:|---:|
| **threshold 0.25** | | | |
| Dice 2D, tumour slices | 0.452313 | 0.452307 | −5.94×10⁻⁶ |
| Dice 3D, whole volume | 0.1839 | 0.5711 | +0.3872 |
| **threshold 0.40** | | | |
| Dice 2D, tumour slices | 0.445675 | 0.444505 | −1.17×10⁻³ |
| Dice 3D, whole volume | 0.1932 | 0.5686 | +0.3754 |

All three checks pass. **Tumour-slice Dice is protocol-independent at a fixed
threshold** — the two protocols agree to five and six decimal places at 0.25, and
the largest disagreement anywhere is 1.38×10⁻³. Restricting which slices are fed
to a 2D model does not change its prediction on the slices it is fed either way,
so the evaluation sets are being indexed correctly. **Whole-volume Dice differs,
and the difference is the zero-fill** — +0.20 to +0.39, tracking how much each
model fires on negative slices. **The optimum is protocol-specific**: `all → 0.40`
against `positives → 0.10` on the first run, `all → 0.90` against
`positives → 0.25` on the second.

The residual 10⁻³ disagreement is not an indexing fault. Reconstruction returns
the 1 mm stack to each patient's original spacing, and wherever that spacing is
not 1 mm the result is a blend of neighbouring preprocessed slices; at the
tumour's Z edges one of those neighbours is a slice the restricted protocol
zero-filled, so the fill reaches the slices being scored. Seven of the ten test
patients resample. `protocol_matrix.py` splits patients by `z_interpolates()` and
reports a failure only when a patient whose reconstruction does *not* resample
disagrees — which would be a genuine indexing bug.

### 5. Every crop was stretched into a square

Body crops have different aspect ratios per patient, so resizing each directly to
a square scaled the two axes by different factors and undid part of the benefit of
the 1 mm isotropic resampling. `--resize_mode pad` pads the crop to square before
resizing, and `--target_size` takes 256 and 320. Both are measured in
[section 18](#18--square-padding-and-higher-resolution).

### 6. The schedule was defined in epochs, not steps

At batch size 16, `all` gives about 898 batches per epoch and `positives` about
89. With the scheduler and early stopping defined in epochs, the positives run
reached a small learning rate after roughly ten times fewer parameter updates, so
comparisons across sampling modes were never budget-matched. Everything moved to
steps: `schedule_unit="step"`, `max_steps=49_390`, `lr_t_max=44_900` (91% of the
budget, then held at `eta_min`), `patience_steps`.

A run at double budget put its best epoch at **46,696 steps**, before the
reference budget is spent, with validation averaging 0.3171 before that point and
0.3952 after — so 49,390 steps is sufficient and probably generous.

Across ten runs the **best-on-validation checkpoint beats the final annealed one
in nine**, mean `final − best = −0.0210`. The `eta_min` hold halves the penalty
but does not reverse it (six of seven, mean −0.0094). Both are reported for every
run, each with its own threshold sweep, since the optimal threshold moves with the
weights.

### 7. Non-finite losses were detected after the step

The `NaN`/`Inf` check ran after `backward()` and `optimizer.step()`, so a batch
with a non-finite loss could move the weights before being counted as skipped.
The check now sits **before** both, and skips the batch
([`train.py`](src/training/train.py)). Under AMP `GradScaler` would have declined
the step anyway; on CPU it was the difference between poisoning the weights and
not. A finite loss can still backpropagate into non-finite gradients, so the
scaler's own gradient inspection remains the second half of the guard.

---

## Metric granularity

Every number above is 3D Dice per patient: one score per reconstructed volume,
where a false positive anywhere in the scan counts against it.

`src/evaluation/hierarchical_report.py` scores every checkpoint at four
granularities at once, from the same reconstructions, so the comparison needs no
re-training:

| view | what it measures | what it cannot see |
|---|---|---|
| 2D, tumour slices | delineation, given that a lesion is present | every false positive on an empty slice |
| 2D, every slice | the above, plus the empty slices | nothing — but see the ceiling below |
| 3D, per lesion | one score per ground-truth component | false positives away from every lesion |
| 3D, per patient | what this project quotes | — |

```bash
python -m src.evaluation.hierarchical_report --runs all
```

| Configuration | 2D, tumour slices | 2D, every slice | 3D, per lesion | 3D, per patient | slices failed | false alarms |
|---|---:|---:|---:|---:|---:|---:|
| **baseline** — `all` + `anatomic` + `dice_ce` | 0.3981 | 0.9053 | 0.1387 | 0.5070 | 43% | 5% |
| U-Net, `lr=3e-4` | 0.3597 | 0.9042 | 0.1512 | 0.4642 | 44% | 5% |
| 2.5D, `n_adjacent=3` | 0.3687 | 0.9060 | 0.1735 | 0.4259 | 42% | 5% |
| SegResNet, `lr=3e-4` | 0.3619 | 0.8818 | 0.1060 | 0.4180 | 36% | 7% |
| crop 96, Attention U-Net | 0.3146 | 0.9065 | 0.1091 | 0.3998 | 44% | 4% |
| DiceFocal | 0.3092 | 0.8858 | 0.0876 | 0.3536 | 44% | 6% |
| crop 96, SegResNet | 0.3128 | 0.8606 | 0.0930 | 0.3494 | 47% | 9% |
| `hard_negatives`, SegResNet | 0.2879 | 0.8486 | 0.0868 | 0.3258 | 51% | 10% |
| `hard_negatives`, 2.5D | 0.3168 | 0.8684 | 0.0912 | 0.3203 | 47% | 8% |
| Attention U-Net | 0.2692 | 0.9071 | 0.0952 | 0.3155 | 60% | 4% |
| Attention U-Net, 50 ep / patience 10 | 0.2692 | 0.9071 | 0.0952 | 0.3155 | 60% | 4% |
| `hard_negatives`, Attention U-Net | 0.2561 | 0.8737 | 0.0946 | 0.3050 | 53% | 7% |
| `balanced` sampling | 0.2724 | 0.8865 | 0.0853 | 0.3023 | 51% | 6% |
| `hard_negatives`, U-Net | 0.3033 | 0.8072 | 0.1388 | 0.2659 | 49% | 15% |
| no augmentation | 0.2787 | 0.8328 | 0.1443 | 0.2648 | 55% | 12% |
| crop 96, 2.5D | 0.3448 | 0.4258 | 0.1472 | 0.1601 | 39% | 57% |
| crop 96, U-Net | 0.3319 | 0.2996 | 0.2237 | 0.1398 | 36% | 71% |
| Focal Tversky (β=0.7) | 0.3972 | 0.3782 | 0.1609 | 0.1043 | 40% | 63% |
| Tversky (β=0.6) | 0.4062 | 0.4076 | 0.1631 | 0.0885 | 36% | 59% |
| Focal Tversky (β=0.6) | 0.4167 | 0.2936 | 0.1445 | 0.0864 | 35% | 72% |
| Tversky (β=0.7) | 0.3627 | 0.5140 | 0.1174 | 0.0771 | 44% | 47% |

Sorted by the column this project quotes (3D Dice per pacient). `slices failed` is the share of
tumour-bearing slices scoring below 0.10; `false alarms` is the share of empty
slices carrying any prediction at all.

### The two views rank the models almost independently

Across the 21 runs, per-slice tumour Dice and per-patient Dice correlate at
**r = −0.265** (Spearman **−0.112**). They are not two readings of the same
quantity — they are close to unrelated, and where they do relate, it is the wrong
way round.

The mechanism is visible in the last column. Per-slice tumour Dice correlates
**positively** with the false-alarm rate (**r = +0.586**), while per-patient Dice
correlates negatively with it (**r = −0.880**). A model that paints more widely
scores *better* on tumour slices, because the slices where that costs it are
exactly the ones the metric drops.

The three highest per-slice scores make the point without any statistics:

| Configuration | 2D, tumour slices | rank | 3D, per patient | rank | false alarms |
|---|---:|---:|---:|---:|---:|
| Focal Tversky (β=0.6) | 0.4167 | **1 / 21** | 0.0864 | **20 / 21** | 72% |
| Tversky (β=0.6) | 0.4062 | **2 / 21** | 0.0885 | **19 / 21** | 59% |
| **baseline** | 0.3981 | 3 / 21 | **0.5070** | **1 / 21** | 5% |

The two configurations at the top of the per-slice table are the ones
[shown above](#why-tversky-cannot-be-fixed-by-tuning-β) to have exactly zero
gradient on empty slices — 90% of the training data produces no learning signal,
and nothing stops the network from painting everywhere. Judged per slice, that
pathology reads as the best delineation in the project. Judged per patient, it
reads as the worst model in the project. Both readings are arithmetically
correct.

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

```bash
python -m src.training.train --exp_name hard_negatives_unet \
    --sampling hard_negatives --epochs 100 --lr_t_max 50 --patience 20
```

All three sampling modes keep every positive slice; they differ only in which
negatives they add. `balanced` draws them uniformly from the body, while
`hard_negatives` concentrates them along Z around the tumour — 70% near, 30% far.

The two take **the same number** of negatives, 1419 against 1416 on the training
split, so the controlled comparison is `hard_negatives` against `balanced` rather
than against `all`. Only the choice of negatives changes:

| | median distance to the tumour | within 20 slices |
|---|---:|---:|
| `balanced` | 72 slices | 14.5% |
| `hard_negatives` | 27 slices | **42.2%** |

The reasoning was that a slice immediately above or below the lesion contains
vessels, atelectasis and mediastinal structures that resemble it, and those are
where false positives actually appear. A uniformly drawn negative is usually air
or obvious body wall and teaches almost nothing. If that held, `hard_negatives`
would raise **precision** over `balanced` at comparable sensitivity.

**It does the opposite.**

| U-Net | Dice | sensitivity | precision | FP components |
|---|---:|---:|---:|---:|
| `balanced` | 0.3023 | 0.2928 | **0.3784** | **8.1** |
| `hard_negatives` | 0.2659 | 0.3358 | 0.2704 | 19.6 |

Precision falls by 0.11 and the false-positive components more than double.
Against `balanced` the Dice difference is −0.036 at t = −1.01, so on the metric
itself the two are indistinguishable; it is the direction of the precision and
component counts that refutes the reasoning.

Run across all four architecture configurations, each against the same
architecture trained on its usual sampling:

| architecture | its usual sampling | `hard_negatives` | Δ | t |
|---|---:|---:|---:|---:|
| U-Net | 0.3023 (`balanced`) | 0.2659 | −0.036 | −1.01 |
| 2.5D | 0.4259 (`all`) | 0.3203 | −0.106 | **−5.21** |
| Attention U-Net | 0.3155 (`all`) | 0.3050 | −0.011 | −0.21 |
| SegResNet | 0.4181 (`all`) | 0.3258 | −0.092 | −1.64 |

No configuration improves. The reason the hypothesis fails is visible in the
construction: concentrating negatives near the lesion means the model is never
shown the anatomy that is far from it — liver, shoulders, apex, scanner table.
Only 30% of an already small negative set was drawn from "far", which works out
to roughly ten slices per patient to cover the entire rest of the body. The
difficulty gained did not pay for the coverage lost.

**The finding is therefore about count, not selection.** `all` (0.5069) beats both
subsampling schemes by a wide margin, while `balanced` (0.3023) and
`hard_negatives` (0.2659) — same number of negatives, very different choice of
them — are statistically indistinguishable. How many negatives the model sees
dominates; which ones they are does not measurably matter.

There is no efficiency argument either. At 2835 slices per epoch against 14368,
`hard_negatives` trains roughly five times faster than the baseline, but it buys
nothing over `balanced`, which is equally cheap.

### 8 — Tumour-centred crop against the full slice

```bash
python -m src.training.train --exp_name crop96_unet \
    --crop tumor --crop_size 96 --sampling all \
    --epochs 100 --lr_t_max 50 --patience 20
```

**The crop applies to training only.** At inference the tumour location is exactly
what is being predicted, so a window centred on the ground-truth centroid would
feed the label to the model. Validation and test always run on the full 192×192
slice. This works because all four configurations are fully convolutional: they
accept 96×96 while training and 192×192 at evaluation unchanged.

What it changes, measured on the training split:

| | tumour pixels | slices containing tumour |
|---|---:|---:|
| full slice | 0.076% | 10.2% |
| 96 px crop | **0.303%** | 10.2% |

Density rises by exactly the area ratio, 192²/96² = 4. The proportion of slices
containing a tumour is untouched, which keeps this experiment orthogonal to
sampling.

96 px is the smallest window that holds every lesion with context to spare — the
largest tumour bounding box in the training split is 38×57 px at 1 mm — while
staying divisible by 16, as the four downsampling stages require. The centre is
jittered by ±24 px rather than placed exactly on the centroid: cropping precisely
would put the tumour at the middle of every training sample, from which a network
can learn the position instead of the appearance, and that shortcut collapses the
moment it is evaluated on a full slice where the lesion is off-centre. Slices with
no tumour are cropped at a random position rather than dropped, which would
silently turn this into a sampling experiment as well.

Two effects pull against each other. Four times the positive density should help
the Dice term, which now sees a target that is larger relative to its background.
But the model never sees a whole thorax during training and is then asked to
search an image four times the area, which may cost precision.

#### The first measurement was not measuring the crop

Evaluated on full slices, three of the four configurations collapsed — U-Net to
0.1398 with 172.9 false-positive components per patient. That is not what a model
which failed to learn looks like. Fed a 96 px window instead, the same checkpoints
score 0.45 to 0.52, at or above what the baseline reaches on validation. Every one
of them had learned the task; what they could not do was carry it over to a
different field of view.

| model | fed a 96 px window | fed the full slice | drop | normalisation |
|---|---:|---:|---:|---|
| `crop96_unet` | 0.4520 | 0.2224 | **−0.230** | Instance |
| `crop96_unet_25d` | 0.4516 | 0.3199 | −0.132 | Instance |
| `crop96_segresnet` | 0.5227 | 0.4833 | −0.039 | Group |
| `crop96_attention_unet` | 0.5049 | 0.5111 | **+0.006** | Batch |

Comparing a window-trained model against a slice-trained one therefore changes two
things at once, and the field of view dominates. Separating them needs an
inference path that gives the window-trained model the field of view it was
trained under — see [below](#field-of-view-is-a-variable-in-its-own-right).

#### The crop's actual effect

With `--sw_roi 96`, each slice is covered by overlapping 96 px windows whose
predictions are blended back into a full-size map. Both cells below now match
training and inference field of view, so the difference between them is
attributable to the crop:

| architecture | full slice, trained and tested | 96 px, trained and tested | Δ | t |
|---|---:|---:|---:|---:|
| U-Net | 0.5069 | 0.3666 | **−0.140** | **−2.95** |
| Attention U-Net | 0.3155 | 0.4447 | **+0.129** | 2.22 |
| 2.5D | 0.4259 | 0.3960 | −0.030 | −0.88 |
| SegResNet | 0.4181 | 0.3314 | −0.087 | −1.54 |

**Cropping does not help.** It significantly hurts the baseline U-Net and leaves
2.5D and SegResNet unchanged. Attention U-Net is the exception at +0.129, but
t = 2.22 sits just under the 2.262 threshold, so it is suggestive rather than
established — and it was the weakest configuration to begin with.

Restoring the matched field of view does, however, recover most of what the first
measurement had lost:

| architecture | full-slice inference | sliding window | Δ | t |
|---|---:|---:|---:|---:|
| U-Net | 0.1398 | 0.3666 | **+0.227** | **4.24** (9/10) |
| 2.5D | 0.1602 | 0.3960 | **+0.236** | **3.24** (8/10) |
| Attention U-Net | 0.3999 | 0.4447 | +0.045 | 1.14 |
| SegResNet | 0.3494 | 0.3314 | −0.018 | −0.89 |

False-positive components on U-Net fall from 172.9 per patient to 27.5, and the
threshold the sweep selects drops from 0.975 to 0.80. The over-segmentation was
the mismatch, not the training.

**One result worth singling out.** `crop96_attention_unet` under sliding-window
inference scores **0.5180 on lung_058**, against 0.1847 for the baseline and
0.3811 for the next best configuration. That patient carries an 807 mm³ tumour,
smaller than any in the training split, and is where nearly every configuration
fails. Quadrupling the positive-pixel density helps most exactly where the target
is smallest, which is the mechanism the crop was supposed to exploit.

### Field of view is a variable in its own right

The crop experiment turned up something that has nothing to do with cropping.
Changing the field of view between training and inference costs a large fraction
of a model's performance, in **both** directions, without touching a single
weight.

Each cell below is the share of its own matched-field-of-view score that a model
retains once the field of view changes under it:

| architecture | normalisation | trained on slices, tested on windows | trained on windows, tested on slices |
|---|---|---:|---:|
| U-Net | Instance | **0.387** | **0.381** |
| 2.5D | Instance | 0.608 | 0.404 |
| SegResNet | Group | 0.624 | 1.054 |
| Attention U-Net | Batch | 0.699 | 0.899 |

The baseline U-Net falls from 0.5069 to 0.1961 purely by being handed tiled
windows at inference — the same checkpoint, the same data, only a different field
of view. It loses about 62% of its score in either direction, which is a striking
symmetry.

The normalisation layer is a large part of why. **InstanceNorm and GroupNorm
derive their statistics from the content of each input**, so a window filled with
lung and tumour and a slice dominated by air outside the body normalise the same
anatomy to different values. **BatchNorm applies fixed running statistics at
evaluation**, so what surrounds a structure does not change how it is scaled.
That predicts InstanceNorm as the most fragile and BatchNorm as the most robust,
which is what the extremes show. It does not explain the whole ordering —
SegResNet pays no penalty at all going from windows to slices — so normalisation
is a major factor rather than the only one.

A second mechanism is present and this experiment cannot fully separate it.
Blending nine overlapping windows with Gaussian weights smooths the probability
map. Re-selecting the threshold on validation absorbs the change in scale — the
baseline moves from 0.75 to 0.35 — but not the change in shape.

**The practical consequence** is that training-time field of view and
inference-time field of view are a pair, not two independent choices. Patch-based
training is a legitimate and standard technique, but only alongside an inference
path that reproduces the same field of view. An initial reading of this experiment
attributed the collapse to cropping itself, which the matched-field-of-view cells
show was wrong.

### 9 — Attention U-Net

```bash
python -m src.training.train --exp_name attention_unet \
    --model_type attention_unet --sampling all --epochs 100 --lr_t_max 50 --patience 20
```

Adds attention gates on every skip connection
([models/README.md](src/models/README.md)), which learn to suppress healthy tissue
and air before concatenation with the decoder. Relevant in principle because the
target occupies under 1% of the volume. 1,987,417 parameters against the
baseline's 1,624,844.

### 10 — SegResNet

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

### 11 — 2.5D, three consecutive slices

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

### 12 — Attention U-Net under the earlier epoch budget

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

---

The experiments below were all run **after** the corrections in
[Evaluation protocol](#evaluation-protocol), at a matched budget of 49,390
optimiser steps, and are directly comparable with each other. Sections 1–12 above
predate the corrections.

### 13 — Can the model memorise?

The floor under every other number here. A model with 1.6M parameters facing
eight slices has roughly 200,000 parameters per slice; if it cannot reach a high
Dice on those, the failure is not capacity or data but a fault in the loss, the
masks, or the gradient path. Memorisation needs no learnable pattern — only that
the mapping be representable and that gradients reach it — so it separates "hard
problem" from "broken pipeline", which otherwise both produce a low score.

Augmentation is off throughout, deliberately: showing a different image every step
makes memorisation impossible by construction and would turn any bug into
something indistinguishable from variance.

Four checks live in [`test_overfit_sanity.py`](tests/test_overfit_sanity.py) and
run in the standard suite: a fixed batch can be memorised, every loss drives the
model onto the target, the centre slice is the one being scored under 2.5D, and
real tumour slices can be memorised. All pass.

### 14 — Negative-to-positive ratio at matched optimiser steps

Section 1 compared `balanced` against `all` at equal *epochs*, which gave them
different amounts of training. Repeated at equal steps, with the epoch count
scaled so each arm spends the same 49,390:

| ratio | Dice raw | Dice post | FP raw | FP post |
|---|---:|---:|---:|---:|
| 1 : 1 | 0.3962 | 0.4090 | 9.00 | 2.40 |
| 1 : 3 | 0.4290 | 0.4397 | 6.90 | 1.50 |
| 1 : 5 | 0.3818 | 0.4037 | 11.40 | 2.00 |
| 1 : 9 | 0.4183 | 0.4349 | 10.60 | 2.40 |
| all (the natural distribution) | **0.4554** | **0.4691** | 11.30 | 2.90 |

The whole spread is 0.074 against a ±0.1209 noise band, so **no ordering here is
established**. What the budget-matched form does settle is that the earlier
−0.183 penalty attributed to `balanced` sampling was mostly the missing training,
not the sampling: at equal steps the gap shrinks to 0.059 and stops being
significant.

Training exclusively on positive slices is not included as an arm. Section 3 of
the protocol section measures what it does to a system that will receive whole
volumes.

### 15 — Hard negatives chosen by the model's own errors

Section 7 drew negatives by distance from the tumour. Here, the model is run over the training set, and the negative
slices where it actually produces false positives are oversampled—
[`mine_negatives.py`](src/training/mine_negatives.py).

| second stage | Dice raw | Dice post |
|---|---:|---:|
| stage 1 only, all slices | 0.3940 | 0.4167 |
| distance-based negatives | 0.3982 | — |
| **mined** negatives | 0.4194 | 0.4323 |
| **randomly chosen** negatives | 0.4201 | — |

Mined and random negatives are **indistinguishable** (0.4194 against 0.4201). The
gain over stage 1 comes from a second stage existing at all, not from which
negatives it uses, so the mining earns nothing on this dataset.

### 16 — Two-stage training

First stage on all slices to learn normal anatomy, second stage at a lower
learning rate with more positives and hard negatives. Fine-tuning on positives
alone risks the model forgetting what healthy slices look like, so both mixes were
run.

| second stage | Dice raw | Dice post | false alarm | FP raw |
|---|---:|---:|---:|---:|
| stage 1 only | 0.3940 | 0.4167 | 0.0701 | 10.20 |
| positives, low LR | 0.4166 | 0.4363 | 0.0769 | 10.90 |
| positives, high LR | 0.4226 | 0.4430 | 0.0721 | 10.90 |
| + hard negatives 0.5, low LR | 0.4194 | 0.4323 | **0.0640** | 8.60 |
| + hard negatives 0.5, high LR | 0.3779 | 0.3972 | 0.0785 | 10.60 |
| + hard negatives 1.0, low LR | 0.3983 | 0.4150 | 0.0672 | 9.00 |
| + hard negatives 1.0, high LR | 0.4062 | 0.4200 | **0.0630** | 8.70 |

Best Dice gain is 0.029, inside the noise band. The only arms whose confidence
interval excluded zero did so on **false-alarm rate**, not Dice — adding hard
negatives to the second stage reduces firing on empty slices without changing
agreement on the tumour.

### 17 — Inter-slice context

A tumour persists across slices; a vessel crossing the plane does not. A model
that sees neighbouring slices should be able to tell them apart and stop firing on
isolated structures. Section 11 tried this at `n_adjacent=3` under the old budget;
it is repeated here across four widths.

Preprocessing resamples to 1 mm against a median acquisition of 1.24 mm, so
`n_adjacent=3` spans about ±0.8 of a real acquired slice — two of its three
channels are largely interpolated from the same source data as the centre. That is
why the sweep goes to ±3 mm.

| context | Dice raw | Dice post | FP raw | FP post | % neg slices |
|---|---:|---:|---:|---:|---:|
| 2D, 1 channel | 0.3940 | 0.4167 | 10.20 | 1.80 | 6.08% |
| ±1 mm, 3 channels | 0.4616 | 0.4723 | 5.90 | 1.00 | 5.12% |
| **±2 mm, 5 channels** | **0.5014** | **0.5161** | **3.70** | **0.50** | **2.83%** |
| ±3 mm, 7 channels | 0.4446 | 0.4569 | 4.30 | 0.80 | 3.54% |

**False-positive components fall 64% and the negative-slice firing rate more than
halves.** That is the predicted behaviour, and it is the metric the argument is
about: a model that has stopped firing on structures which appear in one slice and
vanish in the next.

±2 mm is the peak: 3 channels is too narrow to distinguish a vessel from a
tumour, and 7 overshoots. The same ordering appears independently at 320 x 320 in
section 19.

### 18 — Square padding and higher resolution

Padding the crop to square before resizing, at three grids. The round-trip Dice of
the ground truth through preprocessing and back bounds what any model can score,
and was measured first on CPU
([`resolution_ceiling.py`](src/preprocessing/resolution_ceiling.py)):

| grid | ceiling, all | ceiling, small tumours | worst patient |
|---|---:|---:|---:|
| 192 pad | 0.9320 | 0.8979 | 0.8355 |
| 256 pad | 0.9477 | 0.9237 | 0.8792 |
| 320 pad | 0.9549 | 0.9347 | 0.9115 |
| 192 stretch | 0.9386 | 0.9081 | 0.8602 |
| 256 stretch | 0.9512 | 0.9281 | 0.8957 |
| 320 stretch | 0.9583 | 0.9387 | 0.9100 |

Small tumours have the lowest ceiling at 192 and gain most from 320 — +0.037
against +0.023 overall — so the concern about lost information is measurable.

Trained:

| grid | Dice raw | Dice post | FP raw | FP post |
|---|---:|---:|---:|---:|
| 256 pad | 0.4292 | 0.4386 | 9.60 | 3.10 |
| 320 pad | 0.4572 | 0.4664 | 9.80 | 2.70 |

**Resolution raises Dice but leaves false positives untouched** — FP components
sit near 10 at every grid. Against section 17, where context cut them to 3.70, the
two levers are doing different jobs: resolution buys agreement on the tumour,
context buys silence everywhere else.

### 19 — Resolution crossed with context

The two levers that worked, applied together at 320 x 320: a 2 x 2 over resize
mode and context width. **Three of the four cells are measured under the
corrected protocol.** The fourth is filled with the closest run this project has,
`B_res320_25d7`, which is *not* a matched arm — see the note below the table.

| arm | Dice raw | Dice post | sens | prec | FP raw | FP post | % neg |
|---|---:|---:|---:|---:|---:|---:|---:|
| pad, 5 channels | 0.5119 | 0.5176 | 0.5171 | 0.6082 | 5.70 | 1.70 | 7.42% |
| pad, 7 channels | 0.4827 | 0.4910 | 0.4824 | 0.6430 | 4.80 | 0.80 | 5.46% |
| **stretch, 5 channels** | **0.5659** | **0.5834** | 0.5574 | 0.6398 | 5.30 | **0.80** | 5.11% |
| *stretch, 7 channels* † | *0.5451* | *0.5551* | *0.5042* | *0.7211* | *2.50* | *0.20* | — |

† `B_res320_25d7`, and **not comparable with the three rows above it**. It differs
from the arm this cell calls for in three ways at once, any one of which would be
enough to break the comparison:

| | `B_res320_25d7` | what this cell requires |
|---|---|---|
| Z spacing | 2.5 mm — 5,745 training slices | 1.0 mm — 14,368 training slices |
| budget | ~32,300 steps, cosine over 90 epochs, early stopping at patience 25 | 49,390 steps, cosine over steps, no early stopping |
| context | 7 channels at 2.5 mm = **±7.5 mm** | 7 channels at 1.0 mm = **±3 mm** |

The third line is the one that matters most. At 2.5 mm a 7-channel window reaches
two and a half times further through the patient than the same channel count does
in the rows above, so this row is not "7 channels" in the sense the column header
means. Its low false-positive count — 2.50 raw, 0.20 after post-processing, the
lowest in the project — is consistent with the finding that wider context
suppresses false positives, but it cannot be attributed to resize mode, to
context width, or to resolution, because all three moved together along with the
budget.

It is shown here because it is the only evidence that exists for this corner of
the grid, not because it settles it.

**The best configuration in the project: 320 x 320 stretch with 5 input channels,
Dice 0.5659 raw and 0.5834 after post-processing**, with false-positive components
falling from 5.30 to 0.80 per patient and precision rising from 0.6398 to 0.7004
under the same filter. Sensitivity is essentially unchanged (0.5574 → 0.5527), so
the filter is removing spurious components rather than trimming the tumour.

### 20 — Lung mask and two-stage localisation

Restricting the model to the lungs should remove false positives in the
mediastinum, the chest wall and the abdomen. The risk has the same shape as the
benefit: a tumour touching the pleura is not air, so a mask built to exclude the
chest wall excludes that tumour with it, and sensitivity lost that way is lost
before training starts.

So the ceiling was measured first over all 63 patients
([`lung_mask.py`](src/preprocessing/lung_mask.py)):

| dilation | tumour kept | worst patient | image kept | patients under 90% |
|---|---:|---:|---:|---:|
| 0 | 0.697 | 0.030 | 13.1% | 36 |
| 3 | 0.914 | 0.303 | 16.5% | 13 |
| 8 | 0.980 | 0.599 | 22.2% | 4 |
| **12** | **0.994** | 0.782 | 26.7% | 1 |

At 12 voxels the mask keeps 99.4% of all tumour while discarding 73% of the image,
so the approach passes its gate. What the gate cannot say is whether the false
positives live in the discarded part.

They do not. Applying the mask to the control's own predictions — same weights,
same threshold, only the filter differs, so every difference is the mask:

| arm | Dice | FP comp. | Δ Dice | FP removed |
|---|---:|---:|---:|---:|
| control, no mask | 0.4224 | 8.30 | — | — |
| post-hoc, dilation 0 | 0.3463 | 9.20 | **−0.0761** | −11% |
| post-hoc, dilation 3 | 0.4191 | 7.90 | −0.0033 | 5% |
| post-hoc, dilation 5 | 0.4248 | 7.90 | +0.0025 | 5% |
| post-hoc, dilation 8 | 0.4248 | 7.90 | +0.0025 | 5% |
| post-hoc, dilation 12 | 0.4248 | 7.90 | +0.0025 | 5% |
| trained on masked input | 0.4100 | 7.80 | −0.0124 | 6% |

Paired at dilation 12: **+0.0025, 95% CI [−0.0024, +0.0073]**.

The mask discards **73% of the image and removes 5% of the false-positive
components**. Dilations 5, 8 and 12 give identical results, which is the decisive
detail: beyond 5 voxels the mask no longer intersects anything the model
predicted, so the predictions already lie inside the lung fields. At dilation 0
masking actively hurts, −0.0761, by deleting juxtapleural true positives — exactly
the risk the ceiling measurement was built to catch, now observed.

The false positives are vessels and nodules *within* the lungs, where an
air-threshold mask is blind by construction. Training on masked input does not
rescue it. This is also why section 17 works and this does not: separating a
vessel from a tumour needs the third dimension, not a spatial prior.

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

### Round 6 — negative selection and field of view

Two further data-pipeline variables were tested last, each across all four
architecture configurations: which negative slices the model is shown, and how
much of a slice it sees at a time.

**Hard negative sampling refuted its own hypothesis.** The prediction was higher
precision, since false positives cluster next to the lesion. Precision fell
instead, from 0.3784 to 0.2704, and false-positive components rose from 8.1 to
19.6. Nothing improved on any architecture.

What came out of it is a cleaner statement than the one being tested. `balanced`
and `hard_negatives` hold the negative count fixed and vary only the selection,
and they land within noise of each other (t = −1.01) — while `all`, which changes
the count and nothing else, beats both by around 0.2. On this dataset the amount
of negative anatomy the model is shown dominates, and the choice of which
negatives does not measurably register. That also explains the failure: negatives
drawn only from beside the tumour leave the liver, the shoulders and the scanner
table unrepresented, and roughly ten "far" slices per patient is not enough to
cover them.

**Cropping to the lesion looked catastrophic, and was not.** Three of the four
window-trained models scored between 0.14 and 0.35 on full slices, with U-Net
producing 172.9 false-positive components per patient. Fed the 96 px windows they
were trained on, the same checkpoints score 0.45 to 0.52 — every one of them had
learned the task. The first reading blamed the crop; the failure was the field of
view changing between training and inference.

Re-evaluating through overlapping windows, which restores the training field of
view without ever consulting the ground truth, recovered +0.227 on U-Net
(t = 4.24) and +0.236 on 2.5D (t = 3.24). With both cells then matched, the crop's
own effect is negative or neutral everywhere except Attention U-Net, which gains
0.129 at t = 2.22 — under the significance threshold.

Two things came out of it that were not the question being asked. Field of view
turns out to cost a large fraction of a model's score whenever it changes between
training and inference, in both directions, tracking the normalisation layer.
And `crop96_attention_unet` under matched inference reaches **0.518 on lung_058**,
the 807 mm³ tumour smaller than anything in the training split, against 0.185 for
the baseline — the density argument for cropping holding exactly where the target
is smallest, even though it does not hold on average.

### What remains open

**The test set has ten patients.** Only two of the seven configurations carrying
a paired test are separable from the baseline in the direction of being *worse but
not significantly so* — 2.5D and SegResNet sit inside the noise — and one patient
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

**The crop helps small lesions and hurts on average, and only one patient carries
that claim.** lung_058 is the sole small-category case in the test set, so the
0.518 it reaches under a window-trained Attention U-Net rests on a single volume.
Confirming it would need a split with several small tumours in test, or
cross-validation.

**Two mechanisms are tangled in the field-of-view result.** The normalisation
layer explains the extremes but not the whole ordering, and Gaussian blending
across overlapping windows smooths the probability map independently of that.
Separating them would mean re-running the matrix with the blending weights flat,
or with the normalisation layers swapped between architectures.

**The architecture ranking rests on single seeds.** Only the baseline was run
three times, and its spread — 0.5069 / 0.4751 / 0.4737, standard deviation 0.0153
— is the only estimate of run-to-run noise this project has. 2.5D at 0.4259 and
SegResNet at 0.4181 sit 0.008 apart on one seed each, half a standard deviation,
and the paired tests already place both inside the noise against the baseline. The
ordering between them is therefore not established, and reading the table as a
ranking of the three non-baseline architectures reads more into it than the
measurements support. Two more seeds on 2.5D would be the cheapest way to find out
how wide the intervals actually are.

---

## Where the results live

Every run writes to `output/experiments/{exp_name}/seed_{n}/`:
`benchmark_report.json`, `config.json`, `threshold_sweep.json`,
`training_history.csv`, `test_results_per_patient.csv` and `best_model.pt`.
96 runs are stored.

`baseline/` additionally holds the three artifacts that describe the baseline as a
group rather than as single runs: `multi_seed_summary.json`,
`threshold_rescore.json` — the three seeds re-swept in original geometry, which is
why the same checkpoints appear in this document with two sets of scores — and
`ensemble_report.json`.

Every experiment reports, on the all-slice evaluation: **Dice 3D, sensitivity,
precision, predicted-to-true volume ratio, false-positive component count, and the
share of negative slices given at least one tumour pixel.** Each catches a failure
the others hide — a model can hold Dice steady while over-painting, or look precise
per patient while firing on empty slices. The volume ratio is reported as a median,
because on one run the mean read 1.809 while eight of ten patients were
under-segmenting: a single patient with 188 ground-truth voxels produced a ratio of
9.27 on its own.

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
| 2.5D with 3, 5 or 7 slices | `--n_adjacent {1,3,5,7}`; edges replicate the end slice rather than zero-padding |
| every slice predicted before scoring | `metrics.py` — `stack_slice_predictions(require_full_coverage=True)` raises otherwise |
| restricted evaluation labelled as such | `train.py` — `oracle_positive_slices_evaluation` in every report |
| threshold chosen in original geometry | `metrics.py` — `threshold_sweep_original_geometry` |
| selection on the full validation set | `train.py` — `eval_sampling` defaults to `"all"`, the val loader is built from it |
| threshold x protocol matrix | `src/evaluation/protocol_matrix.py` → `output/protocol_matrices/` |
| square padding, 256 and 320 grids | `preprocessing.py` — `--resize_mode pad`, `--target_size` |
| budget and schedule in optimiser steps | `train.py` — `schedule_unit="step"`, `max_steps`, `lr_t_max`, `patience_steps` |
| non-finite loss caught before the step | `train.py` — checked before `backward()`, batch skipped |
| memorisation sanity check | `tests/test_overfit_sanity.py` — 4 tests in the standard suite |
| preprocessing ceiling per grid | `src/preprocessing/resolution_ceiling.py` → `output/resolution_ceiling.csv` |
| lung-mask coverage ceiling | `src/preprocessing/lung_mask.py` → `output/lung_mask_coverage.json` |
| cross-validation over all 63 patients | `src/training/cross_validation.py` → `output/cv_folds.json` |
| seed ensemble, probabilities averaged | `src/evaluation/ensemble.py` → `output/ensemble_report.json` |
| seed ensemble, probabilities averaged | `output/experiments/baseline/ensemble_report.json` |
| baseline re-swept in original geometry | `output/experiments/baseline/threshold_rescore.json` |
| every run's artifacts | `output/experiments/{exp_name}/seed_{n}/` — 96 runs |
| every measured result, one row per arm | `output/all_experiment_results.csv` |
| Z spacing for 2.5D | uniform by construction — 1 mm isotropic resampling precedes stacking |
| per-model reporting | `benchmark_report.json`: parameters, time per epoch, GPU memory, inference time per volume |
| regression suite | `tests/` — 107 tests |

Two items are handled differently from their original phrasing, both deliberately.
`DiceMetric` is no longer used at all, so the question of `ignore_empty` does not
arise: metrics are computed directly, with empty-mask handling verified by the
artificial tests. And the abrupt jump in the Dice curve was a thresholding
artifact — early stopping now watches *soft* Dice, which rises smoothly from the
first epoch, with `hard@0.5` still reported alongside for comparison.
