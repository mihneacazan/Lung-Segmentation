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
| regression suite | `tests/` — 107 tests |

Two items are handled differently from their original phrasing, both deliberately.
`DiceMetric` is no longer used at all, so the question of `ignore_empty` does not
arise: metrics are computed directly, with empty-mask handling verified by the
artificial tests. And the abrupt jump in the Dice curve was a thresholding
artifact — early stopping now watches *soft* Dice, which rises smoothly from the
first epoch, with `hard@0.5` still reported alongside for comparison.
