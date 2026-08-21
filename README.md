# Lung Tumor Segmentation on CT — Medical Segmentation Decathlon, Task 06

An end-to-end deep learning pipeline for segmenting lung tumours in CT volumes,
built so that every reported number is measured in the patient's original NIfTI
geometry against the untouched ground-truth mask.

The dataset is 63 annotated CT volumes, split 44 / 9 / 10 by patient. Tumours
occupy well under 1% of a volume, which is what makes the task hard and what most
of the design decisions below are responding to.

**Best configuration: 2D U-Net, DiceCE loss, every slice, anatomically valid
augmentation — Dice 0.4853 ± 0.0153 over three seeds.** 21 configurations
were compared under identical conditions. The two effects that mattered were the
slice sampling strategy (+0.18 Dice) and the augmentation policy (+0.22); the
choice of architecture trailed far behind both, and at 10 test patients the
alternatives to the baseline U-Net were not statistically separable from it.

---

## Data

The raw scans are not in this repository — 31 GB of NIfTI volumes, individually
past GitHub's file size limit. Only `archive/dataset.json`, the manifest listing
the 63 training cases and their labels, is committed so the expected layout is
visible without downloading anything.

Task06_Lung is available from the [Medical Segmentation
Decathlon](http://medicaldecathlon.com/) directly, or as a
[Kaggle mirror](https://www.kaggle.com/datasets/vivekprajapati2048/medical-segmentation-decathlon-lung)
(the copy this project was developed against). Extract it into `archive/` at the
repository root:

```
archive/
  dataset.json          already present
  imagesTr/             63 training volumes
  labelsTr/             63 tumour masks
  imagesTs/             test volumes (unlabelled, unused here)
```

The Decathlon archive circulates in several layouts — plain `.nii.gz`,
decompressed `.nii`, and files nested one level deep inside a directory named
after themselves. `resolve_nifti_path` in [src/config.py](src/config.py) tries
each in turn, so any of them works without renaming files. On Kaggle the dataset
directory is located automatically by searching for `dataset.json`.

---

## Quick start

```bash
pip install -r requirements.txt

python -m src.eda.eda_report                  # dataset audit and statistics
python -m src.preprocessing.create_split      # deterministic 44 / 9 / 10 patient split
python -m src.preprocessing.preprocessing     # ~40 min, includes geometry QC
python -m src.visualization.overlay_check     # visual image/mask alignment check
python -m pytest tests/ -q                    # 107 regression tests

# the winning configuration
python -m src.training.train --exp_name baseline --loss_type dice_ce \
    --sampling all --epochs 100 --lr_t_max 50 --patience 20 --seeds 42,43,44
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the full results, the reasoning behind
each configuration, and a chronological journal of what was tried.

---

## Pipeline

### 1. Split (`src/preprocessing/create_split.py`)

44 train / 9 validation / 10 test, **at patient level**, stratified by tumour
volume, positive-slice count, and size category (small / medium / large). The
split is deterministic — seeded once and written to `output/patient_split.json`,
so every experiment trains and is scored on exactly the same patients. **The test
set is never used for model selection**, including threshold choice.

### 2. Preprocessing (`src/preprocessing/preprocessing.py`)

Per patient, in this order:

| Step | Detail |
|---|---|
| Integrity checks | shape, affine, orientation, label domain, NaN/Inf, empty volumes |
| Canonical reorientation | to RAS, **recording the exact transform for inversion** |
| Resampling | 1.0 mm isotropic — trilinear for CT, nearest for masks |
| Body crop | thresholded in raw HU (> −500), largest connected component only |
| HU windowing | clip to [−1000, +400], normalize to [0, 1] |
| Slice resize | 192 × 192 per axial slice |

Output is **one `.npy` volume per patient**, not one file per slice:

```
output/preprocessed/
  volumes/lung_001_img.npy      float16 (192, 192, D)
  volumes/lung_001_lbl.npy      uint8   (192, 192, D)
  index.json                    splits, per-case positive/body slice indices
output/metadata/lung_001.json   everything needed to invert the pipeline
output/preprocessing_qc.csv     per-case QC measurements
```

Storing whole volumes is what makes positive/negative sampling a *runtime* choice
rather than something frozen on disk, giving a 2.5D model access to a slice's
Z-neighbours.

### 3. Geometry and its inverse (`src/geometry.py`)

Every forward step above is inverted, in reverse order, to put a prediction back
into the source NIfTI geometry:

```
slice resize  →  body crop  →  1 mm resampling  →  canonical reorientation
     ↑               ↑                ↑                        ↑
   invert         invert           invert                   invert
```

All 63 volumes in this dataset are stored
**LAS**, so reorienting to RAS flips the left–right axis. Measured on real
patients, reconstructing *with* the inverse flip and *without* it:

| Patient | With the inverse flip | Without it |
|---|---|---|
| lung_053 | **0.9848** | 0.0000 |
| lung_022 | **0.9519** | 0.0000 |
| lung_041 | **0.9762** | 0.0000 |

Nothing crashes when the inverse is missing. The prediction is simply mirrored
onto the opposite lung, overlaps nothing, and the evaluation reports zero — which
is indistinguishable from a model that failed to learn.

Preprocessing measures this round-trip for every patient and writes it to
`output/preprocessing_qc.csv`. It is the accuracy ceiling no model can exceed.
Use `--skip_qc` to skip it when the geometry is trusted.

### 4. Sampling and augmentation (`src/training/dataset.py`)

| `--sampling` | Training slices |
|---|---|
| `balanced` | all positives + an equal number of negatives, **redrawn every epoch** |
| `all` | every slice, at the real ~9% positive rate |
| `hard_negatives` | positives + negatives concentrated near the tumour in Z |

**Validation and test always use every slice.** Balancing them would raise the
positive rate from the real ~9% to ~33% and make the reported Dice
unrepresentative of deployment.

| `--augment` | Transforms |
|---|---|
| `none` | — |
| `standard` | horizontal flip, vertical flip, 90° rotations |
| `anatomic` | rotation ±15°, translation ±8%, scale ±10%, gamma 0.8–1.25, Gaussian noise |

`standard` is the naive computer-vision recipe and is **anatomically invalid** for
axial chest CT: a vertical flip puts the spine in front of the sternum, and a
quarter turn lays the patient on their side. No scanner produces such an image.
`anatomic` is restricted to transforms corresponding to real acquisition
variation. The two exist to quantify the difference.

| `--crop` | Training input |
|---|---|
| `none` | the full 192 × 192 slice |
| `tumor` | a `--crop_size` window around the lesion, jittered, at 4× the positive-pixel density |

Cropping is **training-only, and deliberately so**: at inference the tumour
location is the unknown being predicted, so centring a window on it would feed the
label to the model.

A window-trained model still has to be evaluated somehow, and handing it a full
slice changes its field of view — which costs more than the crop itself is worth.
`--sw_roi` covers each slice with overlapping windows and blends the predictions
back into a full-size map, restoring the field of view the model trained under. It
consults no ground truth: the grid is fixed, every pixel is covered, and it runs
identically on an unannotated patient. This is the standard inference path for
patch-trained segmentation networks.

```bash
python -m src.evaluation.evaluate \
    --checkpoint output/experiments/crop96_unet/seed_42/best_model.pt \
    --split test --sw_roi 96
```

### 5. Training (`src/training/train.py`)

Mixed precision, cosine LR schedule, full checkpointing with resume, per-epoch
history CSV, and multi-seed runs.

Early stopping monitors **soft Dice**, not hard Dice at a fixed 0.5 threshold. A
freshly initialised network outputs probabilities below 0.5 everywhere, so hard
Dice reads exactly 0.0 for many epochs and then jumps the moment probabilities
cross the threshold. Watching that flat line, early stopping fires while the
network is still learning, and the later jump looks like a breakthrough when it
is only a thresholding artifact. Soft Dice moves smoothly from epoch one. A
`--min_epochs` floor guards the warm-up on top of that.

### 6. Evaluation (`src/evaluation/`)

`metrics.py` holds the metric implementations; `evaluate.py` is the standalone
CLI and is the same code training calls, so an experiment and a re-evaluation
cannot silently disagree.

```bash
# choose a threshold — validation only, refused on test
python -m src.evaluation.evaluate \
    --checkpoint output/experiments/baseline/seed_42/best_model.pt \
    --split val --sweep_threshold

# apply it to the held-out test set
python -m src.evaluation.evaluate \
    --checkpoint output/experiments/baseline/seed_42/best_model.pt \
    --split test --threshold 0.45 --save_nifti
```

Reported per patient: Dice, IoU, sensitivity, precision, specificity, HD95 (mm),
ASD (mm), false-positive connected components, failure flag, inference time.
Reported in aggregate: macro-average, micro-average, and a breakdown by tumour
size. `--save_nifti` writes predictions carrying the patient's original affine,
so they overlay on the source CT in any viewer.

---

## Experiments

All share the same split, the same evaluation code, the same checkpoint selection
rule, and the same code version. Each changes **one variable** against the
baseline. Full results, per-experiment reasoning and the journal are in
[EXPERIMENTS.md](EXPERIMENTS.md).

```bash
COMMON="--sampling all --epochs 100 --lr_t_max 50 --patience 20"

# baseline — DiceCE + every slice + anatomic augmentation
python -m src.training.train --exp_name baseline --loss_type dice_ce \
    $COMMON --seeds 42,43,44

# 1. balanced sampling            (isolates the effect of class imbalance)
python -m src.training.train --exp_name dicece_balanced --loss_type dice_ce \
    --sampling balanced --epochs 100 --lr_t_max 50 --patience 20

# 2. DiceFocal
python -m src.training.train --exp_name dice_focal --loss_type dice_focal $COMMON

# 3. Tversky / Focal Tversky      (beta > alpha penalises misses harder)
python -m src.training.train --exp_name tversky --loss_type tversky \
    --tversky_alpha 0.3 --tversky_beta 0.7 $COMMON
python -m src.training.train --exp_name focal_tversky --loss_type focal_tversky \
    --tversky_alpha 0.3 --tversky_beta 0.7 $COMMON

# 4. No augmentation
python -m src.training.train --exp_name no_augment --augment none \
    --loss_type dice_ce $COMMON

# 5. Hard negative sampling       (negatives drawn from beside the tumour,
#    same count as balanced, so only their selection differs)
python -m src.training.train --exp_name hard_negatives_unet --loss_type dice_ce \
    --sampling hard_negatives --epochs 100 --lr_t_max 50 --patience 20

# 6. Tumour-centred crop          (training only — validation and test stay
#    full-size, since the tumour location is what is being predicted)
python -m src.training.train --exp_name crop96_unet --loss_type dice_ce \
    --crop tumor --crop_size 96 $COMMON

# Architectures
python -m src.training.train --exp_name attention_unet \
    --model_type attention_unet $COMMON
python -m src.training.train --exp_name segresnet \
    --model_type segresnet --lr 3e-4 $COMMON      # 1e-3 diverges to NaN

# 2.5D: 3 consecutive slices as input channels
python -m src.training.train --exp_name unet_25d --n_adjacent 3 $COMMON
```

The headline result: `all` sampling and anatomic augmentation are the two large
effects (+0.18 and +0.22 Dice); architecture choice trails far behind, and at ten
test patients 2.5D and SegResNet are statistically indistinguishable from the
baseline. Within sampling, it is the *amount* of negative anatomy that counts:
`balanced` and `hard_negatives` keep the same number of negatives and differ only
in where they are drawn from, and they land within noise of each other, while
`all` beats both by roughly 0.2. Training on tumour-centred windows does not help
either, once inference is given the same field of view — and the size of that
qualifier is itself a finding: changing field of view between training and
inference costs the baseline more than half its Dice.

The baseline was run on three seeds (42, 43, 44) to measure that
run-to-run noise; every other configuration above changes one variable against
it on a single seed. A handful of follow-up runs — Tversky and Focal Tversky at a
gentler `beta`, a U-Net at SegResNet's learning rate, and Attention U-Net under
an earlier epoch budget — exist to isolate a specific question raised by these
results rather than to test a new idea; they're covered in EXPERIMENTS.md rather
than repeated here.

Each run writes to `output/experiments/{exp_name}/seed_{seed}/`:
`config.json`, `best_model.pt`, `checkpoint.pt`, `training_history.csv`,
`threshold_sweep.json`, `test_results_per_patient.csv`, `benchmark_report.json`.
`baseline/`, having three seeds, additionally has a `multi_seed_summary.json` one
level up.

---

## Tests

```bash
python -m pytest tests/ -v
```

65 tests, written against bugs that produced plausible-looking wrong numbers
rather than crashes:

- `test_geometry.py` — orientation inversion across all 48 permutation/flip
  combinations; full preprocess-and-reconstruct round-trip on an asymmetric
  phantom in four orientations; proof the phantom is asymmetric enough for a
  mirror to be detectable; body crop excludes air and rejects the scanner table.
- `test_metrics.py` — 6 scenarios (perfect, disjoint, both empty,
  false alarm, missed tumour, partial overlap); threshold sweep recovers a
  planted optimum, refuses to run without ground truth, and is not trivially flat.
- `test_dataset.py` — `balanced` and `all` produce genuinely different training
  sets; validation keeps the real class distribution; negatives are redrawn each
  epoch; 2.5D replicates edges rather than zero-padding and never crosses patient
  boundaries; `standard` and `anatomic` are different policies and `anatomic`
  never flips.
- `test_training_smoke.py` — a complete experiment on a tiny synthetic dataset,
  on CPU, in seconds: every architecture trains, every loss trains, every
  sampling mode trains, the threshold sweep produces a real curve, a checkpoint
  resumes rather than restarts, and a perfect prediction round-trips to Dice 1.0
  through the reconstruction path.

---

## Repository layout

```
src/
  config.py                  paths and Kaggle auto-detection
  geometry.py                reorientation, resampling, crop, and their inverses
  eda/eda_report.py          dataset audit, topology, distortion analysis
  preprocessing/
    create_split.py          stratified patient-level split
    preprocessing.py         forward pipeline + per-case QC
  training/
    dataset.py               sampling, augmentation, 2D and 2.5D
    losses.py                loss factory
    train.py                 training loop, early stopping, benchmarking
  models/
    factory.py               --model_type dispatch
    unet_2d.py  attention_unet.py  segresnet.py
  evaluation/
    metrics.py               3D metrics, threshold sweep, reporting
    evaluate.py              standalone evaluation CLI
  visualization/
    overlay_check.py         image/mask alignment overlays
    visualizer.py            interactive raw-CT browser (Streamlit)
    prediction_viewer.py     interactive checkpoint/prediction browser (Streamlit)
tests/                       regression suite
output/                      everything the pipeline produces — see below
```

Each package under `src/` carries its own README with the design rationale for
that stage.

`output/` ships pre-populated with the results this report describes, so the
experiments and figures can be inspected without re-running anything:

```
output/
  eda_figures/, eda_report.md, eda_statistics.csv   EDA (section "Pipeline")
  patient_split.json                                the 44/9/10 split
  metadata/                                         per-patient data for inverting the pipeline
  preprocessing_qc.csv                               per-case QC (round-trip Dice, etc.)
  experiments/{exp_name}/seed_{seed}/                one directory per run — see EXPERIMENTS.md
```

`output/preprocessed/` (the `.npy` volumes training reads from) is regenerated
by `python -m src.preprocessing.preprocessing` and is not included, since it is
fully determined by the raw archive and the code above and adds no information
of its own.
