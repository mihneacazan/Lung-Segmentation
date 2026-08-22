# Preprocessing

Turns the raw Decathlon archive into the patient split and the preprocessed
volumes that every experiment trains on.

| File | Role |
|---|---|
| `create_split.py` | Fixed, stratified 44 / 9 / 10 patient split |
| `preprocessing.py` | Geometry pipeline, per-patient volume export, per-case QC |

The geometric transforms themselves live in `src/geometry.py`, not here.
`preprocessing.py` only orchestrates them over the 63 patients, so that the
transforms and their inverses can be tested in isolation
(`tests/test_geometry.py`).

---

## create_split.py

**44 train / 9 validation / 10 test, at patient level**, stratified by tumour
volume, positive-slice count, and size category (small < 1000 mm³, medium
1000–5000 mm³, large > 5000 mm³). Written to `output/patient_split.json` with a
fixed seed, and asserted to have zero overlap between splits.

Splitting at slice level instead would put slices from the same patient in both
train and validation. Neighbouring axial slices of one scan are nearly identical
images, so validation would be scoring the model on data it had effectively
already seen, and the reported number would not describe a new patient at all.

**The test split is never used for model selection**, including the choice of
binarization threshold — that is swept on validation only.

---

## preprocessing.py

Per patient, in this order:

| Step | Detail |
|---|---|
| Integrity checks | shape, affine, orientation, label domain, NaN/Inf, empty volumes |
| Canonical reorientation | to RAS, **recording the exact transform so it can be inverted** |
| Resampling | 1.0 mm isotropic — trilinear for CT, linear + threshold for masks |
| Body crop | thresholded in raw HU (> −500), largest connected component only |
| HU windowing | clip to [−1000, +400], normalize to [0, 1] |
| Slice resize | 192 × 192 per axial slice |

### Why HU windowing at [−1000, +400]

CT records attenuation from −1000 HU (air) to roughly +3000 HU (dense bone and
metal implants). Lung parenchyma and tumour tissue both sit inside
[−1000, +400]. Clipping the rest removes signal the model has no use for, and
rescaling to [0, 1] puts the input in the range the network expects.

### Why masks are resized with linear interpolation, not nearest

Nearest-neighbour resizing of a small tumour on a 512 → 192 downscale can drop it
entirely: if no output pixel centre lands inside the lesion, the slice silently
becomes negative. Resizing with `order=1` and thresholding at 0.5 afterwards
preserves those lesions. Measured across all 63 patients, this reduced the count
of lost positive slices to 10 in total.

### Output layout

```
output/preprocessed/
  volumes/lung_001_img.npy      float16 (192, 192, D)
  volumes/lung_001_lbl.npy      uint8   (192, 192, D)
  index.json                    splits, per-case positive/body slice indices
output/metadata/lung_001.json   everything needed to invert the pipeline
output/preprocessing_qc.csv     per-case QC measurements
```

**One file per patient, not one per slice.** Whole volumes are what make
positive/negative sampling a *runtime* choice rather than something frozen on
disk, giving a 2.5D model access to a slice's Z-neighbours.

### QC

For every patient, the pipeline runs the full forward transform and its inverse
on the ground-truth mask, then measures the Dice between the round-tripped mask
and the original. This is the accuracy ceiling no model trained on these volumes
can exceed, and it is the check that catches a flipped or permuted axis — a bug
that otherwise stays invisible until evaluation reports zero and looks like a bad
model.

Measured on the current dataset: minimum 0.8602, mean 0.9386, no patient below
0.80. Skip with `--skip_qc` when the geometry is already trusted.

---

## Running

```bash
python -m src.preprocessing.create_split     # writes output/patient_split.json
python -m src.preprocessing.preprocessing    # ~40 min with QC, ~8 min without
```

`create_split.py` reads `output/eda_statistics.csv`, so the EDA has to run first.
