# Exploratory data analysis

Audits the raw Decathlon Task06_Lung archive before any of it is preprocessed:
geometry, intensity, mask topology, and the distortion a naive resize would
introduce. Its job is to establish what the data actually is, and to justify the
choices the preprocessing pipeline then makes.

```bash
python -m src.eda.eda_report
```

Writes `output/eda_statistics.csv` (54 columns × 63 patients),
`output/eda_report.md`, and five figures under `output/eda_figures/`.

---

## What it checks

**Integrity** — image and mask exist, shapes match, affines match, mask values
are binary, no NaN or Inf, no empty volumes, and no duplicate volumes by MD5.
All 63 patients pass; there are no duplicates.

**Geometry** — orientation, per-axis spacing, voxel volume. Two findings drive
the whole preprocessing design:

- **Every volume is stored LAS.** Reorienting to canonical RAS therefore flips
  the left–right axis on all 63 patients. This is the transform whose inverse
  must be applied when a prediction is put back into the source geometry —
  omitting it silently mirrors every prediction and reports Dice 0.
- **Spacing varies widely between patients**: 0.598–0.977 mm in-plane, and
  0.625–2.500 mm along Z, a four-fold range. A model trained without resampling
  would see the same lesion at different apparent sizes depending on the scanner
  protocol, so resampling to 1 mm isotropic is not optional.

**Mask topology** — connected components in 3D, size of the largest component,
tumour position along Z. **24 of 63 patients have more than one component**
(up to 14), so "the tumour" is not always a single blob, and a post-processing
rule that keeps only the largest component will cost recall on those cases.

**Class balance** — **9.38% of slices contain tumour** (1657 of 17,657). This is
the real rate that validation and test are evaluated at, and the number any
sampling strategy has to be judged against.

**Tumour size distribution** — 738 mm³ to 370,384 mm³, median 5220 mm³: a
500-fold range. This is what the small / medium / large stratification in
`create_split.py` and the stratified metric report are built on.

---

## Preprocessing distortion analysis

The EDA also measures what a **naive** 512 → 192 resize would cost, by resizing
masks with nearest-neighbour interpolation and comparing against the original:
**9 positive slices disappear entirely** across 5 patients, because no output
pixel centre lands inside a small lesion.

This is the measurement that motivated resizing masks with linear interpolation
followed by a 0.5 threshold in `src/preprocessing/preprocessing.py`, and the
`preprocessing_dice_ceiling.png` figure records the resulting accuracy ceiling
per patient.

This figure quantifies the *problem* — what a naive resize would cost — and is
deliberately not re-run against the pipeline that solves it. The corresponding
measurement for the pipeline as built is the round-trip Dice column in
`output/preprocessing_qc.csv`, which reports what the full forward-and-inverse
chain actually preserves, per patient.

---

## Implementation notes

**Header-only scanning where possible.** Shape and spacing come from the NIfTI
header via `nibabel.load()` without touching voxel data, which takes under 0.01 s
per file.

**`np.asanyarray(img.dataobj)` rather than `img.get_fdata()`.** `get_fdata()`
promotes the native int16 CT array to float64, quadrupling memory for volumes
that are already hundreds of megabytes. `dataobj` keeps the native dtype.

---

## Figures

| File | Content |
|---|---|
| `tumor_volume_distribution.png` | Histogram of tumour volumes across patients |
| `slices_comparison.png` | Total vs. positive slices per patient |
| `sample_overlay.png` | CT slice with ground-truth mask overlaid |
| `connected_components_distribution.png` | How many patients have fragmented masks |
| `preprocessing_dice_ceiling.png` | Per-patient accuracy ceiling from resizing |
