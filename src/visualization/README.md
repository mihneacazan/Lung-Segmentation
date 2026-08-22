# Visualization

Three tools with different jobs: one runs as part of the pipeline and writes
figures to disk, the other two are interactive browsers — one for the raw scans,
one for what a trained model does to them.

| File | Type | Job |
|---|---|---|
| `overlay_check.py` | Script, part of the pipeline | Verifies image/mask alignment after preprocessing |
| `visualizer.py` | Streamlit app, manual use | Browses raw NIfTI volumes with clinical HU presets |
| `prediction_viewer.py` | Streamlit app, manual use | Compares a checkpoint's prediction against ground truth |

---

## overlay_check.py

Samples slices from the preprocessed volumes and draws the CT slice with its mask
overlaid, one figure per split:

```bash
python -m src.visualization.overlay_check
```

Writes `output/eda_figures/overlay_train.png`, `overlay_val.png`,
`overlay_test.png`.

**Why this exists as a pipeline step, not an afterthought.** A misaligned mask —
off by a flip, a transposed axis, or a resampling that moved image and label
differently — produces a preprocessed dataset that looks entirely normal in every
numeric check, and trains a model that can never work. The numeric guard against
this is the round-trip Dice in `output/preprocessing_qc.csv`; this is the visual
one, and the two fail in different ways, which is why both are kept.

Slices are sampled across several patients rather than all from one, so a single
unlucky volume cannot make the whole split look fine.

---

## visualizer.py

An interactive Streamlit browser for the **raw** archive — it reads NIfTI files
directly and does not depend on anything the pipeline produces:

```bash
streamlit run src/visualization/visualizer.py
```

- Z-axis slider through the axial stack, with ground-truth mask overlay.
- Hounsfield window presets: lung [−1000, 400], mediastinum [−150, 250], and the
  unclipped raw range — the same presets a radiologist switches between on a PACS
  workstation.
- Jump-to-tumour selector built from the positive slice indices, so inspecting a
  lesion does not mean scrolling through 300 empty slices.

Useful for sanity-checking a specific patient by eye — for example the test cases
that score Dice 0.0, where the question is whether the anatomy is unusual or the
prediction simply missed.

---

## prediction_viewer.py

```bash
streamlit run src/visualization/prediction_viewer.py
```

Loads any checkpoint under `output/experiments/`, runs inference on the selected
patient, and draws the CT with ground truth in red and the prediction in blue.
The repository ships with every experiment's checkpoint already in place, so no
setup beyond the raw archive is needed before running this.

**The prediction is reconstructed into the patient's original NIfTI geometry
before anything is drawn.** Working in the 192×192 preprocessed space would be
far cheaper, but the numbers in `benchmark_report.json` are computed in original
geometry against a ground truth read straight from the source archive — so a
viewer in preprocessed space would show a different object than the one that was
scored, and a disagreement between the two would be impossible to interpret. The
Dice above the image is the same Dice as in the CSV, and the app shows the delta
against the reported value to prove it.

- **Cases are sorted worst-first** when the split has a scored CSV, so the
  failures are at the top of the selector rather than buried alphabetically.
- **Jump targets** — largest tumour, best match, largest miss, largest false
  positive. Reaching the informative slice should not require dragging a slider
  through 300 of them.
- **Z-axis profile** below the image plots ground-truth and predicted voxel
  counts per slice. This is the fastest way to read a Dice of 0.0: if the two
  areas do not overlap at all, the model is segmenting something else. `lung_036`
  is the example — ground truth in slices 88–101, prediction in 132–196, zero
  shared slices under every one of the eight checkpoints.
- **Threshold slider** starts at the value the run chose on validation, so moving
  it shows what that choice bought.
