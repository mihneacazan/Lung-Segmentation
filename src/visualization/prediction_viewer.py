"""
Interactive viewer for comparing a trained model's prediction against the
ground truth, one patient and one slice at a time.

    streamlit run src/visualization/prediction_viewer.py

Reads the checkpoints written under `output/experiments/`, runs inference on the
selected patient, and reconstructs the prediction into that patient's original
NIfTI geometry before drawing anything. That last part is the point: the
per-patient numbers in `benchmark_report.json` are computed in original geometry
against a ground truth read straight from the source archive, so a viewer working
in the 192x192 preprocessed space would show a different object than the one that
was scored. Here the Dice printed above the image is the same Dice as in the CSV.

Inference runs on whatever device is available; the models are around 1.6M
parameters and a single patient is a few hundred slices, so CPU is fine.
"""

import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.evaluation.metrics import (
    compute_asd_3d,
    compute_dice_3d,
    compute_hd95_3d,
    compute_precision_3d,
    compute_sensitivity_3d,
    count_false_positive_components,
    filter_predicted_components,
    reconstruct_patient_3d_volume,
)
from src.models.factory import build_model

st.set_page_config(page_title="Prediction Viewer", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

LUNG_WINDOW = (-1000.0, 400.0)
MEDIASTINUM_WINDOW = (-150.0, 250.0)


# ============================================================================
#  DISCOVERY
# ============================================================================

def find_experiments():
    """Returns {label: run_dir} for every checkpoint under output/experiments."""
    runs = {}
    pattern = os.path.join(OUTPUT_DIR, "experiments", "*", "seed_*", "best_model.pt")
    for ckpt in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(ckpt)
        exp = os.path.basename(os.path.dirname(run_dir))
        seed = os.path.basename(run_dir).replace("seed_", "")
        runs[f"{exp} (seed {seed})"] = run_dir
    return runs


@st.cache_data(show_spinner=False)
def load_run_metadata(run_dir):
    """Config, reported metrics and per-patient scores for one run."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    report = {}
    report_path = os.path.join(run_dir, "benchmark_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

    scores = {}
    csv_path = os.path.join(run_dir, "test_results_per_patient.csv")
    if os.path.exists(csv_path):
        import csv as csv_mod
        with open(csv_path) as f:
            for row in csv_mod.DictReader(f):
                scores[row["case_id"]] = float(row["dice_3d"])

    return config, report, scores


@st.cache_data(show_spinner=False)
def load_split(split):
    """Case ids belonging to one split."""
    with open(os.path.join(OUTPUT_DIR, "preprocessed", "index.json")) as f:
        return json.load(f)["splits"][split]


# ============================================================================
#  INFERENCE
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model(run_dir):
    """Builds the architecture recorded in config.json and loads its weights."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model_type"], in_channels=config["n_adjacent"],
                        out_channels=1).to(device)

    state = torch.load(os.path.join(run_dir, "best_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return model, device, config


@st.cache_data(show_spinner=False)
def predict_patient(run_dir, case_id):
    """
    Runs the model over every slice of one patient.

    Neighbour selection for the 2.5D models replicates `LungSliceDataset`
    exactly, edge clipping included: out-of-range neighbours repeat the edge
    slice rather than being zero-filled, because zero is a real intensity in this
    normalized space and means air.

    Returns:
        np.ndarray: (192, 192, D) probabilities, before thresholding.
    """
    model, device, config = load_model(run_dir)
    n_adjacent = config["n_adjacent"]
    half = n_adjacent // 2

    volumes = os.path.join(OUTPUT_DIR, "preprocessed", "volumes")
    img = np.load(os.path.join(volumes, f"{case_id}_img.npy"), mmap_mode="r")
    n_slices = img.shape[2]

    probs = np.zeros((img.shape[0], img.shape[1], n_slices), dtype=np.float32)
    progress = st.progress(0.0, text=f"Running inference on {case_id}...")

    with torch.no_grad():
        for start in range(0, n_slices, 16):
            stop = min(start + 16, n_slices)
            batch = []
            for s in range(start, stop):
                idx = [int(np.clip(s + o, 0, n_slices - 1))
                       for o in range(-half, half + 1)]
                batch.append(np.stack(
                    [np.asarray(img[:, :, k], dtype=np.float32) for k in idx]))

            tensor = torch.from_numpy(np.stack(batch)).to(device)
            out = torch.sigmoid(model(tensor)).cpu().numpy()[:, 0]
            probs[:, :, start:stop] = np.moveaxis(out, 0, -1)
            progress.progress(stop / n_slices, text=f"Running inference on {case_id}...")

    progress.empty()
    return probs


@st.cache_data(show_spinner=False)
def reconstruct(run_dir, case_id, threshold, postproc_fraction):
    """
    Brings the prediction back into the patient's original NIfTI geometry and
    scores it against the untouched ground truth.

    Only the overlap metrics are computed here. On a 512x512x271 volume the
    surface distances cost 44 s for HD95 and 38 s for ASD, and the connected
    component count another 19 s, against 8 s for the whole overlap set — too slow
    to sit between moving a slider and seeing an image. They are available on
    demand through `surface_metrics`, and for the reported threshold they are
    already in the per-patient CSV.

    Returns:
        tuple: (ct, ground_truth, prediction, metrics, components_removed)
    """
    probs = predict_patient(run_dir, case_id)

    with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
        metadata = json.load(f)

    pred = reconstruct_patient_3d_volume(probs, metadata, threshold=threshold,
                                         binarize=True)

    removed = 0
    if postproc_fraction > 0:
        pred, removed = filter_predicted_components(
            pred, min_fraction=postproc_fraction)

    ct = np.asanyarray(
        nib.load(resolve_nifti_path(f"./imagesTr/{case_id}.nii.gz")).dataobj)
    gt = (np.asanyarray(
        nib.load(resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")).dataobj)
        > 0.5).astype(np.uint8)

    metrics = {
        "dice_3d": compute_dice_3d(pred, gt),
        "sensitivity_3d": compute_sensitivity_3d(pred, gt),
        "precision_3d": compute_precision_3d(pred, gt),
    }
    return ct, gt, pred, metrics, removed


@st.cache_data(show_spinner="Computing surface distances (~1.5 min)...")
def surface_metrics(run_dir, case_id, threshold, postproc_fraction):
    """HD95, ASD and the false-positive component count, on request only."""
    _, gt, pred, _, _ = reconstruct(run_dir, case_id, threshold, postproc_fraction)
    with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
        spacing = tuple(json.load(f)["original_spacing"])
    return {
        "hd95_3d": compute_hd95_3d(pred, gt, spacing),
        "asd_3d": compute_asd_3d(pred, gt, spacing),
        "fp_components": count_false_positive_components(pred, gt),
    }


# ============================================================================
#  SLICE SELECTION
# ============================================================================

def slice_summary(gt, pred):
    """Per-slice voxel counts and Dice, used to drive the jump selectors."""
    axis = (0, 1)
    gt_counts = gt.sum(axis=axis)
    pred_counts = pred.sum(axis=axis)
    overlap = np.logical_and(gt, pred).sum(axis=axis)
    denominator = gt_counts + pred_counts
    dice = np.divide(2.0 * overlap, denominator,
                     out=np.zeros_like(denominator, dtype=float),
                     where=denominator > 0)
    return gt_counts, pred_counts, overlap, dice


def interesting_slices(gt_counts, pred_counts, overlap, dice):
    """
    Named jump targets. The failure cases are the ones worth reaching in one
    click: a slice where the tumour is large and the model saw nothing, and a
    slice where the model marked a lot of tissue that is not tumour.
    """
    targets = {}
    if gt_counts.max() > 0:
        targets["Largest tumour (ground truth)"] = int(gt_counts.argmax())
        matched = np.where(gt_counts > 0, dice, -1.0)
        if matched.max() > 0:
            targets["Best match"] = int(matched.argmax())
        missed = np.where(gt_counts > 0, gt_counts - overlap, -1)
        if missed.max() > 0:
            targets["Largest miss (false negative)"] = int(missed.argmax())
    false_positive = pred_counts - overlap
    if false_positive.max() > 0:
        targets["Largest false positive"] = int(false_positive.argmax())
    return targets


# ============================================================================
#  UI
# ============================================================================

st.title("🔬 Prediction Viewer")

experiments = find_experiments()
if not experiments:
    st.error(
        f"No checkpoint under {os.path.join(OUTPUT_DIR, 'experiments')}. "
        "Unpack `results_*.zip` there first.")
    st.stop()

st.sidebar.header("Model")
run_label = st.sidebar.selectbox("Experiment", list(experiments))
run_dir = experiments[run_label]
config, report, scores = load_run_metadata(run_dir)

st.sidebar.caption(
    f"{config['model_type']} · {config['loss_type']} · sampling {config['sampling']} "
    f"· n_adjacent {config['n_adjacent']} · lr {config['lr']}")

st.sidebar.header("Patient")
split = st.sidebar.radio("Split", ["test", "val"], horizontal=True)
cases = load_split(split)

# Sorting by score puts the failures at the top, which is where the interesting
# cases are; the reported number is only available for the split that was scored.
if scores and split == "test":
    cases = sorted(cases, key=lambda c: scores.get(c, 1.0))
    labels = {c: f"{c}  —  Dice {scores[c]:.3f}" if c in scores else c for c in cases}
else:
    cases = sorted(cases)
    labels = {c: c for c in cases}

case_id = st.sidebar.selectbox("Case", cases, format_func=lambda c: labels[c])

st.sidebar.header("Threshold and post-processing")
default_threshold = float(report.get("optimal_threshold", 0.5))
threshold = st.sidebar.slider("Binarisation threshold", 0.05, 0.99,
                              default_threshold, step=0.01)
st.sidebar.caption(f"Threshold chosen on validation for this run: {default_threshold}")

apply_postproc = st.sidebar.checkbox(
    "Filter small components", value=False,
    help="Drops connected components below a fraction of the largest one — "
         "the post-processing reported under the pp_ keys in the CSV.")
postproc_fraction = st.sidebar.slider(
    "Minimum fraction", 0.01, 0.50, float(config.get("postproc_min_fraction", 0.10)),
    step=0.01, disabled=not apply_postproc) if apply_postproc else 0.0

ct, gt, pred, metrics, removed = reconstruct(
    run_dir, case_id, threshold, postproc_fraction)

# --- Volume-level numbers -------------------------------------------------

reported = scores.get(case_id)
at_reported_threshold = (abs(threshold - default_threshold) < 1e-9
                         and postproc_fraction == 0)

columns = st.columns(4)
columns[0].metric(
    "Dice 3D", f"{metrics['dice_3d']:.4f}",
    delta=f"{metrics['dice_3d'] - reported:+.4f} vs reported"
    if reported is not None and at_reported_threshold else None)
columns[1].metric("Sensitivity", f"{metrics['sensitivity_3d']:.4f}")
columns[2].metric("Precision", f"{metrics['precision_3d']:.4f}")
columns[3].metric("Predicted / true volume",
                  f"{pred.sum() / max(gt.sum(), 1):.2f}x")

if removed:
    st.caption(f"Post-processing: {removed} components removed.")

if st.checkbox("Compute HD95, ASD and false-positive components "
               "(~1.5 min per patient)"):
    surface = surface_metrics(run_dir, case_id, threshold, postproc_fraction)
    surface_columns = st.columns(3)
    surface_columns[0].metric("HD95 (mm)", f"{surface['hd95_3d']:.1f}")
    surface_columns[1].metric("ASD (mm)", f"{surface['asd_3d']:.1f}")
    surface_columns[2].metric("FP components", int(surface["fp_components"]))
elif at_reported_threshold and case_id in scores:
    st.caption(
        "HD95 and the false-positive count for the reported threshold are already in "
        f"`{os.path.basename(run_dir)}/test_results_per_patient.csv`; tick the box "
        "above only if you have moved the threshold.")

# --- Slice navigation -----------------------------------------------------

gt_counts, pred_counts, overlap, slice_dice = slice_summary(gt, pred)
targets = interesting_slices(gt_counts, pred_counts, overlap, slice_dice)

navigation, options = st.columns([3, 1])

with options:
    st.subheader("Display")
    window_name = st.radio("HU window",
                           ["Lung", "Mediastinum", "Raw"], index=0)
    hu_min, hu_max = {
        "Lung": LUNG_WINDOW,
        "Mediastinum": MEDIASTINUM_WINDOW,
        "Raw": (float(ct.min()), float(ct.max())),
    }[window_name]

    show_gt = st.checkbox("Ground truth (red)", value=True)
    show_pred = st.checkbox("Prediction (blue)", value=True)
    fill = st.checkbox("Fill, not just outline", value=False)

    st.divider()
    st.write("**Jump to slice**")
    if targets:
        target_name = st.selectbox("Notable slices", list(targets))
        if st.button("Go there", use_container_width=True):
            st.session_state.slice_index = targets[target_name]
    else:
        st.caption("No tumour and no prediction in this volume.")

    positive = np.where(gt_counts > 0)[0]
    if len(positive):
        st.caption(f"Ground truth in slices {positive.min()}–{positive.max()} "
                   f"({len(positive)} slices)")
    predicted = np.where(pred_counts > 0)[0]
    if len(predicted):
        st.caption(f"Prediction in slices {predicted.min()}–{predicted.max()} "
                   f"({len(predicted)} slices)")

with navigation:
    # Streamlit rejects a slider given both a value and a key already present in
    # session state, and the jump buttons need to write into that key. So the
    # default is seeded here instead, and reseeded whenever the patient changes —
    # otherwise a slice index from a 337-slice volume survives into a 271-slice
    # one and lands out of range.
    default_slice = (int(gt_counts.argmax()) if gt_counts.max() > 0
                     else ct.shape[2] // 2)
    if st.session_state.get("viewer_case") != (run_dir, case_id):
        st.session_state.viewer_case = (run_dir, case_id)
        st.session_state.slice_index = default_slice
    st.session_state.slice_index = int(
        np.clip(st.session_state.get("slice_index", default_slice),
                0, ct.shape[2] - 1))

    slice_index = st.slider("Slice", 0, ct.shape[2] - 1, key="slice_index")

    ct_slice = np.clip(ct[:, :, slice_index], hu_min, hu_max)
    gt_slice = gt[:, :, slice_index]
    pred_slice = pred[:, :, slice_index]

    n_gt, n_pred = int(gt_slice.sum()), int(pred_slice.sum())
    n_overlap = int(np.logical_and(gt_slice, pred_slice).sum())
    st.markdown(
        f"**Slice {slice_index}** · ground truth `{n_gt}` px · prediction `{n_pred}` px · "
        f"overlap `{n_overlap}` px · 2D Dice `{slice_dice[slice_index]:.3f}`")

    figure, axis = plt.subplots(figsize=(8, 8))
    figure.patch.set_facecolor("#0e1117")
    axis.imshow(ct_slice.T, cmap="gray", origin="lower")

    if show_gt and n_gt:
        if fill:
            axis.imshow(np.ma.masked_where(gt_slice == 0, gt_slice).T,
                        cmap="Reds", origin="lower", alpha=0.35, vmin=0, vmax=1)
        axis.contour(gt_slice.T, colors="#ff3b3b", linewidths=1.4,
                     origin="lower", levels=[0.5])
    if show_pred and n_pred:
        if fill:
            axis.imshow(np.ma.masked_where(pred_slice == 0, pred_slice).T,
                        cmap="Blues", origin="lower", alpha=0.35, vmin=0, vmax=1)
        axis.contour(pred_slice.T, colors="#00b4ff", linewidths=1.4,
                     origin="lower", levels=[0.5])

    axis.axis("off")
    plt.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

# --- Where the two masks live along Z -------------------------------------

with st.expander("Distribution along Z", expanded=True):
    figure, axis = plt.subplots(figsize=(12, 2.6))
    z = np.arange(len(gt_counts))
    axis.fill_between(z, gt_counts, color="#ff3b3b", alpha=0.55, label="ground truth")
    axis.fill_between(z, pred_counts, color="#00b4ff", alpha=0.45, label="prediction")
    axis.axvline(slice_index, color="#ffffff", linewidth=1.0, linestyle="--")
    axis.set_xlabel("slice")
    axis.set_ylabel("pixels")
    axis.legend(loc="upper right")
    st.pyplot(figure)
    plt.close(figure)

    st.caption(
        "If the two areas do not overlap at all, the model is segmenting something "
        "other than the tumour — a Dice of 0 with a non-zero false-positive count "
        "means it is predicting, just in the wrong place.")
