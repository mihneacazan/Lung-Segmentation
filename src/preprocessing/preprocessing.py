"""
Anatomically-Correct Preprocessing Pipeline for Lung Tumor Segmentation.

Transforms raw NIfTI CT volumes and their ground-truth masks into preprocessed
2D axial slice stacks, stored one .npy volume per patient.

Pipeline per patient:
    1. Load NIfTI image + label, and run per-case integrity checks.
    2. Reorient to canonical RAS, recording the exact transform so it can be
       inverted at evaluation time.
    3. Resample to 1.0mm isotropic spacing (trilinear for CT, nearest for masks).
    4. Crop to the body bounding box, in Hounsfield Units.
    5. Apply HU lung windowing [-1000, +400] and normalize to [0, 1].
    6. Resize each axial slice to 192x192.
    7. Save the slice stack, reconstruction metadata, and a QC record that
       measures how much the preprocessing itself costs.

Storage layout
--------------
One .npy per patient rather than one per slice. This matters for three reasons:
    - Positive/negative slice sampling becomes a runtime choice instead of being
      baked into the files on disk, so `--sampling balanced` and `--sampling all`
      can share a single preprocessed dataset.
    - A 2.5D model needs a slice's Z-neighbours, which only exist if the whole
      volume is stored.
    - 63 files instead of ~35,000 is dramatically faster to upload to and read
      from Kaggle.

Quality control
---------------
Every patient's label mask is pushed through the full forward pipeline and then
reconstructed back into original NIfTI geometry. The resulting Dice is the
ceiling no model can beat, and any geometry bug (a missing flip, a bad crop
inverse) drives it towards zero and is caught here rather than being mistaken
for poor model performance.

Usage:
    python -m src.preprocessing.preprocessing
"""

import os
import json
import shutil
import csv
import numpy as np
import nibabel as nib
from tqdm import tqdm

from src.config import resolve_nifti_path, OUTPUT_DIR, DATA_DIR
from src.preprocessing.create_split import load_patient_split
from src.geometry import (
    TARGET_SPACING,
    get_ornt,
    reorient_to_canonical,
    permute_spacing,
    resample_volume,
    resize_slice_2d,
    crop_body_3d,
    apply_crop,
    reconstruct_to_original_geometry,
)


# HU windowing range for lung CT imaging
HU_MIN = -1000.0   # Air / lung parenchyma lower bound
HU_MAX = 400.0     # Soft tissue / tumor upper bound

# Target 2D slice size after cropping and resizing
TARGET_SLICE_SIZE = (192, 192)

# A slice counts as containing body (rather than pure air) above this mean
# normalized intensity. Used to mark which negative slices are worth sampling.
BODY_SLICE_MIN_MEAN = 0.05


def get_preprocessed_dir():
    """Returns the base path for preprocessed output: output/preprocessed/"""
    return os.path.join(OUTPUT_DIR, "preprocessed")


def get_volumes_dir():
    """Returns the path holding one .npy slice stack per patient."""
    return os.path.join(get_preprocessed_dir(), "volumes")


def get_metadata_dir():
    """Returns the path for per-patient reconstruction metadata: output/metadata/"""
    return os.path.join(OUTPUT_DIR, "metadata")


def clean_preprocessed_directory():
    """
    Removes output/preprocessed/ before regenerating.

    Without this, changing the patient split and re-running would leave slices
    from a patient who moved between splits sitting in their old folder, and the
    model would be evaluated on data it had already trained on.
    """
    preprocessed_dir = get_preprocessed_dir()
    if os.path.exists(preprocessed_dir):
        print(f"[CLEAN] Removing stale preprocessed directory: {preprocessed_dir}")
        shutil.rmtree(preprocessed_dir)
    print("[CLEAN] Preprocessed directory is clean.")


def apply_hu_windowing(img_hu):
    """
    Clips a CT volume to the lung-relevant Hounsfield window and normalizes to
    [0, 1].

    CT density reference points:
        -1000 HU = air (inside lungs and outside the body)
            0 HU = water
         +400 HU = soft tissue and tumor
        +1000 HU = bone (not relevant here)

    Args:
        img_hu (np.ndarray): Raw CT volume in Hounsfield Units.

    Returns:
        np.ndarray: float32 volume with values in [0.0, 1.0].
    """
    windowed = np.clip(img_hu, HU_MIN, HU_MAX)
    normalized = (windowed - HU_MIN) / (HU_MAX - HU_MIN)
    return normalized.astype(np.float32)


def verify_case(case_id, img_nii, lbl_nii, img_data, lbl_data):
    """
    Runs the per-case integrity checks required before a volume may enter the
    dataset.

    Checks shape agreement, affine agreement, orientation agreement, label value
    domain, NaN/Inf contamination, and empty volumes or masks.

    Args:
        case_id (str): Patient identifier.
        img_nii, lbl_nii: nibabel image objects for image and label.
        img_data, lbl_data (np.ndarray): Their raw voxel arrays.

    Returns:
        tuple: (checks_dict, list_of_problem_strings)
    """
    from nibabel.orientations import aff2axcodes

    problems = []

    shape_match = tuple(img_nii.shape) == tuple(lbl_nii.shape)
    if not shape_match:
        problems.append(f"shape mismatch {img_nii.shape} vs {lbl_nii.shape}")

    affine_diff = float(np.abs(img_nii.affine - lbl_nii.affine).max())
    if affine_diff > 1e-4:
        problems.append(f"affine mismatch (max diff {affine_diff:.2e})")

    img_axcodes = "".join(aff2axcodes(img_nii.affine))
    lbl_axcodes = "".join(aff2axcodes(lbl_nii.affine))
    if img_axcodes != lbl_axcodes:
        problems.append(f"orientation mismatch {img_axcodes} vs {lbl_axcodes}")

    label_values = np.unique(lbl_data)
    if not np.all(np.isin(label_values, [0, 1])):
        problems.append(f"label values outside {{0,1}}: {label_values[:10]}")

    if not np.isfinite(img_data).all():
        problems.append("image contains NaN or Inf")
    if not np.isfinite(lbl_data).all():
        problems.append("label contains NaN or Inf")

    if float(img_data.std()) == 0.0:
        problems.append("image volume is constant (empty)")

    n_tumor_voxels = int((lbl_data > 0.5).sum())
    if n_tumor_voxels == 0:
        problems.append("label mask is empty")

    checks = {
        "shape_match": shape_match,
        "affine_max_diff": affine_diff,
        "orientation": img_axcodes,
        "label_values": [float(v) for v in label_values[:5]],
        "n_tumor_voxels_original": n_tumor_voxels,
    }
    return checks, problems


def preprocess_case(case_id, img_path, lbl_path, run_qc=True):
    """
    Runs the full forward preprocessing pipeline for one patient.

    Args:
        case_id (str): Patient identifier.
        img_path, lbl_path (str): Paths to the NIfTI image and label.
        run_qc (bool): Whether to run the reconstruction round-trip. It costs
            roughly 30 seconds per patient because it resamples the full volume
            back to original resolution, but it is what catches geometry bugs.

    Returns:
        tuple: (img_stack, lbl_stack, metadata, qc)
            img_stack (np.ndarray): float16 (192, 192, D) normalized CT slices.
            lbl_stack (np.ndarray): uint8 (192, 192, D) binary masks.
            metadata (dict): everything needed to invert the pipeline.
            qc (dict): quality-control measurements for this case.
    """
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    original_affine = img_nii.affine.copy()
    original_shape = tuple(img_nii.shape)
    original_spacing = tuple(float(z) for z in img_nii.header.get_zooms()[:3])

    img_hu = np.asanyarray(img_nii.dataobj).astype(np.float32)
    lbl_raw = np.asanyarray(lbl_nii.dataobj).astype(np.float32)

    checks, problems = verify_case(case_id, img_nii, lbl_nii, img_hu, lbl_raw)

    # --- Step 1: reorient to canonical RAS, recording the exact transform ---
    ornt = get_ornt(original_affine)
    img_hu = reorient_to_canonical(img_hu, ornt)
    lbl_raw = reorient_to_canonical(lbl_raw, ornt)
    canonical_shape = tuple(img_hu.shape)
    canonical_spacing = permute_spacing(original_spacing, ornt)

    # --- Step 2: resample to 1mm isotropic ---
    # Trilinear for CT intensities, nearest-neighbour for the binary mask.
    img_hu = resample_volume(img_hu, canonical_spacing, TARGET_SPACING, order=1)
    lbl_res = resample_volume(lbl_raw, canonical_spacing, TARGET_SPACING, order=0)
    lbl_res = (lbl_res > 0.5).astype(np.uint8)
    resampled_shape = tuple(img_hu.shape)

    # --- Step 3: crop to the body bounding box (thresholded in HU) ---
    crop_bbox = crop_body_3d(img_hu, margin=5)
    img_hu = apply_crop(img_hu, crop_bbox)
    lbl_crop = apply_crop(lbl_res, crop_bbox)
    cropped_shape = tuple(img_hu.shape)

    n_tumor_voxels_cropped = int(lbl_crop.sum())
    n_pos_slices_before_resize = int((lbl_crop.sum(axis=(0, 1)) > 0).sum())

    # --- Step 4: HU windowing and normalization to [0, 1] ---
    img_norm = apply_hu_windowing(img_hu)
    del img_hu

    # --- Step 5: resize every axial slice to 192x192 ---
    n_slices = cropped_shape[2]
    img_stack = np.zeros((*TARGET_SLICE_SIZE, n_slices), dtype=np.float16)
    lbl_stack = np.zeros((*TARGET_SLICE_SIZE, n_slices), dtype=np.uint8)

    for s in range(n_slices):
        img_stack[:, :, s] = resize_slice_2d(
            img_norm[:, :, s], TARGET_SLICE_SIZE, order=1).astype(np.float16)
        resized_lbl = resize_slice_2d(
            lbl_crop[:, :, s].astype(np.float32), TARGET_SLICE_SIZE, order=1)
        lbl_stack[:, :, s] = (resized_lbl > 0.5).astype(np.uint8)

    # Per-slice bookkeeping used later for sampling
    pos_per_slice = lbl_stack.sum(axis=(0, 1))
    positive_indices = [int(i) for i in np.where(pos_per_slice > 0)[0]]
    slice_mean = img_stack.astype(np.float32).mean(axis=(0, 1))
    body_indices = [int(i) for i in np.where(slice_mean > BODY_SLICE_MIN_MEAN)[0]]

    metadata = {
        "case_id": case_id,
        "original_affine": original_affine.tolist(),
        "original_shape": list(original_shape),
        "original_spacing": list(original_spacing),
        "ornt": np.asarray(ornt).tolist(),
        "canonical_shape": list(canonical_shape),
        "canonical_spacing": list(canonical_spacing),
        "target_spacing": list(TARGET_SPACING),
        "resampled_shape": list(resampled_shape),
        "crop_bbox": crop_bbox,
        "cropped_shape": list(cropped_shape),
        "target_slice_size": list(TARGET_SLICE_SIZE),
        "hu_min": HU_MIN,
        "hu_max": HU_MAX,
    }

    # --- Step 6: quality control -------------------------------------------
    # Push the preprocessed label back through the full inverse chain and
    # compare against the untouched original mask. This is the accuracy ceiling
    # imposed by preprocessing alone, and it is also a live assertion that the
    # geometry inverse is correct.
    gt_original = (np.asanyarray(lbl_nii.dataobj) > 0.5).astype(np.uint8)
    voxel_mm3 = float(np.prod(original_spacing))
    original_volume_mm3 = float(gt_original.sum() * voxel_mm3)
    n_pos_slices_original = int((gt_original.sum(axis=(0, 1)) > 0).sum())

    if run_qc:
        lbl_reconstructed = reconstruct_to_original_geometry(
            lbl_stack.astype(np.float32), metadata, threshold=0.5)

        inter = int(np.logical_and(lbl_reconstructed > 0, gt_original > 0).sum())
        denom = int((lbl_reconstructed > 0).sum() + (gt_original > 0).sum())
        roundtrip_dice = float(2.0 * inter / denom) if denom > 0 else 1.0
        reconstructed_volume_mm3 = float(lbl_reconstructed.sum() * voxel_mm3)
    else:
        roundtrip_dice = float("nan")
        reconstructed_volume_mm3 = float("nan")

    qc = {
        "case_id": case_id,
        "orientation": checks["orientation"],
        "shape_match": checks["shape_match"],
        "affine_max_diff": checks["affine_max_diff"],
        "original_shape": "x".join(str(s) for s in original_shape),
        "original_spacing": "x".join(f"{s:.3f}" for s in original_spacing),
        "resampled_shape": "x".join(str(s) for s in resampled_shape),
        "cropped_shape": "x".join(str(s) for s in cropped_shape),
        "crop_retained_pct": round(
            100.0 * float(np.prod(cropped_shape)) / float(np.prod(resampled_shape)), 2),
        "n_slices_kept": n_slices,
        "n_tumor_voxels_original": checks["n_tumor_voxels_original"],
        "n_tumor_voxels_cropped": n_tumor_voxels_cropped,
        "n_pos_slices_original": n_pos_slices_original,
        "n_pos_slices_before_resize": n_pos_slices_before_resize,
        "n_pos_slices_after_resize": len(positive_indices),
        "pos_slices_lost_to_resize": n_pos_slices_before_resize - len(positive_indices),
        "tumor_volume_original_mm3": round(original_volume_mm3, 2),
        "tumor_volume_reconstructed_mm3": round(reconstructed_volume_mm3, 2),
        "tumor_volume_change_pct": round(
            100.0 * (reconstructed_volume_mm3 - original_volume_mm3) / original_volume_mm3, 3
        ) if original_volume_mm3 > 0 else 0.0,
        "roundtrip_dice": round(roundtrip_dice, 4),
        "problems": "; ".join(problems) if problems else "",
    }

    return img_stack, lbl_stack, metadata, qc


def preprocess_all(run_qc=True):
    """
    Runs preprocessing over every patient in the dataset, writing:
        output/preprocessed/volumes/{case_id}_img.npy   float16 (192,192,D)
        output/preprocessed/volumes/{case_id}_lbl.npy   uint8   (192,192,D)
        output/preprocessed/index.json                  split + slice index
        output/metadata/{case_id}.json                  reconstruction metadata
        output/preprocessing_qc.csv                     per-case QC measurements

    Args:
        run_qc (bool): Whether to measure the reconstruction round-trip for every
            patient. Adds roughly 30-50 seconds per case (about 40 minutes for
            the full dataset) and is what verifies the geometry inverse.
    """
    print("=== ANATOMICALLY-CORRECT PREPROCESSING PIPELINE ===\n")
    if not run_qc:
        print("[!] QC round-trip disabled. Geometry errors will not be detected.\n")

    clean_preprocessed_directory()

    split = load_patient_split()
    split_of = {}
    for mode in ("train", "val", "test"):
        for case_id in split[mode]:
            split_of[case_id] = mode

    volumes_dir = get_volumes_dir()
    os.makedirs(volumes_dir, exist_ok=True)
    os.makedirs(get_metadata_dir(), exist_ok=True)

    with open(os.path.join(DATA_DIR, "dataset.json"), "r") as f:
        dataset_info = json.load(f)
    training_cases = dataset_info["training"]

    index = {
        "target_slice_size": list(TARGET_SLICE_SIZE),
        "target_spacing": list(TARGET_SPACING),
        "hu_window": [HU_MIN, HU_MAX],
        "splits": {"train": [], "val": [], "test": []},
        "cases": {},
    }
    qc_records = []

    print(f"Processing {len(training_cases)} CT volumes...\n")

    for case in tqdm(training_cases, desc="Preprocessing"):
        case_id = os.path.basename(case["image"]).replace(".nii.gz", "").replace(".nii", "")

        mode = split_of.get(case_id)
        if mode is None:
            print(f"  [WARNING] {case_id} is in no split, skipping.")
            continue

        try:
            img_stack, lbl_stack, metadata, qc = preprocess_case(
                case_id,
                resolve_nifti_path(case["image"]),
                resolve_nifti_path(case["label"]),
                run_qc=run_qc,
            )
        except Exception as e:
            print(f"\n  [ERROR] {case_id} failed: {e}")
            continue

        if qc["problems"]:
            print(f"\n  [QC] {case_id}: {qc['problems']}")
        if qc["roundtrip_dice"] < 0.80:
            print(f"\n  [QC] {case_id}: round-trip Dice is only "
                  f"{qc['roundtrip_dice']:.4f} — geometry may be wrong.")

        np.save(os.path.join(volumes_dir, f"{case_id}_img.npy"), img_stack)
        np.save(os.path.join(volumes_dir, f"{case_id}_lbl.npy"), lbl_stack)

        with open(os.path.join(get_metadata_dir(), f"{case_id}.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        pos_per_slice = lbl_stack.sum(axis=(0, 1))
        positive_indices = [int(i) for i in np.where(pos_per_slice > 0)[0]]
        slice_mean = img_stack.astype(np.float32).mean(axis=(0, 1))
        body_indices = [int(i) for i in np.where(slice_mean > BODY_SLICE_MIN_MEAN)[0]]

        index["splits"][mode].append(case_id)
        index["cases"][case_id] = {
            "split": mode,
            "n_slices": int(lbl_stack.shape[2]),
            "positive_slices": positive_indices,
            "body_slices": body_indices,
            "tumor_voxels": int(lbl_stack.sum()),
        }
        qc_records.append(qc)

    with open(os.path.join(get_preprocessed_dir(), "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    qc_path = os.path.join(OUTPUT_DIR, "preprocessing_qc.csv")
    if qc_records:
        with open(qc_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(qc_records[0].keys()))
            writer.writeheader()
            writer.writerows(qc_records)

    _print_summary(index, qc_records, qc_path)


def _print_summary(index, qc_records, qc_path):
    """Prints the preprocessing summary and the accuracy ceiling it implies."""
    print("\n" + "=" * 72)
    print("        PREPROCESSING SUMMARY")
    print("=" * 72)

    for mode in ("train", "val", "test"):
        cases = index["splits"][mode]
        total = sum(index["cases"][c]["n_slices"] for c in cases)
        pos = sum(len(index["cases"][c]["positive_slices"]) for c in cases)
        pct = (100.0 * pos / total) if total else 0.0
        print(f"  {mode.upper():6s}: {len(cases):2d} patients | {total:6d} slices "
              f"| {pos:5d} positive ({pct:.2f}%)")

    print("\n  All splits store every slice. Positive/negative balancing is a")
    print("  training-time sampling choice, so validation and test keep the")
    print("  real class distribution.")

    if qc_records:
        dices = [q["roundtrip_dice"] for q in qc_records
                 if not np.isnan(q["roundtrip_dice"])]
        lost = sum(q["pos_slices_lost_to_resize"] for q in qc_records)
        affected = sum(1 for q in qc_records if q["pos_slices_lost_to_resize"] > 0)
        vol_change = [q["tumor_volume_change_pct"] for q in qc_records
                      if not np.isnan(q["tumor_volume_change_pct"])]
        retained = [q["crop_retained_pct"] for q in qc_records]

        print("\n  --- Preprocessing cost (the ceiling no model can beat) ---")
        if dices:
            print(f"  Round-trip Dice:        mean {np.mean(dices):.4f} | "
                  f"min {np.min(dices):.4f} | max {np.max(dices):.4f}")
        print(f"  Positive slices lost:   {lost} across {affected} patients")
        if vol_change:
            print(f"  Tumor volume change:    mean {np.mean(vol_change):+.2f}%")
        print(f"  Volume kept after crop: mean {np.mean(retained):.1f}%")
        print(f"\n  QC report: {qc_path}")

        if dices and np.min(dices) < 0.80:
            print("\n  [!] At least one case has a low round-trip Dice. That is a")
            print("      geometry bug, not a data property — investigate before training.")

    print("=" * 72)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess the Decathlon Lung dataset into 2D slice stacks")
    parser.add_argument("--skip_qc", action="store_true",
                        help="Skip the reconstruction round-trip check (much "
                             "faster, but geometry errors go undetected)")
    args = parser.parse_args()

    preprocess_all(run_qc=not args.skip_qc)
