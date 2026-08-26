"""
Per-slice overlay export at two points in the preprocessing pipeline.

For every tumour-bearing slice, writes one figure holding the same slice twice:
immediately after the body crop, at native 1mm isotropic resolution, and after
the resize to 192x192 that the network is actually trained on. The pair is what
makes the resize legible — everything before it preserves anatomy at the
acquisition scale, and the resize is the one step that throws detail away.

Two things the comparison surfaces that a single-stage overlay cannot:

    - The crop is not square. Cropped shapes run from roughly 370x270 to
      490x420, and all of them are mapped onto 192x192, so each patient's chest
      is stretched by a different amount. The two panels sit side by side at the
      same display size, so the distortion is visible directly.
    - Ten slices across five patients hold a tumour before the resize and none
      after it, the lesion being small enough to vanish under the downsampling.
      Those slices are exported too and titled LOST TO RESIZE, since they are
      the clearest statement of what the 192x192 choice costs.

The post-crop volume is not stored by the preprocessing stage — it only exists
inside `preprocess_case` — so it is recomputed here by replaying reorientation,
resampling and cropping. The crop bounding box is read from the per-patient
metadata rather than recomputed, which keeps this export pinned to the same crop
that produced the stored volumes.

Cost is dominated by the 1mm resampling, at roughly 30 s per patient, and runs
once per patient rather than once per slice.

Usage:
    python -m src.visualization.slice_overlays
    python -m src.visualization.slice_overlays --split test
    python -m src.visualization.slice_overlays --cases lung_028,lung_074
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.geometry import (
    TARGET_SPACING,
    apply_crop,
    get_ornt,
    permute_spacing,
    reorient_to_canonical,
    resample_volume,
    resize_plan,
    to_network_grid,
)
from src.preprocessing.preprocessing import apply_hu_windowing


# The resize schemes --compare draws side by side. The crop panel is shared
# rather than repeated: cropping happens before any of them, so it is the one
# reference all three are measured against.
COMPARE_MODES = (
    ("192 x 192 deformed", "stretch", 192, None),
    ("192 x 192 with padding", "pad", 192, None),
    ("256 x 256", "fixed_mm", 256, 2.0),
)

# Panel titles are two lines. Named rather than written inline because an f-string
# cannot carry a backslash escape in the versions this has to run under.
NL = "\n"


def compose_rgb(img, lbl):
    """
    Builds one RGB array holding the slice, its mask and the mask outline.

    Drawing three separate translucent layers through `imshow` costs more than
    the rest of the figure put together, so the blend is done here in numpy and
    handed to matplotlib as a single image.

    Args:
        img (np.ndarray): 2D slice, already windowed and normalized to [0, 1].
        lbl (np.ndarray): 2D binary mask, same shape.

    Returns:
        np.ndarray: (H, W, 3) float array in [0, 1].
    """
    rgb = np.repeat(np.clip(img, 0.0, 1.0)[:, :, None], 3, axis=2)
    mask = lbl > 0.5
    if mask.any():
        rgb[mask] = 0.65 * rgb[mask] + 0.35 * np.array([1.0, 0.0, 0.0])
        rgb[mask & ~binary_erosion(mask)] = [0.0, 1.0, 0.0]
    return rgb


def pipeline_to_crop(case_id, metadata):
    """
    Replays the forward pipeline as far as the body crop.

    Mirrors `preprocess_case` steps 1 to 4 — reorient to canonical RAS, resample
    to 1mm isotropic, crop, apply the HU window — and stops before the resize.
    The crop box comes from `metadata` instead of being recomputed, so this
    cannot drift from the crop that produced the stored 192x192 volumes.

    Returns:
        tuple: (img, lbl) at the cropped 1mm isotropic size.
    """
    img_nii = nib.load(resolve_nifti_path(f"./imagesTr/{case_id}.nii.gz"))
    lbl_nii = nib.load(resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz"))

    img = np.asanyarray(img_nii.dataobj).astype(np.float32)
    lbl = np.asanyarray(lbl_nii.dataobj).astype(np.float32)

    ornt = get_ornt(img_nii.affine)
    spacing = permute_spacing(
        tuple(float(z) for z in img_nii.header.get_zooms()[:3]), ornt)
    img = reorient_to_canonical(img, ornt)
    lbl = reorient_to_canonical(lbl, ornt)

    img = resample_volume(img, spacing, TARGET_SPACING, order=1)
    lbl = (resample_volume(lbl, spacing, TARGET_SPACING, order=0) > 0.5).astype(np.uint8)

    bbox = metadata["crop_bbox"]
    return apply_hu_windowing(apply_crop(img, bbox)), apply_crop(lbl, bbox)


def export_case(case_id, out_dir, dpi=110, figure=None):
    """
    Writes one two-panel figure per tumour-bearing slice of a single patient.

    A slice qualifies if it carries a tumour at either stage, so the handful that
    are positive before the resize and empty after it are exported rather than
    silently skipped — they are the point of the comparison.

    Returns:
        tuple: (slices written, slices whose tumour did not survive the resize)
    """
    with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
        metadata = json.load(f)

    img_crop, lbl_crop = pipeline_to_crop(case_id, metadata)

    volumes = os.path.join(OUTPUT_DIR, "preprocessed", "volumes")
    img_final = np.load(os.path.join(volumes, f"{case_id}_img.npy"), mmap_mode="r")
    lbl_final = np.load(os.path.join(volumes, f"{case_id}_lbl.npy"), mmap_mode="r")

    # The resize leaves the Z axis alone, so slice s means the same slice at both
    # stages and the two positive sets are directly comparable.
    positive_crop = set(np.where(lbl_crop.sum(axis=(0, 1)) > 0)[0].tolist())
    positive_final = set(np.where(np.asarray(lbl_final).sum(axis=(0, 1)) > 0)[0].tolist())
    lost = sorted(positive_crop - positive_final)

    os.makedirs(out_dir, exist_ok=True)
    axes = figure.axes

    for slice_index in sorted(positive_crop | positive_final):
        panels = (
            ("after crop", img_crop[:, :, slice_index], lbl_crop[:, :, slice_index]),
            ("after resize to 192x192",
             np.asarray(img_final[:, :, slice_index], dtype=np.float32),
             np.asarray(lbl_final[:, :, slice_index])),
        )
        for axis, (stage, img, lbl) in zip(axes, panels):
            axis.clear()
            axis.imshow(compose_rgb(img, lbl), interpolation="nearest")
            axis.set_title(
                f"{stage}\n{img.shape[0]}x{img.shape[1]} · "
                f"{int((lbl > 0.5).sum())} tumour px", fontsize=9)
            axis.axis("off")

        heading = f"{case_id} · slice {slice_index}"
        if slice_index in lost:
            heading += "  —  LOST TO RESIZE"
        figure.suptitle(heading, fontsize=11, fontweight="bold")
        figure.savefig(
            os.path.join(out_dir, f"{case_id}_s{slice_index:03d}.png"), dpi=dpi)

    return len(positive_crop | positive_final), len(lost)


def export_case_compare(case_id, out_dir, dpi=110, figure=None):
    """
    Writes one figure per tumour-bearing slice showing every resize scheme.

    The first panel is the cropped slice at its native 1mm resolution, which is
    what the patient actually looks like; the rest are the same slice as each
    scheme would hand it to the network. Because every panel is drawn at the
    same height with `imshow` preserving each array's own aspect, a scheme that
    distorts anatomy shows up as a panel of a different width.

    The variant grids are computed here rather than read from disk. They are a
    pure function of the cropped slice and the plan, so materialising three
    preprocessed datasets to draw them would cost hours and add nothing.

    Returns:
        tuple: (slices written, {scheme label: slices whose tumour it drops})
    """
    with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
        metadata = json.load(f)

    img_crop, lbl_crop = pipeline_to_crop(case_id, metadata)
    cropped_shape = img_crop.shape
    height, width = cropped_shape[0], cropped_shape[1]

    plans = [(label, resize_plan(cropped_shape, mode=mode, target_size=size,
                                 mm_per_px=mm), size)
             for label, mode, size, mm in COMPARE_MODES]

    # Millimetres per pixel on each axis, which is what "distortion" means
    # concretely: the two numbers differ exactly when anatomy is being stretched.
    scales = {}
    for label, plan, size in plans:
        if plan["mode"] == "stretch":
            scales[label] = (height / size, width / size)
        elif plan["mode"] == "pad":
            side = plan["square_side"] / size
            scales[label] = (side, side)
        else:
            scales[label] = (height / plan["inner_h"], width / plan["inner_w"])

    positive = set(np.where(lbl_crop.sum(axis=(0, 1)) > 0)[0].tolist())
    dropped = {label: [] for label, _, _ in plans}

    os.makedirs(out_dir, exist_ok=True)
    axes = figure.axes

    for slice_index in sorted(positive):
        img_slice = img_crop[:, :, slice_index]
        lbl_slice = lbl_crop[:, :, slice_index].astype(np.float32)

        panels = [(f"after crop"
                   f"{NL}{height}x{width} · 1.00 mm/px",
                   img_slice, lbl_slice > 0.5)]
        for label, plan, size in plans:
            grid_img = to_network_grid(img_slice, plan, order=1)
            grid_lbl = to_network_grid(lbl_slice, plan, order=1) > 0.5
            if not grid_lbl.any():
                dropped[label].append(slice_index)
            mm_v, mm_h = scales[label]
            panels.append(
                (f"{label}"
                 f"{NL}{size}x{size} · {mm_v:.2f} / {mm_h:.2f} mm/px",
                 grid_img, grid_lbl))

        for axis, (title, img, lbl) in zip(axes, panels):
            axis.clear()
            axis.imshow(compose_rgb(img, lbl), interpolation="nearest")
            axis.set_title(f"{title} · {int(lbl.sum())} px", fontsize=8)
            axis.axis("off")

        vanished = [label for label, _, _ in plans if slice_index in dropped[label]]
        heading = f"{case_id} · slice {slice_index}"
        if vanished:
            heading += "  —  tumour lost in: " + ", ".join(vanished)
        figure.suptitle(heading, fontsize=11, fontweight="bold")
        figure.savefig(
            os.path.join(out_dir, f"{case_id}_s{slice_index:03d}.png"), dpi=dpi)

    return len(positive), dropped


def main():
    parser = argparse.ArgumentParser(
        description="Export per-slice overlays before and after the 192x192 resize.")
    parser.add_argument("--split", default="all",
                        choices=["all", "train", "val", "test"],
                        help="Which split to export. Default: every patient.")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated case ids, overriding --split.")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "slice_overlays"),
                        help="Destination directory.")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--compare", action="store_true",
                        help="Draw every resize scheme against the cropped "
                             "slice, rather than only the one the stored "
                             "dataset used. This is what shows whether a scheme "
                             "distorts anatomy.")
    args = parser.parse_args()

    if args.compare and args.out == os.path.join(OUTPUT_DIR, "slice_overlays"):
        args.out = os.path.join(OUTPUT_DIR, "slice_overlays_compare")

    index_path = os.path.join(OUTPUT_DIR, "preprocessed", "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing {index_path}\n"
            f"Run 'python -m src.preprocessing.preprocessing' first.")

    with open(index_path) as f:
        index = json.load(f)

    if args.cases:
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    elif args.split == "all":
        case_ids = [c for split in index["splits"].values() for c in split]
    else:
        case_ids = list(index["splits"][args.split])

    print(f"=== SLICE OVERLAY EXPORT: {len(case_ids)} patients ===\n")
    print(f"  Destination: {args.out}")
    print(f"  Roughly 30 s per patient, dominated by the 1mm resampling.\n")

    # One figure reused for every slice. Building and destroying it per slice
    # costs about as much as drawing into it. The margins are set once here
    # rather than through `tight_layout`, which would re-solve the layout on
    # every slice and roughly double the cost of the export.
    n_panels = 1 + len(COMPARE_MODES) if args.compare else 2
    figure, _ = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.4))
    figure.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.01,
                           wspace=0.05)

    total_slices = 0
    total_lost = 0
    dropped_by_scheme = {label: 0 for label, _, _, _ in COMPARE_MODES}
    started = time.time()

    try:
        for position, case_id in enumerate(case_ids, start=1):
            case_started = time.time()
            if args.compare:
                written, dropped = export_case_compare(
                    case_id, os.path.join(args.out, case_id), dpi=args.dpi,
                    figure=figure)
                note = ""
                for label, slices in dropped.items():
                    dropped_by_scheme[label] += len(slices)
                    if slices:
                        note += f"  [{label}: -{len(slices)}]"
            else:
                written, lost = export_case(
                    case_id, os.path.join(args.out, case_id), dpi=args.dpi,
                    figure=figure)
                total_lost += lost
                note = f"  [{lost} lost to resize]" if lost else ""
            total_slices += written
            print(f"  [{position:2d}/{len(case_ids)}] {case_id}: {written:3d} slices "
                  f"in {time.time() - case_started:.0f}s{note}", flush=True)
    finally:
        plt.close(figure)

    print(f"\n=== DONE: {total_slices} figures in "
          f"{(time.time() - started) / 60:.1f} min ===")
    if args.compare:
        print("\n  Tumour-bearing slices each scheme loses entirely:")
        for label, count in dropped_by_scheme.items():
            print(f"    {label:22} {count}")
    elif total_lost:
        print(f"  {total_lost} slices carry a tumour before the resize and none "
              f"after it; they are titled LOST TO RESIZE.")


if __name__ == "__main__":
    main()
