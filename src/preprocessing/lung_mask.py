"""
A lung mask, and the ceiling it puts on everything downstream.

Restricting the model to the lungs should mean that predictions in the
mediastinum, the chest wall and the abdomen stop counting. The committed baseline
leaves 7.3 false-positive components per patient and fires on about 5% of empty
slices, and much of that sits outside any lung.

The risk is the same shape as the benefit. A tumour touching the pleura is not
air, so it is not inside a threshold-grown lung region; a mask built to exclude
the chest wall excludes that tumour with it. Masking then removes false positives
and true positives together, and the sensitivity lost is lost before training
starts - no model can recover a voxel the mask deleted.

So the mask is measured before it is used. `coverage_report` computes, per
patient, the fraction of ground-truth tumour that survives the mask. That is a
hard ceiling on masked sensitivity, exactly as the round-trip Dice in
`resolution_ceiling` is a ceiling on reconstruction. If it does not stay high,
the approach is refuted on CPU and needs no GPU time.

The mask is classical rather than learned: thresholding, connected components and
morphology. Nothing here is trained, so it costs no data and cannot overfit.

Usage:
    python -m src.preprocessing.lung_mask                # coverage report
    python -m src.preprocessing.lung_mask --dilate 0,3,5,8
"""

import argparse
import json
import os

import numpy as np
from scipy import ndimage

from src.config import OUTPUT_DIR

# Air in the lung parenchyma sits near -1000 HU and soft tissue near +40. The
# usual split for lung field segmentation is around -320; it is well below any
# soft tissue and well above the parenchyma, so it is not a sensitive choice.
AIR_HU = -320.0

# Preprocessing stored a window of [-1000, 400] HU scaled into [0, 1], so the
# threshold has to be carried into the same units rather than applied raw.
HU_MIN, HU_MAX = -1000.0, 400.0


def hu_threshold_in_normalised_units(hu=AIR_HU):
    return (hu - HU_MIN) / (HU_MAX - HU_MIN)


def lung_mask_from_normalised(vol, dilate_mm=3, min_component_frac=0.005):
    """
    A lung-field mask for a preprocessed volume in [0, 1].

    Args:
        vol (np.ndarray): (H, W, D) float volume as stored by preprocessing.
        dilate_mm (int): Radius, in voxels, the finished mask is grown by. This
            is the juxtapleural allowance - it is what lets a tumour sitting on
            the pleural surface stay inside the mask - and it is the parameter
            `coverage_report` sweeps.
        min_component_frac (float): Air components smaller than this fraction of
            the volume are discarded as bowel gas or noise.

    Returns:
        np.ndarray: bool mask, same shape as `vol`.
    """
    air = vol < hu_threshold_in_normalised_units()

    # Lung air has to be separated from the air around the patient. Discarding
    # the air components that touch a border looks like it would do it, and does
    # on most patients - but the lungs open onto the trachea, and where the
    # airway reaches the edge of the volume the two become one component. On
    # lung_037 that single component held 57.8% of the volume and the border
    # rule deleted the lungs with the exterior, leaving an empty mask.
    #
    # The body silhouette does not have that failure mode. Filling the tissue
    # outline per slice closes the lungs into it, so intersecting it with the air
    # keeps exactly the air inside the patient whatever the airway does.
    body = np.empty_like(air)
    tissue = ~air
    for z in range(vol.shape[2]):
        body[:, :, z] = ndimage.binary_fill_holes(tissue[:, :, z])
    internal_air = air & body

    labels, n = ndimage.label(internal_air)
    if n == 0:
        return np.zeros_like(vol, dtype=bool)
    sizes = ndimage.sum(internal_air, labels, index=np.arange(1, n + 1))
    keep = [i for i in range(1, n + 1)
            if sizes[i - 1] >= min_component_frac * internal_air.size]
    if not keep:
        return np.zeros_like(vol, dtype=bool)
    mask = np.isin(labels, keep)

    # Everything denser than air *inside* the lungs - vessels, airways and the
    # tumour itself - failed the air threshold and is currently a hole. Filling
    # per slice is what puts the tumour back inside its own lung.
    for z in range(mask.shape[2]):
        if mask[:, :, z].any():
            mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])

    if dilate_mm > 0:
        mask = ndimage.binary_dilation(
            mask, ndimage.generate_binary_structure(3, 1), iterations=int(dilate_mm))
    return mask


def coverage_report(case_ids=None, dilations=(0, 3, 5, 8), volumes_dir=None,
                    index_path=None, verbose=True):
    """
    The ceiling, per patient and per dilation.

    Returns:
        dict: dilation -> {"per_case": {case_id: {...}}, "tumour_retained": float,
              "volume_kept": float, "cases_below_90": int}
    """
    volumes_dir = volumes_dir or os.path.join(OUTPUT_DIR, "preprocessed", "volumes")
    index_path = index_path or os.path.join(OUTPUT_DIR, "preprocessed", "index.json")
    index = json.load(open(index_path))
    case_ids = case_ids or sorted(index["cases"])

    out = {d: {"per_case": {}} for d in dilations}
    for case_id in case_ids:
        vol = np.load(os.path.join(volumes_dir, f"{case_id}_img.npy")).astype(np.float32)
        lbl = np.load(os.path.join(volumes_dir, f"{case_id}_lbl.npy")) > 0
        tumour = float(lbl.sum())
        for d in dilations:
            mask = lung_mask_from_normalised(vol, dilate_mm=d)
            inside = float((lbl & mask).sum())
            out[d]["per_case"][case_id] = {
                "tumour_retained": inside / tumour if tumour else float("nan"),
                "volume_kept": float(mask.sum()) / mask.size,
            }
        if verbose:
            r = " ".join(f"d{d}={out[d]['per_case'][case_id]['tumour_retained']:.3f}"
                         for d in dilations)
            print(f"  {case_id}  {r}", flush=True)
        del vol, lbl

    for d in dilations:
        per = out[d]["per_case"]
        ret = [v["tumour_retained"] for v in per.values()
               if not np.isnan(v["tumour_retained"])]
        out[d]["tumour_retained"] = float(np.mean(ret))
        out[d]["tumour_retained_worst"] = float(np.min(ret))
        out[d]["volume_kept"] = float(np.mean([v["volume_kept"] for v in per.values()]))
        out[d]["cases_below_90"] = int(sum(1 for x in ret if x < 0.90))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dilate", type=str, default="0,3,5,8",
                        help="Comma-separated dilation radii in voxels.")
    parser.add_argument("--out", type=str,
                        default=os.path.join(OUTPUT_DIR, "lung_mask_coverage.json"))
    args = parser.parse_args()
    dil = tuple(int(x) for x in args.dilate.split(","))

    print(f"Lung-mask coverage, dilations {dil}")
    print(f"Air threshold {AIR_HU:g} HU "
          f"(= {hu_threshold_in_normalised_units():.4f} normalised)\n")
    rep = coverage_report(dilations=dil)

    print(f"\n{'dilate':>7} {'tumour kept':>12} {'worst case':>11} "
          f"{'volume kept':>12} {'cases <90%':>11}")
    print("-" * 58)
    for d in dil:
        r = rep[d]
        print(f"{d:>7} {r['tumour_retained']:12.4f} "
              f"{r['tumour_retained_worst']:11.4f} "
              f"{r['volume_kept']:12.2%} {r['cases_below_90']:11}")
    print("\n  tumour kept is the ceiling on masked sensitivity.")
    print("  volume kept is how much of the image the mask lets through -")
    print("  the smaller it is, the more false-positive room is removed.")

    json.dump({str(k): v for k, v in rep.items()}, open(args.out, "w"), indent=2)
    print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
