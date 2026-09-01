"""
Separates the effect of the threshold from the effect of which slices are scored.

Notebook N reported the same checkpoint two ways: 0.3652 tumour-slice Dice under
the positives protocol at threshold 0.25, and 0.1667 under the all-slice protocol
at 0.40. Two things moved at once, so neither number attributes its difference to
anything. This computes the full matrix instead - both protocols at every
threshold on one grid - which makes three separate claims checkable:

  1. At a fixed threshold, tumour-slice Dice must be identical across protocols
     for every patient whose reconstruction does not interpolate along Z. A 2D
     model with n_adjacent=1 predicts slice i from slice i alone, so restricting
     which slices are fed to it cannot change the prediction on the slices that
     are fed either way.

     The Z caveat is not a hedge, it is measured. Reconstruction returns the
     preprocessed 1 mm stack to the patient's original spacing, and wherever
     that spacing is not 1 mm the result is a blend of neighbouring preprocessed
     slices. At the tumour's Z edges one of those neighbours is a non-positive
     slice, which the oracle protocol zero-fills - so the fill bleeds into the
     slices being scored.

     On this test set 7 of 10 patients resample: lung_023 upsamples (0.625 mm,
     332 preprocessed slices to 531 original) and six downsample (1.245-2.5 mm).
     Only lung_001, lung_020 and lung_074 are already at 1 mm and can be held to
     the exact identity. The largest disagreement measured is -1.4e-02, on
     lung_023.

     So the check is made per patient and split by that criterion. A
     disagreement on a patient whose reconstruction does *not* interpolate is a
     real indexing bug. One on a patient whose reconstruction does interpolate
     is the zero-fill reaching the scored slices - which is not a bug in the
     code, but is one more reason the oracle protocol does not measure what it
     appears to.

  2. Whole-volume Dice must *not* be identical, and the gap is the zero-fill: a
     slice the model was never run on reconstructs as empty, which is free
     credit on the ~92% of slices that hold no tumour.

  3. The optimal threshold differs per protocol, so quoting one protocol's
     threshold against the other's slices measures neither.

Claim 1 is also pinned as a unit test (`test_metrics.py`), because it is the one
that would silently invalidate every dual-protocol comparison in the project.

Usage:
    python -m src.evaluation.protocol_matrix --run baseline/seed_42
    python -m src.evaluation.protocol_matrix --run positives_only/seed_42 --split test
"""

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.evaluation.evaluate import collect_predictions
from src.evaluation.hierarchical_report import load_run
from src.evaluation.metrics import (
    compute_2d_slice_metrics,
    compute_all_3d_metrics,
    reconstruct_patient_3d_volume,
)
from src.training.dataset import LungSliceDataset, load_index
from torch.utils.data import DataLoader

GRID = [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]


def build_loader(preprocessed_dir, index, case_ids, sampling, config,
                 batch_size=16):
    """One loader per protocol, differing only in which slices it yields."""
    dataset = LungSliceDataset(
        os.path.join(preprocessed_dir, "volumes"), index, case_ids,
        sampling=sampling, augment="none", crop="none",
        n_adjacent=config.get("n_adjacent", 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0)


KEYS = ("dice_2d_tumour_slices", "dice_2d_all_slices", "dice_3d",
        "sensitivity_3d", "precision_3d", "volume_ratio_3d",
        "fp_components", "false_alarm_rate_2d")


def score_grid(probs, metadata_dir, thresholds, require_full_coverage):
    """
    Scores one set of predictions at every threshold on the grid.

    The reconstruction is the expensive step - a 512 x 512 x 304 volume per
    patient, inverting four geometry stages - and it does not depend on the
    threshold. Calling `evaluate_full` once per threshold would redo it seven
    times per patient, which on CPU is the difference between minutes and
    hours. So the probability volume is reconstructed once and cut seven times,
    the same trick `threshold_sweep_original_geometry` uses.
    """
    per_patient = {threshold: {} for threshold in thresholds}

    for case_id, slice_dict in probs.items():
        meta_path = os.path.join(metadata_dir, f"{case_id}.json")
        if not os.path.exists(meta_path):
            print(f"  [WARNING] No metadata for {case_id}, skipping.")
            continue
        with open(meta_path) as f:
            metadata = json.load(f)

        probs_3d = reconstruct_patient_3d_volume(
            slice_dict, metadata, binarize=False,
            require_full_coverage=require_full_coverage)
        gt = (np.asanyarray(nib.load(
            resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")).dataobj)
            > 0.5).astype(np.uint8)

        for threshold in thresholds:
            pred = (probs_3d > threshold).astype(np.uint8)
            scores = compute_all_3d_metrics(pred, gt, surface_metrics=False)
            scores.update(compute_2d_slice_metrics(pred, gt))
            per_patient[threshold][case_id] = scores
        del probs_3d, gt

    rows = {}
    for threshold in thresholds:
        patients = per_patient[threshold]
        row = {}
        for key in KEYS:
            vals = [m[key] for m in patients.values()
                    if key in m and not np.isnan(m[key])]
            row[key] = float(np.mean(vals)) if vals else float("nan")
        rows[threshold] = row
    return rows, per_patient


def z_interpolates(metadata):
    """
    True when reconstruction resamples along Z, so a scored slice is a blend of
    preprocessed neighbours rather than one of them.

    That is the case whenever the preprocessed stack has a different number of
    slices from the original, which happens for any original spacing other than
    the 1 mm the pipeline resamples to - in *either* direction. A finer original
    (0.625 mm) means reconstruction upsamples; a coarser one (2.5 mm) means it
    downsamples. Both blend neighbouring preprocessed slices under linear
    interpolation, so both can carry a zero-filled neighbour into a scored slice.

    Only patients whose original spacing is already 1 mm are held to the exact
    identity. On this test set that is 3 of 10.
    """
    return int(metadata["cropped_shape"][2]) != int(metadata["original_shape"][2])


def run_matrix(run, split="test", preprocessed_name="preprocessed",
               metadata_name="metadata", out=None, output_dir=None):
    """
    Computes and prints the matrix for one run, and returns it.

    Callable in-process so a notebook does not have to shell out: on Kaggle
    `OUTPUT_DIR` is repointed at import time by `config.set_data_dir`, and a
    subprocess would start from the module default instead and look for the
    experiments inside the read-only code dataset. `output_dir` overrides it for
    the same reason.
    """
    base = output_dir or OUTPUT_DIR
    run_dir = os.path.join(base, "experiments", *run.split("/"))
    preprocessed_dir = os.path.join(base, preprocessed_name)
    metadata_dir = os.path.join(base, metadata_name)

    model, device, config, _ = load_run(run_dir)
    index = load_index(preprocessed_dir)
    case_ids = index["splits"][split]

    print(f"=== THRESHOLD x PROTOCOL MATRIX: {run} ({split}) ===\n")
    print(f"  {len(case_ids)} patients, {len(GRID)} thresholds, 2 protocols")
    print(f"  Same weights and same reconstruction throughout. The only thing")
    print(f"  that changes between the two columns is which slices the model")
    print(f"  is asked about.\n", flush=True)

    results, detail = {}, {}
    for sampling in ("all", "positives"):
        loader = build_loader(preprocessed_dir, index, case_ids, sampling, config)
        n_slices = len(loader.dataset)
        print(f"  Protocol '{sampling}': {n_slices} slices ... ", end="",
              flush=True)
        probs, _, _ = collect_predictions(model, loader, device)
        # A protocol that skips slices cannot demand full coverage: that is the
        # oracle path, and it is exactly the thing being quantified here.
        rows, per_patient = score_grid(
            probs, metadata_dir, GRID,
            require_full_coverage=(sampling == "all"))
        results[sampling] = rows
        detail[sampling] = per_patient
        results[sampling + "_n_slices"] = n_slices
        del probs
        print("done", flush=True)

    print(f"\n{'':6s} {'Dice 2D, felii cu tumoare':>34s}   "
          f"{'Dice 3D, volum intreg':>28s}")
    print(f"{'prag':>6s} {'all':>10s} {'positives':>10s} {'delta':>10s}   "
          f"{'all':>10s} {'positives':>8s} {'delta':>8s}")
    print("-" * 74)
    max_gap = 0.0
    for threshold in GRID:
        a, p = results["all"][threshold], results["positives"][threshold]
        gap_2d = p["dice_2d_tumour_slices"] - a["dice_2d_tumour_slices"]
        gap_3d = p["dice_3d"] - a["dice_3d"]
        max_gap = max(max_gap, abs(gap_2d))
        print(f"{threshold:6.2f} {a['dice_2d_tumour_slices']:10.4f} "
              f"{p['dice_2d_tumour_slices']:10.4f} {gap_2d:+10.6f}   "
              f"{a['dice_3d']:10.4f} {p['dice_3d']:8.4f} {gap_3d:+8.4f}")

    # Per patient, split by whether reconstruction resamples along Z. Only the
    # patients it does not resample can be held to the exact identity; see the
    # module docstring.
    flat, interpolating = {}, {}
    for case_id in detail["all"][GRID[0]]:
        meta = json.load(open(os.path.join(metadata_dir, f"{case_id}.json")))
        worst = max(
            abs(detail["positives"][t][case_id]["dice_2d_tumour_slices"]
                - detail["all"][t][case_id]["dice_2d_tumour_slices"])
            for t in GRID if case_id in detail["positives"][t])
        (interpolating if z_interpolates(meta) else flat)[case_id] = worst

    print(f"\n  Claim 1 - tumour-slice Dice is protocol-independent at a fixed")
    print(f"  threshold, for every patient whose reconstruction does not")
    print(f"  resample along Z.\n")
    print(f"    {len(flat)} patients, Z untouched by reconstruction: "
          f"largest disagreement {max(flat.values(), default=0.0):.2e}")
    print(f"    {len(interpolating)} patients, Z resampled: "
          f"largest disagreement {max(interpolating.values(), default=0.0):.2e}")

    broken = {c: v for c, v in flat.items() if v > 1e-9}
    if broken:
        print("\n  FAILED - these patients disagree with no interpolation to "
              "explain it,")
        print("  which means slices are being indexed or assembled wrongly:")
        for case_id, value in sorted(broken.items(), key=lambda x: -x[1]):
            print(f"    {case_id}  {value:.2e}")
    else:
        print("\n  OK - every patient reconstructed without Z resampling scores")
        print("  identically under both protocols, so the slices line up.")

    bleeding = {c: v for c, v in interpolating.items() if v > 1e-9}
    if bleeding:
        print(f"\n  {len(bleeding)} of the {len(interpolating)} resampled "
              f"patients do disagree. That is not a")
        print("  code fault: an original slice is a blend of preprocessed ones, "
              "and the")
        print("  oracle protocol zero-fills the neighbours, so the fill reaches "
              "the")
        print("  scored slices. One more way the oracle number is not volume "
              "performance.")
        for case_id, value in sorted(bleeding.items(), key=lambda x: -x[1])[:5]:
            print(f"    {case_id}  {value:.2e}")

    best = {k: max(GRID, key=lambda t: results[k][t]["dice_3d"])
            for k in ("all", "positives")}
    print(f"\n  Claim 3 - the optimum is protocol-specific: "
          f"all -> {best['all']:.2f}, positives -> {best['positives']:.2f}")
    print(f"  Quoting one against the other's slices measures neither.")
    # These two are argmaxes over the *test* set, which no honest procedure can
    # reach: a run picks its threshold on validation. They are here to show that
    # the two protocols peak in different places, not as values to adopt. The
    # run's own validation-chosen threshold is in its benchmark report.
    print(f"  [!] Both are oracle values, chosen on test. They demonstrate that")
    print(f"      the protocols disagree; they are not thresholds to use.")

    out = out or os.path.join(
        base, f"protocol_matrix_{run.replace('/', '__')}.json")
    payload = {"run": run, "split": split, "grid": GRID,
               "results": {k: ({str(t): v for t, v in r.items()}
                               if isinstance(r, dict) else r)
                           for k, r in results.items()},
               "max_2d_disagreement": max_gap,
               "max_2d_disagreement_no_z_resampling": max(flat.values(), default=0.0),
               "per_patient_2d_disagreement": {"z_untouched": flat,
                                               "z_resampled": interpolating},
               # Named for what it is: an argmax over test, not a usable choice.
               "oracle_best_threshold_on_test": best,
               "best_threshold": best}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n  {out}")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Threshold x slice-protocol matrix for one run.")
    parser.add_argument("--run", required=True, help="exp_name/seed_dir")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--preprocessed_name", default="preprocessed")
    parser.add_argument("--metadata_name", default="metadata")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_matrix(args.run, split=args.split,
               preprocessed_name=args.preprocessed_name,
               metadata_name=args.metadata_name, out=args.out)


if __name__ == "__main__":
    main()
