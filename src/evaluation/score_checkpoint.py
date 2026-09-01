"""
Scores an arbitrary checkpoint of a finished run, threshold chosen on validation.

Exists for one question: how much of a reported number is selection luck?

Every result in this project comes from the epoch that scored best on validation.
But validation jitter is not constant across training - measured on the committed
baseline, the mean epoch-to-epoch swing is 0.056 while the learning rate is high
and 0.005 once the cosine has annealed, a factor of twelve. An argmax over a
curve like that lands in the noisy phase almost regardless of which model is
genuinely better, and every committed best-epoch does: seeds 42, 43 and 44 kept
epochs 35, 27 and 36, at learning rates of 2.1e-4, 4.4e-4 and 1.9e-4, never from
the annealed tail.

So the selected number is an upper-biased draw. Scoring the *final* annealed
weights next to it puts a size on that bias, and costs no training - the
checkpoint is already on disk.

The threshold is re-swept on validation for whichever weights are being scored,
because a threshold chosen for one set of weights is not the right threshold for
another. Test is touched only once the threshold is fixed.

Usage:
    python -m src.evaluation.score_checkpoint --run baseline/seed_42
    python -m src.evaluation.score_checkpoint --run baseline/seed_42 \
        --weights checkpoint.pt
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
from src.evaluation.hierarchical_report import load_run, predict_patient, score_prediction
from src.evaluation.metrics import (
    reconstruct_patient_3d_volume,
    threshold_sweep_original_geometry,
)

REPORT_KEYS = ("dice_3d_patient", "sensitivity_3d_patient",
               "precision_3d_patient", "volume_ratio_3d_patient",
               "fp_components")


def score_weights(run, weights="best_model.pt", preprocessed_name="preprocessed",
                  metadata_name="metadata", output_dir=None):
    """
    Sweeps the threshold on validation for these weights, then scores test.

    Returns a plain dict, so a notebook can put two of them side by side.
    """
    base = output_dir or OUTPUT_DIR
    run_dir = os.path.join(base, "experiments", *run.split("/"))
    metadata_dir = os.path.join(base, metadata_name)

    with open(os.path.join(base, preprocessed_name, "index.json")) as f:
        index = json.load(f)

    model, device, config, _ = load_run(run_dir, weights=weights)

    val_probs = {c: predict_patient(model, device, config, c, preprocessed_name)
                 for c in sorted(index["splits"]["val"])}
    threshold, _ = threshold_sweep_original_geometry(
        val_probs, metadata_dir, verbose=False)
    del val_probs

    rows = []
    for case_id in sorted(index["splits"]["test"]):
        with open(os.path.join(metadata_dir, f"{case_id}.json")) as f:
            metadata = json.load(f)
        probs = predict_patient(model, device, config, case_id, preprocessed_name)
        pred = reconstruct_patient_3d_volume(probs, metadata,
                                             threshold=threshold, binarize=True)
        gt = (np.asanyarray(nib.load(
            resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")).dataobj)
            > 0.5).astype(np.uint8)
        scores, _ = score_prediction(pred, gt)
        gt_voxels = float(gt.sum())
        scores["volume_ratio_3d_patient"] = (float(pred.sum()) / gt_voxels
                                             if gt_voxels else float("nan"))
        rows.append(scores)
        del probs, pred, gt

    out = {"run": run, "weights": weights, "threshold": float(threshold)}
    for key in REPORT_KEYS:
        vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["volume_ratio_median"] = float(np.median(
        [r["volume_ratio_3d_patient"] for r in rows
         if not np.isnan(r["volume_ratio_3d_patient"])]))
    out["per_patient_dice"] = {
        c: r["dice_3d_patient"]
        for c, r in zip(sorted(index["splits"]["test"]), rows)}
    return out


def compare_best_and_final(run, **kwargs):
    """
    Both checkpoints of one run, with the gap between them.

    A positive `delta` means the annealed weights beat the ones selection kept,
    which would say the selection is costing rather than gaining.
    """
    best = score_weights(run, weights="best_model.pt", **kwargs)
    final = score_weights(run, weights="checkpoint.pt", **kwargs)
    return {"best": best, "final": final,
            "delta": final["dice_3d_patient"] - best["dice_3d_patient"]}


def print_comparison(result):
    best, final = result["best"], result["final"]
    print(f"\n  {'checkpoint':16s} {'prag':>6s} {'Dice 3D':>9s} {'sens':>7s} "
          f"{'prec':>7s} {'vol med':>8s} {'FP':>6s}")
    print("  " + "-" * 62)
    for label, row in (("best pe validare", best), ("final, recopt", final)):
        print(f"  {label:16s} {row['threshold']:6.2f} "
              f"{row['dice_3d_patient']:9.4f} "
              f"{row['sensitivity_3d_patient']:7.4f} "
              f"{row['precision_3d_patient']:7.4f} "
              f"{row['volume_ratio_median']:8.3f} {row['fp_components']:6.2f}")
    delta = result["delta"]
    print(f"\n  final - best: {delta:+.4f}")
    if delta > 0:
        print("  Greutatile recoapte bat epoca aleasa pe validare. Selectia pe")
        print("  argmax nu castiga aici, pierde - ceea ce e de asteptat cand")
        print("  jitterul curbei scade de doisprezece ori pe parcurs.")
    else:
        print("  Epoca aleasa pe validare rezista si pe test, deci selectia nu")
        print("  a fost doar noroc pe o curba zgomotoasa.")


def main():
    parser = argparse.ArgumentParser(
        description="Score one run's checkpoints, threshold swept on validation.")
    parser.add_argument("--run", required=True, help="exp_name/seed_dir")
    parser.add_argument("--weights", default=None,
                        help="A single weights file. Omit to compare "
                             "best_model.pt against checkpoint.pt.")
    parser.add_argument("--preprocessed_name", default="preprocessed")
    parser.add_argument("--metadata_name", default="metadata")
    args = parser.parse_args()

    kwargs = dict(preprocessed_name=args.preprocessed_name,
                  metadata_name=args.metadata_name)
    if args.weights:
        out = score_weights(args.run, weights=args.weights, **kwargs)
        print(json.dumps(out, indent=2, default=float))
    else:
        print_comparison(compare_best_and_final(args.run, **kwargs))


if __name__ == "__main__":
    main()
