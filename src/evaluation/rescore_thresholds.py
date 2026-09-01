"""
Re-selects every committed run's binarization threshold in the space it is scored in.

Every experiment in this project chose its threshold with `threshold_sweep`, which
searches the preprocessed 192x192 grid, and then reported its result in original
NIfTI geometry. Those are different objectives, and the argmax of one is not the
argmax of the other — `threshold_sweep`'s docstring explains the mechanism and why
the claim that they agree was wrong.

This script re-runs the selection with `threshold_sweep_original_geometry` from the
saved checkpoints. No model is retrained and no weight changes: the threshold is a
post-training parameter, chosen after `best_model.pt` is already frozen. What
changes is only the number the probability map is cut at.

Both thresholds are scored on test so the delta is explicit rather than implied. The
new threshold still comes from validation alone, so this is a correction, not a leak.

Results are written to a fresh report rather than overwriting each run's
`benchmark_report.json`, so the original numbers stay auditable and the migration can
be reviewed before anything committed is touched.

Usage:
    python -m src.evaluation.rescore_thresholds
    python -m src.evaluation.rescore_thresholds --runs baseline/seed_42,segresnet/seed_42
    python -m src.evaluation.rescore_thresholds --workers 4
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.evaluation.hierarchical_report import load_run, predict_patient, score_prediction
from src.evaluation.metrics import (
    filter_predicted_components,
    reconstruct_patient_3d_volume,
    threshold_sweep_original_geometry,
)


def discover_runs(experiments_dir):
    """Finds every seed directory holding both a checkpoint and a config."""
    runs = []
    for exp in sorted(os.listdir(experiments_dir)):
        exp_dir = os.path.join(experiments_dir, exp)
        if not os.path.isdir(exp_dir):
            continue
        for seed_dir in sorted(os.listdir(exp_dir)):
            run_dir = os.path.join(exp_dir, seed_dir)
            if (os.path.exists(os.path.join(run_dir, "best_model.pt"))
                    and os.path.exists(os.path.join(run_dir, "config.json"))):
                runs.append(f"{exp}/{seed_dir}")
    return runs


def committed_threshold(run_dir, default=0.5):
    """Reads the threshold this run originally selected, for the comparison."""
    path = os.path.join(run_dir, "benchmark_report.json")
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return float(json.load(f).get("optimal_threshold", default))


def score_at(probs, case_ids, threshold, postproc_min_fraction=0.10):
    """Reconstructs into original geometry at one threshold and macro-averages."""
    rows = []
    for case_id in case_ids:
        with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
            metadata = json.load(f)
        pred = reconstruct_patient_3d_volume(probs[case_id], metadata,
                                             threshold=threshold, binarize=True)
        gt = (np.asanyarray(nib.load(
            resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")).dataobj)
            > 0.5).astype(np.uint8)
        scores, _ = score_prediction(pred, gt)
        if postproc_min_fraction:
            cleaned, _ = filter_predicted_components(
                pred, min_fraction=postproc_min_fraction)
            pp, _ = score_prediction(cleaned, gt)
            scores["pp_dice_3d_patient"] = pp["dice_3d_patient"]
        rows.append(scores)

    keys = ("dice_3d_patient", "sensitivity_3d_patient", "precision_3d_patient",
            "dice_2d_tumour_slices", "fp_components", "pp_dice_3d_patient")
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def rescore_one(run, preprocessed_name="preprocessed", postproc_min_fraction=0.10):
    """
    Re-selects and re-scores one run. Returns a plain dict so it survives the
    process boundary when run in parallel.
    """
    run_dir = os.path.join(OUTPUT_DIR, "experiments", *run.split("/"))
    with open(os.path.join(OUTPUT_DIR, preprocessed_name, "index.json")) as f:
        index = json.load(f)
    val_ids = sorted(index["splits"]["val"])
    test_ids = sorted(index["splits"]["test"])

    started = time.time()
    model, device, config, _ = load_run(run_dir)
    old_threshold = committed_threshold(run_dir)

    val_probs = {c: predict_patient(model, device, config, c, preprocessed_name)
                 for c in val_ids}
    new_threshold, sweep = threshold_sweep_original_geometry(
        val_probs, os.path.join(OUTPUT_DIR, "metadata"), verbose=False)
    del val_probs

    test_probs = {c: predict_patient(model, device, config, c, preprocessed_name)
                  for c in test_ids}
    old_scores = score_at(test_probs, test_ids, old_threshold, postproc_min_fraction)
    new_scores = (old_scores if new_threshold == old_threshold
                  else score_at(test_probs, test_ids, new_threshold,
                                postproc_min_fraction))

    return {
        "run": run,
        "model_type": config["model_type"],
        "n_adjacent": config["n_adjacent"],
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "val_sweep": {str(k): v for k, v in sweep.items()},
        "old": old_scores,
        "new": new_scores,
        "delta_dice_3d": new_scores["dice_3d_patient"] - old_scores["dice_3d_patient"],
        "seconds": time.time() - started,
    }


def _worker(run):
    """
    Top-level so ProcessPoolExecutor can pickle it.

    Each result is written to its own file the moment it exists. The first
    version of this script accumulated everything in memory and wrote once at
    the end; the process was killed after four hours with sixteen of
    twenty-three runs finished, and only the console log survived. A run costs
    tens of minutes, so losing a completed one to a crash is the expensive
    failure here, not the few milliseconds this write costs.
    """
    try:
        result = rescore_one(run)
    except Exception as exc:                                # noqa: BLE001
        result = {"run": run, "error": f"{type(exc).__name__}: {exc}"}

    partial_dir = os.path.join(OUTPUT_DIR, "rescored_partial")
    os.makedirs(partial_dir, exist_ok=True)
    with open(os.path.join(partial_dir, run.replace("/", "__") + ".json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Re-select every run's threshold in original geometry.")
    parser.add_argument("--runs", default=None,
                        help="Comma-separated exp/seed pairs. Default: all of them.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes. Reconstruction is single-threaded "
                             "and CPU-bound, so this scales close to linearly until "
                             "memory runs out; each worker holds one ~345 MB volume.")
    parser.add_argument("--postproc_min_fraction", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already written to output/rescored_partial/, "
                             "so a killed job is continued rather than restarted.")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "rescored_thresholds"))
    args = parser.parse_args()

    experiments_dir = os.path.join(OUTPUT_DIR, "experiments")
    runs = (args.runs.split(",") if args.runs else discover_runs(experiments_dir))

    partial_dir = os.path.join(OUTPUT_DIR, "rescored_partial")
    if args.resume and os.path.isdir(partial_dir):
        done = {f[:-5].replace("__", "/") for f in os.listdir(partial_dir)
                if f.endswith(".json")}
        skipped = [r for r in runs if r in done]
        runs = [r for r in runs if r not in done]
        if skipped:
            print(f"  Resuming: {len(skipped)} already on disk, skipping them.")

    print(f"=== THRESHOLD RE-SELECTION IN ORIGINAL GEOMETRY ===\n")
    print(f"  {len(runs)} runs, {args.workers} worker(s)")
    print(f"  No retraining: every checkpoint is read as committed.\n", flush=True)

    started = time.time()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for res in pool.map(_worker, runs):
                results.append(res)
                _report_one(res)
    else:
        for run in runs:
            res = _worker(run)
            results.append(res)
            _report_one(res)

    ok = [r for r in results if "error" not in r]
    moved = [r for r in ok if r["new_threshold"] != r["old_threshold"]]
    deltas = [r["delta_dice_3d"] for r in ok]

    print(f"\n{'=' * 78}")
    print(f"  {len(ok)}/{len(runs)} runs rescored, {len(moved)} changed threshold")
    if deltas:
        print(f"  mean delta Dice 3D : {np.mean(deltas):+.4f}")
        print(f"  median             : {np.median(deltas):+.4f}")
        print(f"  worst / best       : {min(deltas):+.4f} / {max(deltas):+.4f}")
        print(f"  runs that got worse: {sum(1 for d in deltas if d < -1e-9)}")
    print(f"  total {(time.time() - started) / 60:.1f} min")
    print("=" * 78)

    with open(args.out + ".json", "w") as f:
        json.dump({"results": results}, f, indent=2, default=float)

    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "old_threshold", "new_threshold",
                         "old_dice_3d", "new_dice_3d", "delta",
                         "old_sensitivity", "new_sensitivity",
                         "old_precision", "new_precision"])
        for r in sorted(ok, key=lambda x: -x["delta_dice_3d"]):
            writer.writerow([
                r["run"], r["old_threshold"], r["new_threshold"],
                round(r["old"]["dice_3d_patient"], 6),
                round(r["new"]["dice_3d_patient"], 6),
                round(r["delta_dice_3d"], 6),
                round(r["old"]["sensitivity_3d_patient"], 6),
                round(r["new"]["sensitivity_3d_patient"], 6),
                round(r["old"]["precision_3d_patient"], 6),
                round(r["new"]["precision_3d_patient"], 6)])

    print(f"\n  {args.out}.json")
    print(f"  {args.out}.csv")


def _report_one(res):
    if "error" in res:
        print(f"  [FAIL] {res['run']:44s} {res['error']}", flush=True)
        return
    arrow = "->" if res["new_threshold"] != res["old_threshold"] else "=="
    print(f"  {res['run']:44s} {res['old_threshold']:.2f} {arrow} "
          f"{res['new_threshold']:.2f}   Dice {res['old']['dice_3d_patient']:.4f} "
          f"-> {res['new']['dice_3d_patient']:.4f}  "
          f"({res['delta_dice_3d']:+.4f})  [{res['seconds']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
