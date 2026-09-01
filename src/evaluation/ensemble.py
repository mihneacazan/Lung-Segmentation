"""
Averages several seeds of the same experiment into one prediction, and scores it.

The three baseline seeds disagree far more than their scores suggest. On one test
patient, 970 voxels are called tumour by at least one of them and only 392 by all
three: forty per cent of the union comes from a single model. That disagreement is
run-to-run noise rather than signal, and averaging the probability maps cancels it
while leaving what the models agree on intact.

Two things about how the averaging is done matter more than they look.

**Probabilities are averaged, not masks.** Thresholding each model first and then
taking a majority vote is a different operation: a voxel read 0.45 / 0.45 / 0.95
averages to 0.617 and survives, but loses a 2-of-3 vote at threshold 0.5. Neither
rule is obviously right, and this module implements averaging because it keeps the
models' confidence rather than discarding it at the first step.

**The threshold has to be swept again.** Averaging compresses values toward the
middle — a voxel read 0.9 / 0.0 / 0.0 becomes 0.3 — so a threshold chosen for a
single model's distribution cuts far too aggressively into an averaged one.
Measured on lung_023, the ensemble scores 0.4102 at the committed threshold of
0.75, below two of the three individual models, and 0.4543 at 0.50, above all of
them at any threshold of their own. Inheriting the threshold is the difference
between the ensemble looking worse and looking better.

Averaging happens in the 192x192 grid, before reconstruction. Reconstruction
interpolates and only then thresholds, and interpolation is linear, so averaging
before it gives the same result as averaging after and costs one reconstruction
instead of three.

Usage:
    python -m src.evaluation.ensemble
    python -m src.evaluation.ensemble --exp_name baseline --seeds 42,43,44
"""

import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.evaluation.hierarchical_report import predict_patient, score_prediction
from src.evaluation.metrics import (
    DEFAULT_SWEEP_THRESHOLDS,
    filter_predicted_components,
    reconstruct_patient_3d_volume,
)
from src.models.factory import build_model


def load_seed_models(exp_name, seeds):
    """
    Loads every seed of one experiment, checking they are the same architecture.

    Returns:
        tuple: (list of models, device, config, {seed: its own threshold})
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, thresholds, config = [], {}, None

    for seed in seeds:
        run_dir = os.path.join(OUTPUT_DIR, "experiments", exp_name, f"seed_{seed}")
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)

        if config is None:
            config = cfg
        elif (cfg["model_type"], cfg["n_adjacent"]) != (config["model_type"],
                                                        config["n_adjacent"]):
            raise ValueError(
                f"seed {seed} is a {cfg['model_type']} with n_adjacent="
                f"{cfg['n_adjacent']}, but seed {seeds[0]} is a "
                f"{config['model_type']} with n_adjacent={config['n_adjacent']}. "
                f"Averaging predictions from different architectures is a "
                f"different experiment from averaging seeds of one.")

        model = build_model(cfg["model_type"], in_channels=cfg["n_adjacent"],
                            out_channels=1).to(device)
        state = torch.load(os.path.join(run_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        models.append(model)

        report_path = os.path.join(run_dir, "benchmark_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                thresholds[seed] = float(json.load(f).get("optimal_threshold", 0.5))

    return models, device, config, thresholds


def predict_all(models, device, config, case_ids, preprocessed_name="preprocessed"):
    """
    Runs every model over every patient, keeping each seed's map separately.

    Returns:
        dict: {case_id: array of shape (n_models, H, W, D)}
    """
    out = {}
    for case_id in case_ids:
        out[case_id] = np.stack([
            predict_patient(m, device, config, case_id, preprocessed_name)
            for m in models])
    return out


def sweep_threshold(probs, case_ids, preprocessed_name="preprocessed",
                    thresholds=None, verbose=True):
    """
    Picks the threshold maximising mean per-patient Dice on the given split.

    Scored in the 192x192 grid against the preprocessed labels, which is what
    `threshold_sweep` does for single models and is two orders of magnitude
    cheaper than reconstructing every candidate.

    This must only ever be called on validation. A threshold chosen on test
    would leak the test set into model selection.
    """
    thresholds = thresholds or DEFAULT_SWEEP_THRESHOLDS
    volumes = os.path.join(OUTPUT_DIR, preprocessed_name, "volumes")
    labels = {c: np.load(os.path.join(volumes, f"{c}_lbl.npy"), mmap_mode="r")
              for c in case_ids}

    results = {}
    for t in thresholds:
        scores = []
        for c in case_ids:
            pred = probs[c] > t
            gt = np.asarray(labels[c]) > 0.5
            total = pred.sum() + gt.sum()
            scores.append(1.0 if total == 0
                          else 2.0 * float(np.logical_and(pred, gt).sum()) / total)
        results[float(t)] = float(np.mean(scores))
        if verbose:
            print(f"    [{t:.3f}] mean val Dice = {results[float(t)]:.4f}", flush=True)

    best = max(results, key=results.get)
    return best, results


def score_split(probs, case_ids, threshold, postproc_min_fraction=0.10):
    """Reconstructs into original geometry and scores, raw and post-processed."""
    rows, pp_rows = [], []
    for case_id in case_ids:
        with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
            metadata = json.load(f)

        pred = reconstruct_patient_3d_volume(probs[case_id], metadata,
                                             threshold=threshold, binarize=True)
        gt = (np.asanyarray(nib.load(
            resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")).dataobj)
            > 0.5).astype(np.uint8)

        scores, _ = score_prediction(pred, gt)
        rows.append({"case_id": case_id, **scores})

        cleaned, removed = filter_predicted_components(
            pred, min_fraction=postproc_min_fraction)
        pp_scores, _ = score_prediction(cleaned, gt)
        pp_rows.append({"case_id": case_id, "components_removed": removed,
                        **pp_scores})
    return rows, pp_rows


def summarise(rows):
    """Macro-averages the per-patient rows, skipping undefined entries."""
    keys = ("dice_3d_patient", "sensitivity_3d_patient", "precision_3d_patient",
            "dice_2d_tumour_slices", "dice_2d_all_slices", "failure_rate_2d",
            "false_alarm_rate_2d", "fp_components")
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["failures"] = sum(1 for r in rows if r["dice_3d_patient"] < 0.10)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Average several seeds of one experiment and score the result.")
    parser.add_argument("--exp_name", default="baseline")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--preprocessed_name", default="preprocessed")
    parser.add_argument("--postproc_min_fraction", type=float, default=0.10)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "ensemble_report.json"))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    with open(os.path.join(OUTPUT_DIR, args.preprocessed_name, "index.json")) as f:
        index = json.load(f)
    val_ids = sorted(index["splits"]["val"])
    test_ids = sorted(index["splits"]["test"])

    print(f"=== SEED ENSEMBLE: {args.exp_name}, seeds {seeds} ===\n")
    models, device, config, own_thresholds = load_seed_models(args.exp_name, seeds)
    print(f"  {len(models)} x {config['model_type']} "
          f"(n_adjacent={config['n_adjacent']}) on {device}")
    print(f"  Their own thresholds: "
          f"{', '.join(f'{s}:{own_thresholds.get(s, 0.5):.2f}' for s in seeds)}")
    print(f"  {len(val_ids)} validation, {len(test_ids)} test patients\n")

    started = time.time()
    print("--- Inference on validation ---", flush=True)
    val_raw = predict_all(models, device, config, val_ids, args.preprocessed_name)
    val_mean = {c: v.mean(axis=0) for c, v in val_raw.items()}
    print(f"  {time.time() - started:.0f}s\n")

    print("--- Threshold sweep on validation, for the ensemble ---", flush=True)
    threshold, sweep = sweep_threshold(val_mean, val_ids, args.preprocessed_name)
    print(f"  => ensemble threshold {threshold:.3f} "
          f"(val Dice {sweep[threshold]:.4f})\n", flush=True)
    del val_raw, val_mean

    print("--- Inference on test ---", flush=True)
    t0 = time.time()
    test_raw = predict_all(models, device, config, test_ids, args.preprocessed_name)
    print(f"  {time.time() - t0:.0f}s\n", flush=True)

    results = {"exp_name": args.exp_name, "seeds": seeds,
               "ensemble_threshold": threshold,
               "ensemble_val_sweep": {str(k): v for k, v in sweep.items()},
               "individual": {}, "ensemble": {}}

    # Each seed alone, at the threshold its own run selected on validation. This
    # is the fair comparison: the ensemble is not being credited for a fresh
    # threshold that the individuals were denied.
    print("--- Each seed alone, at its own validation threshold ---", flush=True)
    for i, seed in enumerate(seeds):
        t = own_thresholds.get(seed, 0.5)
        probs = {c: test_raw[c][i] for c in test_ids}
        rows, pp_rows = score_split(probs, test_ids, t, args.postproc_min_fraction)
        results["individual"][str(seed)] = {
            "threshold": t, "summary": summarise(rows),
            "postprocessed_summary": summarise(pp_rows),
            "per_patient": rows}
        s = summarise(rows)
        print(f"  seed {seed} @ {t:.2f}   Dice 3D {s['dice_3d_patient']:.4f} | "
              f"2D tumour {s['dice_2d_tumour_slices']:.4f} | "
              f"sens {s['sensitivity_3d_patient']:.4f} | "
              f"prec {s['precision_3d_patient']:.4f}", flush=True)

    print("\n--- Ensemble ---", flush=True)
    mean = {c: test_raw[c].mean(axis=0) for c in test_ids}
    del test_raw
    rows, pp_rows = score_split(mean, test_ids, threshold, args.postproc_min_fraction)
    results["ensemble"] = {"threshold": threshold, "summary": summarise(rows),
                           "postprocessed_summary": summarise(pp_rows),
                           "per_patient": rows}
    s, pp = summarise(rows), summarise(pp_rows)
    print(f"  ensemble @ {threshold:.2f}   Dice 3D {s['dice_3d_patient']:.4f} | "
          f"2D tumour {s['dice_2d_tumour_slices']:.4f} | "
          f"sens {s['sensitivity_3d_patient']:.4f} | "
          f"prec {s['precision_3d_patient']:.4f}", flush=True)
    print(f"  post-processed            Dice 3D {pp['dice_3d_patient']:.4f}",
          flush=True)

    best_single = max(results["individual"].values(),
                      key=lambda r: r["summary"]["dice_3d_patient"])
    mean_single = float(np.mean([r["summary"]["dice_3d_patient"]
                                 for r in results["individual"].values()]))
    results["comparison"] = {
        "best_single_seed": best_single["summary"]["dice_3d_patient"],
        "mean_of_seeds": mean_single,
        "ensemble": s["dice_3d_patient"],
        "gain_over_best_seed": s["dice_3d_patient"] - best_single["summary"]["dice_3d_patient"],
        "gain_over_mean_of_seeds": s["dice_3d_patient"] - mean_single,
    }

    print(f"\n{'=' * 70}")
    print(f"  mean of the three seeds : {mean_single:.4f}")
    print(f"  best single seed        : {best_single['summary']['dice_3d_patient']:.4f}")
    print(f"  ensemble                : {s['dice_3d_patient']:.4f}")
    print(f"  gain over best seed     : "
          f"{results['comparison']['gain_over_best_seed']:+.4f}")
    print(f"  gain over seed average  : "
          f"{results['comparison']['gain_over_mean_of_seeds']:+.4f}")
    print("=" * 70)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  {args.out}")
    print(f"  total {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
