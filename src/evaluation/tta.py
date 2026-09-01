"""
Test-time augmentation over the intensity transform the model was trained with.

A convolutional network is not exactly invariant to the variations it was
augmented against: the same slice, brightened slightly, produces a slightly
different probability map. Neither map is the right one. Averaging over a few
variants is more stable than trusting any single pass, and it costs no training.

Which transforms are usable is decided by what training used, not by convention.
`_augment_anatomic` applies rotation, translation, scale, gamma and noise, and
**deliberately excludes horizontal flip** because mirroring a chest produces
dextrocardia, which occurs in about 1 in 10,000 people and appears nowhere in
this dataset. Flip TTA — the reflex choice, and the one most papers use — would
put every second pass outside the distribution the model has ever seen.

This module implements the gamma variants only, and the reason is geometric. An
intensity change needs no undoing: the probability map comes back already aligned
with the original slice, so the average is exact. A rotation does need undoing,
and that inverse is a second interpolation across the probability map. On a
lesion 20 to 30 pixels across in a 192 grid, that blurs precisely the boundary
the score is most sensitive to. Gamma TTA is therefore free of the cost that
makes geometric TTA ambiguous, and measuring it alone says whether the averaging
helps before any blurring is introduced.

The gamma grid matches training: `_augment_anatomic` draws from U(0.8, 1.25) and
applies `x ** gamma` to an image already normalised to [0, 1]. Identity is
included, so the model's ordinary prediction is one of the votes.

As with the seed ensemble, the threshold must be swept again. Averaging
compresses values toward the middle, and a threshold chosen for a single pass
cuts too deep into an averaged map.

Usage:
    python -m src.evaluation.tta
    python -m src.evaluation.tta --exp_name baseline --seed 42
    python -m src.evaluation.tta --gammas 0.9,1.0,1.1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR
from src.evaluation.ensemble import score_split, summarise, sweep_threshold
from src.evaluation.hierarchical_report import load_run, predict_patient


# Training draws gamma from U(0.8, 1.25). These five span it and include the
# identity, so the ordinary prediction is one of the votes rather than being
# replaced by a set of distorted ones.
DEFAULT_GAMMAS = (0.8, 0.9, 1.0, 1.1, 1.25)


def predict_with_tta(model, device, config, case_id, gammas,
                     preprocessed_name="preprocessed"):
    """
    Averages one model's probability maps over several gamma-adjusted inputs.

    Neighbour selection for the 2.5D models replicates `LungSliceDataset`, edge
    clipping included, because `predict_patient` does the work and gamma is
    applied to the stored volume before that.

    Args:
        model, device, config: As returned by `load_run`.
        case_id (str): Patient identifier.
        gammas (sequence): Exponents applied as `x ** gamma` to the normalised
            image. 1.0 leaves it unchanged.
        preprocessed_name (str): Which preprocessed dataset to read.

    Returns:
        np.ndarray: (H, W, D) mean probability over the gamma variants.
    """
    volumes = os.path.join(OUTPUT_DIR, preprocessed_name, "volumes")
    original = np.clip(
        np.asarray(np.load(os.path.join(volumes, f"{case_id}_img.npy"),
                           mmap_mode="r"), dtype=np.float32), 0.0, 1.0)

    total = None
    for gamma in gammas:
        # Clipping happens once above, matching training, where the augmented
        # image is clipped to [0, 1] before the exponent is applied. Gamma 1.0
        # is left as an explicit identity rather than computed, so the ordinary
        # prediction is bit-identical to what the model reports without TTA.
        variant = original if gamma == 1.0 else original ** float(gamma)
        probs = predict_patient(model, device, config, case_id,
                                preprocessed_name, img=variant)
        total = probs if total is None else total + probs

    return total / len(gammas)


def predict_all_tta(model, device, config, case_ids, gammas,
                    preprocessed_name="preprocessed"):
    """Runs `predict_with_tta` over a split."""
    return {c: predict_with_tta(model, device, config, c, gammas,
                                preprocessed_name)
            for c in case_ids}


def main():
    parser = argparse.ArgumentParser(
        description="Score one checkpoint with and without gamma test-time augmentation.")
    parser.add_argument("--exp_name", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gammas", default=",".join(str(g) for g in DEFAULT_GAMMAS),
                        help="Comma-separated exponents. Include 1.0.")
    parser.add_argument("--preprocessed_name", default="preprocessed")
    parser.add_argument("--postproc_min_fraction", type=float, default=0.10)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "tta_report.json"))
    args = parser.parse_args()

    gammas = tuple(float(g) for g in args.gammas.split(","))
    if 1.0 not in gammas:
        print("  [!] 1.0 is not in the gamma grid, so the model's ordinary "
              "prediction is not one of the votes.")

    run_dir = os.path.join(OUTPUT_DIR, "experiments", args.exp_name,
                           f"seed_{args.seed}")
    with open(os.path.join(OUTPUT_DIR, args.preprocessed_name, "index.json")) as f:
        index = json.load(f)
    val_ids = sorted(index["splits"]["val"])
    test_ids = sorted(index["splits"]["test"])

    model, device, config, own_threshold = load_run(run_dir)
    print(f"=== GAMMA TTA: {args.exp_name} seed {args.seed} ===\n")
    print(f"  {config['model_type']} (n_adjacent={config['n_adjacent']}) on {device}")
    print(f"  Gammas: {gammas}")
    print(f"  Its own threshold, swept on validation without TTA: {own_threshold:.2f}")
    print(f"  {len(val_ids)} validation, {len(test_ids)} test patients\n")

    started = time.time()
    print("--- Inference on validation, with TTA ---", flush=True)
    val_probs = predict_all_tta(model, device, config, val_ids, gammas,
                                args.preprocessed_name)
    print(f"  {time.time() - started:.0f}s\n", flush=True)

    print("--- Threshold sweep on validation, for the TTA prediction ---", flush=True)
    threshold, sweep = sweep_threshold(val_probs, val_ids, args.preprocessed_name)
    print(f"  => TTA threshold {threshold:.3f} (val Dice {sweep[threshold]:.4f})\n",
          flush=True)
    del val_probs

    print("--- Inference on test ---", flush=True)
    t0 = time.time()
    tta_probs = predict_all_tta(model, device, config, test_ids, gammas,
                                args.preprocessed_name)
    plain_probs = {c: predict_patient(model, device, config, c,
                                      args.preprocessed_name) for c in test_ids}
    print(f"  {time.time() - t0:.0f}s\n", flush=True)

    results = {"exp_name": args.exp_name, "seed": args.seed,
               "gammas": list(gammas), "tta_threshold": threshold,
               "plain_threshold": own_threshold,
               "tta_val_sweep": {str(k): v for k, v in sweep.items()}}

    for tag, probs, thr in (("plain", plain_probs, own_threshold),
                            ("tta", tta_probs, threshold)):
        rows, pp_rows = score_split(probs, test_ids, thr,
                                    args.postproc_min_fraction)
        results[tag] = {"threshold": thr, "summary": summarise(rows),
                        "postprocessed_summary": summarise(pp_rows),
                        "per_patient": rows}
        s = summarise(rows)
        print(f"  {tag:6} @ {thr:.2f}   Dice 3D {s['dice_3d_patient']:.4f} | "
              f"2D tumour {s['dice_2d_tumour_slices']:.4f} | "
              f"sens {s['sensitivity_3d_patient']:.4f} | "
              f"prec {s['precision_3d_patient']:.4f}", flush=True)

    # TTA at the un-augmented threshold as well, because the whole point of the
    # separate sweep is that inheriting the threshold is what makes averaging
    # look worse than it is. Reporting both keeps that visible.
    rows_inherit, _ = score_split(tta_probs, test_ids, own_threshold,
                                  args.postproc_min_fraction)
    results["tta_at_inherited_threshold"] = {
        "threshold": own_threshold, "summary": summarise(rows_inherit)}

    plain = results["plain"]["summary"]["dice_3d_patient"]
    tta = results["tta"]["summary"]["dice_3d_patient"]
    inherit = summarise(rows_inherit)["dice_3d_patient"]

    print(f"\n{'=' * 70}")
    print(f"  no TTA                        : {plain:.4f}")
    print(f"  TTA at the inherited threshold: {inherit:.4f}  ({inherit - plain:+.4f})")
    print(f"  TTA with its own threshold    : {tta:.4f}  ({tta - plain:+.4f})")
    print("=" * 70)

    results["comparison"] = {"plain": plain, "tta": tta,
                             "tta_at_inherited_threshold": inherit,
                             "gain": tta - plain}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  {args.out}")
    print(f"  total {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
