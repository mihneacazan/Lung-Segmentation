"""
Standalone evaluation for a trained segmentation model.

Takes a checkpoint and a split, reconstructs every prediction into the patient's
original NIfTI geometry, and reports the full metric set per patient and in
aggregate. Kept separate from training so that a model can be re-evaluated
without retraining, and so that the same evaluation code is provably shared by
every experiment.

Reported per patient: Dice, IoU, precision, sensitivity, specificity, HD95, ASD,
number of false-positive connected components, and a failure flag.
Reported in aggregate: macro-average, micro-average, and a breakdown by tumour
size category.

Threshold selection is allowed on validation only. Sweeping on test would make
the test set part of model selection and the reported number would no longer be
an estimate of held-out performance.

Usage:
    python -m src.evaluation.evaluate \
        --checkpoint output/experiments/baseline/seed_42/best_model.pt --split test

    python -m src.evaluation.evaluate \
        --checkpoint output/experiments/baseline/seed_42/best_model.pt \
        --split val --sweep_threshold
"""

import os
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import nibabel as nib

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.models.factory import build_model, MODEL_TYPES
from src.training.dataset import build_dataloaders
from src.evaluation.metrics import (
    compute_2d_slice_metrics,
    compute_all_3d_metrics,
    filter_predicted_components,
    reconstruct_patient_3d_volume,
    count_model_parameters,
    measure_gpu_memory_mb,
    export_results_csv,
    compute_macro_micro_averages,
    stratified_report,
    threshold_sweep,
)


# ============================================================================
#  FAST VALIDATION (used during training)
# ============================================================================

def evaluate_fast(model, dataloader, device, threshold=0.5):
    """
    Computes per-patient 3D Dice in preprocessed 192x192 space, accumulating
    streaming sums so no probability volume is ever held in memory.

    Returns both a hard and a soft Dice:

        hard  binarizes at `threshold`, and is the number eventually reported
        soft  uses probabilities directly, 2*sum(p*g) / (sum(p) + sum(g))

    Soft Dice exists because hard Dice is a useless training signal early on. A
    freshly initialised network outputs probabilities well below 0.5 everywhere,
    so hard Dice sits at exactly 0.0 for many epochs and then jumps the moment
    the probabilities cross the threshold. Early stopping watching that flat line
    terminates training during the phase where the network is in fact learning
    steadily, and the later jump looks like a breakthrough when it is only a
    thresholding artifact. Soft Dice moves smoothly from epoch one.

    Returns:
        dict: {case_id: {"dice_hard": float, "dice_soft": float}}
    """
    model.eval()
    acc = defaultdict(lambda: {"inter_h": 0.0, "pred_h": 0.0,
                               "inter_s": 0.0, "pred_s": 0.0, "gt": 0.0})

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            probs = torch.sigmoid(model(images))
            hard = (probs > threshold).float()

            dims = tuple(range(1, probs.ndim))
            stats = {
                "inter_h": (hard * labels).sum(dim=dims).cpu().numpy(),
                "pred_h": hard.sum(dim=dims).cpu().numpy(),
                "inter_s": (probs * labels).sum(dim=dims).cpu().numpy(),
                "pred_s": probs.sum(dim=dims).cpu().numpy(),
                "gt": labels.sum(dim=dims).cpu().numpy(),
            }

            for i, cid in enumerate(batch["case_id"]):
                for key, values in stats.items():
                    acc[cid][key] += float(values[i])

    def _dice(inter, pred, gt):
        if pred == 0 and gt == 0:
            return 1.0
        if pred == 0 or gt == 0:
            return 0.0
        return float(2.0 * inter / (pred + gt))

    return {
        cid: {
            "dice_hard": _dice(a["inter_h"], a["pred_h"], a["gt"]),
            "dice_soft": _dice(a["inter_s"], a["pred_s"], a["gt"]),
        }
        for cid, a in acc.items()
    }


# ============================================================================
#  PREDICTION COLLECTION
# ============================================================================

def collect_predictions(model, dataloader, device, sw_roi=None, sw_overlap=0.5):
    """
    Runs inference over a split, returning per-patient probability and
    ground-truth slice stacks plus wall-clock inference time per patient.

    Probabilities are kept as float16: a 9-patient validation split is roughly
    230 MB that way against 460 MB in float32, and the extra precision is
    irrelevant to a threshold comparison.

    With `sw_roi` set, the slice is covered by overlapping windows of that size
    and the per-window predictions are blended back into a full-size map. This
    exists for models trained on a window rather than a whole slice: a network
    only ever shown 96 px of context behaves differently when handed 192, and
    the effect is severe for the normalization layers that derive their
    statistics from the input itself. Tiling restores the field of view the model
    was trained under.

    Nothing about it consults the ground truth — window positions come from a
    fixed grid over the image, every pixel is covered, and the procedure runs
    identically on a patient with no annotation. It is the standard inference
    path for patch-trained segmentation networks.

    Windows are averaged in **probability** space, not logit space, and weighted
    with a Gaussian toward their centre so that predictions made at a window edge,
    where the model saw the least context, count for less than those made at the
    middle.

    Args:
        sw_roi (int|None): Window side length, or None for whole-slice inference.
        sw_overlap (float): Fraction of overlap between neighbouring windows.

    Returns:
        tuple: (probs, labels, inference_times), each keyed by case_id.
    """
    model.eval()

    if sw_roi is not None:
        from monai.inferers import sliding_window_inference

        def infer(x):
            return sliding_window_inference(
                x, roi_size=(sw_roi, sw_roi), sw_batch_size=8,
                predictor=lambda w: torch.sigmoid(model(w)),
                overlap=sw_overlap, mode="gaussian")
    else:
        def infer(x):
            return torch.sigmoid(model(x))

    probs = defaultdict(dict)
    labels = defaultdict(dict)
    times = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = infer(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            out = out.cpu().numpy().astype(np.float16)
            gts = batch["label"].numpy().astype(np.uint8)
            case_ids = batch["case_id"]
            slice_indices = batch["slice_idx"]

            for i, cid in enumerate(case_ids):
                s_idx = int(slice_indices[i])
                probs[cid][s_idx] = out[i, 0]
                labels[cid][s_idx] = gts[i, 0]
                times[cid] += elapsed / len(case_ids)

    return dict(probs), dict(labels), dict(times)


# ============================================================================
#  FULL EVALUATION IN ORIGINAL NIfTI GEOMETRY
# ============================================================================

def evaluate_full(probs, inference_times, metadata_dir, threshold=0.5,
                  save_nifti_dir=None, postproc_min_fraction=0.10,
                  surface_metrics=True):
    """
    Reconstructs each patient's prediction into their original NIfTI geometry and
    computes the full metric set against the untouched ground-truth mask.

    The ground truth is read directly from the source NIfTI, never from the
    preprocessed stack, so error introduced by preprocessing is charged to the
    result rather than hidden by evaluating in a space where it does not appear.

    When `postproc_min_fraction` is non-zero, every metric is computed twice: once
    on the raw prediction and once after dropping predicted connected components
    small relative to the largest. The second set is stored under `pp_`-prefixed
    keys, so a single run reports both and the effect of post-processing can be
    read off the same checkpoint rather than inferred across runs. Only the
    per-patient numbers and pooled counters are kept for the filtered variant,
    not the filtered volumes, which would double peak memory for no benefit.

    Args:
        probs: {case_id: {slice_index: 2D probability array}}.
        inference_times: {case_id: seconds}.
        metadata_dir (str): Directory of per-patient reconstruction metadata.
        threshold (float): Binarization threshold, chosen on validation.
        save_nifti_dir (str|None): If set, writes each prediction as .nii.gz with
            the patient's original affine.
        postproc_min_fraction (float): Component-size cutoff as a fraction of the
            largest component. 0 skips the post-processed metric set entirely.
        surface_metrics (bool): False skips HD95 and ASD, which dominate the cost
            of an evaluation. Use it for comparisons decided on overlap.

    Returns:
        tuple: (patient_metrics, patient_pred_masks, patient_gt_masks, pp_micro)
            where pp_micro holds pooled voxel counts for the filtered variant, or
            None when post-processing is disabled.
    """
    from src.geometry import save_prediction_nifti

    patient_metrics = {}
    pred_masks = {}
    gt_masks = {}
    pp_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0} if postproc_min_fraction else None

    if save_nifti_dir:
        os.makedirs(save_nifti_dir, exist_ok=True)

    for case_id, slice_dict in sorted(probs.items()):
        meta_path = os.path.join(metadata_dir, f"{case_id}.json")
        if not os.path.exists(meta_path):
            print(f"  [WARNING] No metadata for {case_id}, skipping.")
            continue

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        pred_3d = reconstruct_patient_3d_volume(
            slice_dict, metadata, threshold=threshold, binarize=True)

        gt_path = resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")
        gt_3d = (np.asanyarray(nib.load(gt_path).dataobj) > 0.5).astype(np.uint8)
        spacing = tuple(metadata["original_spacing"])

        metrics = compute_all_3d_metrics(pred_3d, gt_3d, spacing,
                                         surface_metrics=surface_metrics)
        # Per-slice views of the same prediction. Much of the nodule literature
        # reports Dice this way, so carrying both makes this project's numbers
        # comparable with published ones without a second scoring pass. They
        # cost roughly a tenth of a second per patient.
        metrics.update(compute_2d_slice_metrics(pred_3d, gt_3d))
        metrics["inference_time_sec"] = inference_times.get(case_id, 0.0)
        metrics["gt_empty"] = bool(gt_3d.sum() == 0)
        metrics["pred_empty"] = bool(pred_3d.sum() == 0)

        if postproc_min_fraction:
            pred_pp, n_removed = filter_predicted_components(
                pred_3d, min_fraction=postproc_min_fraction)
            pp_metrics = compute_all_3d_metrics(pred_pp, gt_3d, spacing,
                                                surface_metrics=surface_metrics)
            pp_metrics.update(compute_2d_slice_metrics(pred_pp, gt_3d))
            for key, value in pp_metrics.items():
                metrics[f"pp_{key}"] = value
            metrics["pp_components_removed"] = n_removed

            pred_b, gt_b = pred_pp.astype(bool), gt_3d.astype(bool)
            pp_counts["tp"] += int(np.logical_and(pred_b, gt_b).sum())
            pp_counts["fp"] += int(np.logical_and(pred_b, ~gt_b).sum())
            pp_counts["fn"] += int(np.logical_and(~pred_b, gt_b).sum())
            pp_counts["tn"] += int(np.logical_and(~pred_b, ~gt_b).sum())

            if save_nifti_dir:
                save_prediction_nifti(
                    pred_pp, metadata,
                    os.path.join(save_nifti_dir, f"{case_id}_pred_pp.nii.gz"))
            del pred_pp

        patient_metrics[case_id] = metrics
        pred_masks[case_id] = pred_3d
        gt_masks[case_id] = gt_3d

        if save_nifti_dir:
            save_prediction_nifti(
                pred_3d, metadata,
                os.path.join(save_nifti_dir, f"{case_id}_pred.nii.gz"))

    return patient_metrics, pred_masks, gt_masks, pp_counts


# ============================================================================
#  REPORTING
# ============================================================================

def print_report(title, patient_metrics, averages, strat, threshold):
    """Prints the aggregate metric report for a split."""
    def _mean(key):
        vals = [m[key] for m in patient_metrics.values()
                if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else 0.0

    failures = sum(1 for m in patient_metrics.values() if m["is_failure"])
    n = len(patient_metrics)
    empty_gt = sum(1 for m in patient_metrics.values() if m.get("gt_empty"))
    empty_pred = sum(1 for m in patient_metrics.values() if m.get("pred_empty"))

    print(f"\n{'=' * 75}")
    print(f"  {title}  ({n} patients, threshold={threshold:.2f})")
    print(f"{'=' * 75}")
    print(f"  Dice 3D:         {_mean('dice_3d'):.4f}")
    print(f"  IoU 3D:          {_mean('iou_3d'):.4f}")
    print(f"  Sensitivity:     {_mean('sensitivity_3d'):.4f}")
    print(f"  Precision:       {_mean('precision_3d'):.4f}")
    print(f"  Specificity:     {_mean('specificity_3d'):.4f}")
    print(f"  HD95:            {_mean('hd95_3d'):.2f} mm")
    print(f"  ASD:             {_mean('asd_3d'):.2f} mm")
    print(f"  FP components:   {_mean('fp_components'):.2f}")
    print(f"  Failure rate:    {100.0 * failures / max(n, 1):.1f}% ({failures}/{n})")
    print(f"  Inference/volume:{_mean('inference_time_sec'):.3f}s")
    print(f"  Empty GT masks:  {empty_gt} | empty predictions: {empty_pred}")
    print(f"\n  Macro Dice: {averages['macro']['dice_3d']:.4f} | "
          f"Micro Dice: {averages['micro']['dice_3d']:.4f}")

    # The same predictions scored per slice. Printed under its own heading and
    # never on the same line as the 3D numbers, because the two are routinely
    # confused across papers and the all-slice figure is far higher for reasons
    # that have nothing to do with the model being better.
    if any("dice_2d_all_slices" in m for m in patient_metrics.values()):
        n_pos = sum(m.get("n_tumour_slices", 0) for m in patient_metrics.values())
        n_hit = sum(m.get("n_tumour_slices_with_prediction", 0)
                    for m in patient_metrics.values())
        print("\n  Per-slice view of the same predictions:")
        print(f"    tumour slices only   Dice={_mean('dice_2d_tumour_slices'):.4f}  "
              f"IoU={_mean('iou_2d_tumour_slices'):.4f}  "
              f"Sens={_mean('sensitivity_2d_tumour_slices'):.4f}  "
              f"Prec={_mean('precision_2d_tumour_slices'):.4f}")
        print(f"    every slice          Dice={_mean('dice_2d_all_slices'):.4f}  "
              f"IoU={_mean('iou_2d_all_slices'):.4f}  "
              f"Sens={_mean('sensitivity_2d_all_slices'):.4f}  "
              f"Prec={_mean('precision_2d_all_slices'):.4f}")
        print(f"    {n_hit}/{n_pos} tumour slices received a prediction; "
              f"the rest are excluded from precision, not scored 1.0")
        print(f"    slice failure rate {100 * _mean('failure_rate_2d'):.1f}%  |  "
              f"false alarms on {100 * _mean('false_alarm_rate_2d'):.1f}% "
              f"of empty slices")

    print("\n  By tumour size:")
    for cat in ("small", "medium", "large"):
        info = strat.get(cat, {})
        if info.get("n_patients", 0) > 0:
            print(f"    {cat.upper():7s} ({info['n_patients']:2d} pts): "
                  f"Dice={info.get('mean_dice_3d', 0):.4f}  "
                  f"HD95={info.get('mean_hd95_3d', 0):.2f}mm  "
                  f"failure={info.get('failure_rate_pct', 0):.0f}%")
        else:
            print(f"    {cat.upper():7s}:  no patients in this split")

    print_postproc_delta(patient_metrics)
    print("=" * 75)


def print_postproc_delta(patient_metrics):
    """
    Prints the raw-vs-post-processed comparison, when both were computed.

    Shown as a delta rather than a second table because the question it answers
    is specifically whether dropping satellite components is worth it, and that
    is a difference, not a level.
    """
    if not any("pp_dice_3d" in m for m in patient_metrics.values()):
        return

    def _mean(key):
        vals = [m[key] for m in patient_metrics.values()
                if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else 0.0

    pairs = [("Dice 3D", "dice_3d", 4), ("IoU 3D", "iou_3d", 4),
             ("Sensitivity", "sensitivity_3d", 4), ("Precision", "precision_3d", 4),
             ("HD95 (mm)", "hd95_3d", 2), ("ASD (mm)", "asd_3d", 2),
             ("FP components", "fp_components", 2),
             ("Dice 2D tumour", "dice_2d_tumour_slices", 4),
             ("Dice 2D all", "dice_2d_all_slices", 4)]

    removed = _mean("pp_components_removed")
    print(f"\n  Post-processing (components < 10% of largest dropped, "
          f"{removed:.1f} removed/patient):")
    print(f"    {'metric':<15s} {'raw':>10s} {'filtered':>10s} {'delta':>10s}")
    for label, key, digits in pairs:
        raw, pp = _mean(key), _mean(f"pp_{key}")
        print(f"    {label:<15s} {raw:>10.{digits}f} {pp:>10.{digits}f} "
              f"{pp - raw:>+10.{digits}f}")

    fails = sum(1 for m in patient_metrics.values() if m.get("is_failure"))
    pp_fails = sum(1 for m in patient_metrics.values() if m.get("pp_is_failure"))
    print(f"    {'failures':<15s} {fails:>10d} {pp_fails:>10d} "
          f"{pp_fails - fails:>+10d}")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on a split, in original NIfTI geometry")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep_threshold", action="store_true",
                        help="Search for the best threshold. Validation only.")
    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES,
                        help="Defaults to the value in the checkpoint's config.json")
    parser.add_argument("--n_adjacent", type=int, default=None, choices=[1, 3, 5])
    parser.add_argument("--sw_roi", type=int, default=None,
                        help="Tile each slice with overlapping windows of this "
                             "size instead of predicting it whole. Use it to "
                             "match the field of view a window-trained model "
                             "(--crop tumor) saw during training. Uses no ground "
                             "truth: the grid is fixed and covers every pixel.")
    parser.add_argument("--sw_overlap", type=float, default=0.5,
                        help="Overlap fraction between neighbouring windows.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_nifti", action="store_true")
    parser.add_argument("--postproc_min_fraction", type=float, default=0.10,
                        help="Predicted components smaller than this fraction of "
                             "the largest are dropped in the post-processed "
                             "metric set. 0 disables it.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to the checkpoint's directory")
    args = parser.parse_args()

    if args.sweep_threshold and args.split == "test":
        parser.error(
            "Refusing to sweep the threshold on test. Choosing a threshold there "
            "makes the test set part of model selection, and the resulting score "
            "is no longer a held-out estimate. Sweep on --split val, then pass "
            "the chosen value to test with --threshold.")

    exp_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    output_dir = args.output_dir or exp_dir
    os.makedirs(output_dir, exist_ok=True)

    # Recover the architecture from the config saved next to the checkpoint.
    cfg = {}
    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    model_type = args.model_type or cfg.get("model_type", "unet")
    n_adjacent = args.n_adjacent or cfg.get("n_adjacent", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_type} | n_adjacent: {n_adjacent} | device: {device}")

    model = build_model(model_type, in_channels=n_adjacent, out_channels=1).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    trainable, total = count_model_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    preprocessed_dir = os.path.join(OUTPUT_DIR, "preprocessed")
    metadata_dir = os.path.join(OUTPUT_DIR, "metadata")
    loaders, datasets, index = build_dataloaders(
        preprocessed_dir, batch_size=args.batch_size, sampling="all",
        augment="none", n_adjacent=n_adjacent, num_workers=args.num_workers)

    print(f"\nRunning inference on '{args.split}' "
          f"({len(datasets[args.split])} slices, "
          f"{len(index['splits'][args.split])} patients)...")
    probs, labels, times = collect_predictions(
        model, loaders[args.split], device,
        sw_roi=args.sw_roi, sw_overlap=args.sw_overlap)

    threshold = args.threshold
    if args.sweep_threshold:
        print("\n--- Threshold sweep (validation only) ---")
        threshold, sweep = threshold_sweep(probs, labels)
        with open(os.path.join(output_dir, "threshold_sweep.json"), "w") as f:
            json.dump({"best_threshold": threshold,
                       "sweep_results": {str(k): v for k, v in sweep.items()}},
                      f, indent=4)

    print(f"\nReconstructing 3D volumes in original NIfTI geometry...")
    patient_metrics, preds, gts, pp_counts = evaluate_full(
        probs, times, metadata_dir, threshold=threshold,
        save_nifti_dir=os.path.join(output_dir, f"predictions_{args.split}")
        if args.save_nifti else None,
        postproc_min_fraction=args.postproc_min_fraction)

    from src.training.train import load_tumor_categories
    tumor_cats = load_tumor_categories()

    csv_path = os.path.join(output_dir, f"{args.split}_results_per_patient.csv")
    export_results_csv(patient_metrics, csv_path, tumor_cats)

    averages = compute_macro_micro_averages(patient_metrics, preds, gts)
    strat = stratified_report(patient_metrics, tumor_cats)

    print_report(f"EVALUATION: {args.split.upper()}", patient_metrics,
                 averages, strat, threshold)

    summary_path = os.path.join(output_dir, f"{args.split}_evaluation.json")
    with open(summary_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "split": args.split,
            "model_type": model_type,
            "n_adjacent": n_adjacent,
            "threshold": threshold,
            "parameters_trainable": trainable,
            "gpu_memory_peak_mb": measure_gpu_memory_mb(),
            "macro_average": averages["macro"],
            "micro_average": averages["micro"],
            "stratified_report": strat,
            "per_patient": patient_metrics,
        }, f, indent=4, default=float)
    print(f"\n  Summary: {summary_path}")


if __name__ == "__main__":
    main()
