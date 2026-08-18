"""
Training and Experiment Benchmarking Pipeline for Lung Tumor Segmentation.

Every experiment runs through this one entry point so that architectures, losses,
sampling strategies, and augmentation policies are compared under identical
conditions: same patient split, same evaluation code, same checkpoint selection
rule, same seeds.

Features:
    - Configurable architecture      --model_type {unet, attention_unet, segresnet}
    - Configurable loss              --loss_type {dice_ce, dice_focal, tversky, focal_tversky}
    - Configurable slice sampling    --sampling {balanced, all, hard_negatives}
    - Configurable augmentation      --augment {none, standard, anatomic}
    - 2D or 2.5D input               --n_adjacent {1, 3, 5}
    - Mixed precision, early stopping, full checkpointing, resume
    - Validation on every slice of every volume, never a balanced subset
    - Threshold chosen on validation only, then applied to test
    - Final metrics computed on 3D volumes reconstructed into original NIfTI
      geometry, with per-patient CSV export

Usage:
    python -m src.training.train --exp_name exp1 --loss_type dice_ce \
        --sampling balanced --epochs 50 --seeds 42,43,44
"""

import os
import json
import time
import csv
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from src.config import OUTPUT_DIR
from src.models.factory import build_model, MODEL_TYPES
from src.training.losses import build_loss_function, LOSS_TYPES
from src.training.dataset import build_dataloaders
from src.evaluation.metrics import (
    count_model_parameters,
    measure_gpu_memory_mb,
    export_results_csv,
    compute_macro_micro_averages,
    stratified_report,
    threshold_sweep,
)
# Evaluation lives in the evaluation package so that training and the standalone
# evaluation script provably run the same code.
from src.evaluation.evaluate import (
    evaluate_fast, collect_predictions, evaluate_full, print_postproc_delta)


# ============================================================================
#  TRAINING LOOP
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None,
                    max_grad_norm=1.0):
    """
    Runs one training epoch, optionally under mixed precision.

    Gradients are clipped to `max_grad_norm` before every optimizer step. This is
    not a precaution: two runs in the benchmark diverged to a non-finite loss and
    never recovered, both under `--sampling all`. The pattern is consistent with
    Dice-family losses on a target occupying under 1% of the volume: a batch
    holding almost no positive voxels puts a very small number in the denominator
    and produces a disproportionate gradient. Clipping bounds that step.

    Under AMP the gradients must be unscaled before the norm is measured,
    otherwise the norm is computed on `GradScaler`'s inflated values and the clip
    threshold means nothing.

    Non-finite batches are excluded from the running mean rather than allowed to
    turn the reported epoch loss into `nan`, and are counted so the caller can
    abort a diverged run instead of burning the full patience budget on frozen
    weights.

    Returns:
        tuple: (mean training loss per sample, non-finite batch count,
                total batch count).
    """
    model.train()
    total_loss = 0.0
    n_samples = 0
    n_nonfinite = 0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        n_batches += 1

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
            # The criterion is deliberately outside autocast. fp16 flushes any
            # sigmoid output below ~6e-8 to exactly zero, so a network with
            # strongly negative background logits produces a summed probability
            # of exactly 0 and the Dice denominator degenerates. Convolutions are
            # essentially all of the FLOPs here, so evaluating the loss in fp32
            # costs almost nothing while removing that cliff.
            loss = criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            n_nonfinite += 1
            continue

        total_loss += loss_value * images.size(0)
        n_samples += images.size(0)

    return total_loss / max(n_samples, 1), n_nonfinite, n_batches


# ============================================================================
#  EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Stops training when the monitored metric has not improved for `patience`
    consecutive epochs, but never before `min_epochs`.

    The warm-up floor matters for this problem specifically. Segmentation of a
    structure occupying well under 1% of the volume spends its first epochs
    driving down the loss without producing any above-threshold prediction, so a
    patience counter started at epoch one fires long before the model has had a
    chance.

    Args:
        patience (int): Epochs without improvement before stopping.
        min_delta (float): Improvement below this counts as no improvement.
        min_epochs (int): Earliest epoch at which stopping may trigger.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 min_epochs: int = 15):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, epoch: int) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.should_stop = True
        return self.should_stop


class CollapseDetector:
    """
    Stops a run whose validation Dice has fallen to zero and cannot come back.

    This is a different failure from divergence, and it is easy to mistake for
    healthy training. The loss stays finite and in fact drops sharply, so the
    non-finite batch counter never fires — but the network has saturated to an
    all-background prediction and every validation Dice reads exactly 0.0000.

    Recovery is not merely unlikely, it is structurally blocked: once the sigmoid
    output is zero everywhere its derivative is zero too, so the empty slices —
    around 90% of the data under `--sampling all` — supply no gradient at all.

    The loss-side cause is addressed in `src.training.losses`; this detector is
    the backstop. Without it a collapsed run spends its whole patience budget
    training weights that can no longer change, at roughly ten minutes of GPU
    time per run.

    A plain "Dice is zero" test would fire during the opening epochs, when a
    network that has not yet learned anything legitimately scores zero. The
    detector therefore arms only after validation Dice has exceeded `min_peak`
    at least once.

    Args:
        min_peak (float): Validation Dice that must be reached before the
            detector arms.
        patience (int): Consecutive zero epochs required to stop.
        zero_tol (float): Scores at or below this count as zero.
    """

    def __init__(self, min_peak: float = 0.10, patience: int = 3,
                 zero_tol: float = 1e-6):
        self.min_peak = min_peak
        self.patience = patience
        self.zero_tol = zero_tol
        self.armed = False
        self.counter = 0

    def step(self, score: float) -> bool:
        if score > self.min_peak:
            self.armed = True
        if not self.armed:
            return False

        if score <= self.zero_tol:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.patience


# ============================================================================
#  CHECKPOINTS
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_score,
                    config_dict, history, path):
    """Saves a complete training state, sufficient to resume exactly."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_score": best_score,
        "config": config_dict,
        "history": history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """
    Restores a training state saved by `save_checkpoint`.

    Returns:
        tuple: (last_epoch, best_score, history)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt.get("best_score", -np.inf), ckpt.get("history", [])


# ============================================================================
#  TUMOR SIZE CATEGORIES
# ============================================================================

def load_tumor_categories() -> dict:
    """
    Loads per-patient tumour size categories for stratified reporting.

    Returns:
        dict: {case_id: 'small' | 'medium' | 'large'}, empty if the EDA CSV
              has not been generated.
    """
    eda_csv = os.path.join(OUTPUT_DIR, "eda_statistics.csv")
    if not os.path.exists(eda_csv):
        return {}

    from src.preprocessing.create_split import classify_tumor_size

    df = pd.read_csv(eda_csv)
    return {row["case_id"]: classify_tumor_size(row["tumor_volume_mm3"])
            for _, row in df.iterrows()}


# ============================================================================
#  MAIN EXPERIMENT
# ============================================================================

def run_training_experiment(epochs=50, batch_size=16, lr=1e-3,
                            model_type="unet", loss_type="dice_ce",
                            augment="anatomic", sampling="balanced",
                            n_adjacent=1, seed=42, patience=10, min_epochs=15,
                            exp_name="experiment", resume_path=None,
                            num_workers=2, save_nifti=False,
                            lr_t_max=None, max_grad_norm=1.0,
                            postproc_min_fraction=0.10,
                            tversky_alpha=0.3, tversky_beta=0.7):
    """
    Runs one complete training and evaluation experiment for a single seed.

    `lr_t_max` sets the cosine annealing period independently of `epochs`. They
    exist as separate knobs because tying them together makes a larger budget
    change the schedule rather than extend the run. Measured on this dataset, a
    120-epoch budget scored *worse* than a 50-epoch one on the same seed (test
    Dice 0.4177 against 0.4446): the stretched cosine meant the learning rate
    never fell below 5.8e-04 before early stopping fired at epoch 54, while the
    shorter run reached its best epoch at 7.8e-05, deep in the annealed tail.
    Pinning `lr_t_max` keeps the schedule fixed while the budget varies.

    Returns:
        dict: The benchmark report, also written to disk as benchmark_report.json.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    exp_dir = os.path.join(OUTPUT_DIR, "experiments", exp_name, f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    t_max = int(lr_t_max) if lr_t_max else epochs

    exp_config = {
        "exp_name": exp_name, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "lr": lr, "model_type": model_type,
        "loss_type": loss_type, "augment": augment, "sampling": sampling,
        "n_adjacent": n_adjacent, "patience": patience, "min_epochs": min_epochs,
        "lr_t_max": t_max, "max_grad_norm": max_grad_norm,
        "postproc_min_fraction": postproc_min_fraction,
        "tversky_alpha": tversky_alpha, "tversky_beta": tversky_beta,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=4)

    # --- Data ------------------------------------------------------------
    preprocessed_dir = os.path.join(OUTPUT_DIR, "preprocessed")
    metadata_dir = os.path.join(OUTPUT_DIR, "metadata")

    loaders, datasets, index = build_dataloaders(
        preprocessed_dir, batch_size=batch_size, sampling=sampling,
        augment=augment, n_adjacent=n_adjacent, seed=seed,
        num_workers=num_workers)

    n_train = len(datasets["train"])
    n_val = len(datasets["val"])
    n_test = len(datasets["test"])
    val_pos = sum(len(index["cases"][c]["positive_slices"])
                  for c in index["splits"]["val"])

    print(f"\n{'=' * 75}")
    print(f"  EXPERIMENT: {exp_name} (seed={seed})")
    print(f"{'=' * 75}")
    print(f"  Device: {device} | AMP: {use_amp}")
    print(f"  Model: {model_type} | Loss: {loss_type} | "
          f"Sampling: {sampling} | Augment: {augment}")
    print(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | "
          f"n_adjacent: {n_adjacent}")
    print(f"  LR schedule: cosine over {t_max} epochs | "
          f"grad clip: {max_grad_norm or 'off'}")
    print(f"  Early stopping: patience={patience}, min_epochs={min_epochs}")
    print(f"  Slices: {n_train} train ({sampling}) | "
          f"{n_val} val | {n_test} test")
    print(f"  Val positive rate: {100.0 * val_pos / max(n_val, 1):.2f}% "
          f"(real distribution, unbalanced)")

    # --- Model, loss, optimizer ------------------------------------------
    model = build_model(model_type, in_channels=n_adjacent, out_channels=1).to(device)
    trainable_params, total_params = count_model_parameters(model)
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = build_loss_function(loss_type, tversky_alpha=tversky_alpha,
                                    tversky_beta=tversky_beta)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # eta_min keeps the annealed tail useful rather than frozen: with epochs
    # beyond lr_t_max the schedule holds here instead of stopping learning.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=lr * 0.01)
    scaler = GradScaler("cuda") if use_amp else None

    start_epoch = 1
    best_score = -np.inf
    history = []

    if resume_path and os.path.exists(resume_path):
        last_epoch, best_score, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = last_epoch + 1
        print(f"  Resumed from epoch {last_epoch}, best score {best_score:.4f}")

    # --- Training loop ---------------------------------------------------
    early_stopper = EarlyStopping(patience=patience, min_epochs=min_epochs)
    early_stopper.best_score = best_score
    collapse_detector = CollapseDetector()
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print()
    start_total = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Redraw the negative slice sample so that, across epochs, the model
        # still sees the full variety of negative anatomy.
        datasets["train"].set_epoch(epoch)

        train_loss, n_nonfinite, n_batches = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler,
            max_grad_norm=max_grad_norm)

        # CosineAnnealingLR is periodic: stepping past T_max sends the learning
        # rate back up towards its peak. When the epoch budget is larger than the
        # annealing period, hold at eta_min instead of climbing.
        if scheduler.last_epoch < t_max:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # A run that has fully diverged produces a non-finite loss on every batch,
        # and under AMP the scaler then skips every optimizer step, so the weights
        # are frozen. Further epochs cannot change anything, and without this check
        # the run would keep training for the whole patience window before early
        # stopping noticed. Stop at the first fully non-finite epoch instead.
        diverged = n_batches > 0 and n_nonfinite == n_batches

        val = evaluate_fast(model, loaders["val"], device, threshold=0.5)
        val_hard = float(np.mean([v["dice_hard"] for v in val.values()])) if val else 0.0
        val_soft = float(np.mean([v["dice_soft"] for v in val.values()])) if val else 0.0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_dice_soft": round(val_soft, 6),
            "val_dice_hard": round(val_hard, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "epoch_time_sec": round(epoch_time, 2),
            "nonfinite_batches": n_nonfinite,
        })

        marker = ""
        if val_soft > best_score:
            best_score = val_soft
            torch.save(model.state_dict(), best_model_path)
            marker = "  <- best"
        if n_nonfinite:
            marker += f"  [!] {n_nonfinite}/{n_batches} non-finite batches"

        print(f"  Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"Val Dice soft: {val_soft:.4f} | hard@0.5: {val_hard:.4f} | "
              f"{epoch_time:.1f}s{marker}", flush=True)

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_score,
                        exp_config, history, checkpoint_path)

        if diverged:
            print(f"\n  [DIVERGED] Every batch in epoch {epoch} produced a "
                  f"non-finite loss, so no optimizer step was applied. Training "
                  f"stops here; best_model.pt still holds the best epoch "
                  f"(val soft Dice {best_score:.4f}).")
            break

        if collapse_detector.step(val_soft):
            print(f"\n  [COLLAPSED] Validation Dice has read 0.0000 for "
                  f"{collapse_detector.patience} consecutive epochs after "
                  f"having passed {collapse_detector.min_peak:.2f}. The network "
                  f"has saturated to an all-background prediction, which leaves "
                  f"no gradient on the empty slices and so no way back. "
                  f"Training stops here; best_model.pt still holds the best "
                  f"epoch (val soft Dice {best_score:.4f}).")
            break

        if early_stopper.step(val_soft, epoch):
            print(f"\n  [EARLY STOP] No improvement in {patience} epochs "
                  f"(past the {min_epochs}-epoch floor).")
            break

    total_train_time = time.time() - start_total
    gpu_memory_mb = measure_gpu_memory_mb()
    mean_epoch_time = float(np.mean([h["epoch_time_sec"] for h in history]))

    history_path = os.path.join(exp_dir, "training_history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"\n  Training history: {history_path}")

    # --- Threshold selection, on validation only -------------------------
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n--- Threshold sweep on validation ---")
    val_probs, val_labels, _ = collect_predictions(model, loaders["val"], device)
    best_threshold, sweep_results = threshold_sweep(val_probs, val_labels)

    with open(os.path.join(exp_dir, "threshold_sweep.json"), "w") as f:
        json.dump({"best_threshold": best_threshold,
                   "sweep_results": {str(k): v for k, v in sweep_results.items()}},
                  f, indent=4)
    del val_probs, val_labels

    # --- Final test evaluation, in original NIfTI geometry ---------------
    print(f"\n--- Test evaluation (threshold={best_threshold:.2f}) ---")
    test_probs, _, test_times = collect_predictions(model, loaders["test"], device)
    test_metrics, test_preds, test_gts, pp_counts = evaluate_full(
        test_probs, test_times, metadata_dir, threshold=best_threshold,
        save_nifti_dir=os.path.join(exp_dir, "predictions") if save_nifti else None,
        postproc_min_fraction=postproc_min_fraction)
    del test_probs

    tumor_cats = load_tumor_categories()
    export_results_csv(test_metrics, os.path.join(exp_dir, "test_results_per_patient.csv"),
                       tumor_cats)
    averages = compute_macro_micro_averages(test_metrics, test_preds, test_gts)
    strat = stratified_report(test_metrics, tumor_cats)

    def _mean(key):
        vals = [m[key] for m in test_metrics.values() if key in m]
        return float(np.mean(vals)) if vals else 0.0

    failures = sum(1 for m in test_metrics.values() if m["is_failure"])
    failure_rate = (100.0 * failures / len(test_metrics)) if test_metrics else 0.0

    # --- Benchmark report -------------------------------------------------
    print(f"\n{'=' * 75}")
    print(f"  BENCHMARK: {exp_name} (seed={seed})")
    print(f"{'=' * 75}")
    print(f"  Parameters:        {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"  Time/epoch:        {mean_epoch_time:.2f}s")
    print(f"  GPU memory peak:   {gpu_memory_mb:.1f} MB")
    print(f"  Inference/volume:  {_mean('inference_time_sec'):.3f}s")
    print(f"  Dice 3D:           {_mean('dice_3d'):.4f}")
    print(f"  IoU 3D:            {_mean('iou_3d'):.4f}")
    print(f"  HD95:              {_mean('hd95_3d'):.2f} mm")
    print(f"  ASD:               {_mean('asd_3d'):.2f} mm")
    print(f"  Sensitivity:       {_mean('sensitivity_3d'):.4f}")
    print(f"  Precision:         {_mean('precision_3d'):.4f}")
    print(f"  Specificity:       {_mean('specificity_3d'):.4f}")
    print(f"  FP components:     {_mean('fp_components'):.2f}")
    print(f"  Failure rate:      {failure_rate:.1f}% ({failures}/{len(test_metrics)})")
    print(f"  Threshold:         {best_threshold:.2f} (chosen on validation)")
    print(f"\n  Macro Dice: {averages['macro']['dice_3d']:.4f} | "
          f"Micro Dice: {averages['micro']['dice_3d']:.4f}")

    for cat in ("small", "medium", "large"):
        if strat.get(cat, {}).get("n_patients", 0) > 0:
            sr = strat[cat]
            print(f"  {cat.upper():7s} ({sr['n_patients']} pts): "
                  f"Dice={sr.get('mean_dice_3d', 0):.4f}  "
                  f"HD95={sr.get('mean_hd95_3d', 0):.2f}mm  "
                  f"failure={sr.get('failure_rate_pct', 0):.0f}%")

    print_postproc_delta(test_metrics)
    print("=" * 75)

    benchmark = {
        "experiment_name": exp_name,
        "seed": seed,
        "config": exp_config,
        "parameters_trainable": trainable_params,
        "parameters_total": total_params,
        "mean_epoch_time_sec": mean_epoch_time,
        "total_train_time_sec": total_train_time,
        "gpu_memory_peak_mb": gpu_memory_mb,
        "epochs_run": len(history),
        "best_val_dice_soft": best_score,
        "optimal_threshold": best_threshold,
        "threshold_sweep": {str(k): v for k, v in sweep_results.items()},
        "test_metrics_summary": {
            "mean_dice_3d": _mean("dice_3d"),
            "median_dice_3d": float(np.median([m["dice_3d"] for m in test_metrics.values()])),
            "mean_iou_3d": _mean("iou_3d"),
            "mean_hd95_mm": _mean("hd95_3d"),
            "mean_asd_mm": _mean("asd_3d"),
            "mean_sensitivity": _mean("sensitivity_3d"),
            "mean_precision": _mean("precision_3d"),
            "mean_specificity": _mean("specificity_3d"),
            "mean_fp_components": _mean("fp_components"),
            "mean_inference_time_sec": _mean("inference_time_sec"),
            "failure_count": failures,
            "failure_rate_percent": failure_rate,
        },
        "macro_average": averages["macro"],
        "micro_average": averages["micro"],
        "stratified_report": strat,
        "per_patient_test_metrics": test_metrics,
    }

    if pp_counts:
        tp, fp, fn, tn = (pp_counts["tp"], pp_counts["fp"],
                          pp_counts["fn"], pp_counts["tn"])
        pp_failures = sum(1 for m in test_metrics.values() if m.get("pp_is_failure"))
        benchmark["postprocessed"] = {
            "min_fraction": postproc_min_fraction,
            "mean_components_removed": _mean("pp_components_removed"),
            "macro": {
                "dice_3d": _mean("pp_dice_3d"),
                "iou_3d": _mean("pp_iou_3d"),
                "hd95_mm": _mean("pp_hd95_3d"),
                "asd_mm": _mean("pp_asd_3d"),
                "sensitivity": _mean("pp_sensitivity_3d"),
                "precision": _mean("pp_precision_3d"),
                "fp_components": _mean("pp_fp_components"),
            },
            "micro": {
                "dice_3d": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
                "iou_3d": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 1.0,
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 1.0,
                "precision": float(tp / (tp + fp)) if (tp + fp) else 1.0,
                "specificity": float(tn / (tn + fp)) if (tn + fp) else 1.0,
            },
            "failure_count": pp_failures,
            "failure_rate_percent": (100.0 * pp_failures / len(test_metrics)
                                     if test_metrics else 0.0),
        }

    report_path = os.path.join(exp_dir, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(benchmark, f, indent=4, default=float)
    print(f"\n  Benchmark: {report_path}")

    return benchmark


# ============================================================================
#  MULTI-SEED RUNNER
# ============================================================================

def run_multi_seed_experiment(seeds, **kwargs):
    """
    Runs the same experiment across several seeds and reports mean +/- std, so
    that differences between experiments can be read against run-to-run noise.
    """
    exp_name = kwargs.get("exp_name", "experiment")
    all_results = {}

    for seed in seeds:
        all_results[seed] = run_training_experiment(seed=seed, **kwargs)

    if len(all_results) < 2:
        return all_results

    def _collect(key):
        return [r["test_metrics_summary"][key] for r in all_results.values()]

    dices, hd95s = _collect("mean_dice_3d"), _collect("mean_hd95_mm")
    sens, prec = _collect("mean_sensitivity"), _collect("mean_precision")

    print(f"\n{'=' * 75}")
    print(f"  MULTI-SEED SUMMARY: {exp_name} ({len(seeds)} seeds)")
    print(f"{'=' * 75}")
    print(f"  Dice 3D:     {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    print(f"  HD95:        {np.mean(hd95s):.2f} +/- {np.std(hd95s):.2f} mm")
    print(f"  Sensitivity: {np.mean(sens):.4f} +/- {np.std(sens):.4f}")
    print(f"  Precision:   {np.mean(prec):.4f} +/- {np.std(prec):.4f}")
    print("=" * 75)

    agg_dir = os.path.join(OUTPUT_DIR, "experiments", exp_name)
    with open(os.path.join(agg_dir, "multi_seed_summary.json"), "w") as f:
        json.dump({
            "exp_name": exp_name,
            "seeds": list(seeds),
            "config": all_results[seeds[0]]["config"],
            "mean_dice": float(np.mean(dices)), "std_dice": float(np.std(dices)),
            "mean_hd95": float(np.mean(hd95s)), "std_hd95": float(np.std(hd95s)),
            "mean_sensitivity": float(np.mean(sens)), "std_sensitivity": float(np.std(sens)),
            "mean_precision": float(np.mean(prec)), "std_precision": float(np.std(prec)),
            "per_seed_dice": {str(s): d for s, d in zip(seeds, dices)},
        }, f, indent=4)

    return all_results


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Lung tumor segmentation: training and benchmarking")
    parser.add_argument("--exp_name", type=str, default="unet_2d_baseline")
    parser.add_argument("--model_type", type=str, default="unet", choices=MODEL_TYPES)
    parser.add_argument("--loss_type", type=str, default="dice_ce", choices=LOSS_TYPES)
    parser.add_argument("--sampling", type=str, default="balanced",
                        choices=["balanced", "all", "hard_negatives"])
    parser.add_argument("--augment", type=str, default="anatomic",
                        choices=["none", "standard", "anatomic"])
    parser.add_argument("--n_adjacent", type=int, default=1, choices=[1, 3, 5],
                        help="1 = 2D, 3 or 5 = 2.5D with consecutive slices")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_t_max", type=int, default=None,
                        help="Cosine annealing period in epochs. Defaults to "
                             "--epochs. Pin it when raising the epoch budget, "
                             "or the LR decay stretches instead of extending.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient-norm clip before each optimizer step. "
                             "0 disables it.")
    parser.add_argument("--postproc_min_fraction", type=float, default=0.10,
                        help="Predicted connected components smaller than this "
                             "fraction of the largest are dropped in the "
                             "post-processed metric set. 0 disables filtering.")
    parser.add_argument("--tversky_alpha", type=float, default=0.3,
                        help="Weight on false positives. Tversky losses only.")
    parser.add_argument("--tversky_beta", type=float, default=0.7,
                        help="Weight on false negatives. Tversky losses only. "
                             "The default of 0.7 charges a miss 2.33 times a "
                             "false alarm, which over-segments badly on this "
                             "target; alpha = beta = 0.5 reduces to Dice.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated, e.g. 42,43,44")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_nifti", action="store_true",
                        help="Write test predictions as NIfTI in original geometry")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    kwargs = dict(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        model_type=args.model_type, loss_type=args.loss_type,
        augment=args.augment, sampling=args.sampling, n_adjacent=args.n_adjacent,
        patience=args.patience, min_epochs=args.min_epochs,
        exp_name=args.exp_name, num_workers=args.num_workers,
        save_nifti=args.save_nifti, lr_t_max=args.lr_t_max,
        max_grad_norm=args.max_grad_norm,
        postproc_min_fraction=args.postproc_min_fraction,
        tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
    )

    if len(seeds) == 1:
        run_training_experiment(seed=seeds[0], resume_path=args.resume, **kwargs)
    else:
        run_multi_seed_experiment(seeds=seeds, **kwargs)


if __name__ == "__main__":
    main()
