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
    - Full slice or tumour window    --crop {none, tumor}
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
from src.training.dataset import (
    build_dataloaders,
    build_eval_loader,
    CROP_MODES,
    DEFAULT_CROP_SIZE,
    SAMPLING_MODES,
)
from src.evaluation.metrics import (
    count_model_parameters,
    measure_gpu_memory_mb,
    export_results_csv,
    compute_macro_micro_averages,
    stratified_report,
    threshold_sweep,
    threshold_sweep_original_geometry,
)
# Evaluation lives in the evaluation package so that training and the standalone
# evaluation script provably run the same code.
from src.evaluation.evaluate import (
    evaluate_fast, collect_predictions, evaluate_full, print_postproc_delta)


# ============================================================================
#  TRAINING LOOP
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None,
                    max_grad_norm=1.0, scheduler=None, scheduler_t_max=None):
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
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Checked before backward, not after the step. The earlier version scored
        # the loss only once the weights had already moved, which left the
        # non-AMP path free to take a step from a non-finite loss and then count
        # it as skipped. Under AMP `scaler.step` would have declined the step
        # anyway, so this changes nothing there; on CPU it is the difference
        # between poisoning the weights and not.
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            n_nonfinite += 1
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # GradScaler inspects the unscaled gradients and skips the step when
            # any is non-finite, which is the real guard: a finite loss can still
            # backpropagate into non-finite gradients, so the loss check above is
            # necessary but not sufficient.
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           max_grad_norm)
                # Without a GradScaler nothing else inspects the gradients, so
                # the equivalent check is made here. clip_grad_norm_ returns the
                # pre-clip total norm, which is non-finite exactly when some
                # gradient is.
                if not torch.isfinite(grad_norm):
                    n_nonfinite += 1
                    continue
            optimizer.step()

        total_loss += loss_value * images.size(0)
        n_samples += images.size(0)
        # Per-optimizer-step annealing. Passed only when the caller runs a
        # step-unit schedule; otherwise the caller steps once per epoch and this
        # stays None. `scheduler_t_max` holds the schedule at its floor rather
        # than letting CosineAnnealingLR wrap back up towards the peak, the same
        # guard the per-epoch path applies.
        if scheduler is not None:
            if scheduler_t_max is None or scheduler.last_epoch < scheduler_t_max:
                scheduler.step()

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

    An epoch is not a comparable unit across sampling modes: at batch 16 `all`
    runs ~898 optimizer steps per epoch and `balanced` at 1:1 runs ~177, so a
    fixed epoch patience gives one run five times the step-wise grace of the
    other. Pass `patience_steps` and `min_steps` to count in optimizer steps
    instead, which is the unit the two runs actually share. The epoch fields stay
    the default so every committed run reproduces unchanged.

    Args:
        patience (int): Epochs without improvement before stopping.
        min_delta (float): Improvement below this counts as no improvement.
        min_epochs (int): Earliest epoch at which stopping may trigger.
        patience_steps (int|None): Optimizer steps without improvement before
            stopping. Overrides `patience` when set. Zero disables early
            stopping entirely, which is what a step-budget experiment wants:
            stopping at a different point in each run means the runs did not in
            fact receive the same number of updates.
        min_steps (int|None): Earliest step count at which stopping may trigger.
            Overrides `min_epochs` when set.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 min_epochs: int = 15, patience_steps: int = None,
                 min_steps: int = None):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.patience_steps = patience_steps
        self.min_steps = min_steps
        self.best_score = -np.inf
        self.counter = 0
        self.best_steps = 0
        self.should_stop = False

    @property
    def unit(self) -> str:
        return "steps" if self.patience_steps else "epochs"

    def step(self, score: float, epoch: int, steps_taken: int = 0) -> bool:
        # Disabled. An experiment that controls the optimizer-step count cannot
        # also stop at a different point in each run: the first attempt at this
        # baseline spent 53%, 65% and 76% of the same budget, kept checkpoints at
        # learning rates of 7.6e-4, 5.8e-4 and 4.1e-4, and came out both worse
        # and three times noisier than the runs it was meant to reproduce. Where
        # the budget is the control, it has to be spent in full.
        if self.patience_steps == 0 or (self.patience_steps is None
                                        and self.patience == 0):
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_steps = steps_taken
        else:
            self.counter += 1
            if self.patience_steps:
                stalled = steps_taken - self.best_steps
                past_floor = steps_taken >= (self.min_steps or 0)
                if stalled >= self.patience_steps and past_floor:
                    self.should_stop = True
            elif self.counter >= self.patience and epoch >= self.min_epochs:
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
                    config_dict, history, path, steps_taken=0):
    """
    Saves a complete training state, sufficient to resume exactly.

    `steps_taken` is part of that state because the optimizer step count is what
    every comparison in this project is controlled on. Without it a resumed run
    restarts the counter at zero and reports the steps it took after the resume
    as though they were the whole budget.
    """
    torch.save({
        "epoch": epoch,
        "steps_taken": steps_taken,
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
        tuple: (last_epoch, best_score, history, steps_taken)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    # Checkpoints written before the step budget existed carry no count. Fall
    # back to the history, which records it per epoch, rather than to zero.
    steps = ckpt.get("steps_taken")
    if steps is None:
        rows = ckpt.get("history", [])
        steps = int(rows[-1].get("optimizer_steps", 0)) if rows else 0
    return (ckpt["epoch"], ckpt.get("best_score", -np.inf),
            ckpt.get("history", []), int(steps))


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
                            crop="none", crop_size=DEFAULT_CROP_SIZE,
                            n_adjacent=1, seed=42, patience=10, min_epochs=15,
                            exp_name="experiment", resume_path=None,
                            num_workers=2, save_nifti=False,
                            lr_t_max=None, max_steps=None, max_grad_norm=1.0,
                            schedule_unit="epoch", patience_steps=None,
                            min_steps=None, negative_ratio=1.0,
                            postproc_min_fraction=0.10,
                            tversky_alpha=0.3, tversky_beta=0.7,
                            preprocessed_name="preprocessed",
                            metadata_name="metadata",
                            surface_metrics=True,
                            eval_sampling="all", second_eval_sampling=None,
                            model_channels=None, mined_negatives_path=None,
                            init_weights=None):
    """
    Runs one complete training and evaluation experiment for a single seed.

    `eval_sampling` decides which slices validation and test are drawn from, and
    it changes what the reported number means rather than how well the model
    does. The default 'all' faces the real distribution, where about 91% of
    slices hold no tumour. 'positives' removes them, which removes the detection
    half of the task and leaves only delineation; it produces a much higher
    number that is not comparable with anything else in this project.

    `second_eval_sampling` scores the same trained weights a second time under a
    different protocol. It exists so the gap between the two can be read off one
    run instead of being inferred across two, which would confound the protocol
    with the seed. The second protocol gets its own threshold sweep on its own
    validation slices: a threshold picked where every slice contains tumour is
    far too permissive once empty slices are back, and reusing it would charge
    the difference to the model instead of to the calibration.

    `surface_metrics=False` drops HD95 and ASD from the final test evaluation.
    They run distance transforms over each patient's full reconstruction and cost
    roughly 160 s per patient against 7 s for everything else, which is the
    difference between fitting a Kaggle GPU session and not. The overlap metrics
    are bit-identical either way.

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
    # `max_steps` exists because an epoch is not a comparable unit here. At batch
    # 16, `all` sampling gives ~898 optimizer steps per epoch and `positives`
    # about 89, so two runs with the same epoch budget differ tenfold in how far
    # the optimizer actually travels and in how quickly the cosine reaches its
    # floor. Every sampling comparison in this project was made that way, and
    # measured the budget rather than the sampling. Set `max_steps` to make the
    # comparison the intended one; leaving it None reproduces the committed runs.

    exp_config = {
        "exp_name": exp_name, "seed": seed, "epochs": epochs,
        "max_steps": int(max_steps) if max_steps else None,
        "schedule_unit": schedule_unit,
        "patience_steps": int(patience_steps) if patience_steps else None,
        "min_steps": int(min_steps) if min_steps else None,
        "negative_ratio": float(negative_ratio),
        "batch_size": batch_size, "lr": lr, "model_type": model_type,
        "loss_type": loss_type, "augment": augment, "sampling": sampling,
        "eval_sampling": eval_sampling,
        "second_eval_sampling": second_eval_sampling,
        "crop": crop, "crop_size": crop_size,
        "preprocessed_name": preprocessed_name, "metadata_name": metadata_name,
        "n_adjacent": n_adjacent, "patience": patience, "min_epochs": min_epochs,
        "lr_t_max": t_max, "max_grad_norm": max_grad_norm,
        "postproc_min_fraction": postproc_min_fraction,
        "tversky_alpha": tversky_alpha, "tversky_beta": tversky_beta,
        "model_channels": list(model_channels) if model_channels else None,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=4)

    # --- Data ------------------------------------------------------------
    # Named rather than fixed, so a run can be pointed at a dataset built with a
    # different resize mode without disturbing the default one.
    preprocessed_dir = os.path.join(OUTPUT_DIR, preprocessed_name)
    metadata_dir = os.path.join(OUTPUT_DIR, metadata_name)

    # Loaded before the split is built so a missing or stale file fails here,
    # not sixty epochs later when the sampler quietly had nothing to draw from.
    mined_scores = None
    if mined_negatives_path:
        from src.training.mine_negatives import load_mined
        mined_scores, mined_meta = load_mined(mined_negatives_path)
        q = mined_meta.get("quality", {})
        print(f"  Mined negatives: {len(mined_scores)} patients from "
              f"{mined_meta.get('run', '?')}, "
              f"{q.get('fraction_with_false_positive', float('nan')):.1%} of "
              f"negatives carry a false positive")

    loaders, datasets, index = build_dataloaders(
        preprocessed_dir, batch_size=batch_size, sampling=sampling,
        negative_ratio=negative_ratio,
        augment=augment, crop=crop, crop_size=crop_size,
        n_adjacent=n_adjacent, seed=seed, num_workers=num_workers,
        eval_sampling=eval_sampling, mined_scores=mined_scores)

    # The second protocol's validation loader is built now rather than at the
    # end, because it is also read once per epoch as a diagnostic. Its curve is
    # logged and never acted on: early stopping follows the primary protocol,
    # the one training was set up for, and a model selected by one objective and
    # stopped by another is selected by neither.
    second_val_loader = second_test_loader = None
    if second_eval_sampling:
        second_val_loader, _ = build_eval_loader(
            preprocessed_dir, "val", sampling=second_eval_sampling,
            batch_size=batch_size, n_adjacent=n_adjacent, seed=seed,
            num_workers=num_workers)
        second_test_loader, _ = build_eval_loader(
            preprocessed_dir, "test", sampling=second_eval_sampling,
            batch_size=batch_size, n_adjacent=n_adjacent, seed=seed,
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
    if crop == "tumor":
        print(f"  Crop: {crop_size}x{crop_size} tumour-centred (train only; "
              f"val/test stay full-size)")
    # Only the grad-clip half is settled at this point. The annealing period and
    # the stopping rule are both rewritten further down once `max_steps` and
    # `schedule_unit` are known, and printing them here as though they were
    # final produced lines like "cosine over 44900 epochs" in the log.
    print(f"  Grad clip: {max_grad_norm or 'off'}")
    print(f"  Slices: {n_train} train ({sampling}) | "
          f"{n_val} val | {n_test} test  (eval sampling: {eval_sampling})")
    if eval_sampling == "all":
        print(f"  Val positive rate: {100.0 * val_pos / max(n_val, 1):.2f}% "
              f"(real distribution, unbalanced)")
    else:
        print(f"  [!] Validation and test use '{eval_sampling}', not the real "
              f"slice distribution. Scores from this run are not comparable "
              f"with runs evaluated on every slice.")
        # Selection, not just reporting, is restricted here, and that is the
        # part easiest to miss. Early stopping and the checkpoint both read
        # `val_dice_soft` from this same loader, so the epoch kept is the one
        # that looked best on tumour-bearing slices - where false positives on
        # negative slices cannot count against it. For a model that will be
        # handed whole volumes, that is the wrong selection criterion.
        print(f"  [!] Checkpoint selection, early stopping and the threshold "
              f"sweep all read the '{eval_sampling}' validation set. The epoch "
              f"kept is the best on that subset, never penalised for false "
              f"positives on the slices it excludes. Use "
              f"--eval_sampling all if the model is meant for whole volumes.")
    if second_eval_sampling:
        print(f"  Second protocol: '{second_eval_sampling}' "
              f"({len(second_val_loader.dataset)} val | "
              f"{len(second_test_loader.dataset)} test slices), scored after "
              f"training with its own threshold sweep")

    # --- Optimizer-step budget -------------------------------------------
    steps_per_epoch = len(loaders["train"])
    if max_steps:
        epochs = max(1, -(-int(max_steps) // steps_per_epoch))
        t_max = epochs if lr_t_max is None else int(lr_t_max)
        print(f"  Step budget: {int(max_steps):,} optimizer steps at "
              f"{steps_per_epoch} per epoch -> {epochs} epochs")
    planned_steps = epochs * steps_per_epoch
    print(f"  {steps_per_epoch} optimizer steps/epoch, {planned_steps:,} planned "
          f"over {epochs} epochs")

    # The scheduler's period, in whichever unit the caller chose. Epochs are the
    # default because every committed run used them, but they are the wrong unit
    # for comparing sampling modes: the same epoch count is a fivefold different
    # number of updates, so the cosine reaches its floor after fivefold fewer
    # steps. Under "step" the annealing spans `scheduler_t_max` optimizer steps
    # regardless of how they are grouped into epochs.
    if schedule_unit not in ("epoch", "step"):
        raise ValueError(f"schedule_unit must be 'epoch' or 'step', "
                         f"got {schedule_unit!r}")
    per_step_schedule = schedule_unit == "step"
    scheduler_t_max = (int(lr_t_max) if lr_t_max else planned_steps
                       ) if per_step_schedule else t_max
    if per_step_schedule:
        share = 100.0 * scheduler_t_max / max(planned_steps, 1)
        print(f"  LR schedule: cosine over {scheduler_t_max:,} optimizer steps "
              f"(stepped per batch, not per epoch)")
        print(f"               = {share:.0f}% of the {planned_steps:,}-step "
              f"budget"
              + ("" if share >= 99.5 else
                 f", then held at eta_min for the last "
                 f"{planned_steps - scheduler_t_max:,}"))
    else:
        print(f"  LR schedule: cosine over {t_max} epochs "
              f"(stepped per epoch)")
    if patience_steps:
        print(f"  Early stopping: patience={int(patience_steps):,} steps, "
              f"floor={int(min_steps or 0):,} steps")
    elif patience_steps == 0 or patience == 0:
        print(f"  Early stopping: disabled - the budget is spent in full")
    else:
        print(f"  Early stopping: patience={patience}, min_epochs={min_epochs}")

    # The config was assembled before any of this ran, so it still holds the
    # values the caller passed rather than the ones in force. A run whose
    # config.json says epochs=50 and lr_t_max=50 when it actually ran 55 epochs
    # under a 49 390-step cosine cannot be reproduced from its own record, and
    # `base_corrected_seed42` said exactly that. Overwrite with what happened.
    exp_config["epochs"] = epochs
    exp_config["lr_t_max"] = t_max
    exp_config["scheduler_t_max"] = scheduler_t_max
    exp_config["scheduler_t_max_unit"] = "steps" if per_step_schedule else "epochs"
    # `if patience_steps` folds 0 into None, and those mean opposite things:
    # 0 disables early stopping, None falls back to the epoch counter.
    exp_config["patience_steps"] = (int(patience_steps)
                                    if patience_steps is not None else None)
    exp_config["min_steps"] = int(min_steps) if min_steps is not None else None
    exp_config["mined_negatives_path"] = mined_negatives_path
    exp_config["init_weights"] = init_weights
    exp_config["early_stopping_disabled"] = bool(
        patience_steps == 0 or (patience_steps is None and patience == 0))
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # --- Model, loss, optimizer ------------------------------------------
    model = build_model(model_type, in_channels=n_adjacent, out_channels=1,
                        channels=model_channels).to(device)
    if init_weights:
        # Stage two of a two-stage run: take the weights and nothing else.
        # `resume_path` restores the optimizer, scheduler and step counter too,
        # which would continue the first stage rather than start a second with
        # its own schedule and budget.
        model.load_state_dict(torch.load(init_weights, map_location=device))
        print(f"  Initialised from {init_weights} (weights only, fresh "
              f"optimizer and schedule)")

    trainable_params, total_params = count_model_parameters(model)
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = build_loss_function(loss_type, tversky_alpha=tversky_alpha,
                                    tversky_beta=tversky_beta)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # eta_min keeps the annealed tail useful rather than frozen: with epochs
    # beyond lr_t_max the schedule holds here instead of stopping learning.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_t_max, eta_min=lr * 0.01)
    scaler = GradScaler("cuda") if use_amp else None

    start_epoch = 1
    best_score = -np.inf
    history = []
    resumed_steps = 0

    if resume_path and os.path.exists(resume_path):
        last_epoch, best_score, history, resumed_steps = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device)
        start_epoch = last_epoch + 1
        print(f"  Resumed from epoch {last_epoch}, {resumed_steps:,} optimizer "
              f"steps already taken, best score {best_score:.4f}")

    # --- Training loop ---------------------------------------------------
    early_stopper = EarlyStopping(patience=patience, min_epochs=min_epochs,
                                  patience_steps=patience_steps,
                                  min_steps=min_steps)
    early_stopper.best_score = best_score
    collapse_detector = CollapseDetector()
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print()
    start_total = time.time()
    steps_taken = resumed_steps

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Redraw the negative slice sample so that, across epochs, the model
        # still sees the full variety of negative anatomy.
        datasets["train"].set_epoch(epoch)

        train_loss, n_nonfinite, n_batches = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler if per_step_schedule else None,
            scheduler_t_max=scheduler_t_max)
        steps_taken += n_batches - n_nonfinite

        # CosineAnnealingLR is periodic: stepping past T_max sends the learning
        # rate back up towards its peak. When the epoch budget is larger than the
        # annealing period, hold at eta_min instead of climbing. Under a per-step
        # schedule the same guard already ran inside the epoch.
        if not per_step_schedule and scheduler.last_epoch < scheduler_t_max:
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

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_dice_soft": round(val_soft, 6),
            "val_dice_hard": round(val_hard, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
            "epoch_time_sec": round(epoch_time, 2),
            "nonfinite_batches": n_nonfinite,
            # Cumulative optimizer steps, so any two runs can be compared at
            # equal budget after the fact rather than at equal epochs. Skipped
            # batches are excluded, since a step that did not happen did not
            # move the weights.
            "optimizer_steps": steps_taken,
        }

        second_soft = None
        if second_val_loader is not None:
            second = evaluate_fast(model, second_val_loader, device, threshold=0.5)
            second_soft = (float(np.mean([v["dice_soft"] for v in second.values()]))
                           if second else 0.0)
            record[f"val_dice_soft_{second_eval_sampling}"] = round(second_soft, 6)
            record[f"val_dice_hard_{second_eval_sampling}"] = (
                round(float(np.mean([v["dice_hard"] for v in second.values()])), 6)
                if second else 0.0)

        history.append(record)

        marker = ""
        if val_soft > best_score:
            best_score = val_soft
            torch.save(model.state_dict(), best_model_path)
            marker = "  <- best"
        if n_nonfinite:
            marker += f"  [!] {n_nonfinite}/{n_batches} non-finite batches"

        second_note = (f" | {second_eval_sampling}: {second_soft:.4f}"
                       if second_soft is not None else "")
        print(f"  Epoch {epoch:03d}/{epochs:03d} | Loss: {train_loss:.4f} | "
              f"Val Dice soft: {val_soft:.4f} | hard@0.5: {val_hard:.4f}"
              f"{second_note} | {epoch_time:.1f}s{marker}", flush=True)

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_score,
                        exp_config, history, checkpoint_path,
                        steps_taken=steps_taken)

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

        if early_stopper.step(val_soft, epoch, steps_taken):
            if patience_steps:
                print(f"\n  [EARLY STOP] No improvement in "
                      f"{int(patience_steps):,} optimizer steps "
                      f"(past the {int(min_steps or 0):,}-step floor).")
            else:
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

    # Swept in original geometry, which is the space the test numbers below are
    # reported in. Sweeping in network space instead picks a threshold roughly
    # one grid step too high; see `threshold_sweep`'s docstring.
    print("\n--- Threshold sweep on validation ---")
    val_probs, val_labels, _ = collect_predictions(model, loaders["val"], device)
    # A restricted eval_sampling means the model was never run on most slices,
    # so full coverage cannot be required here either. The threshold that comes
    # back is then the best one for that protocol and for no other.
    is_oracle_eval = eval_sampling != "all"
    best_threshold, sweep_results = threshold_sweep_original_geometry(
        val_probs, metadata_dir, require_full_coverage=not is_oracle_eval)

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
        require_full_coverage=not is_oracle_eval,
        save_nifti_dir=os.path.join(exp_dir, "predictions") if save_nifti else None,
        postproc_min_fraction=postproc_min_fraction,
        surface_metrics=surface_metrics)
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
    print(f"  Vol pred/true:     {_mean('volume_ratio_3d'):.3f}")
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
        "optimizer_steps": steps_taken,
        "steps_per_epoch": steps_per_epoch,
        # True when the headline numbers below were computed on a slice subset.
        # They are then an upper bound obtained with oracle knowledge of which
        # slices hold tumour, not volume performance: on this dataset the same
        # checkpoint scored 0.4645 that way and 0.0540 on every slice.
        "oracle_positive_slices_evaluation": is_oracle_eval,
        "eval_sampling": eval_sampling,
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

    # --- Same weights, second slice protocol ------------------------------
    if second_test_loader is not None:
        print(f"\n{'=' * 75}")
        print(f"  SECOND PROTOCOL: evaluation on '{second_eval_sampling}' slices")
        print(f"{'=' * 75}")
        print("  Same checkpoint, same reconstruction. Only which slices the "
              "model is asked about changes.")

        print(f"\n--- Threshold sweep on validation ({second_eval_sampling}) ---")
        v_probs, v_labels, _ = collect_predictions(model, second_val_loader, device)
        second_threshold, second_sweep = threshold_sweep_original_geometry(
            v_probs, metadata_dir,
            require_full_coverage=(second_eval_sampling == "all"))
        del v_probs, v_labels

        print(f"\n--- Test evaluation (threshold={second_threshold:.2f}) ---")
        t_probs, _, t_times = collect_predictions(model, second_test_loader, device)
        second_metrics, second_preds, second_gts, _ = evaluate_full(
            t_probs, t_times, metadata_dir, threshold=second_threshold,
            require_full_coverage=(second_eval_sampling == "all"),
            save_nifti_dir=None, postproc_min_fraction=postproc_min_fraction,
            surface_metrics=surface_metrics)
        del t_probs
        second_averages = compute_macro_micro_averages(
            second_metrics, second_preds, second_gts)
        del second_preds, second_gts

        def _mean2(key):
            vals = [m[key] for m in second_metrics.values()
                    if key in m and not np.isnan(m[key])]
            return float(np.mean(vals)) if vals else 0.0

        print(f"\n  Dice 2D, tumour slices: {_mean2('dice_2d_tumour_slices'):.4f}")
        print(f"  Dice 2D, every slice:   {_mean2('dice_2d_all_slices'):.4f}")
        print(f"  Dice 3D, per patient:   {_mean2('dice_3d'):.4f}")
        print(f"  Sensitivity: {_mean2('sensitivity_3d'):.4f} | "
              f"Precision: {_mean2('precision_3d'):.4f}")
        print(f"  Threshold:   {second_threshold:.2f} "
              f"(swept separately, primary was {best_threshold:.2f})")
        if second_eval_sampling == "positives":
            print("\n  [!] Only tumour-bearing slices were fed to the model, so "
                  "every slice it was not shown reconstructs as empty. Read "
                  "'Dice 2D, tumour slices' here: it is the one figure this "
                  "protocol measures cleanly. The 3D and all-slice numbers "
                  "inherit those untouched slices and are not comparable with "
                  "a run that saw the whole volume.")
        print("=" * 75)

        benchmark["second_protocol"] = {
            "eval_sampling": second_eval_sampling,
            "optimal_threshold": second_threshold,
            "threshold_sweep": {str(k): v for k, v in second_sweep.items()},
            "test_metrics_summary": {
                "mean_dice_2d_tumour_slices": _mean2("dice_2d_tumour_slices"),
                "mean_dice_2d_all_slices": _mean2("dice_2d_all_slices"),
                "mean_dice_3d": _mean2("dice_3d"),
                "mean_sensitivity": _mean2("sensitivity_3d"),
                "mean_precision": _mean2("precision_3d"),
            },
            "macro_average": second_averages["macro"],
            "micro_average": second_averages["micro"],
            "per_patient_test_metrics": second_metrics,
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
                        choices=list(SAMPLING_MODES))
    parser.add_argument("--eval_sampling", type=str, default="all",
                        choices=list(SAMPLING_MODES),
                        help="Which slices validation and test draw from. 'all' "
                             "faces the real distribution and is what every "
                             "reported number in this project uses. Anything "
                             "else measures a restricted task and is not "
                             "comparable with those numbers.")
    parser.add_argument("--second_eval_sampling", type=str, default=None,
                        choices=list(SAMPLING_MODES),
                        help="Score the same trained weights a second time "
                             "under this protocol, with its own threshold "
                             "sweep. Reading both off one run keeps the "
                             "protocol difference from being confounded with "
                             "the seed.")
    parser.add_argument("--augment", type=str, default="anatomic",
                        choices=["none", "standard", "anatomic"])
    parser.add_argument("--crop", type=str, default="none", choices=CROP_MODES,
                        help="'tumor' trains on a window centred on the lesion "
                             "instead of the full slice. Training only: "
                             "validation and test always run full-size, since "
                             "the tumour location is what is being predicted.")
    parser.add_argument("--crop_size", type=int, default=DEFAULT_CROP_SIZE,
                        help="Side length of the --crop tumor window. Must be "
                             "divisible by 16 for the downsampling stages.")
    parser.add_argument("--model_channels", type=str, default=None,
                        help="Comma-separated filters per encoder level, "
                             "e.g. 32,64,128,256,512 to double the default "
                             "width. Omit to keep each architecture's "
                             "benchmark size, which every committed run used.")
    parser.add_argument("--n_adjacent", type=int, default=1, choices=[1, 3, 5, 7, 9],
                        help="1 = 2D, odd values above that = 2.5D with "
                             "consecutive slices. What matters is the span in "
                             "millimetres, not the channel count: at the "
                             "default 1 mm slice spacing, 3 channels cover 3 mm "
                             "of a lesion whose median extent is 23 mm, and the "
                             "neighbours are partly interpolated. Pair a wide "
                             "stack with a preprocessed set built at a coarser "
                             "--slice_spacing.")
    parser.add_argument("--preprocessed_name", type=str, default="preprocessed",
                        help="Which preprocessed dataset under output/ to train "
                             "on. Change it to train against a variant built by "
                             "preprocessing's --out_name rather than the "
                             "default one every committed run used.")
    parser.add_argument("--metadata_name", type=str, default="metadata",
                        help="Reconstruction metadata directory matching "
                             "--preprocessed_name. A mismatched pair "
                             "reconstructs into the wrong geometry and scores "
                             "near zero without erroring.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--negative_ratio", type=float, default=1.0,
                        help="Negative slices drawn per positive under "
                             "--sampling balanced. 1.0 is the committed default; "
                             "the real distribution is roughly 11. Pair it with "
                             "--max_steps, or the ratio changes the epoch size "
                             "and the comparison measures the budget instead.")
    parser.add_argument("--schedule_unit", type=str, default="epoch",
                        choices=["epoch", "step"],
                        help="Unit the cosine anneals in. 'epoch' reproduces "
                             "every committed run. 'step' anneals over "
                             "optimizer steps instead, which is the only unit "
                             "two sampling modes share: the same epoch count "
                             "is ~898 steps under --sampling all and ~177 at "
                             "1:1, so an epoch-unit cosine reaches its floor "
                             "after fivefold fewer updates in one of them.")
    parser.add_argument("--patience_steps", type=int, default=None,
                        help="Early-stopping patience in optimizer steps rather "
                             "than epochs, for the same reason. Overrides "
                             "--patience when set.")
    parser.add_argument("--min_steps", type=int, default=None,
                        help="Earliest step count at which early stopping may "
                             "fire. Overrides --min_epochs when set.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Total optimizer steps, converted to an epoch count "
                             "from the sampling mode's steps per epoch. An epoch "
                             "is not a comparable unit here: at batch 16, "
                             "--sampling all gives ~898 steps per epoch and "
                             "--sampling positives about 89, so equal epochs "
                             "means a tenfold difference in optimizer travel. "
                             "Use this whenever comparing sampling modes; omit "
                             "it to reproduce the committed runs.")
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
        augment=args.augment, sampling=args.sampling,
        eval_sampling=args.eval_sampling,
        second_eval_sampling=args.second_eval_sampling,
        crop=args.crop, crop_size=args.crop_size, n_adjacent=args.n_adjacent,
        preprocessed_name=args.preprocessed_name,
        metadata_name=args.metadata_name,
        model_channels=([int(c) for c in args.model_channels.split(",")]
                        if args.model_channels else None),
        patience=args.patience, min_epochs=args.min_epochs,
        exp_name=args.exp_name, num_workers=args.num_workers,
        save_nifti=args.save_nifti, lr_t_max=args.lr_t_max,
        max_steps=args.max_steps, negative_ratio=args.negative_ratio,
        schedule_unit=args.schedule_unit, patience_steps=args.patience_steps,
        min_steps=args.min_steps,
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
