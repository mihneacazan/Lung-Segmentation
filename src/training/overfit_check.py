"""
Can the model memorise a handful of real slices? The floor under every number.

Every score in this project rests on an assumption nothing else checks: that the
chain from a volume on disk to a gradient is wired correctly. A broken pipeline
and a hard problem both produce low Dice, and a ranking of twenty experiments run
through the same broken pipeline would be exactly as consistent and exactly as
meaningless.

Memorisation separates the two. It needs no learnable pattern - only that the
mapping be representable and that gradients reach it. 1.6M parameters against 32
slices is roughly 50,000 parameters per slice, so a failure here is not capacity
and not data volume. It is a bug in the loss, the labels, the alignment or the
optimiser.

This differs from `tests/test_overfit_sanity.py` in scale rather than intent.
That suite runs in CI and so works on a 48 px window around the tumour, where the
positive class is 1.26% of pixels. This runs at the full 192 px the model is
actually trained at, where the tumour is 0.31%, and on 16-32 slices rather than
6. It is slower and stricter, and it is the version to run when a split changes
or preprocessing is touched.

Two checks, because they fail differently:

  slices   Tumour-bearing slices only. The batch is the whole world here, so an
           empty slice would let the model score a free 1.0 by predicting
           nothing, and the number would mean nothing.

  patient  One patient end to end, empty slices included. Dice is scored on the
           tumour slices; the empty ones are reported separately as a false-alarm
           rate. A model that memorises the tumours by painting everywhere passes
           the first check and fails this one.

Augmentation is off throughout, and that is the point rather than a convenience:
it would show a different image every step and make memorisation impossible by
construction, turning a bug into something indistinguishable from variance.

No GPU required. Measured on 8 CPU threads: 1.3 s/step at 16 slices, 2.7 s at 32.

Usage:
    python -m src.training.overfit_check
    python -m src.training.overfit_check --n_slices 32 --steps 600
    python -m src.training.overfit_check --split output/patient_split_fold2test.json
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
from src.models.factory import build_model
from src.training.losses import build_loss_function

# The U-Net downsamples four times; a side that is not a multiple of 16 fails in
# a skip connection with a shape error that looks like a bug in the test.
GRID_MULTIPLE = 16


def _dice(pred, target, eps=1e-7):
    """Dice of a thresholded prediction against a binary target."""
    p = (pred > 0.5).float()
    inter = (p * target).sum()
    return float((2 * inter + eps) / (p.sum() + target.sum() + eps))


def memorise(images, labels, steps=400, lr=3e-3, loss_type="dice_ce",
             model_type="unet", seed=0, log_every=50, verbose=True):
    """
    Trains one model on one fixed batch until it memorises it, then scores it.

    Deliberately not `run_training_experiment`: augmentation, sampling, early
    stopping and the scheduler are all excluded so a failure has as few possible
    causes as it can.

    Returns:
        tuple: (final Dice, list of (step, loss, dice) samples)
    """
    for side in images.shape[-2:]:
        if side % GRID_MULTIPLE:
            raise ValueError(
                f"side {side} is not a multiple of {GRID_MULTIPLE}; the U-Net's "
                f"four downsamplings would fail in a skip connection")

    torch.manual_seed(seed)
    model = build_model(model_type, in_channels=images.shape[1], out_channels=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = build_loss_function(loss_type)

    history = []
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                d = _dice(torch.sigmoid(model(images)), labels)
            model.train()
            history.append((step, float(loss), d))
            if verbose:
                print(f"    step {step:4}  loss {float(loss):.4f}  dice {d:.4f}",
                      flush=True)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(images))
    return _dice(probs, labels), history, probs


def load_slices(case_ids, index, volumes_dir, n_slices, positives_only=True):
    """
    Gathers up to `n_slices` full-size slices, drawing across patients in turn.

    Spreading over several patients rather than taking them all from one is
    deliberate: a single misaligned volume would otherwise be the only thing
    tested, and a per-patient bug would look like a global one.
    """
    # Round-robin, most-central tumour slice of each patient first. Filling
    # from one patient before moving on - which is what this did originally -
    # put all 8 slices in the smallest tumour in the split, at 0.05% positive
    # pixels, and tested one volume's alignment rather than the pipeline's.
    ranked = []
    for case_id in case_ids:
        info = index["cases"][case_id]
        idxs = (sorted(info["positive_slices"]) if positives_only
                else list(range(info["n_slices"])))
        if not idxs:
            continue
        centre = idxs[len(idxs) // 2]
        ranked.append((case_id, sorted(idxs, key=lambda s: abs(s - centre))))

    picked = []
    depth = 0
    while len(picked) < n_slices and ranked:
        progressed = False
        for case_id, idxs in ranked:
            if depth < len(idxs):
                picked.append((case_id, idxs[depth]))
                progressed = True
                if len(picked) >= n_slices:
                    break
        if not progressed:
            break
        depth += 1

    by_case = {}
    for case_id, s in picked:
        by_case.setdefault(case_id, []).append(s)

    imgs, lbls = [], []
    for case_id, slices in by_case.items():
        img = np.load(os.path.join(volumes_dir, f"{case_id}_img.npy"), mmap_mode="r")
        lbl = np.load(os.path.join(volumes_dir, f"{case_id}_lbl.npy"), mmap_mode="r")
        for s in slices:
            imgs.append(np.asarray(img[:, :, s], dtype=np.float32))
            lbls.append((np.asarray(lbl[:, :, s]) > 0.5).astype(np.float32))
    return (torch.from_numpy(np.stack(imgs)[:, None].copy()),
            torch.from_numpy(np.stack(lbls)[:, None].copy()),
            picked)


def check_slices(index, volumes_dir, train_ids, n_slices, steps, verbose=True):
    """Memorise N tumour-bearing slices at full resolution."""
    images, labels, picked = load_slices(train_ids, index, volumes_dir,
                                         n_slices)
    frac = float(labels.mean())
    print(f"  {images.shape[0]} tumour slices at {images.shape[-1]} px from "
          f"{len({c for c, _ in picked})} patients | "
          f"positive pixels {frac:.2%}")
    t0 = time.time()
    dice, history, _ = memorise(images, labels, steps=steps, verbose=verbose)
    print(f"  final Dice {dice:.4f}  ({time.time() - t0:.0f}s)")
    return dice, history


def check_patient(index, volumes_dir, case_id, steps, verbose=True):
    """
    Memorise one whole patient, empty slices included.

    Dice is scored on the tumour slices only. A model that paints everywhere
    would score well on those alone, so the empty slices are reported as a
    false-alarm rate beside it.
    """
    info = index["cases"][case_id]
    img = np.load(os.path.join(volumes_dir, f"{case_id}_img.npy"), mmap_mode="r")
    lbl = np.load(os.path.join(volumes_dir, f"{case_id}_lbl.npy"), mmap_mode="r")
    n = info["n_slices"]
    images = torch.from_numpy(
        np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[:, None].copy())
    labels = torch.from_numpy(
        (np.asarray(lbl) > 0.5).astype(np.float32).transpose(2, 0, 1)[:, None].copy())
    pos = sorted(info["positive_slices"])
    print(f"  {case_id}: {n} slices, {len(pos)} with tumour "
          f"({len(pos) / n:.1%}), positive pixels {float(labels.mean()):.3%}")

    t0 = time.time()
    _, history, probs = memorise(images, labels, steps=steps, verbose=verbose)
    hard = (probs > 0.5).float()
    tumour_dice = _dice(probs[pos], labels[pos])
    empty = [s for s in range(n) if s not in set(pos)]
    alarms = float((hard[empty].sum(dim=(1, 2, 3)) > 0).float().mean()) if empty else 0.0
    print(f"  Dice on tumour slices {tumour_dice:.4f} | "
          f"empty slices given a prediction {alarms:.1%}  "
          f"({time.time() - t0:.0f}s)")
    return tumour_dice, alarms, history


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--split", default=None,
                   help="A split JSON. Defaults to whatever index.json carries.")
    p.add_argument("--n_slices", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--patient_steps", type=int, default=300)
    p.add_argument("--skip_patient", action="store_true")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Dice below this is reported as a failure.")
    p.add_argument("--preprocessed_name", default="preprocessed")
    args = p.parse_args()

    pre = os.path.join(OUTPUT_DIR, args.preprocessed_name)
    with open(os.path.join(pre, "index.json")) as f:
        index = json.load(f)
    volumes = os.path.join(pre, "volumes")

    if args.split:
        with open(args.split) as f:
            train_ids = sorted(json.load(f)["train"])
        print(f"split: {args.split}")
    else:
        train_ids = sorted(index["splits"]["train"])
        print("split: the one in index.json")
    train_ids = [c for c in train_ids if index["cases"][c]["positive_slices"]]

    print(f"\n=== {args.n_slices} tumour slices, no augmentation ===")
    dice, _ = check_slices(index, volumes, train_ids, args.n_slices, args.steps)

    ok = dice >= args.threshold
    if not ok:
        print(f"\n  FAIL: {dice:.4f} < {args.threshold}. Memorisation needs no "
              f"learnable pattern, so this is not the partial-volume problem.\n"
              f"  Check image/label orientation and slice order, the loss, the "
              f"label dtype, and that gradients reach the output layer.")
    else:
        print(f"\n  PASS: the pipeline can memorise. Low Dice on real data is a "
              f"hard problem, not a broken chain.")

    if not args.skip_patient:
        print(f"\n=== one whole patient, empty slices included ===")
        check_patient(index, volumes, train_ids[0], args.patient_steps)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
