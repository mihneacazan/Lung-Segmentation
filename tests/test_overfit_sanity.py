"""
The floor under every other number in this project: can the model memorise?

Every score reported here rests on an assumption nothing else tests — that the
chain from a volume on disk to a gradient is wired correctly. A broken pipeline
and a hard problem both produce low Dice, and a ranking of twenty-one experiments
run through the same broken pipeline would be exactly as consistent, and exactly
as meaningless.

Memorising a handful of slices separates the two, because memorisation needs no
learnable pattern - only that the mapping be representable and that gradients
reach it. 1.6M parameters against eight slices is roughly 200,000 parameters per
slice; failing there is not a capacity limit and not a data limit.

This matters more here than it would elsewhere. The project's own diagnosis is
that the missing sensitivity sits in a partial-volume shell at air density, where
intensity carries no signal. An overfit test bypasses that argument entirely: the
model does not have to infer where the boundary is, it can memorise it. If it
cannot reach a high Dice even then, the explanation is wrong and something
upstream - most likely image-to-mask alignment - is at fault.

Augmentation is off throughout, and that is the point rather than a convenience.
It would show a different image every step, making memorisation impossible by
construction and turning any bug into something indistinguishable from variance.

Run:
    python -m pytest tests/test_overfit_sanity.py -q
"""

import json
import os

import numpy as np
import pytest
import torch

from src.config import OUTPUT_DIR
from src.models.factory import build_model
from src.training.losses import build_loss_function, LOSS_TYPES


# The budget cannot be trimmed to taste. On real slices the run passes through a
# collapse: it predicts a scattered mask for ~50 steps, saturates to all-background
# around step 100, sits at exactly zero for roughly 250 steps, and only then finds
# the tumour and converges. Measured on lung_003: Dice 0.0146 at step 50, 0.0000
# from step 100 to 300, 0.9552 at 400, 0.9856 at 600. A 250-step budget lands in
# the middle of that basin and reports a failure that is purely impatience.
OVERFIT_STEPS = 200
OVERFIT_LR = 5e-3

# The U-Net downsamples four times, so any side not divisible by 16 fails inside
# the skip connection with a shape mismatch rather than anything informative.
GRID_MULTIPLE = 16


def _overfit_dice(images, labels, loss_type="dice_ce", steps=OVERFIT_STEPS,
                  lr=OVERFIT_LR, seed=0, model_type="unet",
                  return_sensitivity=False):
    """
    Trains one model on one fixed batch until it memorises it, then scores it on
    that same batch.

    Deliberately not `run_training_experiment`: augmentation, sampling, early
    stopping and the scheduler are all excluded so that a failure has as few
    possible causes as it can. Augmentation especially — it would show a
    different image every step and make memorisation impossible by construction,
    turning a bug into something indistinguishable from variance.

    Args:
        images (torch.Tensor): (N, C, H, W) network input.
        labels (torch.Tensor): (N, 1, H, W) binary target.
        loss_type (str): Any key of `LOSS_TYPES`.
        steps (int): Optimizer steps on the same batch.
        lr (float): Adam learning rate. Higher than training's 1e-3 because there
            is nothing to generalise to and no reason to converge slowly.
        seed (int): Torch seed, so a failure is reproducible.
        model_type (str): Any key of `MODEL_TYPES`.

    Returns:
        float: Dice of the thresholded prediction against the target it was
            trained on.
    """
    for side in images.shape[-2:]:
        assert side % GRID_MULTIPLE == 0, (
            f"side {side} is not a multiple of {GRID_MULTIPLE}; the U-Net's four "
            f"downsamplings would fail in a skip connection, which looks like a "
            f"shape bug rather than a test-fixture mistake")

    torch.manual_seed(seed)
    model = build_model(model_type, in_channels=images.shape[1], out_channels=1)
    criterion = build_loss_function(loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            pytest.fail(f"{loss_type} produced a non-finite loss while memorising "
                        f"a fixed batch, which no amount of data difficulty can "
                        f"explain")
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(images)) > 0.5).float()

    intersection = float((pred * labels).sum())
    total = float(pred.sum() + labels.sum())
    dice = 1.0 if total == 0 else 2.0 * intersection / total
    if not return_sensitivity:
        return dice
    truth = float(labels.sum())
    return dice, (1.0 if truth == 0 else intersection / truth)


def _bright_blob_batch(n=8, size=32, seed=0):
    """
    A fixed batch whose target is unambiguous: a bright square on dark noise.

    Every slice carries a blob, and the blobs move between slices so the model
    cannot pass by predicting one constant mask. Both properties matter — a batch
    of empty slices would be "solved" by predicting nothing, and the empty-versus-
    empty convention would score that a perfect 1.0. That is the same trap as
    evaluating on tumour slices only, in a different place.
    """
    rng = np.random.default_rng(seed)
    images = rng.uniform(0.05, 0.25, (n, 1, size, size)).astype(np.float32)
    labels = np.zeros((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        r, c = rng.integers(2, size - 12, size=2)
        h, w = rng.integers(6, 10, size=2)
        labels[i, 0, r:r + h, c:c + w] = 1.0
        images[i, 0, r:r + h, c:c + w] = 0.9
    return torch.from_numpy(images), torch.from_numpy(labels)


def test_model_can_memorise_a_fixed_batch():
    """
    The floor under every other number in this project.

    If a 1.6M-parameter network cannot memorise eight slices whose target is a
    bright square, nothing downstream means anything: the loss, the label
    handling, or the gradient path is broken, and every experiment measured that
    break rather than the problem.
    """
    images, labels = _bright_blob_batch()
    dice = _overfit_dice(images, labels)
    assert dice > 0.95, (
        f"the network reached only Dice {dice:.4f} memorising 8 fixed slices. "
        f"This is not a data or capacity limit — look at the loss, the label "
        f"dtype and threshold, and whether gradients reach the output layer.")


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_every_loss_drives_the_model_onto_the_target(loss_type):
    """
    Separates a loss being wrong for this problem from a loss being implemented
    wrongly.

    The four Tversky-family runs are the four worst results in the project
    (0.077 to 0.104), and the documentation attributes that to the loss
    overweighting false negatives on a target under 1% of the volume. That
    explanation was never checked against the alternative: an implementation bug.

    Measured here, the documentation is right. Given 400 steps every loss in
    LOSS_TYPES reaches Dice 1.0000 on this batch, so none of them is broken. What
    separates them is the route: at 150 steps DiceCE is at 0.94 and Dice+Focal at
    0.98, while Tversky and Focal Tversky sit at 0.72 - both with sensitivity
    exactly 1.0 and precision near 0.57. They have found every tumour pixel and
    are still painting almost twice as many as exist, which is precisely what
    alpha=0.3 / beta=0.7 asks for.

    So the assertion is on sensitivity, not on Dice. Demanding a high Dice here
    would fail a correct Tversky for behaving as designed, and running to 400
    steps to avoid that would triple the cost of the file for no extra coverage:
    a loss wired backwards, reduced wrongly, or double-sigmoided does not reach
    sensitivity 1.0 at any step count.
    """
    images, labels = _bright_blob_batch(n=6)
    dice, sensitivity = _overfit_dice(images, labels, loss_type=loss_type,
                                      steps=150, return_sensitivity=True)
    assert sensitivity > 0.95, (
        f"{loss_type} recovered only {sensitivity:.4f} of a target it was free "
        f"to memorise. That points at the implementation - sign, reduction, or a "
        f"sigmoid applied twice - not at the loss being unsuited to this dataset.")
    assert dice > 0.60, (
        f"{loss_type} reached sensitivity {sensitivity:.4f} but Dice only "
        f"{dice:.4f}, so it is painting far more than it should even on a batch "
        f"it has memorised.")


@pytest.mark.parametrize("n_adjacent", [1, 3])
def test_the_centre_slice_is_the_one_being_scored(n_adjacent):
    """
    Guards the 2.5D neighbour indexing, where an off-by-one is invisible.

    The target is built from the centre channel alone; the neighbouring channels
    carry blobs in different places, deliberately contradicting it. A model that
    reads the wrong channel as centre is asked to predict a mask that does not
    match any input it can see, and cannot memorise the batch. A model with the
    indexing right can, easily.

    Nothing else in the suite would catch this: the shapes are correct either
    way, the loss is finite either way, and the score just comes out lower.
    """
    rng = np.random.default_rng(3)
    n, size = 8, 32
    images = rng.uniform(0.05, 0.25, (n, n_adjacent, size, size)).astype(np.float32)
    labels = np.zeros((n, 1, size, size), dtype=np.float32)
    centre = n_adjacent // 2

    for i in range(n):
        for ch in range(n_adjacent):
            r, c = rng.integers(2, size - 12, size=2)
            images[i, ch, r:r + 9, c:c + 9] = 0.9
            if ch == centre:
                labels[i, 0, r:r + 9, c:c + 9] = 1.0

    dice = _overfit_dice(torch.from_numpy(images), torch.from_numpy(labels))
    assert dice > 0.95, (
        f"with n_adjacent={n_adjacent} the network reached only Dice {dice:.4f} "
        f"on a batch whose target is the centre channel's blob. The neighbour "
        f"stack may be ordered so that the centre is not where it is assumed.")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(OUTPUT_DIR, "preprocessed", "index.json")),
    reason="needs a preprocessed dataset; run the preprocessing stage first")
def test_real_tumour_slices_can_be_memorised():
    """
    The same check on real data, and the only version that can catch a
    misalignment between a real image and its real mask.

    The synthetic tests above prove the training chain works on a target the test
    itself built. They cannot prove that `{case}_img.npy` and `{case}_lbl.npy`
    agree about where the tumour is: a transposed axis or a one-slice offset
    would pass every one of them and fail here.

    Scored on a 48 px window centred on the tumour rather than the full 192 px
    slice. That is not only for speed. At full width the tumour is 0.31% of the
    pixels and the run spends hundreds of steps saturated at all-background
    before escaping; the window raises it to 1.26% while keeping every annotated
    voxel, so the check measures whether the mapping is learnable rather than how
    patient the test is. An assertion below confirms nothing was cropped away.

    Tumour-bearing slices only, and that is not the mistake it is elsewhere in
    this project: here the batch is the whole world, so an empty slice would let
    the model score a free 1.0 by predicting nothing.
    """
    with open(os.path.join(OUTPUT_DIR, "preprocessed", "index.json")) as f:
        index = json.load(f)

    volumes = os.path.join(OUTPUT_DIR, "preprocessed", "volumes")
    case_id = sorted(index["splits"]["train"])[0]
    positives = sorted(index["cases"][case_id]["positive_slices"])[:6]
    assert positives, f"{case_id} has no tumour slices to memorise"

    img = np.asarray(np.load(os.path.join(volumes, f"{case_id}_img.npy"),
                             mmap_mode="r"), dtype=np.float32)
    lbl = (np.asarray(np.load(os.path.join(volumes, f"{case_id}_lbl.npy"),
                              mmap_mode="r")) > 0.5).astype(np.float32)

    tumour = lbl[:, :, positives]
    assert tumour.sum() > 0, (
        f"{case_id}'s label is empty on slices the index calls positive, which "
        f"is itself the bug this test exists to find")

    rows, cols = np.where(tumour.sum(axis=2) > 0)
    side = 48
    top = max(0, min(img.shape[0] - side, int(rows.mean()) - side // 2))
    left = max(0, min(img.shape[1] - side, int(cols.mean()) - side // 2))

    window = tumour[top:top + side, left:left + side]
    assert window.sum() == tumour.sum(), (
        f"the {side} px window holds {int(window.sum())} of "
        f"{int(tumour.sum())} annotated voxels. Widen it, or the test is asking "
        f"the model to memorise a mask it cannot see.")

    images = torch.from_numpy(
        img[top:top + side, left:left + side, positives].transpose(2, 0, 1)[:, None].copy())
    labels = torch.from_numpy(window.transpose(2, 0, 1)[:, None].copy())

    dice = _overfit_dice(images, labels, steps=400)
    assert dice > 0.90, (
        f"the network reached only Dice {dice:.4f} memorising {len(positives)} "
        f"real tumour slices from {case_id}. Memorisation needs no learnable "
        f"pattern, so the partial-volume argument does not explain this. Check "
        f"that the image volume and the label volume agree about orientation "
        f"and slice order."
    )
