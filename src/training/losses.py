"""
Loss function factory for the segmentation experiments.

Usage:
    from src.training.losses import build_loss_function
    criterion = build_loss_function("focal_tversky")
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss


LOSS_TYPES = ("dice_ce", "dice_focal", "tversky", "focal_tversky")


# MONAI's Dice family scores an empty prediction on an empty target as a perfect
# match. On a slice whose target is empty the intersection is identically zero,
# so the entire Dice term reduces to
#
#     smooth_nr / (|P| + smooth_dr)
#
# where |P| is the summed sigmoid output and is entirely under the model's
# control. With the default smooth_nr = smooth_dr = 1e-5 that expression rises
# to 1 as the model silences itself, so the network is handed a dial it can turn
# to collect reward without segmenting anything. Under `--sampling all` about
# 90% of training slices are empty, which makes the dial worth more than the task
# itself — the degenerate solution scores a loss of 0.10 against 0.99 for a
# network that is genuinely learning.
#
# Networks do take the offer. The signature is a run that climbs normally for
# fifteen or twenty epochs and then, within two epochs, drops its training loss
# from 0.93 to 0.115 while validation Dice falls to exactly 0.0000 and stays
# there. Nothing pulls it back, because escaping would mean paying +1.0 Dice on
# every empty slice at once.
#
# Zeroing smooth_nr closes the trapdoor. The numerator on an empty target is
# then identically zero whatever |P| is, so the slice contributes a constant 1.0
# and, more importantly, zero gradient: the Dice term stops seeing empty slices
# altogether and shapes only real lesions. Suppressing false positives is left to
# the cross-entropy term, which does it per pixel — the appropriate mechanism.
#
# smooth_dr stays non-zero; that one is the actual guard against 0/0.
#
# The alternative, `batch=True`, removes the reward just as well but pools the
# intersection across the batch, turning a macro average into a micro one. On a
# mixed batch that shifts the cost of missing a small lesion against a large one
# from 1:8.7 to 1:71, and small lesions are already the hardest category here.
#
# Positive slices are untouched by either choice: 2 * |P and G| counts hundreds to
# thousands of pixels, against which the removed 1e-5 shows up at the sixth
# decimal.
#
# One consequence to be aware of when picking a loss: this leaves `tversky` and
# `focal_tversky` with no mechanism at all for suppressing false positives, since
# they carry no pixel-level term to take over. Both over-segment heavily as a
# result. See EXPERIMENTS.md.
SMOOTH_NR = 0.0
SMOOTH_DR = 1e-5


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss, from Abraham & Khan (2019).

    The Tversky index generalizes Dice by weighting false negatives and false
    positives separately:

        TI = TP / (TP + alpha * FP + beta * FN)

    With beta > alpha, missing a tumour costs more than hallucinating one, which
    pushes the network towards higher sensitivity. That trade-off matters here
    because tumour voxels are a small fraction of every volume and a network
    minimizing plain Dice can do well by simply under-segmenting.

    The focal term then raises the loss to a power:

        FTL = (1 - TI) ^ gamma

    With gamma < 1 the gradient stays large for examples that are already close
    to correct, so training does not stall once easy slices are solved.

    MONAI's TverskyLoss does not accept a gamma argument, so passing one raises
    TypeError. This wrapper composes it explicitly instead.

    Args:
        alpha (float): Weight on false positives.
        beta (float): Weight on false negatives.
        gamma (float): Focal exponent.
        sigmoid (bool): Apply sigmoid to logits before computing the index.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, sigmoid: bool = True):
        super().__init__()
        self.tversky = TverskyLoss(sigmoid=sigmoid, alpha=alpha, beta=beta,
                                   smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MONAI's TverskyLoss already returns (1 - TI).
        tversky_loss = self.tversky(pred, target)
        # Clamped away from zero: d/dx x^gamma diverges at x = 0 for gamma < 1.
        return torch.pow(tversky_loss.clamp(min=1e-7), self.gamma)


def build_loss_function(loss_type: str, tversky_alpha: float = 0.3,
                        tversky_beta: float = 0.7) -> nn.Module:
    """
    Creates the loss function for an experiment.

    `tversky_alpha` and `tversky_beta` are ignored by the Dice-family losses and
    default to the values published with Focal Tversky, so `loss_type="tversky"`
    on its own still reproduces the earlier runs exactly.

    Those defaults turned out to be badly matched to this problem. Penalising a
    false negative 2.33 times a false positive is meant to raise sensitivity on an
    under-segmenting model, but on a target occupying under 1% of the volume it
    overshoots: both Tversky runs reached a sensitivity comparable to the DiceCE
    baseline (0.388 and 0.433 against 0.434) at a precision of 0.046 and 0.068,
    painting roughly eight times more volume than exists. Lowering beta narrows
    that ratio; at alpha = beta = 0.5 the index becomes Dice exactly, so the
    useful range is bounded on both sides.

    Args:
        loss_type (str): One of 'dice_ce', 'dice_focal', 'tversky',
            'focal_tversky'.
        tversky_alpha (float): Weight on false positives.
        tversky_beta (float): Weight on false negatives.

    Returns:
        nn.Module: The loss, expecting raw logits and a binary target.

    Raises:
        ValueError: If loss_type is not recognised.
    """
    if loss_type == "dice_ce":
        return DiceCELoss(sigmoid=True,
                          smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
    if loss_type == "dice_focal":
        return DiceFocalLoss(sigmoid=True,
                             smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
    if loss_type == "tversky":
        return TverskyLoss(sigmoid=True, alpha=tversky_alpha, beta=tversky_beta,
                           smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
    if loss_type == "focal_tversky":
        return FocalTverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                gamma=0.75, sigmoid=True)

    raise ValueError(
        f"Unknown loss type: {loss_type!r}. Choose from: {', '.join(LOSS_TYPES)}"
    )
