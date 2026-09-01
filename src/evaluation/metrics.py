"""
3D Volumetric Evaluation Engine for Lung Tumor Segmentation.

This module provides the complete 3D evaluation pipeline required for model
benchmark reporting. It reconstructs 2D slice predictions back into the patient's
original 3D NIfTI geometry and computes ALL required metrics:

    Volumetric Metrics:
        - 3D Dice Similarity Coefficient (DSC)
        - 3D IoU / Jaccard Index
        - 3D Sensitivity / Recall
        - 3D Precision / Positive Predictive Value
        - 3D Specificity / True Negative Rate
        - Failure Rate (% of cases with 3D Dice < 0.10)

    Surface Distance Metrics:
        - 95th Percentile Hausdorff Distance (HD95 in mm)
        - Average Surface Distance (ASD in mm)

    Topology Metrics:
        - Number of False Positive Connected Components

    Reporting:
        - Per-patient CSV export
        - Macro-average (mean of per-patient scores)
        - Micro-average (pooled voxel-level computation)
        - Stratified report by tumor size (Small/Medium/Large)
        - Threshold sweep on validation set

    Model Info:
        - Number of Model Parameters
        - Peak GPU VRAM Memory (MB)

Usage:
    python -m src.evaluation.metrics
"""

import csv
import json
import os

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label as scipy_label
from typing import Dict, List, Tuple, Optional

from src.geometry import reconstruct_to_original_geometry


# ============================================================================
#  CORE 3D VOLUMETRIC METRICS
# ============================================================================

def compute_dice_3d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes the 3D Volumetric Dice Similarity Coefficient (DSC).
    
    Formula:
        Dice = 2 * |Pred intersection GT| / (|Pred| + |GT|)
        
    Special cases:
        - Both empty: 1.0 (perfect agreement on negative volume).
        - One empty, other not: 0.0.
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    
    intersection = np.logical_and(pred_b, gt_b).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def compute_iou_3d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes the 3D Intersection over Union (IoU / Jaccard Index).
    
    Formula:
        IoU = |Pred intersection GT| / |Pred union GT|
        
    Special cases:
        - Both empty: 1.0.
        - One empty, other not: 0.0.
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    pred_sum = pred_b.sum()
    gt_sum = gt_b.sum()
    
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(intersection / union)


def compute_sensitivity_3d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes 3D Sensitivity (Recall / True Positive Rate).
    
    Formula:
        Sensitivity = TP / (TP + FN)
        
    Returns 1.0 if GT is empty (no tumor to miss).
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    tp = np.logical_and(pred_b, gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    
    if (tp + fn) == 0:
        return 1.0
    return float(tp / (tp + fn))


def compute_precision_3d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes 3D Precision (Positive Predictive Value).
    
    Formula:
        Precision = TP / (TP + FP)
        
    Returns 1.0 if both empty, 0.0 if pred has voxels but GT is empty.
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    
    if (tp + fp) == 0:
        return 1.0 if gt_b.sum() == 0 else 0.0
    return float(tp / (tp + fp))


def compute_specificity_3d(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Computes 3D Specificity (True Negative Rate).
    
    Formula:
        Specificity = TN / (TN + FP)
        
    Returns 1.0 if both masks are fully positive (no negatives to classify).
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    tn = np.logical_and(~pred_b, ~gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    
    if (tn + fp) == 0:
        return 1.0
    return float(tn / (tn + fp))


# ============================================================================
#  SURFACE DISTANCE METRICS
# ============================================================================

def _compute_surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray,
                               voxel_spacing: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes bidirectional surface distances between pred and gt boundaries.
    
    Returns:
        Tuple of (distances_pred_to_gt, distances_gt_to_pred) in mm.
        Returns (empty, empty) arrays if either mask is empty.
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return np.array([]), np.array([])
    
    # EDT of the complement gives distance from each voxel to the nearest foreground voxel
    pred_edt = distance_transform_edt(~pred_b, sampling=voxel_spacing)
    gt_edt = distance_transform_edt(~gt_b, sampling=voxel_spacing)
    
    # Surface voxels: foreground voxels that border at least one background voxel
    from scipy.ndimage import binary_erosion
    pred_surface = pred_b & ~binary_erosion(pred_b)
    gt_surface = gt_b & ~binary_erosion(gt_b)
    
    # Handle edge case where erosion removes all voxels (very thin structures)
    if pred_surface.sum() == 0:
        pred_surface = pred_b
    if gt_surface.sum() == 0:
        gt_surface = gt_b
    
    dist_pred_to_gt = gt_edt[pred_surface]
    dist_gt_to_pred = pred_edt[gt_surface]
    
    return dist_pred_to_gt, dist_gt_to_pred


def compute_hd95_3d(pred_mask: np.ndarray, gt_mask: np.ndarray,
                    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """
    Computes 95th Percentile Hausdorff Distance (HD95) in physical mm.
    
    Returns 100.0 mm penalty if either mask is empty.
    """
    dist_p2g, dist_g2p = _compute_surface_distances(pred_mask, gt_mask, voxel_spacing)
    
    if len(dist_p2g) == 0 or len(dist_g2p) == 0:
        return 100.0
    
    all_distances = np.concatenate([dist_p2g, dist_g2p])
    return float(np.percentile(all_distances, 95))


def compute_asd_3d(pred_mask: np.ndarray, gt_mask: np.ndarray,
                   voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)) -> float:
    """
    Computes Average Surface Distance (ASD) in physical mm.
    
    ASD is the mean of all bidirectional surface distances:
        ASD = (mean(d(pred_surface, gt)) + mean(d(gt_surface, pred))) / 2
        
    Returns 100.0 mm penalty if either mask is empty.
    """
    dist_p2g, dist_g2p = _compute_surface_distances(pred_mask, gt_mask, voxel_spacing)
    
    if len(dist_p2g) == 0 or len(dist_g2p) == 0:
        return 100.0
    
    asd = (np.mean(dist_p2g) + np.mean(dist_g2p)) / 2.0
    return float(asd)


# ============================================================================
#  TOPOLOGY METRICS
# ============================================================================

def count_false_positive_components(pred_mask: np.ndarray, gt_mask: np.ndarray) -> int:
    """
    Counts the number of 3D connected components in the prediction that do NOT
    intersect with the ground-truth mask at all (pure false positive blobs).
    
    Args:
        pred_mask: Binary 3D predicted mask.
        gt_mask: Binary 3D ground-truth mask.
        
    Returns:
        int: Number of false positive connected components.
    """
    pred_b = (pred_mask > 0.5).astype(bool)
    gt_b = (gt_mask > 0.5).astype(bool)
    
    if pred_b.sum() == 0:
        return 0
    
    # Label all connected components in the prediction
    labeled_pred, num_components = scipy_label(pred_b)
    
    fp_count = 0
    for comp_id in range(1, num_components + 1):
        component_mask = (labeled_pred == comp_id)
        # Check if this component has ANY overlap with ground truth
        overlap = np.logical_and(component_mask, gt_b).sum()
        if overlap == 0:
            fp_count += 1
    
    return fp_count


def filter_predicted_components(pred_mask: np.ndarray,
                                min_fraction: float = 0.10,
                                min_voxels: int = 0) -> Tuple[np.ndarray, int]:
    """
    Drops predicted connected components that are small relative to the largest.

    The first full benchmark showed HD95 between 115 and 184 mm on every
    configuration while Dice sat around 0.44, which is not what a poorly
    delineated boundary looks like. Per patient the cause was visible: lung_001
    reached sensitivity 0.969 — the lesion was found almost completely — yet
    carried 7 false-positive components and HD95 189 mm. The surface distance was
    being set by scattered specks most of a lung away from the tumour, not by the
    lesion itself.

    Keeping only the largest component would be the obvious rule and the wrong
    one here: the EDA found 24 of 63 patients with more than one true component,
    up to 14, so that rule discards real disease. Scaling the cutoff to the
    largest component keeps genuine multifocal findings and removes speckle.

    The largest component is always kept, so a non-empty prediction never becomes
    empty — the filter removes satellites, it does not decide whether there is a
    prediction at all.

    Args:
        pred_mask: Binary or probability 3D prediction.
        min_fraction: Components below this fraction of the largest component's
            size are removed. 0 disables filtering.
        min_voxels: Absolute size floor, applied on top of `min_fraction`.

    Returns:
        tuple: (filtered uint8 mask, number of components removed).
    """
    pred_b = pred_mask > 0.5
    if not pred_b.any() or (min_fraction <= 0 and min_voxels <= 0):
        return pred_b.astype(np.uint8), 0

    labeled, n_components = scipy_label(pred_b)
    if n_components <= 1:
        return pred_b.astype(np.uint8), 0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0                                    # background is not a component
    largest_label = int(sizes.argmax())
    cutoff = max(float(min_voxels), float(min_fraction) * int(sizes.max()))

    keep = sizes >= cutoff
    keep[0] = False
    keep[largest_label] = True

    filtered = keep[labeled]
    return filtered.astype(np.uint8), int(n_components - keep.sum())


# ============================================================================
#  COMBINED METRICS COMPUTATION
# ============================================================================

def compute_all_3d_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                           voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
                           surface_metrics: bool = True) -> Dict[str, float]:
    """
    Computes ALL 3D volumetric metrics for a single patient case.

    Returns dict with:
        dice_3d, iou_3d, sensitivity_3d, precision_3d, specificity_3d,
        volume_ratio_3d, hd95_3d, asd_3d, fp_components, is_failure

    `volume_ratio_3d` is predicted volume over true volume: below 1 the model
    under-segments, above 1 it over-paints. It is redundant with the pair above
    it - sensitivity and precision both carry TP on top, one over the true volume
    and one over the predicted one, so their quotient is this ratio - but a
    number nobody has to derive is a number that gets read. NaN when the ground
    truth is empty, since there is no volume to be a ratio of.

    With `surface_metrics=False`, hd95_3d and asd_3d come back as NaN and the two
    distance transforms behind them are skipped. Those transforms run over the
    full reconstructed volume - 512 x 512 x 304 for a typical patient here - and
    dominate the cost of an evaluation by roughly an order of magnitude. Turning
    them off is for comparisons that hinge on overlap rather than boundary
    distance, where paying for them would mean trading the number of
    configurations that fit in a GPU session for numbers nobody reads.
    """
    dice = compute_dice_3d(pred_mask, gt_mask)
    iou = compute_iou_3d(pred_mask, gt_mask)
    sens = compute_sensitivity_3d(pred_mask, gt_mask)
    prec = compute_precision_3d(pred_mask, gt_mask)
    spec = compute_specificity_3d(pred_mask, gt_mask)
    if surface_metrics:
        hd95 = compute_hd95_3d(pred_mask, gt_mask, voxel_spacing)
        asd = compute_asd_3d(pred_mask, gt_mask, voxel_spacing)
    else:
        hd95 = asd = float("nan")
    fp_comps = count_false_positive_components(pred_mask, gt_mask)
    is_failure = bool(dice < 0.10 and (gt_mask > 0.5).sum() > 0)

    gt_voxels = float((gt_mask > 0.5).sum())
    pred_voxels = float((pred_mask > 0.5).sum())
    volume_ratio = pred_voxels / gt_voxels if gt_voxels else float("nan")

    return {
        "dice_3d": dice,
        "iou_3d": iou,
        "sensitivity_3d": sens,
        "precision_3d": prec,
        "specificity_3d": spec,
        "volume_ratio_3d": volume_ratio,
        "hd95_3d": hd95,
        "asd_3d": asd,
        "fp_components": fp_comps,
        "is_failure": is_failure
    }


# ============================================================================
#  PER-SLICE AND PER-LESION METRICS
# ============================================================================
#
# These exist to make this project's numbers comparable with work that reports
# segmentation quality at a different granularity, which most of the nodule
# literature does. The three views answer different questions and are not
# interchangeable:
#
#   per-patient   One Dice over the whole reconstructed volume. Every false
#                 positive anywhere in the scan counts against the score. This is
#                 what `compute_all_3d_metrics` reports and what this project
#                 quotes.
#   per-slice     Dice within a single axial slice, averaged over slices. Taken
#                 over tumour-bearing slices only, false positives on the other
#                 ~91% of slices never enter any average, so the number can be
#                 far higher than the per-patient one for the same prediction.
#   per-lesion    Dice for each ground-truth connected component against the part
#                 of the prediction over it. False positives distant from every
#                 lesion are attributed to none of them, so this also reads
#                 higher than per-patient.
#
# The gap is not a detail. On a published LIDC-IDRI run the same 2.5D U-Net
# scores 0.6154 per tumour-slice and 0.2225 per patient.

def compute_2d_slice_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                             failure_threshold: float = 0.10) -> Dict[str, float]:
    """
    Scores a reconstructed volume slice by slice, aggregated two ways.

    Both aggregations are returned because each is incomplete alone. The
    tumour-slice average measures delineation given that something is there; the
    all-slice average additionally charges for predictions on empty anatomy. A
    model can look strong on the first and weak on the second, and the pair says
    so.

    Two conventions in here decide what the numbers mean, so both are explicit:

    Dice and IoU score a slice with no tumour and no prediction as 1.0. That is
    the usual convention and it is why `dice_2d_all_slices` reads so high: about
    91% of slices in this dataset are empty, so most of that average is the
    model being correctly silent. It is comparable with other work that reports
    per-slice Dice over whole volumes, and it is not comparable with per-patient
    Dice. Never quote it beside the 3D number without saying which is which.

    Precision and sensitivity are left undefined rather than assumed. Precision
    on a slice where nothing was predicted is vacuous, and scoring it 1.0 pays a
    model for its silence: with a third of tumour slices receiving no prediction
    at all, that alone moved measured precision by more than 0.08 here. Those
    slices are excluded from the precision average, and the count that did
    contribute is returned so the exclusion stays visible. Sensitivity is
    handled the same way on slices with no ground truth.

    Args:
        pred_mask, gt_mask: Binary volumes of the same shape, (H, W, D), slice
            axis last.
        failure_threshold: A tumour-bearing slice scoring below this counts as a
            failure.

    Returns:
        dict: dice/iou/sensitivity/precision over tumour slices and over all
              slices, the slice counts including how many carried a prediction,
              the tumour-slice failure rate, and the false-alarm rate, meaning
              the fraction of empty slices on which the model predicted anything.
    """
    pred = np.asarray(pred_mask) > 0.5
    gt = np.asarray(gt_mask) > 0.5

    tumour, everything = [], []
    n_false_alarm = n_empty = n_tumour_predicted = 0

    for index in range(gt.shape[2]):
        p, g = pred[:, :, index], gt[:, :, index]
        tp = int(np.logical_and(p, g).sum())
        p_sum, g_sum = int(p.sum()), int(g.sum())

        if p_sum == 0 and g_sum == 0:
            scores = {"dice": 1.0, "iou": 1.0}
        elif p_sum == 0 or g_sum == 0:
            scores = {"dice": 0.0, "iou": 0.0}
        else:
            scores = {"dice": 2.0 * tp / (p_sum + g_sum),
                      "iou": tp / (p_sum + g_sum - tp)}

        # NaN, not a default: nothing was predicted, so there is no precision to
        # measure, and no ground truth means there is no sensitivity to measure.
        scores["precision"] = (tp / p_sum) if p_sum else float("nan")
        scores["sensitivity"] = (tp / g_sum) if g_sum else float("nan")

        everything.append(scores)
        if g_sum > 0:
            tumour.append(scores)
            n_tumour_predicted += int(p_sum > 0)
        else:
            n_empty += 1
            n_false_alarm += int(p_sum > 0)

    def mean_of(rows, key):
        values = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "n_tumour_slices_with_prediction": n_tumour_predicted,
        "dice_2d_tumour_slices": mean_of(tumour, "dice"),
        "iou_2d_tumour_slices": mean_of(tumour, "iou"),
        "sensitivity_2d_tumour_slices": mean_of(tumour, "sensitivity"),
        "precision_2d_tumour_slices": mean_of(tumour, "precision"),
        "dice_2d_all_slices": mean_of(everything, "dice"),
        "iou_2d_all_slices": mean_of(everything, "iou"),
        "sensitivity_2d_all_slices": mean_of(everything, "sensitivity"),
        "precision_2d_all_slices": mean_of(everything, "precision"),
        "n_tumour_slices": len(tumour),
        "n_slices": len(everything),
        "failure_rate_2d": (float(np.mean([r["dice"] < failure_threshold
                                           for r in tumour])) if tumour
                            else float("nan")),
        "false_alarm_rate_2d": (n_false_alarm / n_empty) if n_empty else 0.0,
    }


def compute_lesion_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                           min_lesion_voxels: int = 0,
                           failure_threshold: float = 0.10) -> List[Dict]:
    """
    Scores each ground-truth lesion separately against the prediction over it.

    A lesion's prediction is every predicted voxel inside that lesion's own
    bounding box: the box is the attribution rule. A false positive elsewhere in
    the scan therefore belongs to no lesion and is invisible here, which is why
    this view reads higher than per-patient Dice and why the two must never be
    quoted as though they were the same quantity.

    `min_lesion_voxels` matters on this dataset specifically. The Decathlon lung
    task annotates one primary tumour per patient, and the extra connected
    components are annotation dust: across the ten test patients the largest
    component holds 99.4% to 100% of the tumour volume, with up to thirteen
    fragments of a few voxels each making up the rest. Averaged in unfiltered,
    those fragments score zero and drag the mean below the per-patient figure,
    inverting the relationship this metric has in the nodule literature.
    Filtering them out leaves one lesion per patient, at which point per-lesion
    Dice is per-patient Dice.

    Args:
        pred_mask, gt_mask: Binary volumes of the same shape.
        min_lesion_voxels: Ground-truth components smaller than this are ignored.
        failure_threshold: A lesion scoring below this counts as missed.

    Returns:
        list of dict: one row per lesion, carrying its voxel count, Dice, IoU,
        sensitivity, precision, and whether it was missed.
    """
    pred = np.asarray(pred_mask) > 0.5
    gt = np.asarray(gt_mask) > 0.5

    labelled, n_components = scipy_label(gt)
    lesions = []

    for component in range(1, n_components + 1):
        lesion = labelled == component
        n_voxels = int(lesion.sum())
        if n_voxels < min_lesion_voxels:
            continue

        coords = np.argwhere(lesion)
        lower = coords.min(axis=0)
        upper = coords.max(axis=0) + 1
        box = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))

        lesion_gt = lesion[box]
        lesion_pred = pred[box]

        tp = int(np.logical_and(lesion_pred, lesion_gt).sum())
        p_sum, g_sum = int(lesion_pred.sum()), int(lesion_gt.sum())
        dice = 2.0 * tp / (p_sum + g_sum) if (p_sum + g_sum) else 1.0
        union = p_sum + g_sum - tp

        lesions.append({
            "lesion_voxels": n_voxels,
            "dice": dice,
            "iou": tp / union if union else 1.0,
            "sensitivity": tp / g_sum if g_sum else 1.0,
            "precision": tp / p_sum if p_sum else 0.0,
            "is_missed": bool(dice < failure_threshold),
        })

    return lesions


# ============================================================================
#  3D VOLUME RECONSTRUCTION
# ============================================================================

def stack_slice_predictions(slice_predictions: Dict[int, np.ndarray],
                            n_slices: int,
                            require_full_coverage: bool = True) -> np.ndarray:
    """
    Assembles a {slice_index: 2D array} mapping into a dense 3D stack.

    A missing slice becomes an all-zero prediction, and that is not a neutral
    default. It is a confident claim that the slice holds no tumour, made without
    running the model. Evaluated on a set restricted to tumour-bearing slices,
    the 91.7% of this dataset's test slices that are negative would all be filled
    in that way: the false positives the model would have produced there simply
    never appear, and on the per-slice views an empty prediction against an empty
    label scores a free 1.0. The same checkpoint measured both ways came out at
    Dice 0.4645 and 0.0540.

    So the fill is refused by default. `require_full_coverage=False` is the
    deliberate opt-out for an oracle-positive-slices evaluation, which is a real
    thing to measure but is not volume performance and must not be reported as
    though it were.

    Args:
        slice_predictions: Mapping of slice index to a 2D (192, 192) array.
        n_slices: Depth of the output stack, from metadata["cropped_shape"][2].
        require_full_coverage: Raise unless every slice in range has a
            prediction. Pass False only for an evaluation that is reported as
            restricted.

    Returns:
        np.ndarray: float32 stack of shape (192, 192, n_slices).

    Raises:
        ValueError: If slices are missing and full coverage was required.
    """
    any_slice = next(iter(slice_predictions.values()))
    stack = np.zeros((*any_slice.shape, n_slices), dtype=np.float32)
    in_range = 0
    for slice_idx, slice_2d in slice_predictions.items():
        if 0 <= slice_idx < n_slices:
            stack[:, :, slice_idx] = slice_2d
            in_range += 1

    if require_full_coverage and in_range < n_slices:
        missing = n_slices - in_range
        raise ValueError(
            f"{missing} of {n_slices} slices have no prediction and would be "
            f"filled with zeros, which scores them as confidently empty without "
            f"the model having seen them. Either predict every slice, or pass "
            f"require_full_coverage=False and report the result as an oracle "
            f"positive-slices evaluation rather than as volume performance.")
    return stack


def reconstruct_patient_3d_volume(slice_predictions, metadata: Dict,
                                  threshold: float = 0.5,
                                  binarize: bool = True,
                                  require_full_coverage: bool = True) -> np.ndarray:
    """
    Reconstructs a 3D prediction in the patient's ORIGINAL NIfTI geometry.

    Delegates the geometry work to src.geometry, which inverts all four forward
    steps in reverse order: slice resize, body crop, 1mm resampling, and the
    canonical reorientation. That last inverse is what makes the prediction
    comparable to a ground truth read straight from the source NIfTI; without it
    every volume in this dataset comes back mirrored along the left-right axis.

    Args:
        slice_predictions: Either a {slice_index: 2D array} mapping or a dense
            (192, 192, D) array of per-slice probabilities.
        metadata: Per-patient metadata written by the preprocessing stage.
        threshold: Binarization threshold, applied after all interpolation.
        binarize: If False, returns float probabilities instead of a binary mask.
        require_full_coverage: Refuse to zero-fill slices the model was never run
            on. See `stack_slice_predictions`; pass False only for an evaluation
            reported as restricted to oracle positive slices.

    Returns:
        np.ndarray: Volume of shape metadata["original_shape"].
    """
    n_slices = int(metadata["cropped_shape"][2])

    if isinstance(slice_predictions, dict):
        slice_stack = stack_slice_predictions(
            slice_predictions, n_slices,
            require_full_coverage=require_full_coverage)
    else:
        slice_stack = np.asarray(slice_predictions, dtype=np.float32)

    return reconstruct_to_original_geometry(
        slice_stack, metadata, threshold=threshold, binarize=binarize)


# ============================================================================
#  THRESHOLD SWEEP
# ============================================================================

# The grid runs to 0.99 rather than stopping at 0.90, and is sampled more finely
# in the tail. A model that spreads low probabilities over a large area reaches its
# best Dice at an unusually high threshold, and a grid that ends early reports the
# edge value while the curve is still climbing — scoring the run at a threshold
# that is demonstrably not its best. The fine tail is where such models separate.
DEFAULT_SWEEP_THRESHOLDS = (
    [round(float(t), 3) for t in np.arange(0.10, 0.95, 0.05)]
    + [0.925, 0.95, 0.975, 0.99]
)


def threshold_sweep(patient_probs: Dict[str, Dict[int, np.ndarray]],
                    patient_labels: Dict[str, Dict[int, np.ndarray]],
                    thresholds: Optional[List[float]] = None,
                    verbose: bool = True) -> Tuple[float, Dict]:
    """
    Finds the binarization threshold that maximizes mean per-patient 3D Dice on
    the validation set.

    The sweep runs in preprocessed 192x192 space, which is not the space the
    result is reported in. An earlier version of this docstring claimed the two
    pick the same optimum "since reconstruction is a fixed monotone resampling
    shared by every candidate". That is wrong, and measurably so.

    Reconstruction is monotone in a single voxel's value but is not a bijection:
    it upsamples roughly 2.6x in plane by linear interpolation and only then
    thresholds. A boundary between a 0.9 voxel and a 0.1 voxel becomes a smooth
    ramp, and that ramp contributes almost no area at threshold 0.75 but a great
    deal at 0.40. The reconstructed mask therefore grows faster as the threshold
    falls than the 192x192 mask does, and the optimum moves down. Measured on
    this project's validation split, it moves down by exactly one grid step for
    both a single model (0.75 -> 0.65) and a three-seed ensemble (0.65 -> 0.55),
    worth +0.013 and +0.017 test Dice respectively.

    Prefer `threshold_sweep_original_geometry`, which optimises the quantity
    actually reported. This function is kept because it is two orders of
    magnitude cheaper and is the right tool when the comparison itself lives in
    network space, such as the per-epoch validation signal.

    This must only ever be called on validation data. Choosing a threshold on
    test would leak the test set into model selection.

    Args:
        patient_probs: {case_id: {slice_index: 2D probability array}}.
        patient_labels: {case_id: {slice_index: 2D ground-truth array}} — the
            REAL labels for those same slices.
        thresholds: Candidate thresholds. Defaults to 0.10 ... 0.90 step 0.05.
        verbose: Whether to print each candidate's score.

    Returns:
        tuple: (best_threshold, {threshold: mean_dice}).
    """
    if thresholds is None:
        thresholds = DEFAULT_SWEEP_THRESHOLDS

    missing = set(patient_probs) - set(patient_labels)
    if missing:
        raise ValueError(
            f"Threshold sweep is missing ground truth for: {sorted(missing)[:5]}. "
            f"Without real labels the sweep would compare predictions against "
            f"themselves and always return ~0.5."
        )

    results: Dict[float, float] = {}

    if verbose:
        print(f"  Testing {len(thresholds)} thresholds "
              f"from {thresholds[0]:.2f} to {thresholds[-1]:.2f}...")

    for thresh in thresholds:
        dices = []
        for case_id, probs_dict in patient_probs.items():
            labels_dict = patient_labels[case_id]
            total_intersection = 0
            total_pred = 0
            total_gt = 0

            for s_idx, s_pred in probs_dict.items():
                gt_2d = labels_dict.get(s_idx)
                if gt_2d is None:
                    continue
                p_bin = s_pred > thresh
                g_bin = gt_2d > 0.5
                total_intersection += int(np.logical_and(p_bin, g_bin).sum())
                total_pred += int(p_bin.sum())
                total_gt += int(g_bin.sum())

            if total_pred == 0 and total_gt == 0:
                dice = 1.0
            elif total_pred == 0 or total_gt == 0:
                dice = 0.0
            else:
                dice = float(2.0 * total_intersection / (total_pred + total_gt))
            dices.append(dice)

        results[thresh] = float(np.mean(dices)) if dices else 0.0
        if verbose:
            print(f"    [Threshold {thresh:.2f}] Mean Val Dice = {results[thresh]:.4f}",
                  flush=True)

    best_threshold = max(results, key=results.get)
    if verbose:
        print(f"\n  => Optimal Threshold: {best_threshold:.2f} "
              f"(Val Dice = {results[best_threshold]:.4f})", flush=True)

    return best_threshold, results


def threshold_sweep_original_geometry(patient_probs: Dict[str, Dict[int, np.ndarray]],
                                      metadata_dir: str,
                                      thresholds: Optional[List[float]] = None,
                                      verbose: bool = True,
                                      require_full_coverage: bool = True) -> Tuple[float, Dict]:
    """
    Finds the binarization threshold maximizing mean per-patient 3D Dice in the
    patient's ORIGINAL NIfTI geometry, against the untouched source label.

    This optimises the same quantity `evaluate_full` reports, which
    `threshold_sweep` does not — see that function's docstring for the
    measured discrepancy and why it exists.

    The obvious implementation, reconstructing once per candidate threshold, costs
    about 6.4 seconds per patient per threshold and is unusable. Two observations
    make it cheap instead:

    Reconstruction is threshold-independent when asked for probabilities, so it
    runs once per patient rather than once per candidate.

    Almost none of the volume can ever matter. Only voxels above the lowest
    candidate threshold, or inside the ground truth, can affect any score; on this
    dataset that is 0.04% of an 86-million-voxel volume. Extracting them once and
    sweeping over the compact vector gives bit-identical results roughly 12 times
    faster, and drops the full sweep from 40 seconds per patient to 3.3.

    That second shortcut is only valid because every candidate is at or above
    `min(thresholds)`, which holds by construction. Do not lower a threshold
    afterwards against the returned vector.

    Args:
        patient_probs: {case_id: {slice_index: 2D probability array}}, or a dense
            (H, W, D) array per case — whatever `reconstruct_patient_3d_volume`
            accepts.
        metadata_dir (str): Directory of per-patient reconstruction metadata.
        thresholds: Candidate thresholds. Defaults to 0.10 ... 0.99.
        verbose: Whether to print each candidate's score.
        require_full_coverage: Refuse to zero-fill slices the model was not run
            on. False is for a sweep on an oracle positive-slices protocol, where
            the threshold chosen is the best for that protocol and no other.

    Returns:
        tuple: (best_threshold, {threshold: mean_dice})
    """
    import nibabel as nib

    from src.config import resolve_nifti_path

    if thresholds is None:
        thresholds = DEFAULT_SWEEP_THRESHOLDS
    floor = min(thresholds)

    if verbose:
        print(f"  Testing {len(thresholds)} thresholds from {min(thresholds):.2f} "
              f"to {max(thresholds):.2f}, in original geometry...")

    per_patient: Dict[str, Dict[float, float]] = {}

    for case_id, slice_dict in sorted(patient_probs.items()):
        meta_path = os.path.join(metadata_dir, f"{case_id}.json")
        if not os.path.exists(meta_path):
            print(f"  [WARNING] No metadata for {case_id}, skipping.")
            continue

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        probs_3d = reconstruct_patient_3d_volume(
            slice_dict, metadata, binarize=False,
            require_full_coverage=require_full_coverage)
        gt_path = resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")
        gt_3d = np.asanyarray(nib.load(gt_path).dataobj) > 0.5

        # Every voxel that could ever cross a candidate threshold, plus every
        # voxel of ground truth. Restricting to their union leaves the counts
        # below exact rather than approximate: no candidate mask reaches outside
        # it, and the whole ground truth is inside it.
        relevant = (probs_3d > floor) | gt_3d
        flat = np.flatnonzero(relevant.ravel())
        p_vec = probs_3d.ravel()[flat]
        g_vec = gt_3d.ravel()[flat]
        del probs_3d, relevant, flat

        gt_total = int(g_vec.sum())
        scores = {}
        for thresh in thresholds:
            p_bin = p_vec > thresh
            pred_total = int(p_bin.sum())
            if pred_total == 0 and gt_total == 0:
                scores[thresh] = 1.0
            elif pred_total == 0 or gt_total == 0:
                scores[thresh] = 0.0
            else:
                intersection = int(np.logical_and(p_bin, g_vec).sum())
                scores[thresh] = float(2.0 * intersection /
                                       (pred_total + gt_total))
        per_patient[case_id] = scores
        del p_vec, g_vec

    if not per_patient:
        raise ValueError(
            "No patient could be swept: none of the given cases had metadata in "
            f"{metadata_dir}. Without it a prediction cannot be returned to the "
            f"geometry its ground truth lives in.")

    results = {t: float(np.mean([s[t] for s in per_patient.values()]))
               for t in thresholds}
    if verbose:
        for thresh in thresholds:
            print(f"    [Threshold {thresh:.2f}] Mean Val Dice = "
                  f"{results[thresh]:.4f}", flush=True)

    best_threshold = max(results, key=results.get)
    if verbose:
        print(f"\n  => Optimal Threshold: {best_threshold:.2f} "
              f"(Val Dice = {results[best_threshold]:.4f}, original geometry)",
              flush=True)

    return best_threshold, results


# ============================================================================
#  CSV EXPORT & REPORTING
# ============================================================================

def export_results_csv(patient_metrics: Dict[str, Dict], output_path: str,
                       tumor_categories: Optional[Dict[str, str]] = None):
    """
    Exports per-patient evaluation results to a CSV file.
    
    Args:
        patient_metrics: Dict of {case_id: metrics_dict}.
        output_path: Path to save the CSV file.
        tumor_categories: Optional dict of {case_id: "small"/"medium"/"large"}.
    """
    if not patient_metrics:
        print("  [WARNING] No patient metrics to export.")
        return
    
    fieldnames = ["case_id", "tumor_category",
                  "dice_3d", "iou_3d", "sensitivity_3d", "precision_3d",
                  "specificity_3d", "hd95_3d", "asd_3d", "fp_components",
                  "is_failure", "inference_time_sec"]

    # Post-processed columns appear only when evaluate_full produced them, so a
    # run with filtering disabled still writes exactly the original schema.
    pp_keys = sorted({k for m in patient_metrics.values() for k in m
                      if k.startswith("pp_")})
    fieldnames += pp_keys

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for case_id in sorted(patient_metrics.keys()):
            m = patient_metrics[case_id]
            row = {"case_id": case_id}
            row["tumor_category"] = tumor_categories.get(case_id, "unknown") if tumor_categories else "unknown"
            for key in fieldnames[2:]:
                row[key] = m.get(key, "")
            writer.writerow(row)
    
    print(f"  Exported per-patient results CSV: {output_path}")


def compute_macro_micro_averages(patient_metrics: Dict[str, Dict],
                                 patient_pred_masks: Dict[str, np.ndarray],
                                 patient_gt_masks: Dict[str, np.ndarray]) -> Dict:
    """
    Computes both macro-average and micro-average metrics.
    
    Macro-average: Mean of per-patient scores (each patient weighted equally).
    Micro-average: Pooled voxel-level computation across all patients.
    
    Returns:
        Dict with 'macro' and 'micro' sub-dicts.
    """
    # Macro-average: simple mean of per-patient scores
    metric_keys = ["dice_3d", "iou_3d", "sensitivity_3d", "precision_3d",
                   "specificity_3d", "volume_ratio_3d", "hd95_3d", "asd_3d",
                   # Per-slice views of the same predictions, carried alongside
                   # so a run reports every granularity it was scored at. The
                   # micro block stays 3D only: pooling voxels across patients
                   # has no per-slice counterpart.
                   "dice_2d_tumour_slices", "iou_2d_tumour_slices",
                   "sensitivity_2d_tumour_slices", "precision_2d_tumour_slices",
                   "dice_2d_all_slices", "iou_2d_all_slices",
                   "sensitivity_2d_all_slices", "precision_2d_all_slices",
                   "failure_rate_2d", "false_alarm_rate_2d"]

    macro = {}
    for key in metric_keys:
        # A patient can contribute NaN for the precision views when nothing was
        # predicted anywhere; averaging that in would poison the whole column.
        values = [m[key] for m in patient_metrics.values()
                  if key in m and not np.isnan(m[key])]
        macro[key] = float(np.mean(values)) if values else 0.0
    
    # Micro-average: pool all voxels across all patients
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for case_id in patient_pred_masks:
        pred_b = (patient_pred_masks[case_id] > 0.5).astype(bool)
        gt_b = (patient_gt_masks[case_id] > 0.5).astype(bool)
        
        total_tp += np.logical_and(pred_b, gt_b).sum()
        total_fp += np.logical_and(pred_b, ~gt_b).sum()
        total_fn += np.logical_and(~pred_b, gt_b).sum()
        total_tn += np.logical_and(~pred_b, ~gt_b).sum()
    
    micro = {}
    micro["dice_3d"] = float(2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 1.0
    micro["iou_3d"] = float(total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 1.0
    micro["sensitivity_3d"] = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 1.0
    micro["precision_3d"] = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 1.0
    micro["specificity_3d"] = float(total_tn / (total_tn + total_fp)) if (total_tn + total_fp) > 0 else 1.0
    
    return {"macro": macro, "micro": micro}


def stratified_report(patient_metrics: Dict[str, Dict],
                      tumor_categories: Dict[str, str]) -> Dict:
    """
    Groups patient results by tumor size category and computes per-group statistics.
    
    Args:
        patient_metrics: Per-patient metrics dict.
        tumor_categories: Dict mapping case_id -> "small"/"medium"/"large".
        
    Returns:
        Dict with keys "small", "medium", "large", each containing group averages.
    """
    groups = {"small": [], "medium": [], "large": []}
    
    for case_id, metrics in patient_metrics.items():
        cat = tumor_categories.get(case_id, "unknown")
        if cat in groups:
            groups[cat].append(metrics)
    
    report = {}
    metric_keys = ["dice_3d", "iou_3d", "sensitivity_3d", "precision_3d",
                   "hd95_3d", "asd_3d"]
    
    for cat, metrics_list in groups.items():
        if not metrics_list:
            report[cat] = {"n_patients": 0}
            continue
            
        cat_report = {"n_patients": len(metrics_list)}
        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                cat_report[f"mean_{key}"] = float(np.mean(values))
                cat_report[f"median_{key}"] = float(np.median(values))
                cat_report[f"std_{key}"] = float(np.std(values))
        
        failures = sum(1 for m in metrics_list if m.get("is_failure", False))
        cat_report["failure_rate_pct"] = float(failures / len(metrics_list) * 100)
        
        report[cat] = cat_report
    
    return report


# ============================================================================
#  MODEL UTILITIES
# ============================================================================

def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Counts trainable and total parameters in a PyTorch model.
    
    Returns:
        tuple: (trainable_params, total_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def measure_gpu_memory_mb() -> float:
    """
    Measures current peak GPU memory allocation in megabytes (MB).
    
    Returns:
        float: GPU VRAM in MB, or 0.0 if running on CPU.
    """
    if torch.cuda.is_available():
        memory_bytes = torch.cuda.max_memory_allocated()
        return float(memory_bytes / (1024.0 * 1024.0))
    return 0.0


# ============================================================================
#  ARTIFICIAL TEST CASES (6 REQUIRED SCENARIOS)
# ============================================================================

def run_artificial_metric_tests():
    """
    Validates all metric functions against 6 controlled artificial test scenarios.
    Each scenario has a known expected output, verifying correctness of the
    metric implementations.
    """
    print("=== RUNNING 6 ARTIFICIAL METRIC TEST CASES ===\n")
    
    spacing = (1.0, 1.0, 1.0)
    all_passed = True
    
    # --- Test 1: Perfect Prediction (pred == gt) ---
    print("  Test 1: Perfect Prediction (pred == gt)")
    gt1 = np.zeros((50, 50, 50), dtype=np.uint8)
    gt1[20:30, 20:30, 20:30] = 1
    pred1 = gt1.copy()
    m1 = compute_all_3d_metrics(pred1, gt1, spacing)
    assert m1["dice_3d"] == 1.0, f"FAIL: Dice={m1['dice_3d']}, expected 1.0"
    assert m1["iou_3d"] == 1.0, f"FAIL: IoU={m1['iou_3d']}, expected 1.0"
    assert m1["sensitivity_3d"] == 1.0, f"FAIL: Sens={m1['sensitivity_3d']}"
    assert m1["precision_3d"] == 1.0, f"FAIL: Prec={m1['precision_3d']}"
    assert m1["hd95_3d"] == 0.0, f"FAIL: HD95={m1['hd95_3d']}, expected 0.0"
    assert m1["asd_3d"] == 0.0, f"FAIL: ASD={m1['asd_3d']}, expected 0.0"
    assert m1["fp_components"] == 0, f"FAIL: FP comps={m1['fp_components']}"
    assert m1["is_failure"] == False
    print("    [PASS] All metrics correct for perfect prediction.\n")
    
    # --- Test 2: Complete Non-Overlap (pred and gt do not intersect) ---
    print("  Test 2: Complete Non-Overlap (disjoint pred and gt)")
    gt2 = np.zeros((50, 50, 50), dtype=np.uint8)
    gt2[5:15, 5:15, 5:15] = 1
    pred2 = np.zeros((50, 50, 50), dtype=np.uint8)
    pred2[35:45, 35:45, 35:45] = 1
    m2 = compute_all_3d_metrics(pred2, gt2, spacing)
    assert m2["dice_3d"] == 0.0, f"FAIL: Dice={m2['dice_3d']}"
    assert m2["iou_3d"] == 0.0, f"FAIL: IoU={m2['iou_3d']}"
    assert m2["sensitivity_3d"] == 0.0, f"FAIL: Sens={m2['sensitivity_3d']}"
    assert m2["precision_3d"] == 0.0, f"FAIL: Prec={m2['precision_3d']}"
    assert m2["fp_components"] == 1, f"FAIL: FP comps={m2['fp_components']}, expected 1"
    assert m2["is_failure"] == True
    print("    [PASS] All metrics correct for non-overlapping masks.\n")
    
    # --- Test 3: Both Masks Empty ---
    print("  Test 3: Both Masks Empty (no tumor in GT, no prediction)")
    gt3 = np.zeros((50, 50, 50), dtype=np.uint8)
    pred3 = np.zeros((50, 50, 50), dtype=np.uint8)
    m3 = compute_all_3d_metrics(pred3, gt3, spacing)
    assert m3["dice_3d"] == 1.0, f"FAIL: Dice={m3['dice_3d']}"
    assert m3["iou_3d"] == 1.0, f"FAIL: IoU={m3['iou_3d']}"
    assert m3["sensitivity_3d"] == 1.0, f"FAIL: Sens={m3['sensitivity_3d']}"
    assert m3["precision_3d"] == 1.0, f"FAIL: Prec={m3['precision_3d']}"
    assert m3["fp_components"] == 0
    assert m3["is_failure"] == False
    print("    [PASS] All metrics correct for both-empty case.\n")
    
    # --- Test 4: GT Empty + Prediction Non-Empty (False Positive) ---
    print("  Test 4: GT Empty + Prediction Non-Empty (false alarm)")
    gt4 = np.zeros((50, 50, 50), dtype=np.uint8)
    pred4 = np.zeros((50, 50, 50), dtype=np.uint8)
    pred4[20:30, 20:30, 20:30] = 1
    m4 = compute_all_3d_metrics(pred4, gt4, spacing)
    assert m4["dice_3d"] == 0.0, f"FAIL: Dice={m4['dice_3d']}"
    assert m4["iou_3d"] == 0.0, f"FAIL: IoU={m4['iou_3d']}"
    assert m4["precision_3d"] == 0.0, f"FAIL: Prec={m4['precision_3d']}"
    assert m4["fp_components"] == 1, f"FAIL: FP comps={m4['fp_components']}"
    assert m4["is_failure"] == False  # No GT tumor, so not a segmentation failure
    print("    [PASS] All metrics correct for false positive case.\n")
    
    # --- Test 5: Prediction Empty + GT Non-Empty (Complete Miss) ---
    print("  Test 5: Prediction Empty + GT Non-Empty (missed tumor)")
    gt5 = np.zeros((50, 50, 50), dtype=np.uint8)
    gt5[20:30, 20:30, 20:30] = 1
    pred5 = np.zeros((50, 50, 50), dtype=np.uint8)
    m5 = compute_all_3d_metrics(pred5, gt5, spacing)
    assert m5["dice_3d"] == 0.0, f"FAIL: Dice={m5['dice_3d']}"
    assert m5["iou_3d"] == 0.0, f"FAIL: IoU={m5['iou_3d']}"
    assert m5["sensitivity_3d"] == 0.0, f"FAIL: Sens={m5['sensitivity_3d']}"
    assert m5["fp_components"] == 0
    assert m5["is_failure"] == True  # Missed real tumor
    print("    [PASS] All metrics correct for complete miss case.\n")
    
    # --- Test 6: Partial Overlap ---
    print("  Test 6: Partial Overlap (shifted prediction)")
    gt6 = np.zeros((50, 50, 50), dtype=np.uint8)
    gt6[20:30, 20:30, 20:30] = 1  # 10x10x10 = 1000 voxels
    pred6 = np.zeros((50, 50, 50), dtype=np.uint8)
    pred6[25:35, 25:35, 25:35] = 1  # Shifted by 5 in all directions
    m6 = compute_all_3d_metrics(pred6, gt6, spacing)
    # Overlap: [25:30, 25:30, 25:30] = 5x5x5 = 125 voxels
    # Dice = 2*125/(1000+1000) = 0.125
    assert abs(m6["dice_3d"] - 0.125) < 0.001, f"FAIL: Dice={m6['dice_3d']}, expected ~0.125"
    assert m6["sensitivity_3d"] == 0.125, f"FAIL: Sens={m6['sensitivity_3d']}"
    assert m6["precision_3d"] == 0.125, f"FAIL: Prec={m6['precision_3d']}"
    assert m6["is_failure"] == False  # Dice 0.125 > 0.10
    print("    [PASS] All metrics correct for partial overlap case.\n")
    
    print("=" * 50)
    print("  ALL 6 ARTIFICIAL TEST CASES PASSED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    run_artificial_metric_tests()
