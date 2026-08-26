"""
Tests for the evaluation metrics and the validation threshold sweep.

Covers the six controlled scenarios every metric implementation must get right
(perfect prediction, total miss, both empty, false alarm, missed tumour, partial
overlap), plus a regression test for the threshold sweep.

The sweep bug these guard against: it was called with an empty ground-truth
dictionary, fell back to comparing each prediction against itself thresholded at
0.5, and therefore always reported 0.5 as optimal with a Dice near 1.0. It also
referenced an uninitialised `results` name, so it raised NameError before any of
that could be noticed.

Run:
    python -m pytest tests/ -v
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_3d_metrics,
    compute_dice_3d,
    compute_iou_3d,
    filter_predicted_components,
    threshold_sweep,
    stack_slice_predictions,
    run_artificial_metric_tests,
    DEFAULT_SWEEP_THRESHOLDS,
)


SPACING = (1.0, 1.0, 1.0)


def cube(shape=(50, 50, 50), lo=20, hi=30):
    """A binary volume containing a single axis-aligned cube."""
    v = np.zeros(shape, dtype=np.uint8)
    v[lo:hi, lo:hi, lo:hi] = 1
    return v


# ============================================================================
#  THE SIX REQUIRED SCENARIOS
# ============================================================================

def test_perfect_prediction():
    gt = cube()
    m = compute_all_3d_metrics(gt.copy(), gt, SPACING)
    assert m["dice_3d"] == 1.0
    assert m["iou_3d"] == 1.0
    assert m["sensitivity_3d"] == 1.0
    assert m["precision_3d"] == 1.0
    assert m["hd95_3d"] == 0.0
    assert m["asd_3d"] == 0.0
    assert m["fp_components"] == 0
    assert m["is_failure"] is False


def test_no_overlap_at_all():
    gt = cube(lo=5, hi=15)
    pred = cube(lo=35, hi=45)
    m = compute_all_3d_metrics(pred, gt, SPACING)
    assert m["dice_3d"] == 0.0
    assert m["iou_3d"] == 0.0
    assert m["sensitivity_3d"] == 0.0
    assert m["precision_3d"] == 0.0
    assert m["fp_components"] == 1
    assert m["is_failure"] is True


def test_both_masks_empty():
    """No tumour and no prediction is perfect agreement, not a failure."""
    empty = np.zeros((50, 50, 50), dtype=np.uint8)
    m = compute_all_3d_metrics(empty.copy(), empty, SPACING)
    assert m["dice_3d"] == 1.0
    assert m["iou_3d"] == 1.0
    assert m["sensitivity_3d"] == 1.0
    assert m["precision_3d"] == 1.0
    assert m["is_failure"] is False


def test_empty_gt_with_nonempty_prediction():
    """A false alarm scores zero, but is not a segmentation failure."""
    gt = np.zeros((50, 50, 50), dtype=np.uint8)
    m = compute_all_3d_metrics(cube(), gt, SPACING)
    assert m["dice_3d"] == 0.0
    assert m["precision_3d"] == 0.0
    assert m["fp_components"] == 1
    assert m["is_failure"] is False


def test_empty_prediction_with_nonempty_gt():
    """A completely missed tumour is a failure."""
    pred = np.zeros((50, 50, 50), dtype=np.uint8)
    m = compute_all_3d_metrics(pred, cube(), SPACING)
    assert m["dice_3d"] == 0.0
    assert m["sensitivity_3d"] == 0.0
    assert m["fp_components"] == 0
    assert m["is_failure"] is True


def test_partial_overlap_matches_hand_computation():
    """
    A 10^3 cube against the same cube shifted 5 voxels on every axis overlaps in
    5^3 = 125 voxels, so Dice = 2*125 / (1000 + 1000) = 0.125.
    """
    gt = cube(lo=20, hi=30)
    pred = cube(lo=25, hi=35)
    m = compute_all_3d_metrics(pred, gt, SPACING)
    assert m["dice_3d"] == pytest.approx(0.125, abs=1e-6)
    assert m["sensitivity_3d"] == pytest.approx(0.125, abs=1e-6)
    assert m["precision_3d"] == pytest.approx(0.125, abs=1e-6)
    assert m["is_failure"] is False


def test_builtin_artificial_suite_runs():
    """The module's own self-check must pass unchanged."""
    run_artificial_metric_tests()


def test_hd95_scales_with_physical_spacing():
    """Surface distances are reported in mm, so anisotropic spacing changes them."""
    gt = cube(lo=20, hi=30)
    pred = cube(lo=22, hi=32)
    iso = compute_all_3d_metrics(pred, gt, (1.0, 1.0, 1.0))["hd95_3d"]
    coarse = compute_all_3d_metrics(pred, gt, (1.0, 1.0, 5.0))["hd95_3d"]
    assert coarse > iso


# ============================================================================
#  THRESHOLD SWEEP
# ============================================================================

def test_threshold_sweep_finds_the_planted_optimum():
    """
    Builds probabilities whose best cut is unambiguously 0.70 and checks the
    sweep recovers it.

    Tumour voxels get probability 0.8; a decoy band of background voxels gets
    0.62. Thresholding is strict (`prob > t`), so every candidate at or above
    0.65 drops the decoy and scores a perfect Dice, while every lower candidate
    picks it up and cannot.
    """
    decoy_prob = 0.62
    labels = {"c1": {}}
    probs = {"c1": {}}

    for s in range(10):
        gt = np.zeros((16, 16), dtype=np.uint8)
        p = np.full((16, 16), 0.05, dtype=np.float32)
        if 3 <= s <= 6:
            gt[4:10, 4:10] = 1
            p[4:10, 4:10] = 0.8
            p[10:14, 4:10] = decoy_prob
        labels["c1"][s] = gt
        probs["c1"][s] = p

    best, results = threshold_sweep(probs, labels, verbose=False)

    assert best > decoy_prob, (
        f"optimum {best} does not exclude the decoy at {decoy_prob}")
    assert results[best] == pytest.approx(1.0, abs=1e-6)
    assert results[0.50] < results[best], "decoy band should hurt a 0.5 cut"
    # Every candidate below the decoy must score strictly worse.
    for t, score in results.items():
        if t < decoy_prob:
            assert score < 1.0


def test_threshold_sweep_rejects_missing_ground_truth():
    """
    The sweep must refuse to run without real labels rather than silently
    comparing predictions against themselves.
    """
    probs = {"c1": {0: np.random.rand(8, 8).astype(np.float32)}}
    with pytest.raises(ValueError, match="missing ground truth"):
        threshold_sweep(probs, {}, verbose=False)


def test_threshold_sweep_is_not_trivially_flat():
    """
    A sweep that returns the same Dice at every threshold is the signature of the
    old self-comparison bug.
    """
    rng = np.random.default_rng(0)
    labels, probs = {"c1": {}}, {"c1": {}}
    for s in range(6):
        gt = np.zeros((16, 16), dtype=np.uint8)
        gt[5:11, 5:11] = 1
        p = rng.uniform(0.0, 0.55, size=(16, 16)).astype(np.float32)
        p[5:11, 5:11] = rng.uniform(0.45, 1.0, size=(6, 6))
        labels["c1"][s], probs["c1"][s] = gt, p

    _, results = threshold_sweep(probs, labels, verbose=False)
    assert len(set(round(v, 6) for v in results.values())) > 1


def test_sweep_grid_reaches_past_the_old_ceiling():
    """
    Four of six runs in the first benchmark selected exactly 0.90 — the last
    candidate in the old grid — with the Dice curve still rising at the edge,
    so the reported optimum was a grid artifact rather than a real maximum.
    """
    assert max(DEFAULT_SWEEP_THRESHOLDS) >= 0.99
    assert 0.95 in DEFAULT_SWEEP_THRESHOLDS
    assert DEFAULT_SWEEP_THRESHOLDS == sorted(DEFAULT_SWEEP_THRESHOLDS)
    assert len(set(DEFAULT_SWEEP_THRESHOLDS)) == len(DEFAULT_SWEEP_THRESHOLDS)


def test_sweep_can_select_a_threshold_above_the_old_ceiling():
    """
    With an optimum genuinely above 0.90, the sweep must be able to reach it.
    Under the old grid this case returned 0.90 and scored below its own best.
    """
    labels, probs = {"c1": {}}, {"c1": {}}
    for s in range(6):
        gt = np.zeros((16, 16), dtype=np.uint8)
        p = np.full((16, 16), 0.93, dtype=np.float32)   # diffuse over-prediction
        gt[4:10, 4:10] = 1
        p[4:10, 4:10] = 0.999                           # the lesion is confident
        labels["c1"][s], probs["c1"][s] = gt, p

    best, results = threshold_sweep(probs, labels, verbose=False)
    assert best > 0.90, f"sweep stopped at {best}, so the grid is still capped"
    assert results[best] > results[0.90]


# ============================================================================
#  CONNECTED-COMPONENT POST-PROCESSING
# ============================================================================

def test_component_filter_drops_satellites_and_keeps_the_lesion():
    """The core case: one real lesion plus scattered specks."""
    pred = np.zeros((40, 40, 40), dtype=np.uint8)
    pred[10:20, 10:20, 10:20] = 1        # lesion, 1000 voxels
    pred[30, 30, 30] = 1                 # speck
    pred[35, 5, 5] = 1                   # speck
    pred[2:4, 2:4, 2:4] = 1              # small blob, 8 voxels

    filtered, removed = filter_predicted_components(pred, min_fraction=0.10)

    assert removed == 3
    assert filtered[10:20, 10:20, 10:20].all(), "the lesion must survive intact"
    assert filtered.sum() == 1000
    assert filtered[30, 30, 30] == 0


def test_component_filter_keeps_genuine_multifocal_disease():
    """
    24 of 63 patients in this dataset have more than one true component, so a
    keep-the-largest rule would discard real disease. A second lesion of
    comparable size must survive.
    """
    pred = np.zeros((40, 40, 40), dtype=np.uint8)
    pred[5:15, 5:15, 5:15] = 1           # 1000 voxels
    pred[25:33, 25:33, 25:33] = 1        # 512 voxels, 51% of the largest

    filtered, removed = filter_predicted_components(pred, min_fraction=0.10)

    assert removed == 0
    assert filtered.sum() == pred.sum()


def test_component_filter_never_empties_a_prediction():
    """
    The filter removes satellites; it does not decide whether a prediction
    exists. Even an absurd floor must leave the largest component standing,
    otherwise a marginal case silently turns into Dice 0.
    """
    pred = np.zeros((20, 20, 20), dtype=np.uint8)
    pred[5:7, 5:7, 5:7] = 1              # 8 voxels, the only component

    filtered, removed = filter_predicted_components(
        pred, min_fraction=0.10, min_voxels=10_000)

    assert filtered.sum() == 8
    assert removed == 0


def test_component_filter_is_a_no_op_when_disabled_or_empty():
    """min_fraction=0 must pass the mask through unchanged."""
    pred = np.zeros((20, 20, 20), dtype=np.uint8)
    pred[5:10, 5:10, 5:10] = 1
    pred[15, 15, 15] = 1

    unchanged, removed = filter_predicted_components(pred, min_fraction=0.0)
    assert removed == 0
    assert unchanged.sum() == pred.sum()

    empty, removed = filter_predicted_components(np.zeros((8, 8, 8)))
    assert empty.sum() == 0 and removed == 0


def test_component_filter_improves_hd95_on_the_observed_failure_pattern():
    """
    Reproduces what the benchmark actually showed: the lesion is found almost
    completely, yet HD95 reads in the hundreds of millimetres. lung_001 scored
    sensitivity 0.969 with HD95 189 mm and 7 false-positive components.

    The false positive has to be a blob, not a speck. HD95 is the 95th
    percentile precisely so that a handful of stray voxels cannot move it — a
    single isolated voxel against this lesion leaves HD95 at 0.0. That the real
    runs reported 115-184 mm therefore says the false positives were substantial
    structures far from the tumour, which is what makes them worth removing.
    """
    lesion, blob = 24, 10                # blob is 7.2% of the lesion by volume
    gt = np.zeros((90, 90, 90), dtype=np.uint8)
    gt[10:10 + lesion, 10:10 + lesion, 10:10 + lesion] = 1

    pred = gt.copy()
    pred[78:78 + blob, 78:78 + blob, 78:78 + blob] = 1

    raw = compute_all_3d_metrics(pred, gt, SPACING)
    filtered, removed = filter_predicted_components(pred, min_fraction=0.10)
    post = compute_all_3d_metrics(filtered, gt, SPACING)

    assert removed == 1
    assert raw["hd95_3d"] > 50.0, "the distant blob should dominate raw HD95"
    assert post["hd95_3d"] == 0.0
    assert raw["fp_components"] == 1 and post["fp_components"] == 0
    assert post["dice_3d"] > raw["dice_3d"]
    assert post["sensitivity_3d"] == raw["sensitivity_3d"], (
        "removing a false positive must not cost recall")


# ============================================================================
#  SLICE STACKING
# ============================================================================

def test_stack_slice_predictions_places_slices_correctly():
    """Slices land at their own index; gaps stay zero."""
    slices = {0: np.ones((4, 4)), 3: np.full((4, 4), 2.0)}
    stack = stack_slice_predictions(slices, n_slices=5)
    assert stack.shape == (4, 4, 5)
    assert stack[:, :, 0].mean() == 1.0
    assert stack[:, :, 3].mean() == 2.0
    assert stack[:, :, 1].sum() == 0.0
    assert stack[:, :, 4].sum() == 0.0

def test_surface_metrics_can_be_skipped_without_touching_overlap():
    """
    HD95 and ASD dominate the cost of an evaluation, so they can be turned off
    for comparisons decided on overlap. Everything else must come back bit for
    bit identical, or the switch would quietly change the numbers it is supposed
    to leave alone.
    """
    gt = np.zeros((40, 40, 20), dtype=np.uint8)
    gt[10:25, 10:22, 5:14] = 1
    pred = np.zeros_like(gt)
    pred[12:27, 12:24, 6:15] = 1
    pred[2:6, 2:6, 2:5] = 1                     # a detached false positive

    full = compute_all_3d_metrics(pred, gt, (1.0, 1.0, 1.0))
    fast = compute_all_3d_metrics(pred, gt, (1.0, 1.0, 1.0), surface_metrics=False)

    for key in ("dice_3d", "iou_3d", "sensitivity_3d", "precision_3d",
                "specificity_3d", "fp_components", "is_failure"):
        assert fast[key] == full[key], f"{key} changed when surface metrics were skipped"

    assert np.isfinite(full["hd95_3d"]) and np.isfinite(full["asd_3d"])
    assert np.isnan(fast["hd95_3d"]) and np.isnan(fast["asd_3d"])


# ============================================================================
#  PER-SLICE METRICS
# ============================================================================

def _slices(spec):
    """
    Builds a (16, 16, N) pred/gt pair from a compact per-slice description.

    Each entry is (gt_box, pred_box), where a box is None for empty or a
    (row0, row1, col0, col1) tuple.
    """
    from src.evaluation.metrics import compute_2d_slice_metrics  # noqa: F401

    n = len(spec)
    gt = np.zeros((16, 16, n), dtype=np.uint8)
    pred = np.zeros((16, 16, n), dtype=np.uint8)
    for i, (g, p) in enumerate(spec):
        for box, vol in ((g, gt), (p, pred)):
            if box is not None:
                r0, r1, c0, c1 = box
                vol[r0:r1, c0:c1, i] = 1
    return pred, gt


def test_missed_tumour_slice_does_not_earn_perfect_precision():
    """
    The regression guard for the convention that inflated this project's reported
    2D precision: a tumour slice with no prediction has no precision to measure,
    so it must be excluded from the average rather than scored 1.0.
    """
    from src.evaluation.metrics import compute_2d_slice_metrics

    # One slice segmented perfectly, one tumour slice missed entirely.
    pred, gt = _slices([((4, 8, 4, 8), (4, 8, 4, 8)),
                        ((4, 8, 4, 8), None)])
    m = compute_2d_slice_metrics(pred, gt)

    assert m["n_tumour_slices"] == 2
    assert m["n_tumour_slices_with_prediction"] == 1
    # Averaging the missed slice in as 1.0 would also give 1.0 here, so the
    # discriminating case is the imperfect one below.
    assert m["dice_2d_tumour_slices"] == pytest.approx(0.5)
    assert m["sensitivity_2d_tumour_slices"] == pytest.approx(0.5)
    assert m["precision_2d_tumour_slices"] == pytest.approx(1.0)


def test_precision_average_ignores_slices_without_predictions():
    """Only the slices that actually predicted something set the precision."""
    from src.evaluation.metrics import compute_2d_slice_metrics

    # Slice 0: 16 true positives out of 32 predicted -> precision 0.5.
    # Slice 1: tumour present, nothing predicted -> no precision to measure.
    pred, gt = _slices([((4, 8, 4, 8), (4, 8, 4, 12)),
                        ((4, 8, 4, 8), None)])
    m = compute_2d_slice_metrics(pred, gt)

    assert m["precision_2d_tumour_slices"] == pytest.approx(0.5)
    # The old convention averaged 0.5 with a vacuous 1.0 and reported 0.75.
    assert m["precision_2d_tumour_slices"] < 0.75


def test_empty_slice_left_empty_scores_as_agreement():
    """
    Dice keeps the usual convention, and this is why the all-slice average reads
    so high: most slices in a lung volume hold no tumour.
    """
    from src.evaluation.metrics import compute_2d_slice_metrics

    pred, gt = _slices([(None, None), (None, None), ((4, 8, 4, 8), (4, 8, 4, 8))])
    m = compute_2d_slice_metrics(pred, gt)

    assert m["dice_2d_all_slices"] == pytest.approx(1.0)
    assert m["dice_2d_tumour_slices"] == pytest.approx(1.0)
    assert m["n_tumour_slices"] == 1 and m["n_slices"] == 3
    assert m["false_alarm_rate_2d"] == pytest.approx(0.0)


def test_false_alarms_are_invisible_to_the_tumour_slice_view():
    """
    The reason both aggregations are reported. A prediction on an empty slice
    cannot change the tumour-slice average, and must change the all-slice one.
    """
    from src.evaluation.metrics import compute_2d_slice_metrics

    clean, gt = _slices([((4, 8, 4, 8), (4, 8, 4, 8)), (None, None)])
    noisy, _ = _slices([((4, 8, 4, 8), (4, 8, 4, 8)), (None, (0, 3, 0, 3))])

    a = compute_2d_slice_metrics(clean, gt)
    b = compute_2d_slice_metrics(noisy, gt)

    assert a["dice_2d_tumour_slices"] == pytest.approx(b["dice_2d_tumour_slices"])
    assert b["dice_2d_all_slices"] < a["dice_2d_all_slices"]
    assert b["false_alarm_rate_2d"] == pytest.approx(1.0)
    assert a["false_alarm_rate_2d"] == pytest.approx(0.0)


def test_3d_dice_is_the_slice_weighted_mean_of_2d_dice():
    """
    Ties the two granularities together numerically, which is what makes the gap
    between them interpretable rather than mysterious: 3D Dice is the per-slice
    Dice weighted by |pred| + |gt|, while the tumour-slice average is unweighted
    and drops the slices where only a false positive appears.
    """
    from src.evaluation.metrics import compute_dice_3d

    rng = np.random.default_rng(11)
    gt = (rng.random((16, 16, 12)) > 0.85).astype(np.uint8)
    pred = (rng.random((16, 16, 12)) > 0.85).astype(np.uint8)

    weighted = weights = 0.0
    for i in range(gt.shape[2]):
        p, g = pred[:, :, i], gt[:, :, i]
        w = float(p.sum() + g.sum())
        if w == 0:
            continue
        weighted += w * (2.0 * float(np.logical_and(p, g).sum()) / w)
        weights += w

    assert weighted / weights == pytest.approx(compute_dice_3d(pred, gt), abs=1e-12)
