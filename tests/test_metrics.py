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

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import src.config
from src.evaluation.metrics import (
    compute_2d_slice_metrics,
    compute_all_3d_metrics,
    compute_dice_3d,
    compute_iou_3d,
    filter_predicted_components,
    reconstruct_patient_3d_volume,
    threshold_sweep,
    threshold_sweep_original_geometry,
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
    """
    Slices land at their own index; gaps stay zero.

    The opt-out is explicit because this fixture is deliberately incomplete: it
    tests placement, and full coverage would leave no gap to check.
    """
    slices = {0: np.ones((4, 4)), 3: np.full((4, 4), 2.0)}
    stack = stack_slice_predictions(slices, n_slices=5,
                                    require_full_coverage=False)
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

# ----------------------------------------------------------------------------
#  THRESHOLD SWEEP IN ORIGINAL GEOMETRY
# ----------------------------------------------------------------------------

# Not a multiple of the 32/8 upsampling factor, which is the whole point.
GT_BOUNDARY_COLUMN = 17


def _sweep_fixture(tmp_path, monkeypatch, net_size=8, orig_size=32, n_slices=4):
    """
    Builds one patient whose reconstruction upsamples in plane, which is the
    condition under which the two sweep spaces can disagree at all.

    The prediction ramps linearly across the slice, so every candidate threshold
    places the boundary in a different column and neither sweep sees a flat
    curve. The ground truth ends at original column 17, deliberately not a
    multiple of the 4x upsampling factor: network space can only put a boundary
    on a multiple of four original columns, while reconstruction can put it
    anywhere. That quantisation is what makes the two optima separable.

    Returns:
        tuple: (probs dict, metadata_dir as str, ground truth in original geometry)
    """
    data_dir = tmp_path / "archive"
    metadata_dir = tmp_path / "metadata"
    labels_dir = data_dir / "labelsTr"
    for d in (metadata_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    case_id = "c_sweep"
    affine = np.eye(4)
    ornt = [[0, 1], [1, 1], [2, 1]]

    probs_net = np.empty((net_size, net_size, n_slices), dtype=np.float32)
    for col in range(net_size):
        probs_net[:, col, :] = 0.95 - 0.9 * col / (net_size - 1)

    gt = np.zeros((orig_size, orig_size, n_slices), dtype=np.uint8)
    gt[:, :GT_BOUNDARY_COLUMN, :] = 1
    nib.save(nib.Nifti1Image(gt, affine), str(labels_dir / f"{case_id}.nii.gz"))

    (metadata_dir / f"{case_id}.json").write_text(json.dumps({
        "case_id": case_id,
        "original_affine": affine.tolist(),
        "original_shape": [orig_size, orig_size, n_slices],
        "original_spacing": [1.0, 1.0, 1.0],
        "ornt": ornt,
        "canonical_shape": [orig_size, orig_size, n_slices],
        "canonical_spacing": [1.0, 1.0, 1.0],
        "target_spacing": [1.0, 1.0, 1.0],
        "resampled_shape": [orig_size, orig_size, n_slices],
        "crop_bbox": {"x_min": 0, "x_max": orig_size, "y_min": 0,
                      "y_max": orig_size, "z_min": 0, "z_max": n_slices},
        "cropped_shape": [orig_size, orig_size, n_slices],
        "target_slice_size": [net_size, net_size],
        "hu_min": -1000.0, "hu_max": 400.0,
    }))

    monkeypatch.setattr(src.config, "DATA_DIR", str(data_dir))
    probs = {case_id: {s: probs_net[:, :, s] for s in range(n_slices)}}
    return probs, str(metadata_dir), gt


def test_original_geometry_sweep_matches_a_naive_full_volume_computation():
    """
    The sweep restricts itself to voxels above the lowest candidate or inside the
    ground truth, which is what makes it twelve times faster than scoring the
    whole volume. That shortcut is only exact because no candidate mask reaches
    outside that union. This pins the claim: the fast path must agree with the
    slow one to the last bit, not merely closely.
    """
    rng = np.random.default_rng(11)
    probs = rng.uniform(0.0, 1.0, size=(40, 40, 6)).astype(np.float32)
    gt = np.zeros((40, 40, 6), dtype=bool)
    gt[8:24, 10:26, 1:5] = True
    grid = [0.10, 0.25, 0.50, 0.75, 0.90]

    naive = {}
    for t in grid:
        pred = probs > t
        total = int(pred.sum()) + int(gt.sum())
        naive[t] = 2.0 * int(np.logical_and(pred, gt).sum()) / total

    relevant = (probs > min(grid)) | gt
    flat = np.flatnonzero(relevant.ravel())
    p_vec, g_vec = probs.ravel()[flat], gt.ravel()[flat]
    gt_total = int(g_vec.sum())

    for t in grid:
        p_bin = p_vec > t
        total = int(p_bin.sum()) + gt_total
        fast = 2.0 * int(np.logical_and(p_bin, g_vec).sum()) / total
        assert fast == naive[t], (
            f"restricting to the relevant union changed the score at {t}: "
            f"{fast} vs {naive[t]}")

    assert flat.size < probs.size, "the fixture does not exercise any restriction"


def test_original_geometry_sweep_optimises_what_evaluate_full_reports(
        tmp_path, monkeypatch):
    """
    The whole point of this function is that its argmax is the argmax of the
    score actually reported. Recomputing that score independently, through the
    same reconstruction `evaluate_full` uses, must agree on the winner.
    """
    probs, metadata_dir, gt = _sweep_fixture(tmp_path, monkeypatch)
    grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    best, results = threshold_sweep_original_geometry(
        probs, metadata_dir, thresholds=grid, verbose=False)

    metadata = json.loads(
        (Path(metadata_dir) / "c_sweep.json").read_text())
    independent = {}
    for t in grid:
        pred = reconstruct_patient_3d_volume(
            probs["c_sweep"], metadata, threshold=t, binarize=True) > 0.5
        total = int(pred.sum()) + int(gt.sum())
        independent[t] = 2.0 * int(np.logical_and(pred, gt > 0).sum()) / total

    for t in grid:
        assert results[t] == pytest.approx(independent[t], abs=1e-9), (
            f"threshold {t}: swept {results[t]} but reconstruction scores "
            f"{independent[t]}")
    assert best == max(independent, key=independent.get)


def test_the_two_sweep_spaces_can_disagree(tmp_path, monkeypatch):
    """
    `threshold_sweep`'s docstring used to assert the two spaces pick the same
    optimum. They do not, and the mechanism is interpolation across a boundary:
    upsampling turns a step into a ramp, and the ramp only enters the mask at a
    lower threshold. This is the regression guard for that claim ever coming
    back.
    """
    probs, metadata_dir, gt = _sweep_fixture(tmp_path, monkeypatch)
    grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    # Network space scores against the label resized into network space, which
    # is what the training pipeline hands `threshold_sweep`.
    net_gt = {s: (gt[::4, ::4, s] > 0).astype(np.uint8) for s in range(gt.shape[2])}
    net_best, net_results = threshold_sweep(
        probs, {"c_sweep": net_gt}, thresholds=grid, verbose=False)
    orig_best, orig_results = threshold_sweep_original_geometry(
        probs, metadata_dir, thresholds=grid, verbose=False)

    # A flat curve would make the comparison meaningless: both would return the
    # first candidate and agree for the wrong reason.
    for name, res in (("network", net_results), ("original", orig_results)):
        assert len(set(round(v, 9) for v in res.values())) > 1, (
            f"the {name} sweep is flat, so this fixture proves nothing")

    assert orig_best != net_best, (
        f"both spaces chose {net_best}, so the disagreement this function "
        f"exists to fix is not being exercised")


def test_original_geometry_sweep_refuses_without_metadata(tmp_path, monkeypatch):
    """
    Without metadata a prediction cannot be returned to the geometry its ground
    truth lives in, so there is nothing to sweep. Failing loudly beats returning
    a default threshold that looks like a measurement.
    """
    probs, metadata_dir, _ = _sweep_fixture(tmp_path, monkeypatch)
    empty = tmp_path / "no_metadata"
    empty.mkdir()
    with pytest.raises(ValueError, match="No patient could be swept"):
        threshold_sweep_original_geometry(probs, str(empty), verbose=False)


def test_missing_slices_are_refused_rather_than_zero_filled():
    """
    A slice with no prediction used to become an all-zero one, which is a
    confident claim of "no tumour here" made without running the model.

    Measured on this dataset: under an evaluation restricted to tumour slices,
    2 999 of 3 272 test slices would be filled that way, and the same checkpoint
    scored 0.4645 that way against 0.0540 on every slice. The fill has to be
    asked for, not inherited.
    """
    predictions = {0: np.ones((4, 4), dtype=np.float32),
                   2: np.ones((4, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="no prediction"):
        stack_slice_predictions(predictions, n_slices=5)


def test_the_oracle_opt_out_still_zero_fills():
    """
    The restricted evaluation is a real thing to measure, so the escape hatch has
    to work — and has to be the only way to get the old behaviour.
    """
    predictions = {0: np.ones((4, 4), dtype=np.float32),
                   2: np.ones((4, 4), dtype=np.float32)}
    stack = stack_slice_predictions(predictions, n_slices=5,
                                    require_full_coverage=False)
    assert stack.shape == (4, 4, 5)
    assert stack[..., 0].sum() == 16 and stack[..., 2].sum() == 16
    for empty in (1, 3, 4):
        assert stack[..., empty].sum() == 0


def test_full_coverage_passes_unchanged():
    """Complete coverage must not be affected by the guard."""
    predictions = {s: np.full((4, 4), 0.5, dtype=np.float32) for s in range(5)}
    stack = stack_slice_predictions(predictions, n_slices=5)
    assert stack.shape == (4, 4, 5)
    assert np.allclose(stack, 0.5)


def test_out_of_range_indices_do_not_count_as_coverage():
    """
    An index past the end is silently dropped, so counting keys rather than
    in-range writes would let a stack pass the guard with a hole in it.
    """
    predictions = {s: np.ones((4, 4), dtype=np.float32) for s in range(4)}
    predictions[99] = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="1 of 5 slices"):
        stack_slice_predictions(predictions, n_slices=5)


# ============================================================================
#  THRESHOLD x SLICE-PROTOCOL: THE IDENTITY THAT MUST HOLD
# ============================================================================
#
# Notebook N reported the same checkpoint as 0.3652 tumour-slice Dice under the
# positives protocol at threshold 0.25 and 0.1667 under the all-slice protocol
# at 0.40. Two variables moved at once. Separating them rests on one property:
# at a *fixed* threshold, the tumour-slice Dice cannot depend on which protocol
# selected the slices, because a 2D model predicts slice i from slice i alone.
# If it ever does depend on it, slices are being indexed or assembled wrongly
# and every dual-protocol comparison in the project is void - so it is pinned
# here rather than left as an argument.

def _two_protocol_volumes(n_slices=12, size=16, tumour_slices=(3, 4, 8)):
    """
    One patient's predictions, delivered twice: once for every slice, once for
    the tumour-bearing slices only. The per-slice arrays are the same objects,
    which is exactly the situation a 2D model produces.
    """
    rng = np.random.default_rng(7)
    gt = np.zeros((size, size, n_slices), dtype=np.uint8)
    for index in tumour_slices:
        gt[4:11, 4:11, index] = 1

    per_slice = {}
    for index in range(n_slices):
        slab = rng.uniform(0.0, 0.45, size=(size, size)).astype(np.float32)
        if index in tumour_slices:
            slab[3:10, 5:12] = rng.uniform(0.55, 0.99, size=(7, 7))
        else:
            # A false alarm on an empty slice, so the all-slice protocol has
            # something the positives protocol structurally cannot see.
            if index % 4 == 0:
                slab[1:4, 1:4] = 0.8
        per_slice[index] = slab

    everything = {index: per_slice[index] for index in range(n_slices)}
    positives_only = {index: per_slice[index] for index in tumour_slices}
    return everything, positives_only, gt, tumour_slices


@pytest.mark.parametrize("threshold", [0.10, 0.25, 0.40, 0.50, 0.75, 0.90])
def test_tumour_slice_dice_does_not_depend_on_the_protocol(threshold):
    """
    Claim 1 of the matrix. Same slices, same threshold, same predictions: the
    tumour-slice score has to come out bit-identical whichever protocol fed the
    model, because the untouched slices carry no tumour and so enter no tumour
    slice average.
    """
    everything, positives_only, gt, tumour_slices = _two_protocol_volumes()
    n_slices = gt.shape[2]

    full = stack_slice_predictions(everything, n_slices) > threshold
    # The oracle path: slices the model never saw come back as zeros.
    oracle = stack_slice_predictions(positives_only, n_slices,
                                     require_full_coverage=False) > threshold

    from_full = compute_2d_slice_metrics(full, gt)
    from_oracle = compute_2d_slice_metrics(oracle, gt)

    assert from_full["dice_2d_tumour_slices"] == pytest.approx(
        from_oracle["dice_2d_tumour_slices"], abs=1e-12), (
        "the same tumour slices scored differently under the two protocols")
    assert from_full["n_tumour_slices"] == from_oracle["n_tumour_slices"] \
        == len(tumour_slices)


@pytest.mark.parametrize("threshold", [0.25, 0.40, 0.50])
def test_whole_volume_dice_does_depend_on_the_protocol(threshold):
    """
    Claim 2, and the reason the identity above is not a triviality. The empty
    slices the oracle protocol never runs on reconstruct as empty, so its false
    alarms there vanish and its volume Dice is inflated. If this ever stopped
    being true the guard against zero-filling would have gone missing.
    """
    everything, positives_only, gt, _ = _two_protocol_volumes()
    n_slices = gt.shape[2]

    full = stack_slice_predictions(everything, n_slices) > threshold
    oracle = stack_slice_predictions(positives_only, n_slices,
                                     require_full_coverage=False) > threshold

    honest = compute_dice_3d(full, gt)
    inflated = compute_dice_3d(oracle, gt)

    assert inflated > honest, (
        f"the oracle protocol did not gain from its unscored slices "
        f"({inflated:.4f} vs {honest:.4f}) - the fixture has no false alarms "
        f"on empty slices and proves nothing")
    # And the all-slice view is where that difference shows up.
    assert (compute_2d_slice_metrics(full, gt)["false_alarm_rate_2d"]
            > compute_2d_slice_metrics(oracle, gt)["false_alarm_rate_2d"])


def test_the_volume_ratio_is_sensitivity_over_precision():
    """
    The reported ratio has to be the quantity it claims, not an independent
    estimate that could drift from the two numbers printed beside it.
    """
    everything, _, gt, _ = _two_protocol_volumes()
    pred = stack_slice_predictions(everything, gt.shape[2]) > 0.4

    metrics = compute_all_3d_metrics(pred, gt, surface_metrics=False)
    assert metrics["volume_ratio_3d"] == pytest.approx(
        metrics["sensitivity_3d"] / metrics["precision_3d"], rel=1e-9)
    assert metrics["volume_ratio_3d"] == pytest.approx(
        float(pred.sum()) / float(gt.sum()), rel=1e-9)


def test_the_volume_ratio_is_nan_without_a_ground_truth_volume():
    """No true volume means no ratio; 0.0 would read as perfect agreement."""
    empty = np.zeros((8, 8, 4), dtype=np.uint8)
    pred = np.zeros((8, 8, 4), dtype=np.uint8)
    pred[1:3, 1:3, 1] = 1
    metrics = compute_all_3d_metrics(pred, empty, surface_metrics=False)
    assert np.isnan(metrics["volume_ratio_3d"])


def _z_resampling_fixture(tmp_path, monkeypatch, n_prep=8, n_orig=16, size=8):
    """
    One patient plus predictions that make the Z blend visible.

    `n_orig > n_prep` is the real case: an original spacing finer than the 1 mm
    the pipeline resamples to, so returning the stack to original geometry
    blends neighbouring preprocessed slices. lung_023 in this dataset is exactly
    this, 332 preprocessed slices to 531 original.

    The neighbours of the tumour slices carry 0.45 - below the 0.5 cut on their
    own, but enough that blending them with a confident 0.95 lands above it.
    That is what makes the zero-fill visible: replace 0.45 with 0.0 and the same
    blend falls below the cut. With near-zero neighbours the fixture proves
    nothing, which is how the first version of this test passed while measuring
    nothing.

    Returns:
        tuple: (per-slice probabilities, metadata dir, ground truth, the slice
        indices the oracle protocol would be given)
    """
    data_dir = tmp_path / "archive"
    metadata_dir = tmp_path / "metadata"
    labels_dir = data_dir / "labelsTr"
    for d in (metadata_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    case_id = "c_zresample"
    affine = np.eye(4)
    scale = n_prep / n_orig
    resamples = scale != 1
    # The tumour occupies the original slices the confident preprocessed ones
    # map onto, so the blend at its Z edges is what gets scored.
    z_start, z_stop = (4, 12) if resamples else (2, 6)
    gt = np.zeros((size, size, n_orig), dtype=np.uint8)
    gt[2:6, 2:6, z_start:z_stop] = 1
    nib.save(nib.Nifti1Image(gt, affine), str(labels_dir / f"{case_id}.nii.gz"))

    (metadata_dir / f"{case_id}.json").write_text(json.dumps({
        "case_id": case_id,
        "original_affine": affine.tolist(),
        "original_shape": [size, size, n_orig],
        "original_spacing": [1.0, 1.0, scale],
        "ornt": [[0, 1], [1, 1], [2, 1]],
        "canonical_shape": [size, size, n_orig],
        "canonical_spacing": [1.0, 1.0, scale],
        "target_spacing": [1.0, 1.0, 1.0],
        "resampled_shape": [size, size, n_prep],
        "crop_bbox": {"x_min": 0, "x_max": size, "y_min": 0, "y_max": size,
                      "z_min": 0, "z_max": n_prep},
        "cropped_shape": [size, size, n_prep],
        "target_slice_size": [size, size],
        "hu_min": -1000.0, "hu_max": 400.0,
    }))
    monkeypatch.setattr(src.config, "DATA_DIR", str(data_dir))

    confident = (3, 4) if resamples else (2, 3, 4, 5)
    per_slice = {}
    for index in range(n_prep):
        slab = np.full((size, size), 0.01, dtype=np.float32)
        if index in confident:
            slab[2:6, 2:6] = 0.95
        elif index in (min(confident) - 1, max(confident) + 1):
            slab[2:6, 2:6] = 0.45
        per_slice[index] = slab
    return per_slice, str(metadata_dir), gt, set(confident)


def test_the_protocol_identity_survives_reconstruction_without_z_resampling(
        tmp_path, monkeypatch):
    """
    The earlier version of this test stacked slices and scored them, which only
    checked the metric. The claim is about the whole pipeline, so it has to go
    through reconstruction - and with Z left alone, it still holds exactly.
    """
    per_slice, metadata_dir, gt, positive = _z_resampling_fixture(
        tmp_path, monkeypatch, n_prep=16, n_orig=16)
    metadata = json.loads((Path(metadata_dir) / "c_zresample.json").read_text())
    subset = {z: per_slice[z] for z in positive}

    full = reconstruct_patient_3d_volume(per_slice, metadata, threshold=0.5)
    oracle = reconstruct_patient_3d_volume(subset, metadata, threshold=0.5,
                                           require_full_coverage=False)
    a = compute_2d_slice_metrics(full, gt)["dice_2d_tumour_slices"]
    b = compute_2d_slice_metrics(oracle, gt)["dice_2d_tumour_slices"]
    assert a == pytest.approx(b, abs=1e-12), (
        f"identity broke with Z untouched: {a} vs {b}")


def test_z_resampling_lets_the_zero_fill_reach_the_scored_slices(
        tmp_path, monkeypatch):
    """
    And with Z resampled it stops holding, which is the finding the matrix
    reports rather than an indexing bug. An original slice is a blend of
    preprocessed neighbours; the oracle protocol zero-fills those neighbours,
    so the fill dilutes the tumour slices it *is* scored on.

    Discovered because the matrix flagged claim 1 as failed on real data, and
    the disagreement turned out to sit entirely on lung_023 - the one test
    patient whose reconstruction upsamples along Z.
    """
    per_slice, metadata_dir, gt, positive = _z_resampling_fixture(
        tmp_path, monkeypatch, n_prep=8, n_orig=16)
    metadata = json.loads((Path(metadata_dir) / "c_zresample.json").read_text())
    subset = {z: per_slice[z] for z in positive}

    full = reconstruct_patient_3d_volume(per_slice, metadata, threshold=0.5)
    oracle = reconstruct_patient_3d_volume(subset, metadata, threshold=0.5,
                                           require_full_coverage=False)
    a = compute_2d_slice_metrics(full, gt)["dice_2d_tumour_slices"]
    b = compute_2d_slice_metrics(oracle, gt)["dice_2d_tumour_slices"]

    assert a != pytest.approx(b, abs=1e-9), (
        "the fixture does not actually resample along Z, so it cannot show the "
        "effect it exists to show")
    assert b < a, (
        f"zero-filled neighbours should dilute the oracle score, not raise it: "
        f"{b} vs {a}")
