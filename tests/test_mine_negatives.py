"""
Tests for false-positive mining and the sampling mode it feeds.

The experiment these support asks whether negatives chosen by the model's own
errors beat negatives chosen by distance to the tumour. That comparison is only
meaningful if the two modes differ in *which* slices they pick and in nothing
else, so most of what is checked here is sameness: equal pool sizes, equal draw
counts, equal structure.
"""

import json
import os

import numpy as np
import pytest

from src.training.dataset import LungSliceDataset
from src.training.mine_negatives import (
    load_mined,
    pool_quality,
    rank_slices,
    score_volume,
    write_mined,
)


def _index(n_slices=40, positives=(18, 19, 20), case_id="lung_001"):
    return {"cases": {case_id: {
        "n_slices": n_slices,
        "positive_slices": list(positives),
        "body_slices": list(range(4, n_slices - 4)),
        "tumor_voxels": 500,
    }}, "splits": {"train": [case_id]}}


def _volumes(tmp_path, n_slices=40, case_id="lung_001", size=8):
    rng = np.random.default_rng(0)
    img = rng.random((size, size, n_slices)).astype(np.float16)
    lbl = np.zeros((size, size, n_slices), dtype=np.uint8)
    lbl[2:5, 2:5, 18:21] = 1
    d = tmp_path / "volumes"
    d.mkdir(exist_ok=True)
    np.save(d / f"{case_id}_img.npy", img)
    np.save(d / f"{case_id}_lbl.npy", lbl)
    return str(d)


# -- scoring ---------------------------------------------------------------

def test_score_volume_skips_positive_slices():
    """A predicted pixel on a tumour slice may be right, so it is not mined."""
    probs = np.zeros((10, 4, 4), dtype=np.float32)
    probs[3] = 0.9          # negative slice: a real false positive
    probs[5] = 0.9          # positive slice: not a false positive
    scored = score_volume(probs, positive_slices=[5], n_slices=10, threshold=0.5)
    assert 5 not in scored
    assert scored[3]["fp_pixels"] == 16
    assert scored[0]["fp_pixels"] == 0


def test_score_volume_accepts_either_axis_order():
    """predict_patient returns (H, W, S) in places and (S, H, W) in others."""
    hw_s = np.zeros((4, 4, 10), dtype=np.float32)
    hw_s[:, :, 3] = 0.9
    s_hw = np.moveaxis(hw_s, -1, 0)
    a = score_volume(hw_s, [], 10, 0.5)
    b = score_volume(s_hw, [], 10, 0.5)
    assert a == b
    assert a[3]["fp_pixels"] == 16


def test_prob_mass_breaks_ties_below_the_threshold():
    """
    Most negatives sit at zero false-positive pixels, so without a second key
    the ranking below the threshold would be index order rather than badness.
    """
    probs = np.zeros((4, 4, 4), dtype=np.float32)
    probs[1] = 0.4          # under threshold, but the model is uneasy here
    probs[2] = 0.1
    scored = score_volume(probs, [], 4, threshold=0.5)
    assert all(v["fp_pixels"] == 0 for v in scored.values())
    assert rank_slices(scored)[:2] == [1, 2]


def test_pool_quality_reports_the_unbacked_remainder():
    """
    Asking for more slices than the model actually errs on yields random ones.
    The experiment has to be able to say so rather than claim a mined pool.
    """
    mined = {"a": {0: {"fp_pixels": 5, "prob_mass": 5.0},
                   1: {"fp_pixels": 0, "prob_mass": 0.0},
                   2: {"fp_pixels": 0, "prob_mass": 0.0}}}
    q = pool_quality(mined, pool_sizes={"a": 3})
    assert q["n_negative_slices"] == 3
    assert q["n_with_false_positive"] == 1
    assert q["pool_backed_fraction"] == pytest.approx(1 / 3)


def test_round_trip_through_disk(tmp_path):
    mined = {"lung_001": {7: {"fp_pixels": 3, "prob_mass": 3.5},
                          2: {"fp_pixels": 9, "prob_mass": 9.5}}}
    path = str(tmp_path / "mined.json")
    write_mined(mined, {"run": "r", "quality": pool_quality(mined)}, path)
    scores, meta = load_mined(path)
    assert scores["lung_001"] == {2: 9, 7: 3}     # magnitudes, for weighting
    from src.training.mine_negatives import load_mined_ranking
    assert load_mined_ranking(path)["lung_001"] == [2, 7]   # worst first
    assert meta["run"] == "r"


# -- the sampling mode -----------------------------------------------------

def test_mined_mode_requires_a_ranking(tmp_path):
    """Silently falling back to random would make the experiment measure nothing."""
    with pytest.raises(ValueError, match="mine_negatives"):
        LungSliceDataset(_volumes(tmp_path), _index(), ["lung_001"],
                         sampling="mined_negatives")


def test_mined_and_distance_draw_the_same_amount(tmp_path):
    """
    The two arms must draw the same number of slices, or a difference between
    them could just be a difference in how much data each one saw. Only the
    selection rule differs: distance truncates to a pool and draws flat inside
    it, mined weights the draw by measured error.
    """
    vols, index = _volumes(tmp_path), _index()
    scores = {"lung_001": {s: 50 for s in range(4, 18)}}
    hard = LungSliceDataset(vols, index, ["lung_001"], sampling="hard_negatives")
    mined = LungSliceDataset(vols, index, ["lung_001"],
                             sampling="mined_negatives", mined_scores=scores)
    assert len(hard) == len(mined)


def test_mined_mode_prefers_high_error_slices_over_nearby_ones(tmp_path):
    """
    The whole point: selection follows the model's errors, not proximity.

    Ten positives so the weighted portion is int(0.7 * 10) = 7 slices, enough
    for the preference to be visible. The high-error slices sit far from the
    tumour, where the distance rule would never look.
    """
    pos = tuple(range(30, 40))
    vols = _volumes(tmp_path, n_slices=60)
    index = _index(n_slices=60, positives=pos)
    far = [5, 6, 7, 8, 9, 10, 11, 12]
    ds = LungSliceDataset(vols, index, ["lung_001"],
                          sampling="mined_negatives",
                          mined_scores={"lung_001": {s: 500 for s in far}})
    drawn = {s for _, s in ds.samples if s not in pos}
    n_weighted = int(0.7 * len(pos))
    assert len(drawn & set(far)) >= n_weighted - 1, (
        f"weighted draw took only {len(drawn & set(far))} of its {n_weighted} "
        "slices from the high-error set")

    hard = LungSliceDataset(vols, index, ["lung_001"], sampling="hard_negatives")
    hard_drawn = {s for _, s in hard.samples if s not in pos}
    assert drawn != hard_drawn, "mined and distance produced the same draw"


def test_mined_mode_never_draws_a_positive_slice(tmp_path):
    """
    Scores mined against one preprocessing run must not smuggle a tumour slice
    into the negatives when applied to another, and nothing may be drawn twice.
    """
    vols, index = _volumes(tmp_path), _index()
    ds = LungSliceDataset(vols, index, ["lung_001"],
                          sampling="mined_negatives",
                          mined_scores={"lung_001": {18: 9, 19: 9, 20: 9,
                                                     6: 500, 7: 500}})
    counts = {}
    for _, s in ds.samples:
        counts[s] = counts.get(s, 0) + 1
    assert all(v == 1 for v in counts.values()), "a slice was drawn twice"


def test_mined_mode_redraws_across_epochs(tmp_path):
    """
    Like the other balanced modes, the negative half is redrawn each epoch, so
    training still sees variety rather than one fixed subset forever.
    """
    vols, index = _volumes(tmp_path, n_slices=60), _index(n_slices=60)
    ds = LungSliceDataset(vols, index, ["lung_001"], sampling="mined_negatives",
                          mined_scores={"lung_001":
                                        {s: 20 for s in range(4, 40)}})
    first = list(ds.samples)
    ds.set_epoch(1)
    assert list(ds.samples) != first


def test_negative_ratio_scales_the_hard_modes(tmp_path):
    """
    A second training stage weights the mix towards positives while keeping hard
    negatives, which needs `negative_ratio` to reach the hard modes. It used to
    be accepted and silently ignored there.
    """
    pos = tuple(range(30, 40))
    vols = _volumes(tmp_path, n_slices=60)
    index = _index(n_slices=60, positives=pos)
    scores = {"lung_001": {s: 100 for s in range(4, 20)}}

    for mode, kw in (("hard_negatives", {}),
                     ("mined_negatives", {"mined_scores": scores})):
        counts = {}
        for ratio in (1.0, 0.5):
            ds = LungSliceDataset(vols, index, ["lung_001"], sampling=mode,
                                  negative_ratio=ratio, **kw)
            counts[ratio] = sum(1 for _, s in ds.samples if s not in pos)
        assert counts[0.5] < counts[1.0], f"{mode} ignored negative_ratio"
        assert counts[0.5] == pytest.approx(counts[1.0] / 2, abs=1), mode


def test_default_ratio_reproduces_the_committed_draw(tmp_path):
    """
    Every hard_negatives run committed so far used the default ratio, so the
    default must still produce one negative per positive.
    """
    pos = tuple(range(30, 40))
    vols = _volumes(tmp_path, n_slices=60)
    index = _index(n_slices=60, positives=pos)
    ds = LungSliceDataset(vols, index, ["lung_001"], sampling="hard_negatives")
    negatives = sum(1 for _, s in ds.samples if s not in pos)
    assert negatives == len(pos)
