"""
Tests for the k-fold split machinery.

The failure these guard against is silent: a fold whose training set overlaps
its test set produces a better number, not an error, and the number looks
plausible. So the partition properties are asserted directly rather than
inferred from a run finishing.
"""

import json
import math
import os

import pytest

from src.training.cross_validation import (
    check_folds,
    fold_assignment,
    make_folds,
    pooled_stats,
    prepare_shared_preprocessed,
    write_fold_index,
)


def _categories(n_small=3, n_medium=26, n_large=34):
    """A category map shaped like this dataset: 3 / 26 / 34 across 63 patients."""
    cats = {}
    i = 0
    for cat, n in (("small", n_small), ("medium", n_medium), ("large", n_large)):
        for _ in range(n):
            cats[f"lung_{i:03d}"] = cat
            i += 1
    return cats


def test_folds_partition_every_patient_exactly_once():
    cats = _categories()
    folds = make_folds(sorted(cats), cats, k=5, seed=42)
    flat = sorted(c for f in folds for c in f)
    assert flat == sorted(cats), "folds must cover every patient exactly once"
    assert check_folds(folds, cats)["n_patients"] == 63


def test_fold_sizes_are_as_even_as_possible():
    cats = _categories()
    folds = make_folds(sorted(cats), cats, k=5, seed=42)
    sizes = sorted(len(f) for f in folds)
    assert sizes == [12, 12, 13, 13, 13], sizes


def test_folds_are_reproducible_across_calls():
    cats = _categories()
    a = make_folds(sorted(cats), cats, k=5, seed=42)
    b = make_folds(sorted(cats), cats, k=5, seed=42)
    assert a == b
    assert make_folds(sorted(cats), cats, k=5, seed=43) != a


def test_categories_spread_across_folds_rather_than_clumping():
    cats = _categories()
    folds = make_folds(sorted(cats), cats, k=5, seed=42)
    counts = check_folds(folds, cats)["category_counts"]
    for cat, total in (("medium", 26), ("large", 34)):
        per_fold = [c[cat] for c in counts]
        assert max(per_fold) - min(per_fold) <= 1, (cat, per_fold)
        assert sum(per_fold) == total


def test_the_three_small_patients_land_in_different_folds():
    """
    With 3 small patients and 5 folds, no fold should get two of them - that
    would leave only two folds able to say anything about the stratum.
    """
    cats = _categories()
    folds = make_folds(sorted(cats), cats, k=5, seed=42)
    per_fold = [sum(1 for c in f if cats[c] == "small") for f in folds]
    assert sorted(per_fold) == [0, 0, 1, 1, 1], per_fold


def test_train_val_test_are_disjoint_in_every_fold():
    cats = _categories()
    folds = make_folds(sorted(cats), cats, k=5, seed=42)
    for i in range(5):
        s = fold_assignment(folds, i)
        assert set(s["train"]) & set(s["test"]) == set()
        assert set(s["train"]) & set(s["val"]) == set()
        assert set(s["val"]) & set(s["test"]) == set()
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 63


def test_every_patient_is_tested_once_and_validated_once():
    cats = _categories()
    folds = make_folds(sorted(cats), cats, k=5, seed=42)
    tested, validated = [], []
    for i in range(5):
        s = fold_assignment(folds, i)
        tested += s["test"]
        validated += s["val"]
    assert sorted(tested) == sorted(cats)
    assert sorted(validated) == sorted(cats)


def test_fold_assignment_rejects_out_of_range():
    folds = make_folds([f"p{i}" for i in range(10)], {}, k=5, seed=0)
    with pytest.raises(IndexError):
        fold_assignment(folds, 5)


def test_make_folds_rejects_impossible_requests():
    with pytest.raises(ValueError):
        make_folds(["a", "b"], {}, k=5)
    with pytest.raises(ValueError):
        make_folds(["a", "b", "c"], {}, k=1)


def test_missing_categories_degrade_to_unstratified():
    ids = [f"p{i}" for i in range(20)]
    folds = make_folds(ids, {}, k=4, seed=1)
    assert sorted(c for f in folds for c in f) == sorted(ids)
    assert [len(f) for f in folds] == [5, 5, 5, 5]


# --------------------------------------------------------------------------
#  index.json rewriting
# --------------------------------------------------------------------------

def _base_index(case_ids):
    return {
        "splits": {"train": list(case_ids), "val": [], "test": []},
        "cases": {c: {"split": "train", "n_slices": 300,
                      "positive_slices": [1, 2], "body_slices": [0, 1, 2],
                      "tumor_voxels": 500}
                  for c in case_ids},
    }


def test_write_fold_index_sets_splits_and_preserves_slice_records(tmp_path):
    ids = [f"lung_{i:03d}" for i in range(10)]
    base = _base_index(ids)
    split = {"train": ids[:6], "val": ids[6:8], "test": ids[8:]}
    path = os.path.join(tmp_path, "preprocessed", "index.json")

    write_fold_index(base, split, path)

    with open(path) as f:
        written = json.load(f)
    assert written["splits"] == {k: sorted(v) for k, v in split.items()}
    for case_id in ids:
        assert written["cases"][case_id]["positive_slices"] == [1, 2]
        assert written["cases"][case_id]["n_slices"] == 300
    assert written["cases"][ids[9]]["split"] == "test"


def test_write_fold_index_does_not_mutate_the_base_index(tmp_path):
    """
    The base index is reused for all k folds, so writing fold 0 must not leave
    fold 1 reading fold 0's split.
    """
    ids = [f"lung_{i:03d}" for i in range(6)]
    base = _base_index(ids)
    write_fold_index(base, {"train": ids[:4], "val": ids[4:5], "test": ids[5:]},
                     os.path.join(tmp_path, "index.json"))
    assert base["splits"] == {"train": ids, "val": [], "test": []}
    assert base["cases"][ids[5]]["split"] == "train"


def test_write_fold_index_rejects_unknown_patients(tmp_path):
    base = _base_index(["lung_000", "lung_001"])
    with pytest.raises(KeyError):
        write_fold_index(base, {"train": ["lung_000"], "val": ["lung_001"],
                                "test": ["lung_999"]},
                         os.path.join(tmp_path, "index.json"))


def test_prepare_shared_preprocessed_exposes_volumes_and_writable_index(tmp_path):
    source = tmp_path / "src_pre"
    (source / "volumes").mkdir(parents=True)
    (source / "volumes" / "lung_000_img.npy").write_bytes(b"payload")
    (source / "index.json").write_text(json.dumps(_base_index(["lung_000"])))

    dest = str(tmp_path / "out" / "preprocessed")
    base = prepare_shared_preprocessed(str(source), dest)

    assert base["cases"]["lung_000"]["n_slices"] == 300
    assert os.path.exists(os.path.join(dest, "volumes", "lung_000_img.npy"))

    write_fold_index(base, {"train": ["lung_000"], "val": [], "test": []},
                     os.path.join(dest, "index.json"))
    with open(os.path.join(dest, "index.json")) as f:
        assert json.load(f)["splits"]["train"] == ["lung_000"]


def test_prepare_shared_preprocessed_is_idempotent(tmp_path):
    """Re-running a notebook cell must not fail on the directory already existing."""
    source = tmp_path / "src_pre"
    (source / "volumes").mkdir(parents=True)
    (source / "volumes" / "a_img.npy").write_bytes(b"x")
    (source / "index.json").write_text(json.dumps(_base_index(["a"])))

    dest = str(tmp_path / "out" / "preprocessed")
    prepare_shared_preprocessed(str(source), dest)
    prepare_shared_preprocessed(str(source), dest)
    assert os.path.exists(os.path.join(dest, "volumes", "a_img.npy"))


def test_prepare_shared_preprocessed_reports_a_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_shared_preprocessed(str(tmp_path / "nope"),
                                    str(tmp_path / "dest"))


# --------------------------------------------------------------------------
#  pooling
# --------------------------------------------------------------------------

def test_pooled_stats_matches_hand_computed_values():
    s = pooled_stats([0.2, 0.4, 0.6, 0.8])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.5)
    assert s["sd"] == pytest.approx(0.2581988897, abs=1e-9)
    assert s["se"] == pytest.approx(0.1290994449, abs=1e-9)
    assert s["lo"] == pytest.approx(0.5 - 1.96 * s["se"])


def test_pooled_stats_skips_nan_without_biasing_the_count():
    s = pooled_stats([0.5, float("nan"), 0.5, None])
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(0.5)


def test_pooled_stats_on_63_patients_tightens_the_interval_versus_10():
    """
    The reason for the whole exercise: holding the spread fixed, 63 patients
    give an interval sqrt(6.3) = 2.51x tighter than 10.

    Both samples are built to have the same standard deviation, so the
    comparison isolates the count. Drawing them at random would instead compare
    two noisy sd estimates, and a 10-point draw can easily look tighter than a
    63-point one by luck.
    """
    ten = [0.45 + 0.282 * v for v in (-1, 1) * 5]
    sixty_three = [0.45 + 0.282 * v for v in (-1, 1) * 31] + [0.45 - 0.282,
                                                              0.45 + 0.282,
                                                              0.45]
    wide, narrow = pooled_stats(ten), pooled_stats(sixty_three)
    assert wide["n"] == 10 and narrow["n"] == 65

    assert wide["se"] == pytest.approx(wide["sd"] / math.sqrt(10))
    assert narrow["se"] == pytest.approx(narrow["sd"] / math.sqrt(65))
    assert narrow["se"] < wide["se"] / 2


def test_pooled_stats_handles_degenerate_input():
    assert pooled_stats([])["n"] == 0
    single = pooled_stats([0.3])
    assert single["n"] == 1 and single["ci95"] == 0.0
