"""
K-fold cross-validation over all 63 patients.

Every comparison in this project so far has been scored on the same 10-patient
test set. Measured on it, the per-patient Dice has a standard deviation of
0.282, so the macro mean carries a standard error of 0.089 and the smallest
paired difference it can resolve at 95% is 0.138. Almost every effect measured
here is smaller than that, which is why so many of them came back "the interval
contains zero" - the test set, not the model, was the limit.

Rotating the test set over all 63 patients scores every patient exactly once and
cuts the standard error to roughly 0.282/sqrt(63) = 0.036, tightening the
smallest resolvable difference to about 0.055.

The folds share one preprocessed directory. That is safe here because
preprocessing does not depend on the split: intensities go through a fixed
[-1000, +400] HU window rather than statistics estimated on the training
patients, and each volume is resampled on its own geometry. Nothing is fitted
across patients, so a patient's .npy file is identical whichever fold it lands
in, and only index.json has to be rewritten between folds.

Usage:
    from src.training.cross_validation import make_folds, fold_assignment
"""

import json
import math
import os
import shutil

import numpy as np

CATEGORIES = ("small", "medium", "large")


def make_folds(case_ids, categories, k=5, seed=42):
    """
    Splits patients into k folds, balancing tumour size categories across them.

    Patients are dealt round-robin within each category so that the categories
    spread as evenly as k allows, then the deal is rotated per category so the
    same fold does not collect the leftover of every category.

    A caveat this dataset forces: only 3 of the 63 patients have a small tumour.
    With k=5 at most three folds can hold one, so the "small" stratum stays
    almost unmeasurable however the folds are drawn. Cross-validation still
    improves it from the single small patient the fixed split tests to all 3,
    but a per-stratum number on n=3 is a description, not a measurement.

    Args:
        case_ids (list): Every patient to distribute.
        categories (dict): {case_id: 'small' | 'medium' | 'large'}. Patients
            missing from it are treated as their own category, so an incomplete
            mapping degrades to unstratified rather than crashing.
        k (int): Number of folds.
        seed (int): Fixed so the folds are reproducible across sessions.

    Returns:
        list: k lists of case_ids, each sorted.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if len(case_ids) < k:
        raise ValueError(f"{len(case_ids)} patients cannot fill {k} folds")

    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]

    known = [c for c in CATEGORIES
             if any(categories.get(c_id) == c for c_id in case_ids)]
    unknown = sorted({categories.get(c_id, "unknown") for c_id in case_ids}
                     - set(CATEGORIES))

    offset = 0
    for cat in list(known) + unknown:
        members = sorted(c for c in case_ids if categories.get(c, "unknown") == cat)
        rng.shuffle(members)
        for i, case_id in enumerate(members):
            folds[(i + offset) % k].append(case_id)
        # Rotate the starting fold by however many patients spilled past a whole
        # round, so the next category begins where this one stopped.
        offset = (offset + len(members)) % k

    return [sorted(f) for f in folds]


def fold_assignment(folds, i):
    """
    Turns fold i into a train/val/test split.

    Test is fold i and validation is fold i+1, both rotating, so across the k
    folds every patient is tested exactly once and validated exactly once. That
    matters because validation is not a passive set here - it picks the
    checkpoint and the threshold - so a patient held fixed in validation would
    influence all k results while never being scored.

    Args:
        folds (list): Output of make_folds.
        i (int): Which fold is the test set.

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}, each sorted.
    """
    k = len(folds)
    if not 0 <= i < k:
        raise IndexError(f"fold {i} out of range for {k} folds")

    test = list(folds[i])
    val = list(folds[(i + 1) % k])
    train = [c for j, f in enumerate(folds) if j not in (i, (i + 1) % k)
             for c in f]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def write_fold_index(base_index, split, path):
    """
    Writes an index.json carrying this fold's split and nothing else changed.

    The per-slice records are copied through untouched - they describe the
    volume on disk, which is the same file in every fold.

    Args:
        base_index (dict): The preprocessed index.json as generated.
        split (dict): Output of fold_assignment.
        path (str): Destination index.json.
    """
    known = set(base_index["cases"])
    for mode, ids in split.items():
        missing = sorted(set(ids) - known)
        if missing:
            raise KeyError(f"{mode} names patients absent from the index: {missing}")

    index = dict(base_index)
    index["splits"] = {mode: sorted(ids) for mode, ids in split.items()}
    index["cases"] = {c: dict(rec) for c, rec in base_index["cases"].items()}
    for mode, ids in index["splits"].items():
        for case_id in ids:
            index["cases"][case_id]["split"] = mode

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f)
    return index


def prepare_shared_preprocessed(source_dir, dest_dir):
    """
    Builds a writable preprocessed directory whose volumes stay where they are.

    On Kaggle the preprocessed dataset is mounted read-only under /kaggle/input,
    so index.json cannot be rewritten in place. The volumes are the bulk and are
    only ever read, so they are linked rather than copied; index.json is what
    ends up writable.

    Falls back to copying the volumes where symlinks are unavailable (Windows
    without developer mode), which is correct but costs disk.

    Args:
        source_dir (str): Directory holding volumes/ and index.json.
        dest_dir (str): Writable directory to create.

    Returns:
        dict: The parsed source index.json, to pass to write_fold_index.
    """
    src_volumes = os.path.join(source_dir, "volumes")
    if not os.path.isdir(src_volumes):
        raise FileNotFoundError(f"No volumes/ under {source_dir}")

    if os.path.islink(dest_dir) or os.path.isfile(dest_dir):
        os.unlink(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    dst_volumes = os.path.join(dest_dir, "volumes")
    if os.path.islink(dst_volumes):
        os.unlink(dst_volumes)
    elif os.path.isdir(dst_volumes):
        shutil.rmtree(dst_volumes)

    try:
        os.symlink(os.path.abspath(src_volumes), dst_volumes)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copytree(src_volumes, dst_volumes)

    with open(os.path.join(source_dir, "index.json")) as f:
        return json.load(f)


def check_folds(folds, categories=None):
    """
    Verifies the folds partition the patients, and describes their makeup.

    A silent overlap between train and test would inflate every fold, so this is
    checked rather than assumed.

    Returns:
        dict: Counts per fold, plus 'sizes' and 'category_counts'.
    """
    flat = [c for f in folds for c in f]
    duplicates = sorted({c for c in flat if flat.count(c) > 1})
    if duplicates:
        raise ValueError(f"patients appear in more than one fold: {duplicates}")

    out = {"sizes": [len(f) for f in folds], "n_patients": len(flat)}
    if categories:
        out["category_counts"] = [
            {cat: sum(1 for c in f if categories.get(c) == cat)
             for cat in CATEGORIES}
            for f in folds]
    return out


def pooled_stats(values):
    """
    Mean, spread and interval for the pooled per-patient scores.

    Pooling is the point of the exercise: k fold means averaged together would
    have the same expectation but would hide that the folds differ in size, and
    would leave the interval resting on k=5 points instead of 63.

    Returns:
        dict: n, mean, sd, se, ci95 (half-width), lo, hi.
    """
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "se": float("nan"), "ci95": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    mean = sum(vals) / n
    if n == 1:
        return {"n": 1, "mean": mean, "sd": 0.0, "se": 0.0,
                "ci95": 0.0, "lo": mean, "hi": mean}
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci95": 1.96 * se,
            "lo": mean - 1.96 * se, "hi": mean + 1.96 * se}
