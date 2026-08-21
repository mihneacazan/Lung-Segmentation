"""
Tests for slice sampling and augmentation.

Both settings this file covers are ones that can be silently inert — accepted at
the CLI, reported in the logs, and never actually applied. Two failure modes in
particular:

    Sampling is the single largest effect in the whole benchmark, so it has to be
    a live argument rather than a decorative one. A `--sampling` flag that is
    accepted by the CLI, passed through the experiment function and then dropped
    by the dataloader would make the balanced and all-slices experiments the same
    run, and the comparison between them vacuous — while every log line kept
    reporting the mode that was requested.

    Augmentation has the same shape of problem. If only 'none' were handled,
    'standard' and 'anatomic' would be identical and the augmentation experiment
    would compare a policy against itself — again with nothing in the output to
    suggest anything was wrong.

Run:
    python -m pytest tests/ -v
"""

import json
import os

import numpy as np
import pytest

from src.training.dataset import (
    LungSliceDataset,
    _augment_standard,
    _augment_anatomic,
)


N_SLICES = 40
POSITIVE = list(range(15, 21))          # 6 positive slices
BODY = list(range(5, 36))               # 31 slices contain body


@pytest.fixture
def fake_dataset(tmp_path):
    """
    Writes a minimal preprocessed dataset: two patients per split, each with a
    small tumour on a handful of slices.

    Returns:
        tuple: (volumes_dir, index)
    """
    volumes = tmp_path / "volumes"
    volumes.mkdir()

    cases = {}
    splits = {"train": ["p_tr1", "p_tr2"], "val": ["p_va1"], "test": ["p_te1"]}

    for split, ids in splits.items():
        for case_id in ids:
            img = np.zeros((192, 192, N_SLICES), dtype=np.float16)
            lbl = np.zeros((192, 192, N_SLICES), dtype=np.uint8)
            img[:, :, BODY] = 0.4                    # body slices
            for s in POSITIVE:
                lbl[80:100, 60:80, s] = 1
                img[80:100, 60:80, s] = 0.9          # tumour is bright
            np.save(volumes / f"{case_id}_img.npy", img)
            np.save(volumes / f"{case_id}_lbl.npy", lbl)

            cases[case_id] = {
                "split": split,
                "n_slices": N_SLICES,
                "positive_slices": list(POSITIVE),
                "body_slices": list(BODY),
                "tumor_voxels": int(lbl.sum()),
            }

    index = {"splits": splits, "cases": cases}
    (tmp_path / "index.json").write_text(json.dumps(index))
    return str(volumes), index


# ============================================================================
#  SAMPLING
# ============================================================================

def test_balanced_sampling_is_one_to_one(fake_dataset):
    """Balanced sampling keeps every positive plus an equal number of negatives."""
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="balanced")
    n_patients = len(index["splits"]["train"])
    assert len(ds) == 2 * len(POSITIVE) * n_patients

    positives = sum(1 for _, s in ds.samples if s in POSITIVE)
    assert positives == len(POSITIVE) * n_patients
    assert positives / len(ds) == pytest.approx(0.5)


def test_all_sampling_keeps_every_slice(fake_dataset):
    """'all' means the whole volume, including pure-air slices."""
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"], sampling="all")
    assert len(ds) == N_SLICES * len(index["splits"]["train"])


def test_balanced_and_all_are_actually_different(fake_dataset):
    """
    The regression guard for the dead --sampling flag: these two configurations
    must not produce the same training set.
    """
    volumes, index = fake_dataset
    balanced = LungSliceDataset(volumes, index, index["splits"]["train"],
                                sampling="balanced")
    every = LungSliceDataset(volumes, index, index["splits"]["train"],
                             sampling="all")

    assert len(balanced) != len(every)
    assert set(balanced.samples) != set(every.samples)
    assert set(balanced.samples).issubset(set(every.samples))


def test_hard_negatives_sit_closer_to_the_tumour(fake_dataset):
    """
    Hard-negative sampling should concentrate negatives near the tumour in Z,
    where false positives actually occur, rather than spreading them uniformly.
    """
    volumes, index = fake_dataset
    hard = LungSliceDataset(volumes, index, index["splits"]["train"],
                            sampling="hard_negatives", seed=7)
    balanced = LungSliceDataset(volumes, index, index["splits"]["train"],
                                sampling="balanced", seed=7)

    centre = (min(POSITIVE) + max(POSITIVE)) / 2

    def mean_distance(ds):
        negs = [s for _, s in ds.samples if s not in POSITIVE]
        return float(np.mean([abs(s - centre) for s in negs]))

    assert mean_distance(hard) <= mean_distance(balanced)


def test_negatives_are_redrawn_each_epoch(fake_dataset):
    """
    Balanced sampling must resample negatives per epoch, otherwise the model only
    ever sees one fixed slice of the negative anatomy.
    """
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="balanced", seed=42)
    epoch0 = set(ds.samples)
    ds.set_epoch(1)
    epoch1 = set(ds.samples)

    assert epoch0 != epoch1, "negatives are frozen across epochs"
    # Positives, however, must always be present in full.
    for epoch_samples in (epoch0, epoch1):
        positives = {(c, s) for c, s in epoch_samples if s in POSITIVE}
        assert len(positives) == len(POSITIVE) * len(index["splits"]["train"])


def test_sampling_is_reproducible_for_a_given_seed(fake_dataset):
    volumes, index = fake_dataset
    a = LungSliceDataset(volumes, index, index["splits"]["train"],
                         sampling="balanced", seed=42)
    b = LungSliceDataset(volumes, index, index["splits"]["train"],
                         sampling="balanced", seed=42)
    assert a.samples == b.samples


def test_validation_keeps_the_real_class_distribution(fake_dataset):
    """
    Validation must never be balanced. Balancing it inflates the positive rate
    from the real ~9% to ~33% and makes the reported Dice unrepresentative.
    """
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["val"], sampling="all")
    positive_rate = sum(1 for _, s in ds.samples if s in POSITIVE) / len(ds)
    assert positive_rate == pytest.approx(len(POSITIVE) / N_SLICES)
    assert positive_rate < 0.20


# ============================================================================
#  2.5D
# ============================================================================

@pytest.mark.parametrize("n_adjacent", [1, 3, 5])
def test_channel_count_matches_n_adjacent(fake_dataset, n_adjacent):
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["val"],
                          sampling="all", n_adjacent=n_adjacent)
    sample = ds[20]
    assert sample["image"].shape == (n_adjacent, 192, 192)
    assert sample["label"].shape == (1, 192, 192)


def test_25d_replicates_edges_instead_of_zero_padding(fake_dataset):
    """
    At the first and last slice the missing neighbours are edge-replicated. Zero
    padding would be wrong here: zero is not "no data" in this normalized space,
    it is -1000 HU, i.e. air, so the network would be told there is air beyond
    the ends of the scan.
    """
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["val"],
                          sampling="all", n_adjacent=3)

    first = ds[0]["image"].numpy()
    assert np.array_equal(first[0], first[1]), "first slice should replicate below"

    last = ds[len(ds) - 1]["image"].numpy()
    assert np.array_equal(last[1], last[2]), "last slice should replicate above"


def test_25d_neighbours_come_from_the_same_patient(fake_dataset):
    """A 2.5D stack must never mix slices from two patients."""
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="all", n_adjacent=5)
    for idx in (0, N_SLICES - 1, N_SLICES, len(ds) - 1):
        case_id, slice_idx = ds.samples[idx]
        img_vol = np.load(f"{volumes}/{case_id}_img.npy", mmap_mode="r")
        stack = ds[idx]["image"].numpy()
        for offset, channel in zip(range(-2, 3), stack):
            expected = int(np.clip(slice_idx + offset, 0, img_vol.shape[2] - 1))
            assert np.allclose(channel, np.asarray(img_vol[:, :, expected],
                                                   dtype=np.float32))


# ============================================================================
#  AUGMENTATION
# ============================================================================

def make_pair():
    """An image whose bright region coincides exactly with the label."""
    img = np.zeros((1, 64, 64), dtype=np.float32)
    lbl = np.zeros((1, 64, 64), dtype=np.float32)
    img[0, 10:24, 40:54] = 1.0
    lbl[0, 10:24, 40:54] = 1.0
    return img, lbl


def test_standard_and_anatomic_are_different_policies():
    """
    Regression guard for the collapsed --augment flag. Applied to the same input
    with the same seed, the two policies must produce different output.
    """
    img, lbl = make_pair()
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    std_img, _ = _augment_standard(img.copy(), lbl.copy(), rng_a)
    ana_img, _ = _augment_anatomic(img.copy(), lbl.copy(), rng_b)
    assert not np.allclose(std_img, ana_img)


def test_anatomic_never_flips_or_rotates_by_90_degrees():
    """
    A vertical flip or a quarter turn of an axial chest CT is anatomically
    impossible. Over many draws, the tumour must stay in the quadrant it started
    in; a flip would move it to the opposite one.
    """
    img, lbl = make_pair()
    start_row, start_col = 17, 47      # centre of the blob

    for seed in range(40):
        rng = np.random.default_rng(seed)
        _, out_lbl = _augment_anatomic(img.copy(), lbl.copy(), rng)
        coords = np.argwhere(out_lbl[0] > 0.5)
        if len(coords) == 0:
            continue
        row, col = coords.mean(axis=0)
        # Rotation up to 15 deg plus 8% shift keeps the centroid nearby; a flip
        # would move it roughly 30 voxels across the image.
        assert abs(row - start_row) < 16, f"seed {seed}: row moved to {row:.1f}"
        assert abs(col - start_col) < 16, f"seed {seed}: col moved to {col:.1f}"


def test_standard_does_flip():
    """
    Confirms the contrast is real: over many draws the naive policy does move the
    tumour to the opposite side, which is exactly the anatomically invalid
    behaviour being measured against.
    """
    img, lbl = make_pair()
    moved_far = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        _, out_lbl = _augment_standard(img.copy(), lbl.copy(), rng)
        coords = np.argwhere(out_lbl[0] > 0.5)
        row, col = coords.mean(axis=0)
        if abs(row - 17) > 16 or abs(col - 47) > 16:
            moved_far += 1
    assert moved_far > 10, "standard augmentation is not flipping at all"


def test_anatomic_keeps_image_and_mask_aligned():
    """
    The geometric transform must be applied identically to both. If they drifted
    apart, the network would be trained on mislabelled voxels.
    """
    img, lbl = make_pair()
    for seed in range(25):
        rng = np.random.default_rng(seed)
        out_img, out_lbl = _augment_anatomic(img.copy(), lbl.copy(), rng)
        mask = out_lbl[0] > 0.5
        if mask.sum() < 20:
            continue
        # Intensity augmentation can dim the blob, but it must remain far
        # brighter inside the mask than outside.
        inside = out_img[0][mask].mean()
        outside = out_img[0][~mask].mean()
        assert inside > outside + 0.3, (
            f"seed {seed}: image and mask are misaligned "
            f"(inside {inside:.3f} vs outside {outside:.3f})")


def test_augmented_intensities_stay_in_range():
    """Gamma and noise must not push the image outside [0, 1]."""
    img, lbl = make_pair()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        out_img, _ = _augment_anatomic(img.copy(), lbl.copy(), rng)
        assert out_img.min() >= 0.0 and out_img.max() <= 1.0


def test_rejects_invalid_configuration(fake_dataset):
    volumes, index = fake_dataset
    with pytest.raises(ValueError):
        LungSliceDataset(volumes, index, index["splits"]["val"], n_adjacent=2)
    with pytest.raises(ValueError):
        LungSliceDataset(volumes, index, index["splits"]["val"], sampling="nope")
    with pytest.raises(ValueError):
        LungSliceDataset(volumes, index, index["splits"]["val"], augment="nope")
    with pytest.raises(ValueError):
        LungSliceDataset(volumes, index, index["splits"]["val"], crop="nope")


# ============================================================================
#  TUMOUR-CENTRED CROPPING
#
#  The crop is a training-time device. Its whole justification rests on the
#  model still being asked to search a full slice afterwards, so the tests that
#  matter most here are the ones pinning down what the crop must NOT do:
#  reach evaluation, or teach the network that lesions live in the middle.
# ============================================================================

def test_crop_produces_the_requested_window(fake_dataset):
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="all", crop="tumor", crop_size=96, seed=1)
    for i in range(0, len(ds), 7):
        sample = ds[i]
        assert sample["image"].shape[-2:] == (96, 96)
        assert sample["label"].shape[-2:] == (96, 96)


def test_crop_keeps_the_tumour_inside_the_window(fake_dataset):
    """A window that misses the lesion would train the model on pure background."""
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="all", crop="tumor", crop_size=96, seed=3)

    positives = [i for i, (_, s) in enumerate(ds.samples) if s in POSITIVE]
    assert positives, "fixture should contain positive slices"

    for i in positives:
        assert ds[i]["label"].sum() > 0, (
            f"sample {i} is a positive slice but its crop contains no tumour")


def test_crop_position_varies_across_samples(fake_dataset):
    """
    The jitter has to actually move the window.

    Without it every training example carries the tumour at the exact centre,
    and the network can satisfy the loss by predicting a central blob — which
    then fails on the full slices used at evaluation.
    """
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="all", crop="tumor", crop_size=96, seed=5)

    positives = [i for i, (_, s) in enumerate(ds.samples) if s in POSITIVE]
    centroids = set()
    for i in positives:
        lbl = ds[i]["label"].numpy()[0]
        ys, xs = np.nonzero(lbl)
        centroids.add((round(float(ys.mean()), 1), round(float(xs.mean()), 1)))

    assert len(centroids) > 1, (
        "the tumour sits at an identical position in every crop — jitter is inert")


def test_crop_moves_rather_than_pads_at_the_edges():
    """
    A lesion near a border must shift the window inward, not pad it.

    Padding would invent background that does not exist in the patient, and the
    fabricated intensity would be indistinguishable from air.
    """
    from src.training.dataset import _crop_tumor_centered

    img = np.full((1, 192, 192), 0.4, dtype=np.float32)
    lbl = np.zeros((1, 192, 192), dtype=np.float32)
    lbl[0, 0:10, 0:10] = 1.0                      # tumour in the top-left corner

    for seed in range(8):
        rng = np.random.default_rng(seed)
        out_img, out_lbl = _crop_tumor_centered(img, lbl, rng, 96)
        assert out_img.shape == (1, 96, 96)
        assert out_lbl.sum() > 0
        # Every value comes from the source slice, so nothing was fabricated.
        assert np.all(out_img == 0.4)


def test_crop_keeps_all_25d_channels_aligned(fake_dataset):
    """The window must be identical across the Z-neighbour channels."""
    volumes, index = fake_dataset
    ds = LungSliceDataset(volumes, index, index["splits"]["train"],
                          sampling="all", crop="tumor", crop_size=96,
                          n_adjacent=3, seed=7)

    sample = ds[len(ds) // 2]
    assert sample["image"].shape[0] == 3
    assert sample["image"].shape[-2:] == (96, 96)


def test_crop_is_reproducible_for_a_given_seed(fake_dataset):
    volumes, index = fake_dataset
    kwargs = dict(sampling="all", crop="tumor", crop_size=96, seed=11)
    a = LungSliceDataset(volumes, index, index["splits"]["train"], **kwargs)
    b = LungSliceDataset(volumes, index, index["splits"]["train"], **kwargs)
    for i in (0, len(a) // 3, len(a) - 1):
        assert np.array_equal(a[i]["image"].numpy(), b[i]["image"].numpy())


def test_crop_never_reaches_validation_or_test(fake_dataset):
    """
    Evaluation must see the full slice.

    Centring a window on the tumour requires knowing where the tumour is, which
    at inference is precisely the unknown. A cropped validation set would report
    a number no deployed model could reproduce.
    """
    import json as _json
    from src.training.dataset import build_dataloaders

    volumes, index = fake_dataset
    preprocessed = os.path.dirname(volumes)
    with open(os.path.join(preprocessed, "index.json"), "w") as f:
        _json.dump(index, f)

    _, datasets, _ = build_dataloaders(
        preprocessed, batch_size=2, sampling="all", augment="none",
        crop="tumor", crop_size=96, seed=13, num_workers=0)

    assert datasets["train"].crop == "tumor"
    assert datasets["val"].crop == "none"
    assert datasets["test"].crop == "none"
    assert datasets["val"][0]["image"].shape[-2:] == (192, 192)
    assert datasets["test"][0]["image"].shape[-2:] == (192, 192)


def test_crop_size_larger_than_the_slice_is_rejected():
    from src.training.dataset import _crop_tumor_centered

    img = np.zeros((1, 192, 192), dtype=np.float32)
    lbl = np.zeros((1, 192, 192), dtype=np.float32)
    with pytest.raises(ValueError):
        _crop_tumor_centered(img, lbl, np.random.default_rng(0), 256)
