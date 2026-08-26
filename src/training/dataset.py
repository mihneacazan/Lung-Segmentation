"""
PyTorch Dataset over the preprocessed per-patient slice stacks.

Reads the .npy volumes written by src.preprocessing.preprocessing through a
memory map and exposes individual axial slices. Keeping whole volumes on disk,
rather than a pre-filtered pile of slice files, is what allows four decisions to
be made at runtime instead of being baked into the preprocessed data:

    sampling    which slices a split draws from (train only)
    augment     which spatial transforms are applied (train only)
    crop        full slice versus a tumour-centred window (train only)
    n_adjacent  2D (1 slice) versus 2.5D (3 or 5 consecutive slices)

Validation and test always iterate every slice of every volume at full size, so
the reported metrics reflect the real ~9% positive-slice distribution rather than
a balanced subset, and are measured on the same field of view a deployed model
would see.

Usage:
    from src.training.dataset import build_dataloaders
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import affine_transform


# ============================================================================
#  AUGMENTATION
# ============================================================================

def _augment_standard(img, lbl, rng):
    """
    The naive computer-vision augmentation recipe: horizontal flip, vertical
    flip, and 90-degree rotations.

    These are anatomically invalid for axial chest CT. A vertical flip puts the
    spine in front of the sternum; a 90-degree rotation lays the patient on
    their side. No such image can be produced by a CT scanner, so the network
    spends capacity learning to be invariant to poses that never occur at test
    time. This mode exists to quantify that cost against `anatomic`.

    Args:
        img (np.ndarray): (C, H, W) image slices.
        lbl (np.ndarray): (1, H, W) mask.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (augmented_img, augmented_lbl)
    """
    if rng.random() > 0.5:
        img = np.flip(img, axis=2).copy()
        lbl = np.flip(lbl, axis=2).copy()
    if rng.random() > 0.5:
        img = np.flip(img, axis=1).copy()
        lbl = np.flip(lbl, axis=1).copy()
    if rng.random() > 0.5:
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k, axes=(1, 2)).copy()
        lbl = np.rot90(lbl, k, axes=(1, 2)).copy()
    return img, lbl


def _augment_anatomic(img, lbl, rng):
    """
    Augmentation restricted to transforms that yield an anatomically plausible
    chest CT.

    Included, with the real-world variation each one stands in for:
        rotation +/- 15 deg     patient roll on the table
        translation +/- 8%      patient not centred in the bore
        scale +/- 10%           body habitus and field-of-view differences
        gamma 0.8 - 1.25        reconstruction kernel and dose differences
        gaussian noise          low-dose acquisition

    Deliberately excluded:
        vertical flip, 90-degree rotation   physically impossible poses
        horizontal flip                     mirrors the anatomy, producing
                                            dextrocardia, which occurs in
                                            roughly 1 in 10,000 people and is
                                            absent from this dataset

    The geometric part is applied as a single composed affine so the image and
    mask are resampled once, keeping them exactly aligned.

    Args:
        img (np.ndarray): (C, H, W) image slices.
        lbl (np.ndarray): (1, H, W) mask.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (augmented_img, augmented_lbl)
    """
    h, w = img.shape[1], img.shape[2]

    angle = np.deg2rad(rng.uniform(-15.0, 15.0))
    scale = rng.uniform(0.90, 1.10)
    max_shift = 0.08
    shift_y = rng.uniform(-max_shift, max_shift) * h
    shift_x = rng.uniform(-max_shift, max_shift) * w

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64) / scale

    # Rotate and scale about the image centre rather than the corner.
    center = np.array([h / 2.0, w / 2.0])
    offset = center - matrix @ center + np.array([shift_y, shift_x])

    out_img = np.empty_like(img)
    for c in range(img.shape[0]):
        out_img[c] = affine_transform(
            img[c], matrix, offset=offset, order=1, mode="constant", cval=0.0)

    out_lbl = np.empty_like(lbl)
    for c in range(lbl.shape[0]):
        warped = affine_transform(
            lbl[c].astype(np.float32), matrix, offset=offset,
            order=1, mode="constant", cval=0.0)
        out_lbl[c] = (warped > 0.5).astype(lbl.dtype)

    # Intensity augmentation on the image only.
    if rng.random() > 0.5:
        gamma = rng.uniform(0.8, 1.25)
        out_img = np.clip(out_img, 0.0, 1.0) ** gamma
    if rng.random() > 0.5:
        out_img = out_img + rng.normal(0.0, 0.02, size=out_img.shape).astype(np.float32)

    return np.clip(out_img, 0.0, 1.0).astype(np.float32), out_lbl


AUGMENTATIONS = {
    "none": None,
    "standard": _augment_standard,
    "anatomic": _augment_anatomic,
}


# ============================================================================
#  CROPPING
# ============================================================================

CROP_MODES = ("none", "tumor")

# Which slices a split draws from. The first three are training strategies that
# trade class balance against how much negative anatomy the model ever sees.
# 'positives' is different in kind: it removes the detection problem altogether,
# leaving only delineation. A model trained on it has never been shown an empty
# slice and so has no reason to leave one empty, which makes it useful for
# measuring one half of the task in isolation and useless as a deployable model.
SAMPLING_MODES = ("all", "balanced", "hard_negatives", "positives")

# 96 x 96 out of the preprocessed 192 x 192. The largest tumour bounding box in
# the training split measures 38 x 57 px at 1 mm spacing, so this holds every
# lesion in the dataset with surrounding context to spare, and it stays divisible
# by 16 as the four downsampling stages of the U-Net require.
DEFAULT_CROP_SIZE = 96


def _crop_tumor_centered(img, lbl, rng, size):
    """
    Cuts a `size` x `size` window around the tumour, or at a random position when
    the slice has none.

    Applied to training slices only. At inference the tumour location is exactly
    what is being predicted, so centring a window on it would feed the ground
    truth to the model; validation and test therefore always run on the full
    slice. That asymmetry is the point of the experiment: the question is whether
    concentrating the training signal on a tumour-sized window produces a better
    model when it is later asked to search a whole slice.

    The centre is jittered rather than exact. Cropping precisely on the centroid
    would put the tumour at the middle of the window in every single training
    sample, from which a network can learn the position instead of the appearance
    — a shortcut that collapses the moment it is evaluated on a full slice where
    the lesion is off-centre. The jitter spans a quarter of the window, so the
    tumour lands anywhere in the middle half.

    Slices with no tumour are cropped at a uniformly random position instead of
    being dropped. Discarding them would silently turn this into a sampling
    experiment as well, and sampling is a separate variable with its own runs.

    Args:
        img (np.ndarray): (C, H, W) image slices.
        lbl (np.ndarray): (1, H, W) mask.
        rng (np.random.Generator): Seeded random generator.
        size (int): Side length of the square window.

    Returns:
        tuple: (cropped_img, cropped_lbl), each `size` x `size` in the last two
        axes.
    """
    h, w = img.shape[1], img.shape[2]
    if size > h or size > w:
        raise ValueError(
            f"crop_size {size} exceeds the slice, which is {h} x {w}")
    if size % 16 or size < 32:
        # Four downsampling stages halve the window four times, and the
        # normalization layers need at least a 2x2 map at the bottleneck. Below
        # 32 the failure surfaces as an opaque shape error partway through the
        # first epoch instead of here.
        raise ValueError(
            f"crop_size must be a multiple of 16 and at least 32, got {size}")

    half = size // 2
    ys, xs = np.nonzero(lbl[0])

    if len(ys):
        jitter = size // 4
        center_y = int(ys.mean()) + int(rng.integers(-jitter, jitter + 1))
        center_x = int(xs.mean()) + int(rng.integers(-jitter, jitter + 1))
    else:
        center_y = int(rng.integers(half, h - half + 1))
        center_x = int(rng.integers(half, w - half + 1))

    # Clamping the origin keeps the window inside the slice, so a tumour near an
    # edge shifts the window rather than padding it with fabricated background.
    top = int(np.clip(center_y - half, 0, h - size))
    left = int(np.clip(center_x - half, 0, w - size))

    return (img[:, top:top + size, left:left + size],
            lbl[:, top:top + size, left:left + size])


# ============================================================================
#  DATASET
# ============================================================================

class LungSliceDataset(Dataset):
    """
    Axial slices drawn from preprocessed per-patient volumes.

    Args:
        volumes_dir (str): Directory holding {case_id}_img.npy / _lbl.npy.
        index (dict): Parsed output/preprocessed/index.json.
        case_ids (list): Patients belonging to this split.
        sampling (str): 'all', 'balanced', 'hard_negatives', or 'positives'.
            Evaluation splits normally use 'all'; 'positives' is the one
            exception, and only for the deliberately restricted protocol that
            asks how well a model delineates given that a lesion is present.
        augment (str): 'none', 'standard', or 'anatomic'.
        crop (str): 'none' for the full slice, or 'tumor' for a tumour-centred
            window. Training only; evaluation splits must use 'none'.
        crop_size (int): Side length of that window.
        n_adjacent (int): 1 for 2D, 3 or 5 for 2.5D. Must be odd.
        seed (int): Base seed for sampling, augmentation and cropping.
    """

    def __init__(self, volumes_dir, index, case_ids, sampling="all",
                 augment="none", crop="none", crop_size=DEFAULT_CROP_SIZE,
                 n_adjacent=1, seed=42):
        if n_adjacent % 2 != 1:
            raise ValueError(f"n_adjacent must be odd, got {n_adjacent}")
        if sampling not in SAMPLING_MODES:
            raise ValueError(f"Unknown sampling mode: {sampling}")
        if augment not in AUGMENTATIONS:
            raise ValueError(f"Unknown augment mode: {augment}")
        if crop not in CROP_MODES:
            raise ValueError(f"Unknown crop mode: {crop}")

        self.volumes_dir = volumes_dir
        self.index = index
        self.case_ids = [c for c in case_ids if c in index["cases"]]
        self.sampling = sampling
        self.augment_fn = AUGMENTATIONS[augment]
        self.augment_name = augment
        self.crop = crop
        self.crop_size = crop_size
        self.n_adjacent = n_adjacent
        self.half_window = n_adjacent // 2
        self.seed = seed

        # Volumes are memory-mapped lazily and cached per worker process.
        self._cache = {}

        self.samples = []
        self.set_epoch(0)

    # -- sampling ----------------------------------------------------------

    def set_epoch(self, epoch: int):
        """
        Rebuilds the sample list for a new epoch.

        For 'balanced' and 'hard_negatives' the negative slices are redrawn every
        epoch with a different seed, so the model still sees the full variety of
        negative anatomy across training while each individual epoch stays
        class-balanced. A fixed negative subset chosen once at preprocessing time
        would throw most of that anatomy away permanently.
        """
        self.epoch = epoch
        rng = np.random.default_rng(self.seed * 100_003 + epoch)
        samples = []

        for case_id in self.case_ids:
            info = self.index["cases"][case_id]
            n_slices = info["n_slices"]
            positives = info["positive_slices"]

            if self.sampling == "all":
                chosen = range(n_slices)
            elif self.sampling == "positives":
                # Fixed across epochs: there is nothing to redraw, since every
                # slice that qualifies is already in. A patient whose tumour was
                # lost in preprocessing contributes nothing and simply drops out
                # of the split rather than being silently counted as empty.
                chosen = list(positives)
            else:
                body = set(info["body_slices"])
                negatives = sorted(body - set(positives))

                if self.sampling == "hard_negatives" and positives:
                    # Negatives immediately above and below the tumour look most
                    # like it and are where false positives actually appear.
                    lo, hi = min(positives), max(positives)
                    margin = max(10, (hi - lo))
                    near = [s for s in negatives if lo - margin <= s <= hi + margin]
                    near_set = set(near)
                    far = [s for s in negatives if s not in near_set]
                    n_near = min(len(near), int(0.7 * len(positives)))
                    n_far = min(len(far), len(positives) - n_near)
                    picked = list(rng.choice(near, n_near, replace=False)) if n_near else []
                    picked += list(rng.choice(far, n_far, replace=False)) if n_far else []
                else:
                    n_keep = min(len(negatives), len(positives))
                    picked = list(rng.choice(negatives, n_keep, replace=False)) if n_keep else []

                chosen = sorted(positives + [int(s) for s in picked])

            samples.extend((case_id, int(s)) for s in chosen)

        self.samples = samples

    # -- volume access -----------------------------------------------------

    def _get_volume(self, case_id):
        """Returns memory-mapped (image, label) stacks for a patient."""
        if case_id not in self._cache:
            img = np.load(os.path.join(self.volumes_dir, f"{case_id}_img.npy"),
                          mmap_mode="r")
            lbl = np.load(os.path.join(self.volumes_dir, f"{case_id}_lbl.npy"),
                          mmap_mode="r")
            self._cache[case_id] = (img, lbl)
        return self._cache[case_id]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        case_id, slice_idx = self.samples[idx]
        img_vol, lbl_vol = self._get_volume(case_id)
        n_slices = img_vol.shape[2]

        # 2.5D neighbour selection. Out-of-range neighbours replicate the edge
        # slice instead of being zero-filled: zero is a real intensity in this
        # normalized space (it means -1000 HU, i.e. air), so padding with zeros
        # would tell the network there is air above the apex of the lung.
        offsets = range(-self.half_window, self.half_window + 1)
        neighbour_idx = [int(np.clip(slice_idx + o, 0, n_slices - 1)) for o in offsets]

        img = np.stack([np.asarray(img_vol[:, :, s], dtype=np.float32)
                        for s in neighbour_idx], axis=0)
        lbl = np.asarray(lbl_vol[:, :, slice_idx], dtype=np.float32)[None, ...]

        if self.augment_fn is not None or self.crop == "tumor":
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self.epoch * 100_003 + idx) % (2 ** 32))

            # Augmentation runs on the full slice and the crop follows it, so the
            # window is centred on where the tumour ended up after a rotation or
            # shift rather than where it started.
            if self.augment_fn is not None:
                img, lbl = self.augment_fn(img, lbl, rng)
            if self.crop == "tumor":
                img, lbl = _crop_tumor_centered(img, lbl, rng, self.crop_size)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
            "label": torch.from_numpy(np.ascontiguousarray(lbl, dtype=np.float32)),
            "case_id": case_id,
            "slice_idx": slice_idx,
        }


# ============================================================================
#  DATALOADER FACTORY
# ============================================================================

def load_index(preprocessed_dir):
    """Loads output/preprocessed/index.json, with a clear error if it is absent."""
    index_path = os.path.join(preprocessed_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing {index_path}\n"
            f"Run 'python -m src.preprocessing.preprocessing' first."
        )
    with open(index_path, "r") as f:
        return json.load(f)


def build_dataloaders(preprocessed_dir, batch_size=16, sampling="balanced",
                      augment="anatomic", crop="none",
                      crop_size=DEFAULT_CROP_SIZE, n_adjacent=1, seed=42,
                      num_workers=2, eval_sampling="all"):
    """
    Builds the train, validation, and test DataLoaders.

    Augmentation and cropping apply to the training split only. Validation and
    test are unaugmented and at full size, so the final numbers are measured
    against the real field of view.

    They also default to every slice, which is the honest setting: model
    selection and the reported score then face the real class distribution,
    where roughly 91% of slices contain no tumour and a false positive on any of
    them costs something. `eval_sampling` exists to relax that deliberately, for
    the restricted protocol that scores delineation alone. Anything other than
    'all' produces a number that is not comparable with the rest of this
    project's results and has to be labelled as such wherever it is quoted.

    Args:
        preprocessed_dir (str): Path to output/preprocessed.
        batch_size (int): Mini-batch size.
        sampling (str): Training sampling mode.
        augment (str): Training augmentation mode.
        crop (str): Training crop mode, 'none' or 'tumor'.
        crop_size (int): Side length of the tumour-centred window.
        n_adjacent (int): 1 for 2D, 3 or 5 for 2.5D.
        seed (int): Base seed.
        num_workers (int): DataLoader worker processes.
        eval_sampling (str): Slice selection for validation and test.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    index = load_index(preprocessed_dir)
    volumes_dir = os.path.join(preprocessed_dir, "volumes")

    datasets = {
        "train": LungSliceDataset(
            volumes_dir, index, index["splits"]["train"],
            sampling=sampling, augment=augment, crop=crop, crop_size=crop_size,
            n_adjacent=n_adjacent, seed=seed),
        "val": LungSliceDataset(
            volumes_dir, index, index["splits"]["val"],
            sampling=eval_sampling, augment="none", crop="none",
            n_adjacent=n_adjacent, seed=seed),
        "test": LungSliceDataset(
            volumes_dir, index, index["splits"]["test"],
            sampling=eval_sampling, augment="none", crop="none",
            n_adjacent=n_adjacent, seed=seed),
    }

    loaders = {
        # persistent_workers must stay False here. Workers receive a pickled copy
        # of the dataset, so a persistent pool would keep serving the epoch-0
        # sample list forever and set_epoch() would silently do nothing. Respawning
        # workers each epoch costs about a second and keeps the resampling real.
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=False, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=num_workers > 0),
    }
    return loaders, datasets, index


def build_eval_loader(preprocessed_dir, split, sampling="all", batch_size=16,
                      n_adjacent=1, seed=42, num_workers=2):
    """
    Builds one unaugmented evaluation loader, independently of `build_dataloaders`.

    This exists so a single trained model can be scored under two slice
    protocols in one run. Calling `build_dataloaders` twice would work but would
    also rebuild the training split, which for the sampling modes that redraw
    negatives means paying for a sample list nobody reads.

    Args:
        preprocessed_dir (str): Path to output/preprocessed.
        split (str): 'train', 'val', or 'test'.
        sampling (str): Slice selection, see `SAMPLING_MODES`.
        batch_size (int): Mini-batch size.
        n_adjacent (int): 1 for 2D, 3 or 5 for 2.5D.
        seed (int): Base seed.
        num_workers (int): DataLoader worker processes.

    Returns:
        tuple: (DataLoader, LungSliceDataset)
    """
    index = load_index(preprocessed_dir)
    dataset = LungSliceDataset(
        os.path.join(preprocessed_dir, "volumes"), index, index["splits"][split],
        sampling=sampling, augment="none", crop="none",
        n_adjacent=n_adjacent, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    return loader, dataset
