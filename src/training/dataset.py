"""
PyTorch Dataset over the preprocessed per-patient slice stacks.

Reads the .npy volumes written by src.preprocessing.preprocessing through a
memory map and exposes individual axial slices. Keeping whole volumes on disk,
rather than a pre-filtered pile of slice files, is what allows three decisions to
be made at runtime instead of being baked into the preprocessed data:

    sampling    which slices a split draws from (train only)
    augment     which spatial transforms are applied (train only)
    n_adjacent  2D (1 slice) versus 2.5D (3 or 5 consecutive slices)

Validation and test always iterate every slice of every volume, so the reported
metrics reflect the real ~9% positive-slice distribution rather than a balanced
subset.

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
#  DATASET
# ============================================================================

class LungSliceDataset(Dataset):
    """
    Axial slices drawn from preprocessed per-patient volumes.

    Args:
        volumes_dir (str): Directory holding {case_id}_img.npy / _lbl.npy.
        index (dict): Parsed output/preprocessed/index.json.
        case_ids (list): Patients belonging to this split.
        sampling (str): 'all', 'balanced', or 'hard_negatives'. Only meaningful
            for training; evaluation splits must use 'all'.
        augment (str): 'none', 'standard', or 'anatomic'.
        n_adjacent (int): 1 for 2D, 3 or 5 for 2.5D. Must be odd.
        seed (int): Base seed for sampling and augmentation.
    """

    def __init__(self, volumes_dir, index, case_ids, sampling="all",
                 augment="none", n_adjacent=1, seed=42):
        if n_adjacent % 2 != 1:
            raise ValueError(f"n_adjacent must be odd, got {n_adjacent}")
        if sampling not in ("all", "balanced", "hard_negatives"):
            raise ValueError(f"Unknown sampling mode: {sampling}")
        if augment not in AUGMENTATIONS:
            raise ValueError(f"Unknown augment mode: {augment}")

        self.volumes_dir = volumes_dir
        self.index = index
        self.case_ids = [c for c in case_ids if c in index["cases"]]
        self.sampling = sampling
        self.augment_fn = AUGMENTATIONS[augment]
        self.augment_name = augment
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
            else:
                body = set(info["body_slices"])
                negatives = sorted(body - set(positives))

                if self.sampling == "hard_negatives" and positives:
                    # Negatives immediately above and below the tumour look most
                    # like it and are where false positives actually appear.
                    lo, hi = min(positives), max(positives)
                    margin = max(10, (hi - lo))
                    near = [s for s in negatives if lo - margin <= s <= hi + margin]
                    far = [s for s in negatives if s not in set(near)]
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

        if self.augment_fn is not None:
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self.epoch * 100_003 + idx) % (2 ** 32))
            img, lbl = self.augment_fn(img, lbl, rng)

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
                      augment="anatomic", n_adjacent=1, seed=42, num_workers=2):
    """
    Builds the train, validation, and test DataLoaders.

    Sampling and augmentation apply to the training split only. Validation and
    test always use every slice with no augmentation, so that model selection and
    the final numbers are measured against the real class distribution.

    Args:
        preprocessed_dir (str): Path to output/preprocessed.
        batch_size (int): Mini-batch size.
        sampling (str): Training sampling mode.
        augment (str): Training augmentation mode.
        n_adjacent (int): 1 for 2D, 3 or 5 for 2.5D.
        seed (int): Base seed.
        num_workers (int): DataLoader worker processes.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    index = load_index(preprocessed_dir)
    volumes_dir = os.path.join(preprocessed_dir, "volumes")

    datasets = {
        "train": LungSliceDataset(
            volumes_dir, index, index["splits"]["train"],
            sampling=sampling, augment=augment, n_adjacent=n_adjacent, seed=seed),
        "val": LungSliceDataset(
            volumes_dir, index, index["splits"]["val"],
            sampling="all", augment="none", n_adjacent=n_adjacent, seed=seed),
        "test": LungSliceDataset(
            volumes_dir, index, index["splits"]["test"],
            sampling="all", augment="none", n_adjacent=n_adjacent, seed=seed),
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
