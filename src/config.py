"""
Project paths and environment detection.

Holds only what the pipeline actually reads at runtime: the location of the raw
NIfTI dataset, the location of everything the pipeline writes, and a resolver
that copes with the several directory layouts the Decathlon archive ships in.

Hyperparameters are deliberately not kept here. Every value that changes between
experiments (architecture, loss, sampling, augmentation, learning rate, epochs)
is a CLI argument of `src.training.train`, so that an experiment's configuration
is recorded in its own `config.json` next to its checkpoint rather than in a
module that no run can be traced back to.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGES_TR_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_DIR, "labelsTr")
IMAGES_TS_DIR = os.path.join(DATA_DIR, "imagesTs")


def set_data_dir(new_dir):
    """
    Repoints DATA_DIR and everything derived from it.

    On Kaggle the raw dataset is mounted read-only, so OUTPUT_DIR is moved to
    the writable working directory at the same time.
    """
    global DATA_DIR, IMAGES_TR_DIR, LABELS_TR_DIR, IMAGES_TS_DIR, OUTPUT_DIR

    DATA_DIR = new_dir
    IMAGES_TR_DIR = os.path.join(DATA_DIR, "imagesTr")
    LABELS_TR_DIR = os.path.join(DATA_DIR, "labelsTr")
    IMAGES_TS_DIR = os.path.join(DATA_DIR, "imagesTs")

    if os.path.exists("/kaggle/input"):
        OUTPUT_DIR = "/kaggle/working/output"
        os.makedirs(OUTPUT_DIR, exist_ok=True)


# Kaggle mounts each attached dataset under its own directory, so the raw
# archive is located by looking for the file that identifies it.
if os.path.exists("/kaggle/input"):
    for _root, _dirs, _files in os.walk("/kaggle/input"):
        if "dataset.json" in _files:
            set_data_dir(_root)
            break


def resolve_nifti_path(relative_path):
    """
    Resolves a path from dataset.json to a file that exists on disk.

    The archive is distributed in several layouts: plain `.nii.gz`, decompressed
    `.nii`, and files nested inside a directory named after themselves. Rather
    than assume one, each is tried in turn.

    Args:
        relative_path (str): Path as written in dataset.json, e.g.
            "./labelsTr/lung_001.nii.gz".

    Returns:
        str: Absolute path to the file.

    Raises:
        FileNotFoundError: If no layout matches.
    """
    clean_rel = relative_path.lstrip("./")
    abs_path = os.path.join(DATA_DIR, clean_rel)

    if os.path.isfile(abs_path):
        return abs_path

    if abs_path.endswith(".gz") and os.path.isfile(abs_path[:-3]):
        return abs_path[:-3]

    base_name = os.path.basename(clean_rel).split(".")[0]
    dir_name = os.path.dirname(clean_rel)

    for suffix in (".nii", ".nii.gz"):
        nested_dir = os.path.join(DATA_DIR, dir_name, f"{base_name}{suffix}")
        if os.path.isdir(nested_dir):
            nested_file = os.path.join(nested_dir, f"{base_name}{suffix}")
            if os.path.isfile(nested_file):
                return nested_file

    parent_dir = os.path.join(DATA_DIR, dir_name)
    for root, _dirs, files in os.walk(parent_dir):
        for f in files:
            if f.startswith(base_name) and f.endswith((".nii", ".nii.gz")):
                return os.path.join(root, f)

    raise FileNotFoundError(
        f"Could not resolve NIfTI path for: {relative_path} "
        f"(looked under {DATA_DIR})")
