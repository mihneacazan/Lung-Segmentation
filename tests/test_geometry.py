"""
Regression tests for the geometry chain.

The failure these guard against is silent and expensive. Preprocessing reorients
every volume to canonical RAS, which for this dataset — all 63 volumes are stored
LAS — flips the left-right axis. If the reconstruction path does not flip it back,
predictions are compared against a mirrored ground truth and 3D Dice collapses
towards zero no matter how good the model is. Nothing crashes; the numbers simply
look like an ordinary bad result, and the search for the cause goes to the model
rather than to the geometry.

A synthetic asymmetric phantom catches it immediately: if any axis is flipped or
permuted on the way back, the round-trip Dice drops to near zero.

Run:
    python -m pytest tests/ -v
"""

import itertools

import numpy as np
import pytest
import nibabel as nib
from nibabel.orientations import apply_orientation

from src.geometry import (
    get_ornt,
    invert_ornt,
    reorient_to_canonical,
    restore_original_orientation,
    permute_spacing,
    resample_to_shape,
    crop_body_3d,
    apply_crop,
    reconstruct_to_original_geometry,
)


def dice(a, b):
    """Binary Dice between two masks; 1.0 when both are empty."""
    a, b = a > 0.5, b > 0.5
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


# ============================================================================
#  ORIENTATION
# ============================================================================

@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2])))
def test_invert_ornt_roundtrip_all_flips(perm):
    """
    invert_ornt must undo apply_orientation for every axis permutation combined
    with every combination of flips: 6 x 8 = 48 cases in total.
    """
    rng = np.random.default_rng(0)
    arr = rng.random((4, 5, 6))

    for flips in itertools.product([1, -1], repeat=3):
        ornt = np.array([[perm[i], flips[i]] for i in range(3)], dtype=float)
        forward = apply_orientation(arr, ornt)
        back = apply_orientation(forward, invert_ornt(ornt))
        assert back.shape == arr.shape
        assert np.array_equal(back, arr), f"failed for perm={perm} flips={flips}"


def test_reorient_helpers_roundtrip():
    """The named helpers compose into an identity."""
    rng = np.random.default_rng(1)
    arr = rng.random((7, 8, 9))
    # LAS -> RAS: flip the first axis, exactly this dataset's case.
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    ornt = get_ornt(affine)

    canonical = reorient_to_canonical(arr, ornt)
    assert not np.array_equal(canonical, arr), "LAS input should have been flipped"

    restored = restore_original_orientation(canonical, ornt)
    assert np.array_equal(restored, arr)


def test_las_orientation_is_a_left_right_flip():
    """
    Pins down the specific transform this dataset needs, so that a future change
    to the reorientation step cannot silently alter it.
    """
    affine = np.array([
        [-0.693359375, 0.0, 0.0, 182.15],
        [0.0, 0.693359375, 0.0, -40.15],
        [0.0, 0.0, 1.0, -305.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    ornt = get_ornt(affine)
    np.testing.assert_array_equal(ornt, np.array([[0, -1], [1, 1], [2, 1]]))


def test_permute_spacing_follows_axis_permutation():
    """Spacing must follow the same permutation the data axes underwent."""
    # Axis 0 -> 2, axis 1 -> 0, axis 2 -> 1.
    ornt = np.array([[2, 1], [0, 1], [1, -1]], dtype=float)
    assert permute_spacing((0.7, 0.8, 2.5), ornt) == (0.8, 2.5, 0.7)


# ============================================================================
#  FULL PIPELINE ROUND-TRIP
# ============================================================================

def make_asymmetric_phantom(shape=(64, 64, 40)):
    """
    Builds a CT-like phantom that is asymmetric on all three axes.

    Asymmetry is the whole point: a symmetric phantom would survive a flip and
    the test would pass with the flip still unhandled.

    Returns:
        tuple: (image_hu, label_mask)
    """
    img = np.full(shape, -1000.0, dtype=np.float32)   # air
    lbl = np.zeros(shape, dtype=np.uint8)

    # Body: an off-centre block, closer to one side on every axis.
    img[8:52, 6:46, 4:34] = -700.0                    # lung parenchyma
    img[20:44, 16:38, 10:30] = 40.0                   # soft tissue

    # Tumour: a small blob far from every centre line.
    lbl[12:20, 10:16, 8:14] = 1
    img[12:20, 10:16, 8:14] = 60.0

    return img, lbl


@pytest.mark.parametrize("axcodes,affine_diag", [
    ("LAS", [-0.8, 0.8, 1.5]),      # this dataset
    ("RAS", [0.8, 0.8, 1.5]),       # already canonical
    ("LPS", [-0.8, -0.8, 1.5]),     # DICOM-style
    ("RAI", [0.8, 0.8, -1.5]),      # z reversed
])
def test_preprocess_reconstruct_roundtrip(tmp_path, axcodes, affine_diag):
    """
    Runs a phantom through the real preprocessing pipeline and reconstructs it,
    then checks the mask lands back on itself.

    This is the end-to-end guard. Under the old reconstruction, the LAS case
    scored near zero here.
    """
    from src.preprocessing.preprocessing import preprocess_case

    img_hu, lbl = make_asymmetric_phantom()
    affine = np.diag(affine_diag + [1.0])
    affine[:3, 3] = [100.0, -50.0, -200.0]

    img_path = tmp_path / "case.nii.gz"
    lbl_path = tmp_path / "case_lbl.nii.gz"
    nib.save(nib.Nifti1Image(img_hu, affine), str(img_path))
    nib.save(nib.Nifti1Image(lbl, affine), str(lbl_path))

    _, lbl_stack, metadata, qc = preprocess_case("phantom", str(img_path), str(lbl_path))

    reconstructed = reconstruct_to_original_geometry(
        lbl_stack.astype(np.float32), metadata, threshold=0.5)

    assert reconstructed.shape == lbl.shape, (
        f"{axcodes}: reconstruction shape {reconstructed.shape} != {lbl.shape}")

    score = dice(reconstructed, lbl)
    assert score > 0.75, (
        f"{axcodes}: round-trip Dice {score:.4f} is too low. A near-zero score "
        f"means an axis is flipped or permuted on the inverse path.")

    # preprocess_case computes the same number itself; they must agree to the
    # 4 decimals the QC record is rounded to.
    assert abs(qc["roundtrip_dice"] - score) < 1e-4


def test_mirrored_reconstruction_is_detected():
    """
    Confirms the round-trip assertion is actually sensitive to a mirrored axis:
    mirroring
    the reconstruction along the left-right axis must tank the Dice.

    Without this, a round-trip test could pass vacuously on a phantom that
    happens to be symmetric.
    """
    _, lbl = make_asymmetric_phantom()
    mirrored = np.flip(lbl, axis=0)
    assert dice(mirrored, lbl) < 0.10, (
        "phantom is not asymmetric enough for the mirror test to be meaningful")


def test_resample_to_shape_hits_exact_shape():
    """Inverse resampling must land on the requested shape exactly."""
    rng = np.random.default_rng(2)
    vol = rng.random((37, 41, 23)).astype(np.float32)
    for target in [(50, 50, 50), (12, 60, 9), (37, 41, 23)]:
        assert resample_to_shape(vol, target).shape == target


# ============================================================================
#  BODY CROP
# ============================================================================

def test_crop_body_excludes_surrounding_air():
    """
    The crop must actually shrink the volume. Thresholding the normalized image
    at 0.05 (roughly -930 HU) instead of thresholding raw HU let reconstruction
    noise in the air keep the bounding box at full size for 51 of 63 patients,
    making the crop a no-op.
    """
    img = np.full((80, 80, 40), -1000.0, dtype=np.float32)
    img[20:60, 25:55, 5:35] = 50.0                  # body
    rng = np.random.default_rng(3)
    img += rng.normal(0, 30, img.shape)             # scanner noise everywhere

    bbox = crop_body_3d(img, margin=5)
    cropped = apply_crop(img, bbox)

    assert cropped.size < img.size * 0.6, (
        f"crop kept {100 * cropped.size / img.size:.0f}% of the volume; "
        f"it is not excluding air")
    # The body must survive intact, margin included.
    assert bbox["x_min"] <= 20 and bbox["x_max"] >= 60
    assert bbox["y_min"] <= 25 and bbox["y_max"] >= 55


def test_crop_drops_detached_scanner_table():
    """Only the largest connected component is kept, so the table is discarded."""
    img = np.full((80, 80, 40), -1000.0, dtype=np.float32)
    img[20:60, 25:55, 5:35] = 50.0                  # body
    img[70:78, 25:55, 5:35] = 200.0                 # detached table rail

    bbox = crop_body_3d(img, margin=0)
    assert bbox["x_max"] <= 61, (
        f"bounding box reached x={bbox['x_max']}, so it swallowed the table")
