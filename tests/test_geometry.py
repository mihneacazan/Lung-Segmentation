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
    from_network_grid,
    get_ornt,
    plan_from_metadata,
    resize_plan,
    resize_slice_2d,
    to_network_grid,
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


# ============================================================================
#  CROPPED SLICE -> FIXED NETWORK GRID
# ============================================================================

@pytest.mark.parametrize("cropped_hw", [(355, 282), (440, 275), (404, 404),
                                        (320, 306), (490, 422)])
@pytest.mark.parametrize("mode,target,mm", [("stretch", 192, None),
                                            ("pad", 192, None),
                                            ("fixed_mm", 256, 2.0)])
def test_network_grid_round_trips_to_the_cropped_size(cropped_hw, mode, target, mm):
    """
    Every mode must land on the grid and come back to the crop it started from.

    The shapes used are real cropped extents from this dataset, including the
    most distorted patient and a square one, because the padding offsets are
    integer divisions and an odd difference rounds differently from an even one.
    """
    plan = resize_plan((*cropped_hw, 8), mode=mode, target_size=target,
                       mm_per_px=mm)
    slice_2d = np.random.default_rng(0).random(cropped_hw).astype(np.float32)

    on_grid = to_network_grid(slice_2d, plan)
    back = from_network_grid(on_grid, plan, cropped_hw)

    assert on_grid.shape == (target, target)
    assert back.shape == cropped_hw


def test_stretch_is_byte_identical_to_the_previous_direct_resize():
    """
    The default path must reproduce the old behaviour exactly.

    Twenty-one experiments were trained against a dataset built before these
    modes existed. If `stretch` diverged from the plain resize by even a
    rounding step, those runs would no longer be reproducible from this code.
    """
    slice_2d = np.random.default_rng(1).random((355, 282)).astype(np.float32)
    plan = resize_plan((355, 282, 8), mode="stretch", target_size=192)

    assert np.array_equal(to_network_grid(slice_2d, plan),
                          resize_slice_2d(slice_2d, (192, 192), order=1))


def test_pad_removes_the_anisotropy_that_stretch_introduces():
    """
    The point of `pad` is that both axes end up at the same millimetres per
    pixel. On the worst patient in this dataset stretch scales them 1.60 apart.
    """
    height, width = 440, 275
    stretch_ratio = (height / 192) / (width / 192)
    assert stretch_ratio > 1.5, "test shape no longer exercises the problem"

    plan = resize_plan((height, width, 8), mode="pad", target_size=192)
    # A padded square is scaled by one factor, so both axes carry it equally.
    assert plan["square_side"] == height
    assert plan["pad_top"] == 0
    assert plan["pad_left"] == (height - width) // 2


def test_fixed_mm_gives_every_patient_the_same_scale():
    """Two differently shaped patients must come out at identical mm per pixel."""
    plans = [resize_plan((h, w, 8), mode="fixed_mm", target_size=256,
                         mm_per_px=2.0)
             for h, w in [(355, 282), (490, 422)]]

    for plan, (h, w) in zip(plans, [(355, 282), (490, 422)]):
        assert abs(h / plan["inner_h"] - 2.0) < 0.02
        assert abs(w / plan["inner_w"] - 2.0) < 0.02


def test_fixed_mm_refuses_a_scale_that_would_crop_the_patient():
    """
    Silently truncating a body that does not fit would lose anatomy without any
    error, so the plan is rejected instead.
    """
    with pytest.raises(ValueError, match="does not fit"):
        resize_plan((490, 422, 8), mode="fixed_mm", target_size=192,
                    mm_per_px=2.0)


def test_padding_is_air_not_an_invented_value():
    """
    Zero is -1000 HU after the window, which is what surrounds the patient
    anyway. Padding with anything else would put a material in the image that
    the network has to learn to ignore.
    """
    plan = resize_plan((100, 60, 8), mode="pad", target_size=64)
    filled = to_network_grid(np.ones((100, 60), dtype=np.float32), plan)

    assert filled.min() == 0.0, "padding is not air"
    assert filled.max() == pytest.approx(1.0)


def test_missing_plan_in_metadata_reads_as_stretch():
    """Metadata written before the modes existed must keep working unchanged."""
    plan = plan_from_metadata({"target_slice_size": [192, 192]})
    assert plan == {"mode": "stretch", "target_size": 192}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="resize mode"):
        resize_plan((100, 100, 8), mode="squash", target_size=192)


@pytest.mark.parametrize("slice_spacing", [1.0, 2.0, 2.5])
def test_roundtrip_survives_a_coarser_slice_spacing(tmp_path, slice_spacing):
    """
    The slice axis is a free parameter, and the inverse path has to keep closing
    at every value of it.

    The default of 1.0 mm is finer than this dataset's median source thickness,
    so it interpolates slices into existence for most patients. Raising it is the
    fix, but reconstruction inverts by the recorded shapes rather than by
    spacing, and that is exactly the kind of implicit coupling that breaks
    silently: a wrong inverse still returns an array of the right size, just
    with the mask in the wrong place.
    """
    from src.preprocessing.preprocessing import preprocess_case

    img_hu, lbl = make_asymmetric_phantom()
    affine = np.diag([-0.8, 0.8, 1.5, 1.0])
    affine[:3, 3] = [100.0, -50.0, -200.0]

    img_path = tmp_path / "case.nii.gz"
    lbl_path = tmp_path / "case_lbl.nii.gz"
    nib.save(nib.Nifti1Image(img_hu, affine), str(img_path))
    nib.save(nib.Nifti1Image(lbl, affine), str(lbl_path))

    _, lbl_stack, metadata, _ = preprocess_case(
        "phantom", str(img_path), str(lbl_path), slice_spacing=slice_spacing)

    assert metadata["target_spacing"] == [1.0, 1.0, slice_spacing], (
        "the spacing actually used must be recorded, or the inverse path has no "
        "way to know which dataset it is looking at")

    reconstructed = reconstruct_to_original_geometry(
        lbl_stack.astype(np.float32), metadata, threshold=0.5)

    assert reconstructed.shape == lbl.shape
    score = dice(reconstructed, lbl)
    assert score > 0.75, (
        f"round-trip Dice {score:.4f} at {slice_spacing} mm slices. The mask "
        f"came back in the wrong place, not merely blurred.")


def test_coarser_slice_spacing_actually_produces_fewer_slices(tmp_path):
    """
    Guards the reason for the parameter existing. If the slice count did not
    fall, nothing would have changed except a number in the metadata: a 2.5D
    neighbour would still be an interpolated near-duplicate, and epochs would
    still be as long.

    The resampled count is checked exactly, the cropped count only for
    direction. `crop_body_3d` adds a fixed five-voxel margin per side, which does
    not scale with spacing, so a 2.5x coarser volume keeps the same ten slices of
    padding and the cropped ratio always falls short of 2.5. On a phantom this
    dominates; on a real volume of ~314 slices it costs about 5%.
    """
    from src.preprocessing.preprocessing import preprocess_case

    img_hu, lbl = make_asymmetric_phantom()
    affine = np.diag([-0.8, 0.8, 1.5, 1.0])
    nib.save(nib.Nifti1Image(img_hu, affine), str(tmp_path / "c.nii.gz"))
    nib.save(nib.Nifti1Image(lbl, affine), str(tmp_path / "c_lbl.nii.gz"))

    resampled, cropped = {}, {}
    for spacing in (1.0, 2.5):
        _, _, metadata, _ = preprocess_case(
            "phantom", str(tmp_path / "c.nii.gz"), str(tmp_path / "c_lbl.nii.gz"),
            run_qc=False, slice_spacing=spacing)
        resampled[spacing] = metadata["resampled_shape"][2]
        cropped[spacing] = metadata["cropped_shape"][2]

    ratio = resampled[1.0] / resampled[2.5]
    assert 2.4 < ratio < 2.6, (
        f"the resample itself did not honour the spacing: {resampled[1.0]} "
        f"slices at 1 mm against {resampled[2.5]} at 2.5 mm, a ratio of "
        f"{ratio:.2f} where 2.5 was asked for")

    assert cropped[2.5] < cropped[1.0], (
        f"cropped slice count did not fall: {cropped[1.0]} -> {cropped[2.5]}")
