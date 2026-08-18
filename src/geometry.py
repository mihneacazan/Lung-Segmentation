"""
Geometry utilities for NIfTI volumes: canonical reorientation, physical-spacing
resampling, and the exact INVERSE of both, so that a 2D slice prediction can be
reconstructed back into the original NIfTI geometry voxel-for-voxel.

Why this module exists
----------------------
All 63 Decathlon Lung volumes are stored in LAS orientation. Preprocessing
reorients them to canonical RAS, which flips the left-right axis. If that flip is
not undone before comparing against the ground truth (which is read from the raw
NIfTI in LAS), the prediction is mirrored and the 3D Dice collapses to ~0
regardless of how well the model actually segments.

The forward pipeline is:

    original (LAS)  --reorient-->  canonical (RAS)
                    --resample-->  1mm isotropic
                    --crop------>  body bounding box
                    --resize---->  192x192 axial slices

and every one of those four steps is inverted here, in reverse order.

Usage:
    from src.geometry import invert_ornt, reorient_to_canonical, resample_volume
"""

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, apply_orientation, inv_ornt_aff
from scipy.ndimage import zoom as scipy_zoom


# Target isotropic spacing in millimeters (1 voxel = 1mm x 1mm x 1mm)
TARGET_SPACING = (1.0, 1.0, 1.0)


# ============================================================================
#  ORIENTATION
# ============================================================================

def get_ornt(affine):
    """
    Returns the orientation transform that maps the data axes of a volume with
    this affine onto canonical RAS axes.

    This is exactly the transform `nib.as_closest_canonical` applies internally,
    exposed so that it can be stored in metadata and inverted later.

    Args:
        affine (np.ndarray): 4x4 NIfTI affine matrix.

    Returns:
        np.ndarray: Orientation array of shape (3, 2). Row i is [j, flip],
                    meaning input axis i becomes output axis j, reversed if
                    flip == -1.
    """
    return io_orientation(np.asarray(affine))


def invert_ornt(ornt):
    """
    Inverts an orientation transform array.

    `apply_orientation` first flips input axes where flip == -1, then transposes
    so that input axis i lands at output position ornt[i, 0]. To invert, axis
    ornt[i, 0] of the transformed array must be flipped back (the flip is its own
    inverse) and moved back to position i.

    Guarantees:
        apply_orientation(apply_orientation(arr, ornt), invert_ornt(ornt)) == arr

    Args:
        ornt (np.ndarray): Orientation array of shape (N, 2).

    Returns:
        np.ndarray: The inverse orientation array, same shape.
    """
    ornt = np.asarray(ornt)
    inv = np.zeros_like(ornt)
    for in_axis in range(ornt.shape[0]):
        out_axis = int(ornt[in_axis, 0])
        inv[out_axis] = [in_axis, ornt[in_axis, 1]]
    return inv


def reorient_to_canonical(volume, ornt):
    """
    Applies an orientation transform to a raw 3D array (original -> RAS).

    Args:
        volume (np.ndarray): 3D array in the original data orientation.
        ornt (np.ndarray): Orientation array from `get_ornt`.

    Returns:
        np.ndarray: The volume with axes permuted and flipped to canonical RAS.
    """
    return apply_orientation(np.asarray(volume), np.asarray(ornt))


def restore_original_orientation(volume, ornt):
    """
    Undoes `reorient_to_canonical`, mapping a canonical RAS volume back onto the
    original data orientation (RAS -> original).

    Args:
        volume (np.ndarray): 3D array in canonical RAS orientation.
        ornt (np.ndarray): The FORWARD orientation array from `get_ornt`
                           (this function inverts it internally).

    Returns:
        np.ndarray: The volume in the original data orientation.
    """
    return apply_orientation(np.asarray(volume), invert_ornt(ornt))


def canonical_affine(affine, shape, ornt):
    """
    Computes the affine of the volume after reorientation to canonical RAS.

    Args:
        affine (np.ndarray): Original 4x4 affine.
        shape (tuple): Original volume shape, before reorientation.
        ornt (np.ndarray): Orientation array from `get_ornt`.

    Returns:
        np.ndarray: The 4x4 affine of the reoriented volume.
    """
    return np.asarray(affine).dot(inv_ornt_aff(np.asarray(ornt), tuple(shape)))


def permute_spacing(spacing, ornt):
    """
    Reorders a voxel spacing triplet to match the canonical axis order.

    Reorientation permutes the data axes, so the spacing values must follow the
    same permutation or the resampling factors will be applied to the wrong axes.

    Args:
        spacing (tuple): Voxel spacing (sx, sy, sz) in the original axis order.
        ornt (np.ndarray): Orientation array from `get_ornt`.

    Returns:
        tuple: Voxel spacing in canonical RAS axis order.
    """
    spacing = np.asarray(spacing, dtype=float)
    ornt = np.asarray(ornt)
    permuted = np.zeros(3, dtype=float)
    for in_axis in range(3):
        permuted[int(ornt[in_axis, 0])] = spacing[in_axis]
    return tuple(permuted)


# ============================================================================
#  RESAMPLING
# ============================================================================

def resample_volume(volume, original_spacing, target_spacing=TARGET_SPACING, order=1):
    """
    Resamples a 3D volume from its own voxel spacing onto a common physical
    spacing, so that one voxel means the same physical distance for every patient
    regardless of scanner settings.

    Args:
        volume (np.ndarray): 3D array of shape (H, W, D).
        original_spacing (tuple): Current voxel spacing (sx, sy, sz) in mm.
        target_spacing (tuple): Desired voxel spacing in mm.
        order (int): Interpolation order. Use 1 (trilinear) for CT intensities
                     and 0 (nearest-neighbor) for binary masks.

    Returns:
        np.ndarray: The resampled volume.
    """
    zoom_factors = tuple(
        float(orig) / float(tgt)
        for orig, tgt in zip(original_spacing, target_spacing)
    )
    return scipy_zoom(volume, zoom_factors, order=order, prefilter=False)


def resample_to_shape(volume, target_shape, order=1):
    """
    Resamples a 3D volume to an exact target shape.

    Used on the inverse path, where the destination shape is known exactly and
    must be matched voxel-for-voxel (deriving it from zoom factors alone can be
    off by one due to rounding).

    Args:
        volume (np.ndarray): 3D array to resample.
        target_shape (tuple): Exact output shape (H, W, D).
        order (int): Interpolation order.

    Returns:
        np.ndarray: Volume resampled to exactly `target_shape`.
    """
    target_shape = tuple(int(s) for s in target_shape)
    if tuple(volume.shape) == target_shape:
        return volume

    zoom_factors = tuple(
        t / s for t, s in zip(target_shape, volume.shape)
    )
    resampled = scipy_zoom(volume, zoom_factors, order=order, prefilter=False)

    # scipy_zoom rounds each axis independently, so pad or trim to land exactly
    # on the requested shape.
    if tuple(resampled.shape) != target_shape:
        fixed = np.zeros(target_shape, dtype=resampled.dtype)
        overlap = tuple(slice(0, min(a, b)) for a, b in zip(resampled.shape, target_shape))
        fixed[overlap] = resampled[overlap]
        resampled = fixed

    return resampled


def resize_slice_2d(slice_2d, target_size, order=1):
    """
    Resizes a single 2D axial slice to the target dimensions.

    Args:
        slice_2d (np.ndarray): 2D array of shape (H, W).
        target_size (tuple): Target dimensions (target_H, target_W).
        order (int): Interpolation order (1 = bilinear, 0 = nearest).

    Returns:
        np.ndarray: Resized 2D array.
    """
    h, w = slice_2d.shape
    if (h, w) == tuple(target_size):
        return slice_2d

    zoom_factors = (target_size[0] / h, target_size[1] / w)
    return scipy_zoom(slice_2d, zoom_factors, order=order, prefilter=False)


# ============================================================================
#  BODY CROP
# ============================================================================

def crop_body_3d(img_hu, margin=5):
    """
    Computes the bounding box of the patient's body within a CT volume.

    The threshold is applied in raw Hounsfield Units rather than on the
    normalized image: air is around -1000 HU and lung parenchyma around -800 HU,
    so a -500 HU cut cleanly separates body tissue from air. Thresholding the
    normalized volume at 0.05 instead (as an earlier version did) corresponds to
    roughly -930 HU, which is low enough that reconstruction noise in the air
    around the patient trips it and the bounding box degenerates to the full
    volume.

    Only the largest 3D connected component is kept, which discards the scanner
    table and any detached noise speckle.

    Args:
        img_hu (np.ndarray): 3D CT volume in raw Hounsfield Units.
        margin (int): Extra voxels kept around the bounding box.

    Returns:
        dict: Bounding box with keys x_min/x_max/y_min/y_max/z_min/z_max.
    """
    from scipy.ndimage import binary_fill_holes, label as scipy_label

    body = img_hu > -500.0

    if not body.any():
        return {"x_min": 0, "x_max": int(img_hu.shape[0]),
                "y_min": 0, "y_max": int(img_hu.shape[1]),
                "z_min": 0, "z_max": int(img_hu.shape[2])}

    # Keep only the largest connected component (the body), dropping the table.
    labeled, n_components = scipy_label(body)
    if n_components > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # ignore background
        body = labeled == int(counts.argmax())

    body = binary_fill_holes(body)

    coords = np.argwhere(body)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    return {
        "x_min": int(max(0, mins[0] - margin)),
        "x_max": int(min(img_hu.shape[0], maxs[0] + margin + 1)),
        "y_min": int(max(0, mins[1] - margin)),
        "y_max": int(min(img_hu.shape[1], maxs[1] + margin + 1)),
        "z_min": int(max(0, mins[2] - margin)),
        "z_max": int(min(img_hu.shape[2], maxs[2] + margin + 1)),
    }


def apply_crop(volume, bbox):
    """Slices a volume with a bounding box dict from `crop_body_3d`."""
    return volume[
        bbox["x_min"]:bbox["x_max"],
        bbox["y_min"]:bbox["y_max"],
        bbox["z_min"]:bbox["z_max"],
    ]


# ============================================================================
#  FULL INVERSE: preprocessed slice stack -> original NIfTI geometry
# ============================================================================

def reconstruct_to_original_geometry(slice_stack, metadata, threshold=0.5,
                                     binarize=True):
    """
    Reconstructs a stack of 2D predictions back into the original NIfTI geometry,
    undoing every preprocessing step in reverse order.

    Steps (mirror images of the forward pipeline):
        1. Resize each 192x192 slice back to the cropped in-plane size.
        2. Un-crop: paste into the full 1mm-isotropic volume.
        3. Resample from 1mm isotropic back to the canonical (RAS) shape.
        4. Undo the canonical reorientation, returning to the original axis
           order and flips.
        5. Threshold once, at the very end, so interpolation happens on
           probabilities rather than on an already-binarized mask.

    Args:
        slice_stack (np.ndarray): Probability volume of shape (192, 192, D_cropped),
            where D_cropped matches metadata["cropped_shape"][2].
        metadata (dict): Per-patient metadata written by the preprocessing stage.
        threshold (float): Binarization threshold applied at the end.
        binarize (bool): If False, returns the float probability volume instead
            of a binary mask (useful for threshold sweeps).

    Returns:
        np.ndarray: Volume with shape metadata["original_shape"], dtype uint8 if
                    binarize else float32.
    """
    cropped_shape = tuple(metadata["cropped_shape"])
    resampled_shape = tuple(metadata["resampled_shape"])
    canonical_shape = tuple(metadata["canonical_shape"])
    original_shape = tuple(metadata["original_shape"])
    bbox = metadata["crop_bbox"]
    ornt = np.asarray(metadata["ornt"])

    slice_stack = np.asarray(slice_stack, dtype=np.float32)

    # Step 1: resize slices from 192x192 back to the cropped in-plane size
    target_h, target_w, n_slices = cropped_shape
    cropped_3d = np.zeros(cropped_shape, dtype=np.float32)
    for s in range(min(n_slices, slice_stack.shape[2])):
        cropped_3d[:, :, s] = resize_slice_2d(
            slice_stack[:, :, s], (target_h, target_w), order=1
        )

    # Step 2: un-crop into the full 1mm isotropic volume
    resampled_3d = np.zeros(resampled_shape, dtype=np.float32)
    resampled_3d[
        bbox["x_min"]:bbox["x_max"],
        bbox["y_min"]:bbox["y_max"],
        bbox["z_min"]:bbox["z_max"],
    ] = cropped_3d

    # Step 3: back from 1mm isotropic to the canonical-orientation shape
    canonical_3d = resample_to_shape(resampled_3d, canonical_shape, order=1)

    # Step 4: undo the RAS reorientation (this is the flip that was missing)
    original_3d = restore_original_orientation(canonical_3d, ornt)

    assert tuple(original_3d.shape) == original_shape, (
        f"Reconstruction shape {original_3d.shape} != original {original_shape}"
    )

    # Step 5: threshold last, on interpolated probabilities
    if binarize:
        return (original_3d > threshold).astype(np.uint8)
    return original_3d.astype(np.float32)


def save_prediction_nifti(pred_volume, metadata, output_path):
    """
    Saves a reconstructed prediction as a NIfTI file carrying the patient's
    original affine, so it overlays exactly on the source CT in any viewer.

    Args:
        pred_volume (np.ndarray): Binary prediction in original geometry.
        metadata (dict): Per-patient metadata (supplies the original affine).
        output_path (str): Destination .nii.gz path.
    """
    affine = np.asarray(metadata["original_affine"])
    nifti = nib.Nifti1Image(pred_volume.astype(np.uint8), affine)
    nib.save(nifti, output_path)
