"""
Post-Preprocessing Overlay Verification Script.

Generates visual overlay images showing CT slices with semi-transparent tumor
masks superimposed, AFTER the full preprocessing pipeline has been applied.
This verifies that the image-mask alignment is preserved through reorientation,
resampling, cropping, and resizing.

Saves overlay PNG figures to output/eda_figures/overlay_*.png.

Usage:
    python -m src.visualization.overlay_check
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR


def load_random_slices(split_name: str, n_samples: int = 6):
    """
    Loads tumour-bearing slices from a split for visual alignment checking.

    Reads the per-patient volume stacks written by the preprocessing stage,
    spreading the sampled slices across as many different patients as possible so
    the check is not dominated by one case.

    Args:
        split_name (str): One of "train", "val", or "test".
        n_samples (int): Number of sample slices to load.

    Returns:
        list of tuples: [(image_array, mask_array, label), ...]
    """
    preprocessed_dir = os.path.join(OUTPUT_DIR, "preprocessed")
    volumes_dir = os.path.join(preprocessed_dir, "volumes")

    index_path = os.path.join(preprocessed_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing {index_path}\n"
            f"Run 'python -m src.preprocessing.preprocessing' first.")

    with open(index_path, "r") as f:
        index = json.load(f)

    case_ids = index["splits"].get(split_name, [])
    if not case_ids:
        return []

    # One candidate per patient first, cycling through so that every patient is
    # represented before any patient contributes a second slice.
    candidates = []
    per_case = []
    for case_id in case_ids:
        positives = index["cases"][case_id]["positive_slices"]
        if not positives:
            continue
        # Take slices spread through the tumour rather than only its first slice.
        picks = np.linspace(0, len(positives) - 1, min(3, len(positives)), dtype=int)
        per_case.append([(case_id, positives[p]) for p in picks])

    for round_idx in range(3):
        for case_slices in per_case:
            if round_idx < len(case_slices):
                candidates.append(case_slices[round_idx])

    results = []
    for case_id, slice_idx in candidates[:n_samples]:
        img_vol = np.load(os.path.join(volumes_dir, f"{case_id}_img.npy"), mmap_mode="r")
        lbl_vol = np.load(os.path.join(volumes_dir, f"{case_id}_lbl.npy"), mmap_mode="r")
        img = np.asarray(img_vol[:, :, slice_idx], dtype=np.float32)
        lbl = np.asarray(lbl_vol[:, :, slice_idx], dtype=np.uint8)
        results.append((img, lbl, f"{case_id} slice {slice_idx}"))

    return results


def create_overlay_figure(samples, split_name: str, save_path: str):
    """
    Creates a figure with CT slices overlaid with semi-transparent tumor masks.
    
    Each subplot shows:
        - Grayscale CT slice (after HU windowing + normalization)
        - Red semi-transparent overlay where tumor mask == 1
        - Green contour of tumor boundary for precise alignment check
        
    Args:
        samples (list): List of (image, mask, filename) tuples.
        split_name (str): Split name for the figure title.
        save_path (str): Output path for the saved PNG.
    """
    n = len(samples)
    if n == 0:
        print(f"  [WARNING] No positive samples found in {split_name} split for overlay.")
        return
    
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    
    fig.suptitle(f"Post-Preprocessing Overlay Verification: {split_name.upper()} Set",
                 fontsize=14, fontweight="bold")
    
    for i, (img, lbl, filename) in enumerate(samples):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        
        # Display CT slice in grayscale
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        
        # Overlay tumor mask in semi-transparent red
        mask_overlay = np.zeros((*lbl.shape, 4))  # RGBA
        mask_overlay[lbl > 0.5] = [1.0, 0.0, 0.0, 0.35]  # Red, 35% opacity
        ax.imshow(mask_overlay)
        
        # Draw tumor contour in green for precise boundary alignment check
        from scipy.ndimage import binary_erosion
        if np.sum(lbl > 0.5) > 0:
            contour = (lbl > 0.5).astype(float) - binary_erosion(lbl > 0.5).astype(float)
            contour_overlay = np.zeros((*lbl.shape, 4))
            contour_overlay[contour > 0] = [0.0, 1.0, 0.0, 0.9]  # Green contour
            ax.imshow(contour_overlay)
        
        # Extract patient ID and slice index from filename
        case_info = filename.replace(".npy", "")
        tumor_pct = np.sum(lbl > 0.5) / lbl.size * 100
        ax.set_title(f"{case_info}\nTumor: {tumor_pct:.1f}% of slice", fontsize=9)
        ax.axis("off")
    
    # Hide unused subplot slots
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay figure: {save_path}")


def run_overlay_check():
    """
    Runs the complete overlay verification for all 3 splits (Train, Val, Test).
    Generates one overlay figure per split, each showing 6 representative
    positive slices with tumor masks superimposed.
    """
    print("=== RUNNING POST-PREPROCESSING OVERLAY VERIFICATION ===\n")
    
    figures_dir = os.path.join(OUTPUT_DIR, "eda_figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    for split_name in ["train", "val", "test"]:
        print(f"  Processing {split_name.upper()} split...")
        samples = load_random_slices(split_name, n_samples=6)
        
        if len(samples) == 0:
            print(f"  [SKIP] No preprocessed data found for {split_name}")
            continue
            
        save_path = os.path.join(figures_dir, f"overlay_{split_name}.png")
        create_overlay_figure(samples, split_name, save_path)
    
    print("\n=== OVERLAY VERIFICATION COMPLETE ===")


if __name__ == "__main__":
    run_overlay_check()
