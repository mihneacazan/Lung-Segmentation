# Comprehensive Exploratory Data Analysis & Integrity Audit Report

This report presents a thorough analysis of dataset integrity, volume geometry, tumor topology, and preprocessing distortion across all 63 CT scans in the Decathlon Lung Dataset.

## 1. Dataset & Case-Level Verification Summary
- **Total CT Volumes & Masks Analyzed:** 63
- **File Existence:** Image & Mask files exist for 100% of cases.
- **Shape Alignment:** Image shape matches Mask shape for 63/63 cases.
- **Affine Alignment:** Image affine matrix equals Mask affine matrix for 63/63 cases (Maximum observed affine difference: `0.000000e+00`).
- **Orientation Check:**
  - Image Orientation Distribution: `{'LAS': 63}`
  - Label Orientation Distribution: `{'LAS': 63}`
  - Orientation match between Image and Mask: **100% consistent**.
- **Physical Spacing ($mm$):**
  - X Spacing: `0.60` mm to `0.98` mm (Mean: `0.79` mm)
  - Y Spacing: `0.60` mm to `0.98` mm (Mean: `0.79` mm)
  - Z Spacing: `0.62` mm to `2.50` mm (Mean: `1.31` mm, `34` unique Z values across dataset)

## 2. Data Quality, Anomalies & Label Integrity
- **Mask Binary Integrity:** All label masks strictly contain values `(0, 1)`. No invalid labels, floating values, or missing classes detected.
- **NaN / Inf Verification:**
  - NaN values in Images: `0` | Infs in Images: `0`
  - NaN values in Labels: `0` | Infs in Labels: `0`
- **Empty Volumes & Masks:**
  - Empty CT Images: `0`
  - Empty Ground Truth Masks: `0`
- **Volume Duplicates:** No exact MD5 duplicate volumes detected among images or labels.

## 3. Tumor Size Distribution & Topology (3D Connected Components)
- **Tumor Volume Range:** `737.74` $mm^3$ to `370384.12` $mm^3$ (Mean: `21982.52` $mm^3$).
- **Total Slices:** `17657` total 2D axial slices across dataset.
- **Tumor-Positive Slices:** `1657` slices (`9.38%` of total).
- **Connected Components (3D):**
  - Average number of 3D tumor components per patient: `2.60` (Range: `1` to `14`).
  - Largest component volume percentage: `99.90%` of total tumor mass on average.

## 4. Preprocessing Distortion & Information Loss Analysis
Direct 2D resizing from $512 \times 512$ to $192 \times 192$ using Nearest-Neighbor interpolation introduces spatial distortion and voxel loss:
- **Positive Slices Lost:** `9` positive 2D slices become completely empty after resize!
- **Patients Affected by Lost Slices:** `5` patients have at least 1 positive slice eliminated by resize.
- **Volume Alteration:** Mean tumor volume change after resize is `-0.80%`.
- **Pure Preprocessing 3D Dice Score:** Reconstructing the resized $192 \times 192$ mask back to original $512 \times 512$ space yields an upper-bound 3D Dice of **`0.9288`** (Min: `0.8566`).

---
*Report generated automatically by `src/eda/eda_report.py`.*
