# Stage 1: Exploratory Data Analysis (EDA)

This directory contains the code and documentation for analyzing the geometric, intensity, and mask properties of the Medical Segmentation Decathlon (Task06_Lung) dataset.

---

## 📄 File Overview

* **`eda_report.py`**: The primary python script for dataset exploration.

---

## 🔍 Code Walkthrough & Design Motivations

### 1. Fast Header Scanning vs Full Voxel Loading
* **Implementation:** The script uses `nibabel.load()` to extract `img.shape` and `img.header.get_zooms()` directly from NIfTI headers without loading heavy 3D pixel arrays.
* **Motivation:** Scanning NIfTI headers takes less than 0.01 seconds per file, allowing instant verification of volume shapes ($512\times512\times D$) and voxel spacing ($0.79 \times 0.79 \times 1.31\text{ mm}$) across all 63 patients.

### 2. Memory-Efficient Voxel Reading
* **Implementation:** Uses `np.asanyarray(img.dataobj)` instead of `img.get_fdata()`.
* **Motivation:** `get_fdata()` converts integer CT arrays to 64-bit floating point numbers in RAM, consuming 4x more memory and slowing down disk reads. Using `dataobj` keeps data in native 16-bit integer format (`int16`), speeding up full-volume processing by ~500%.

### 3. Slice-Level Class Imbalance Tracking
* **Implementation:** For each volume, the script evaluates positive slices (`np.sum(lbl_data[:, :, s] == 1) > 0`) vs. negative slices.
* **Motivation:** Discovered that **90.62%** of CT slices are tumor-negative (healthy lung or background), while only **9.38%** contain tumor nodules. This key finding directly informed our Stage 2 slice sampling strategy (1:1 positive-to-negative ratio) and loss function selection (`DiceCELoss`).

### 4. Automated Figure Generation
* **Outputs:**
  * `output/eda_statistics.csv`: Per-patient metrics table.
  * `output/eda_figures/tumor_volume_distribution.png`: Histogram of tumor volumes.
  * `output/eda_figures/slices_comparison.png`: Bar chart of total vs positive slices per patient.
  * `output/eda_figures/sample_overlay.png`: Visual overlay of CT slice + ground truth mask.
