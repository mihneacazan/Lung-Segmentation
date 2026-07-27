# Exploratory Data Analysis (EDA) Report
**Project:** Automatic Lung Tumor Segmentation from CT Images (MSD Lung - Task06)

This report presents the statistical and exploratory analysis of the **Medical Segmentation Decathlon (Task06_Lung)** dataset. The analysis covers **all 63 training cases** with full resolution to provide complete geometric and intensity distributions.

---

## 1. Summary Statistics

The analysis was performed on **all 63 training cases**. No sampling was used, ensuring 100% accurate statistics for the entire cohort.

| Metric | Value | Observations / Details |
| :--- | :--- | :--- |
| **Total training cases** | 63 | Specified in `dataset.json` |
| **Total test cases** | 32 | No public ground truth masks |
| **Average volume shape (H x W x D)** | **512 x 512 x 280.3** | All cases have 512x512 axial resolution, with depth varying between 150 and 500+ slices. |
| **Average Voxel Spacing (x, y, z)** | **0.79 x 0.79 x 1.31 mm** | Sub-millimeter resolution in the axial plane, coarser resolution along the Z-axis. |
| **Average tumor volume** | **21,982.52 mm³** | Min: 737.74 mm³, Max: 370,384.12 mm³. Suggests extreme variation in tumor size. |
| **Total slices scanned (all cases)** | 17,657 slices | Total dataset size for 2D slicing |
| **Positive slices (containing tumor)** | **1,657 slices (9.38%)** | Clear indicator of **class imbalance** (only ~9% of slices contain tumor). |
| **Negative slices (healthy lung/tissue)** | **16,000 slices (90.62%)** | Over 90% of all CT slices do not contain tumor voxels. |

---

## 2. Distributions and Visual Analysis

Analysis plots have been generated and saved in the output directory [output/eda_figures/](file:///d:/Decathlon_Lung/output/eda_figures):

1. **Tumor Volume Distribution ([tumor_volume_distribution.png](file:///d:/Decathlon_Lung/output/eda_figures/tumor_volume_distribution.png)):**
   * Shows the distribution of tumor volumes across all 63 patients. The majority of tumors are smaller than 30,000 mm³, while a few extreme outliers have massive tumors exceeding 300,000 mm³.
2. **Slice Counts Comparison ([slices_comparison.png](file:///d:/Decathlon_Lung/output/eda_figures/slices_comparison.png)):**
   * Compares total slices and tumor-positive slices for each of the 63 patients. Highlights the consistency of low tumor-slice ratios, stressing the need for focused training on positive slices.

---

## 3. Sample Segmentation Overlay (CT + Mask)

A visual overlay of the original CT slice, the ground truth mask, and their combination was generated for the case with the largest tumor:

![Sample Overlay CT + Mask](file:///d:/Decathlon_Lung/output/eda_figures/sample_overlay.png)

*The image above shows an axial slice of the lungs where the tumor formation is clearly visible in grayscale and marked in red on the reference mask.*

---

## 4. Conclusions and Implications for Pipeline Design

1. **Class Imbalance:** Since only 9.38% of slices contain tumor nodules, training a model directly on all slices will lead to high false negatives.
   * **Solution:** We will use **DiceCELoss** (combining Dice Loss and Cross Entropy) and implement a random sampler (`RandCropByPosNegLabeld`) to prioritize positive patches during training.
2. **Size Variation:** Tumors vary from millimeter scale to large masses.
   * **Solution:** Preprocessing will crop the lung area (foreground cropping) and extract fixed-size training patches (e.g. 192x192).
