# Stage 6: Interactive Web Visualization Apps

This directory contains the interactive web visualization applications built using Streamlit.

---

## 📄 File Overview

* **`visualizer.py`**: Interactive CT volume browser with Hounsfield window presets and mask overlays.
* **`prediction_viewer.py`**: Side-by-side comparison of model predictions vs ground truth masks.

---

## 🔍 Code Walkthrough & Design Motivations

### visualizer.py — CT Volume Explorer

#### 1. Interactive Slice Navigation
* **Implementation:** Reads 3D NIfTI volumes dynamically and provides a slider to navigate along the Z-axis (axial slices).
* **Motivation:** Allows researchers and clinicians to visually inspect tumor boundaries across the full 3D extent of the lungs.

#### 2. Presets for Hounsfield Unit Windowing
* **Implementation:** Provides interactive radio options for:
  * **Lung Window:** $[-1000, 400]$ HU (optimal for lung parenchyma).
  * **Mediastinum Window:** $[-150, 250]$ HU (optimal for soft tissue).
  * **Full Raw Range:** Unclipped raw values.
* **Motivation:** Mimics clinical PACS workstations where radiologists switch windowing presets to evaluate different anatomical structures.

#### 3. Tumor Slice Auto-Detection & Quick Jump
* **Implementation:** Scans ground truth masks to find positive slice indices and provides a quick-jump selector directly to tumor slices.
* **Motivation:** Saves time by jumping straight to slices containing tumor nodules instead of manually scrolling through hundreds of empty slices.

---

### prediction_viewer.py — Model Evaluation Dashboard

#### 1. Four-Panel Comparison View
* **Implementation:** Displays side-by-side panels showing:
  1. **CT Slice** (grayscale input)
  2. **Ground Truth Mask** (green overlay with lime contours)
  3. **Predicted Mask** (red overlay with red contours)
  4. **Overlay Comparison** (Green=GT, Red=Prediction, Yellow=Overlap)
* **Motivation:** Enables instant visual diagnosis of model behavior: false positives (red-only), false negatives (green-only), and correct detections (yellow overlap).

#### 2. Per-Slice Dice Score
* **Implementation:** Computes Dice coefficient for each individual validation slice and color-codes it (green ≥0.7, orange ≥0.3, red <0.3).
* **Motivation:** Identifies which anatomical patterns or slice positions the model struggles with.

#### 3. Raw Probability Heatmap
* **Implementation:** Shows the continuous sigmoid probability output (0.0–1.0) as a hot colormap.
* **Motivation:** Reveals whether the model is "almost correct" (probabilities near threshold) or completely uncertain, guiding threshold tuning.

#### 4. Adjustable Binarization Threshold
* **Implementation:** Slider to adjust the probability threshold (default 0.5) for converting soft predictions to binary masks.
* **Motivation:** Allows exploring precision-recall tradeoffs without retraining.

#### 5. Batch Dice Summary Table
* **Implementation:** Computes and displays Dice scores for ALL validation slices in a sortable table, with separate means for tumor-positive and tumor-negative slices.
* **Motivation:** Provides aggregate model performance metrics at a glance.
