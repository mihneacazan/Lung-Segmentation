# 3D CT Lung Nodule Segmentation with 2D U-Net and MONAI

A deep learning project for automated binary segmentation of lung nodules from 3D volumetric CT scans using PyTorch and MONAI. The model is trained and evaluated on the Medical Segmentation Decathlon (Task06 Lung) dataset.

---

## Project Overview

This repository implements a baseline 2D U-Net segmentation pipeline for lung CT scans. The primary objective is to evaluate lightweight 2D architectures on volumetric medical imaging while mitigating spatial class imbalance and patient data leakage.

Key technical aspects:
* **Dataset:** 63 volumetric 3D CT scans from Medical Segmentation Decathlon (Task06_Lung).
* **Architecture:** 2D U-Net (1.5M parameters) with residual blocks and skip connections.
* **Preprocessing:** Patient-level zero-leakage split, Hounsfield Unit (HU) windowing ($[-1000, 400]$), 192x192 spatial normalization, and 1:1 positive-to-negative slice balancing.
* **Loss Function:** `DiceCELoss` (combination of Dice Loss and Binary Cross-Entropy).
* **Results:** Reached a validation Sørensen-Dice coefficient of **0.4978 (~50% overlap)** on unseen test patients over 50 epochs.

---

## Repository Structure

```text
Decathlon_Lung/
├── docs/                         # Theoretical & Analytical Documentation
│   ├── eda_report.md             # Dataset Exploratory Analysis Report
│   └── unet_and_metrics_study.md # U-Net Architecture & Metric Formulations
├── output/                       # Training Artifacts & Visualizations
│   ├── eda_figures/              # EDA visualization plots
│   ├── eda_statistics.csv        # Per-patient dataset statistics
│   ├── patient_split.json        # Patient-level train/validation split
│   ├── training_metrics.csv      # Per-epoch metric history
│   └── training_curves.png       # Loss & Dice convergence curves
├── src/                          # Project Python Package
│   ├── config.py                 # Central configurations & path resolution
│   ├── eda/                      # Exploratory Data Analysis routines
│   │   └── eda_report.py
│   ├── preprocessing/            # HU windowing, slicing & PyTorch DataLoader
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/                   # U-Net Neural Network Architectures
│   │   └── unet_2d.py
│   └── visualization/            # Streamlit Dashboards
│       ├── visualizer.py         # CT Volume Explorer
│       └── prediction_viewer.py  # Prediction Evaluation Dashboard
├── train.py                      # Main entrypoint script for model training
└── README.md                     # Project documentation
```

---

## Pipeline & Architecture Details

### Preprocessing & Data Pipeline
1. **HU Windowing:** CT voxel intensities are clamped to $[-1000, 400]$ HU (Lung Window) to isolate lung tissue, followed by min-max normalization to $[0.0, 1.0]$.
2. **Patient-Level Split:** 51 patients allocated for training (80%) and 12 patients for validation (20%) to guarantee zero patient data leakage.
3. **Slice Balancing:** Tumor-positive and negative slices are sampled at a 1:1 ratio to prevent the network from converging to a trivial background predictor.

### Model Architecture
The 2D U-Net uses the following configuration:
* Input dimensions: `[Batch, 1, 192, 192]`
* Encoder-Decoder Channels: `(16, 32, 64, 128, 256)`
* Residual units: 2 residual units per resolution block ($y = f(x) + x$) for stable gradient flow.
* Skip connections: Direct feature concatenation between encoder and decoder resolution levels.

---

## Validation Results

| Metric | Target | Result |
| :--- | :---: | :---: |
| **Best Validation Dice Score** | > 0.40 | **0.4978** |
| **Train Loss (`DiceCELoss`)** | < 1.00 | **0.7404** |
| **Validation Loss (`DiceCELoss`)** | < 1.00 | **0.8891** |

---

## Installation & Execution

### Prerequisites
* Python 3.11+
* PyTorch 2.1+
* MONAI, Nibabel, SimpleITK, Streamlit, Pandas, Matplotlib

### Setup & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mihneacazan/Decathlon_Lung_Segmentation.git
   cd Decathlon_Lung_Segmentation
   ```

2. **Run Model Training:**
   ```bash
   python train.py
   ```

3. **Run Interactive Prediction Viewer (Streamlit):**
   ```bash
   streamlit run src/visualization/prediction_viewer.py
   ```
