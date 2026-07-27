# Implementation Plan: Lung Cancer Segmentation (MSD Lung & LIDC-IDRI)

This document describes the architecture and step-by-step roadmap for the medical imaging artificial intelligence project (lung tumor segmentation from CT scans), in accordance with the internship/project requirements.

---

## 1. Concepts Clarification

### What is MONAI?
**MONAI** (*Medical Open Network for AI*) is an open-source PyTorch-based framework developed by NVIDIA and King's College London specifically for Deep Learning in medical imaging.
* **Why we use it:**
  * **Native 3D Loading:** Built-in support for `.nii`, `.nii.gz`, and DICOM formats.
  * **Advanced Medical Preprocessing:** Hounsfield scale normalization (e.g. `ScaleIntensityRanged` for lung windowing: HU [-1000, 400]), automated anatomical cropping (`CropForegroundd`), and 3D affine transforms.
  * **Pre-implemented Architectures:** `UNet`, `AttentionUNet`, `SegResNet`, `UNETR`, `SwinUNETR`.
  * **Medical Metrics:** `DiceMetric`, `HausdorffDistanceMetric`.

### What does "working 2D at first" mean?
CT scans are **3D volumes** (stacks of hundreds of 2D axial slices).
* A 3D model (e.g., 3D U-Net) processes the entire 3D volume at once, but consumes a large amount of GPU memory (VRAM) and trains slowly.
* **2D Processing** means extracting individual 2D axial slices (e.g., $512 \times 512$ pixels) from the 3D volume.
* We train a 2D U-Net model on these individual slices. At evaluation time, we pass all slices of a patient through the model and reassemble the predictions back into a 3D volume to compute the Dice score on the entire patient.
* **Benefits:** Faster training, low VRAM requirements, and easier visualization/debugging.

### How does your work connect with your colleague's (MSD vs. LIDC-IDRI)?
* **Your Dataset (MSD Lung - Task06):** Contains 63 training CT volumes with a single consensus mask per patient.
* **Colleague's Dataset (LIDC-IDRI):** Contains over 1000 cases where nodules were independently annotated by up to **4 different radiologists**.
* **Integration Points:**
  * Your colleague will handle multi-annotator mask analysis (union, intersection, majority voting, consensus masks).
  * In the end, you will perform **Cross-Dataset Evaluation**: your model trained on MSD will be evaluated on your colleague's LIDC-IDRI data, and vice-versa, to test generalization performance on external, multi-source data.

---

## 2. Repository Structure & Proposed Architecture

The code will be organized as a clean, reproducible, and modular Python module:

```text
Decathlon_Lung/
├── archive/                  # Raw MSD Lung dataset (imagesTr, labelsTr, imagesTs, dataset.json)
├── src/
│   ├── __init__.py
│   ├── config.py             # Hyperparameters & path configurations
│   ├── dataset.py            # PyTorch/MONAI Dataset loader for 2D slices / 3D volumes
│   ├── preprocessing.py      # HU windowing, normalization, cropping, and slicing
│   ├── models/               # Model definitions (2D U-Net, Attention U-Net, SegResNet)
│   │   ├── unet_2d.py
│   │   └── segresnet.py
│   ├── metrics.py            # Dice, IoU, Sensitivity, Specificity, and Failure Rate calculations
│   ├── utils.py              # Visualizations (CT overlay + Ground Truth vs. Prediction)
│   └── inference.py          # Inference script for new cases
├── train.py                  # Training & validation loop pipeline
├── evaluate.py               # Test set evaluation and metrics reporter
├── eda_report.py             # Dataset statistics compiler and plotter script
├── app/                      # Interactive Web App (Streamlit)
│   └── app.py
├── requirements.txt          # Virtual environment dependencies
├── README.md                 # Setup & execution instructions
└── implementation_plan.md    # Project design and roadmap document (this file)
```

---

## 3. Step-by-Step Roadmap

### Stage 1: Setup & Exploratory Data Analysis (EDA)
* Configure environment (`PyTorch`, `MONAI`, `SimpleITK`, `Nibabel`, `Streamlit`, `Matplotlib`).
* Create `eda_report.py` to analyze volumes (voxel dimensions, spacing, Hounsfield Unit distribution, tumor volume vs. background ratio).
* Identify tumor-positive and tumor-negative slices.

### Stage 2: Preprocessing & 2D/3D Dataset (MONAI Pipeline)
* Apply **Hounsfield windowing (Lung Window)**: clamp HU values to $[-1000, 400]$ and normalize to $[0, 1]$.
* Extract 2D slices (implement positive-slice sampling strategy with a controlled ratio of negative slices).
* Implement reproductible **patient-level split** (80% training, 20% validation) to prevent data leakage.
* Add MONAI spatial and intensity augmentations (rotation, flip, Gaussian noise, contrast adjustment).

### Stage 3: Baseline 2D U-Net Training
* Build a 2D U-Net using MONAI (`monai.networks.nets.UNet`).
* Use **DiceCELoss** to handle the severe background/foreground class imbalance.
* Optimizer: `AdamW` with a learning rate scheduler (`CosineAnnealingLR`).
* Monitor training/validation loss and validation Dice score. Save checkpoints for the best performing model.

### Stage 4: Advanced Architectures & Experiments
* Implement and compare **Attention U-Net** and **SegResNet** models, or **2.5D U-Net** (using 3 adjacent slices).
* Save experiment run metrics and hyperparameter configurations (`.yaml` / `.json`).
* Compile a performance comparison matrix (Dice, IoU, Sensitivity, Precision, Failure Rate).

### Stage 5: Evaluation & Cross-Dataset Evaluation
* Evaluate the final models on the MSD Lung internal test partition.
* Perform cross-dataset evaluation: test the MSD model on the LIDC-IDRI dataset prepared by your colleague, and vice-versa.
* Analyze failure modes (stray false positives, missing small nodules).

### Stage 6: Interactive Web Demo
* Develop a **Streamlit** web application.
* Interactivity features:
  * Select case / patient ID.
  * Axial slice viewer slider.
  * Toggle checkboxes for: Ground Truth Mask, Model Prediction.
  * Compare multiple models and configurations with live Dice/IoU indicators.

### Stage 7: Technical Report & Presentation
* Compile the technical report following a scientific paper structure.
* Embed validation curve charts and overlay visualization figures.

---

## 4. Verification & Validation Plan

### Automated Tests
1. **Performance Metrics:**
   * **Dice Similarity Coefficient (DSC)**
   * **Intersection over Union (IoU)**
   * **Sensitivity (Recall) & Precision**
   * **Failure Rate:** percentage of cases with Dice < 0.1.
2. **Pipeline Checks:**
   * Verify slice-to-mask alignment checks.
   * Add dataloader integrity sanity tests.

### Manual Verification
* Run the Streamlit interface and visually inspect model segmentation contours against ground truth masks for at least 10 validation patients.
