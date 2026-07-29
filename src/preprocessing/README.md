# Stage 2: Data Preprocessing & PyTorch Dataloaders

This directory contains the code and justification for data cleaning, Hounsfield Unit (HU) windowing, patient-level splitting, 2D slice extraction, and MONAI-augmented PyTorch DataLoaders.

---

## 📄 File Overview

* **`preprocessing.py`**: Handles windowing, patient-level splitting, 2D slice extraction, and dataset balancing.
* **`dataset.py`**: Custom PyTorch `Dataset` and MONAI transformation pipelines (`DataLoader`).

---

## 🔍 Code Walkthrough & Design Motivations

### 1. Patient-Level Split (`create_patient_split`)
* **Implementation:** Splits the 63 patients into 51 training cases (~80%) and 12 validation cases (~20%) using a fixed random seed (`seed=42`). Saves the mapping to `output/patient_split.json`.
* **Motivation (Zero-Overlap Guarantee):** Splitting data at the slice level would cause slices from the same patient to appear in both training and validation sets (*data leakage*), leading to artificially high, ungeneralizable validation scores. Splitting at the patient level guarantees zero patient overlap.

### 2. Hounsfield Unit Windowing (`apply_lung_window`)
* **Implementation:** Clamps voxel values to $[-1000, 400]$ HU (Lung Window) and scales them linearly to $[0.0, 1.0]$.
* **Motivation:** CT scans record physical attenuation from $-1000$ HU (air) up to $+3000$ HU (dense bone/metal). Lung tissue and tumors exist strictly in the $[-1000, 400]$ range. Clipping irrelevant bone/metal signals enhances tumor contrast and scales network inputs to standard $[0, 1]$ floating-point range.

### 3. Pre-slicing to Disk vs. Dynamic 3D Loading
* **Implementation:** `preprocess_and_slice_all()` extracts 2D axial slices ($192\times192$) and saves them as lightweight binary `.npy` files under `output/preprocessed/`.
* **Motivation (I/O Bottleneck Removal):** Opening a 300MB 3D NIfTI file on every batch iteration during training causes severe disk bottlenecks. Pre-slicing into 140KB `.npy` files makes batch loading instant, accelerating epoch training speeds by ~10x-50x.

### 4. 1:1 Class Imbalance Slice Balancing
* **Implementation:** Keeps all tumor-positive slices and samples a matching 1:1 ratio of negative slices (filtered to exclude empty air).
* **Motivation:** Without balancing, 90%+ of slices would be empty background, training the model to trivially output all zeros. The 1:1 ratio forces the CNN to learn discriminative features separating healthy lung tissue from tumor nodules.

### 5. MONAI Data Transformations & Augmentation (`get_train_transforms`)
* **Transforms Used:**
  * `EnsureChannelFirstd`: Adds explicit channel dimension `[1, H, W]`.
  * `RandFlipd` (X & Y axes): Simulates anatomical symmetry variations.
  * `RandRotated` & `RandZoomd`: Simulates patient positioning and scanner scaling variations.
  * `ToTensord`: Converts numpy arrays to PyTorch tensors.
* **Motivation:** Data augmentation prevents neural network overfitting on small medical cohorts.
