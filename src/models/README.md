# Stage 3: Model Architectures

This directory contains the neural network model definitions for 2D semantic segmentation.

---

## 📄 File Overview

* **`unet_2d.py`**: Baseline 2D U-Net model implementation using MONAI's `UNet`.

---

## 🔍 Code Walkthrough & Design Motivations

### 1. 2D U-Net Architecture (`build_unet_2d`)
* **Implementation:** Built using `monai.networks.nets.UNet` with parameters:
  * `spatial_dims=2`: Operates on 2D axial CT slices.
  * `in_channels=1`: Single-channel grayscale CT slice input.
  * `out_channels=1`: Single-channel binary probability map for tumor segmentation.
  * `channels=(16, 32, 64, 128, 256)`: Feature channel progression across 5 resolution levels.
  * `strides=(2, 2, 2, 2)`: $2\times2$ max-pooling downsampling.
  * `num_res_units=2`: 2 residual convolutional units per resolution block.
  * `dropout=0.1`: Regularization to prevent feature co-adaptation.

### 2. Why 2D U-Net for Baseline?
* **Low Memory Footprint:** Operates efficiently on low-memory GPUs and CPUs.
* **Fast Convergence:** Learns 2D spatial feature representations quickly.
* **Skip Connections:** Concatenates high-resolution encoder features with decoder features, preserving sharp tumor boundary localization.
