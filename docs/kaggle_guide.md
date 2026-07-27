# How to Upload & Run Your Training on Kaggle GPU

This guide provides a step-by-step walkthrough to train your 2D U-Net model on **Kaggle's free GPU (NVIDIA T4 / P100)** in under 3 minutes.

---

## Step 1: Create a New Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com/) and log into your account.
2. Click **Create** ➔ **New Notebook**.
3. Give your notebook a title (e.g., `Lung Cancer 2D UNet Training`).

---

## Step 2: Enable GPU Acceleration & Add Datasets

1. On the right-hand sidebar under **Notebook Options**:
   * Click **Accelerator** (default is `None`).
   * Select **GPU T4 x2** or **GPU P100**.
2. Add the medical CT dataset:
   * Click **+ Add Input** at the top right.
   * Search for: `medical segmentation decathlon lung` (or dataset `vivekprajapati2048/medical-segmentation-decathlon-lung`).
   * Click the **+** (Add) button next to the dataset.
3. Add your code zip archive:
   * Click **+ Add Input** ➔ **Upload New Dataset**.
   * Upload `lung_segmentation_code.zip` from your local machine (**[lung_segmentation_code.zip](file:///d:/Decathlon_Lung/lung_segmentation_code.zip)**).
   * Title the dataset `lung-code`.

---

## Step 3: Cell 1 - Environment & Code Synchronization

Paste this cell to install dependencies and auto-copy code files into `/kaggle/working/`:

```python
# Cell 1: Install Requirements & Sync Code Files
!pip install -q monai nibabel SimpleITK

import os, sys, shutil

# Auto-locate where train.py is located under /kaggle/input
code_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'train.py' in files:
        code_dir = root
        break

print("Found code directory at:", code_dir)

if code_dir:
    for item in os.listdir(code_dir):
        s = os.path.join(code_dir, item)
        d = os.path.join('/kaggle/working', item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copyfile(s, d)
    print("✓ Successfully synchronized code to /kaggle/working/")
```

---

## Step 4: Cell 2 - Auto-Locate CT Dataset, Preprocess & GPU Training

Paste this cell to set the dataset path dynamically and launch GPU training:

```python
# Cell 2: Auto-detect CT Dataset, Preprocess & Train on GPU
import os, sys
import src.config as config

# Auto-locate dataset.json under /kaggle/input
for root, dirs, files in os.walk('/kaggle/input'):
    if 'dataset.json' in files:
        config.set_data_dir(root)
        print("✓ Kaggle CT Dataset set to:", config.DATA_DIR)
        break

# Ensure preprocessing and training import the updated config
import src.stage2_preprocessing.preprocessing as preprocessing
preprocessing.config = config

# Step 1: Run Preprocessing on Kaggle
print("\n--- STAGE 2: PREPROCESSING ---")
preprocessing.preprocess_and_slice_all()

# Step 2: Run Training Pipeline on GPU
print("\n--- STAGE 3: TRAINING BASELINE 2D U-NET ---")
from train import train_model
train_model()
```

Kaggle's GPU will complete 20 epochs of training in **under 2 minutes**!

---

## Step 5: Download the Trained Model Weights

1. Look at the right-hand sidebar under **Output** (`/kaggle/working/output/`).
2. Locate `best_metric_model.pth`.
3. Click the **three dots (...)** next to `best_metric_model.pth` ➔ Click **Download**.
4. Place `best_metric_model.pth` into your local `d:\Decathlon_Lung\output\` directory on your laptop.
