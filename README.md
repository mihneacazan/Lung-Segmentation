# Lung Tumor Segmentation from CT Images (MSD Lung Task06)

This repository contains a modular, stage-based experimental pipeline for automatic semantic segmentation of lung cancer from 3D volumetric CT images using **PyTorch** and **MONAI**.

---

## 📁 Repository Structure

```text
Decathlon_Lung/
├── archive/                      # Raw dataset (imagesTr, labelsTr, imagesTs, dataset.json)
├── docs/                         # Project Documentation & Study Documents
│   ├── implementation_plan.md    # Project roadmap & planning
│   ├── eda_report.md             # Exploratory Data Analysis report
│   ├── unet_and_metrics_study.md # U-Net & medical metrics study
│   └── kaggle_guide.md           # Kaggle GPU execution guide
├── output/                       # Output artifacts (ignored by git, generated at runtime)
│   ├── eda_figures/
│   ├── preprocessed/
│   ├── patient_split.json
│   ├── training_metrics.csv
│   └── best_metric_model.pth
├── src/                          # Modular Python Source Code (Organized by Stages)
│   ├── config.py                 # Global configurations & paths
│   ├── stage1_eda/               # Stage 1: Exploratory Data Analysis
│   │   └── eda_report.py
│   ├── stage2_preprocessing/     # Stage 2: Preprocessing & Data Loaders
│   │   ├── preprocessing.py      # Windowing, cropping, patient-split & 2D slicing
│   │   └── dataset.py            # PyTorch / MONAI Dataset & DataLoader
│   ├── stage3_models/            # Stage 3 & 4: Model Architectures
│   │   └── unet_2d.py            # 2D U-Net baseline model
│   └── stage6_visualization/     # Stage 6: Interactive Demo & Visualization
│       └── visualizer.py         # Streamlit web app
├── train.py                      # Main entrypoint script for model training
├── requirements.txt              # Environment dependencies
├── task.md                       # Task checklist
└── README.md                     # Main entry Readme
```

---

## ⚙️ Installation and Setup

### 1. Prerequisites
* Windows 10/11
* Python 3.11.x (official CPython) installed

### 2. Set Up Virtual Environment
Open a PowerShell terminal in the project root directory:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
Install all required packages:
```powershell
pip install -r requirements.txt
```

---

## 📊 How to Run the Pipeline Stages

### Stage 1: Exploratory Data Analysis (EDA)
```powershell
python -m src.stage1_eda.eda_report
```

### Stage 2: Data Preprocessing & Slicing
```powershell
python -m src.stage2_preprocessing.preprocessing
```

### Stage 3: Baseline 2D U-Net Training
```powershell
python train.py
```

### Stage 6: Interactive Web App (Streamlit)
```powershell
streamlit run src/stage6_visualization/visualizer.py
```
