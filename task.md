# Project Execution Tasks

- `[x]` **Stage 1: Setup & Exploratory Data Analysis (EDA)**
  - `[x]` Initialize Git repository and `.gitignore`
  - `[x]` Install and configure Python 3.11 virtual environment with PyTorch & MONAI
  - `[x]` Create optimized `eda_report.py` and run full stats on all 63 patients
  - `[x]` Write `unet_and_metrics_study.md` (in English)
  - `[x]` Write `eda_report.md` (in English)
  - `[x]` Create interactive visualizer application (`visualizer.py`)

- `[x]` **Stage 2: Preprocessing & Data Pipeline**
  - `[x]` Define patient-level train/validation split script with zero patient overlap (`output/patient_split.json`)
  - `[x]` Create `src/preprocessing.py` implementing Hounsfield windowing, cropping, and normalizations
  - `[x]` Create `src/dataset.py` with custom PyTorch/MONAI Dataset loader for 2D slices
  - `[x]` Verify correctness of the preprocessing pipeline (checked tensor batch shapes `[4, 1, 192, 192]`)

- `[ ]` **Stage 3: Implement & Train Baseline 2D U-Net**
  - `[ ]` Set up PyTorch/MONAI training script (`train.py`)
  - `[ ]` Configure loss function (`DiceCELoss`), optimizer (`AdamW`), and Cosine Annealing learning rate scheduler
  - `[ ]` Implement training loop with automatic checkpoints saving best validation Dice
  - `[ ]` Plot training and validation loss curves and validation Dice curves

- `[ ]` **Stage 4: Advanced Models & Experiments**
  - `[ ]` Implement Attention U-Net model option in training configurations
  - `[ ]` Implement SegResNet model option in training configurations
  - `[ ]` Save experimental hyperparameter config files in `.json` / `.yaml` formats
  - `[ ]` Generate comparative table of models (Dice, IoU, Sensitivity, Precision, Failure Rate)

- `[ ]` **Stage 5: Evaluation & Cross-Dataset Evaluation**
  - `[ ]` Create evaluation script (`evaluate.py`) for validation on test partitions
  - `[ ]` Adapt evaluation helper to support multi-annotator consensus masks for cross-dataset evaluation with LIDC-IDRI

- `[ ]` **Stage 6: Interactive Web App Demo**
  - `[ ]` Integrate model inference into the Streamlit visualizer to allow users to compare predictions and real masks
  - `[ ]` Enable multi-model and multi-annotator visual comparisons

- `[ ]` **Stage 7: Final Documentation & Technical Report**
  - `[ ]` Write technical paper draft summarizing final results, metrics, and limitations
