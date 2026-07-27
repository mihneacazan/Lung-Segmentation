import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global defaults
DATA_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

IMAGES_TR_DIR = os.path.join(DATA_DIR, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_DIR, "labelsTr")
IMAGES_TS_DIR = os.path.join(DATA_DIR, "imagesTs")

# Training configurations (baseline 2D U-Net)
CONFIG = {
    "data_dir": DATA_DIR,
    "images_tr": IMAGES_TR_DIR,
    "labels_tr": LABELS_TR_DIR,
    "output_dir": OUTPUT_DIR,
    "seed": 42,
    "val_split": 0.2,
    "patch_size": (192, 192),
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 50,
    "hu_min": -1000.0,
    "hu_max": 400.0,
}

def set_data_dir(new_dir):
    """
    Dynamically updates DATA_DIR and all dependent directory paths.
    """
    global DATA_DIR, IMAGES_TR_DIR, LABELS_TR_DIR, IMAGES_TS_DIR, OUTPUT_DIR
    DATA_DIR = new_dir
    CONFIG["data_dir"] = new_dir
    IMAGES_TR_DIR = os.path.join(DATA_DIR, "imagesTr")
    LABELS_TR_DIR = os.path.join(DATA_DIR, "labelsTr")
    IMAGES_TS_DIR = os.path.join(DATA_DIR, "imagesTs")
    CONFIG["images_tr"] = IMAGES_TR_DIR
    CONFIG["labels_tr"] = LABELS_TR_DIR
    
    if os.path.exists("/kaggle/input"):
        OUTPUT_DIR = "/kaggle/working/output"
        CONFIG["output_dir"] = OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-detect Kaggle environment vs local environment
if os.path.exists("/kaggle/input"):
    kaggle_data_dir = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "dataset.json" in files:
            kaggle_data_dir = root
            break
    if kaggle_data_dir:
        set_data_dir(kaggle_data_dir)

def resolve_nifti_path(relative_path):
    """
    Resolves the actual path of a NIfTI file from dataset.json.
    Handles situations where NIfTI files are stored inside folders named after them.
    """
    clean_rel = relative_path.lstrip("./")
    abs_path = os.path.join(DATA_DIR, clean_rel)
    
    if os.path.isfile(abs_path):
        return abs_path
        
    if abs_path.endswith(".gz") and os.path.isfile(abs_path[:-3]):
        return abs_path[:-3]
        
    base_name = os.path.basename(clean_rel).split(".")[0]
    dir_name = os.path.dirname(clean_rel)
    
    possible_dir = os.path.join(DATA_DIR, dir_name, f"{base_name}.nii")
    if os.path.isdir(possible_dir):
        possible_file = os.path.join(possible_dir, f"{base_name}.nii")
        if os.path.isfile(possible_file):
            return possible_file
            
    possible_dir_gz = os.path.join(DATA_DIR, dir_name, f"{base_name}.nii.gz")
    if os.path.isdir(possible_dir_gz):
        possible_file = os.path.join(possible_dir_gz, f"{base_name}.nii.gz")
        if os.path.isfile(possible_file):
            return possible_file
            
    parent_dir = os.path.join(DATA_DIR, os.path.dirname(clean_rel))
    for root, dirs, files in os.walk(parent_dir):
        for f in files:
            if f.startswith(base_name) and (f.endswith(".nii") or f.endswith(".nii.gz")):
                return os.path.join(root, f)
                
    raise FileNotFoundError(f"Could not resolve NIfTI path for: {relative_path} (resolved path: {abs_path})")
