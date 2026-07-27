import os
import json
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
import src.config as config
from src.config import resolve_nifti_path

def get_preprocessed_dir():
    """
    Construiește și returnează calea către folderul de ieșire al feliilor preprocesate.
    """
    return os.path.join(config.OUTPUT_DIR, "preprocessed")

def create_patient_split():
    """
    Creează o împărțire reproductibilă a datelor la nivel de pacient (80% train, 20% validation)
    și o salvează într-un fișier JSON pentru a preveni scurgerea de date (data leakage).
    """
    # 1. Definim calea către fișierul JSON de splitare
    split_json_path = os.path.join(config.OUTPUT_DIR, "patient_split.json")
    
    # 2. Dacă fișierul de split există deja pe disc, îl încărcăm direct pentru consistență
    if os.path.exists(split_json_path):
        with open(split_json_path, 'r') as f:
            return json.load(f)
            
    # 3. Citim lista pacienților din dataset.json
    dataset_json_path = os.path.join(config.DATA_DIR, "dataset.json")
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
        
    # 4. Extragem numele curate ale cazurilor (ex: "lung_001")
    training_cases = dataset_info["training"]
    case_names = [os.path.basename(c["image"]).replace(".gz", "") for c in training_cases]
    
    # 5. Fixăm seed-ul aleator pentru reproductibilitate exactă (seed=42)
    random.seed(config.CONFIG["seed"])
    
    # 6. Amestecăm aleatoriu lista numelor pacienților
    random.shuffle(case_names)
    
    # 7. Calculăm numărul de pacienți alocați pentru validare (20%)
    num_val = int(len(case_names) * config.CONFIG["val_split"])
    val_cases = case_names[:num_val]
    train_cases = case_names[num_val:]
    
    # 8. Structurăm dicționarul de splitare
    split = {
        "train": train_cases,
        "val": val_cases
    }
    
    # 9. Salvăm dicționarul de splitare în format JSON pe disc
    with open(split_json_path, 'w') as f:
        json.dump(split, f, indent=4)
        
    print(f"Created patient-level split. Train: {len(train_cases)} patients, Val: {len(val_cases)} patients.")
    return split

def apply_lung_window(img_data):
    """
    Aplică ferestruirea Hounsfield Unit (HU) specifică plămânilor [-1000, 400] HU
    și normalizează valorile în intervalul [0.0, 1.0].
    """
    # 1. Extragem limitele ferestrei pulmonare din configurație
    hu_min = config.CONFIG["hu_min"] # -1000 HU (aer/plămân)
    hu_max = config.CONFIG["hu_max"] # +400 HU (țesut moale)
    
    # 2. Limităm (clamp/clip) valorile pixelilor între hu_min și hu_max
    windowed = np.clip(img_data, hu_min, hu_max)
    
    # 3. Normalizăm min-max pentru a scala valorile în intervalul [0.0, 1.0]
    normalized = (windowed - hu_min) / (hu_max - hu_min)
    
    # 4. Returnăm matricea convertită la float32 (potrivită pentru rețele neuronale)
    return normalized.astype(np.float32)

def crop_and_resize_slice(slice_img, slice_lbl, target_size=None):
    """
    Redimensionează felia CT 2D și masca corespunzătoare la dimensiunea țintă (ex: 192x192).
    """
    # 1. Dacă nu este specificată nicio dimensiune, o luăm din configurație
    if target_size is None:
        target_size = config.CONFIG["patch_size"]
        
    from scipy.ndimage import zoom
    
    # 2. Obținem dimensiunea curentă (h, w) a feliei
    h, w = slice_img.shape
    
    # 3. Dacă dimensiunea feliei este deja egală cu cea țintă, o returnăm direct
    if (h, w) == target_size:
        return slice_img, slice_lbl
        
    # 4. Calculăm factorii de scalare/zoom pe fiecare axă
    zoom_factors = (target_size[0] / h, target_size[1] / w)
    
    # 5. Redimensionăm imaginea CT folosind interpolare biliniară (order=1)
    resized_img = zoom(slice_img, zoom_factors, order=1, prefilter=False)
    
    # 6. Redimensionăm masca folosind interpolare cel mai apropiat vecin (order=0) pentru a păstra valorile binare
    resized_lbl = zoom(slice_lbl, zoom_factors, order=0, prefilter=False)
    
    # 7. Asigurăm binarizarea strictă a măștii (0 sau 1)
    resized_lbl = (resized_lbl > 0.5).astype(np.uint8)
    
    return resized_img, resized_lbl

def preprocess_and_slice_all():
    """
    Taie volumele CT 3D în felii axiale 2D, aplică ferestruirea HU, le redimensionează 
    și le salvează ca array-uri NumPy (.npy). Implementează un raport 1:1 de balancing între felii cu tumoare și felii sănătoase.
    """
    # 1. Obținem split-ul pacienților și folderul de ieșire
    split = create_patient_split()
    preprocessed_dir = get_preprocessed_dir()
    
    # 2. Creăm subdirectoarele pentru imaginile și etichetele de train și val
    for mode in ["train", "val"]:
        os.makedirs(os.path.join(preprocessed_dir, mode, "images"), exist_ok=True)
        os.makedirs(os.path.join(preprocessed_dir, mode, "labels"), exist_ok=True)
        
    # 3. Deschidem dataset.json
    dataset_json_path = os.path.join(config.DATA_DIR, "dataset.json")
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
        
    training_cases = dataset_info["training"]
    
    print("=== PREPROCESSING ALL CT VOLUMES & EXTRACTING 2D SLICES ===")
    
    # 4. Procesăm fiecare volum CT pacient cu pacient
    for case in tqdm(training_cases, desc="Processing cases"):
        img_rel = case["image"]
        lbl_rel = case["label"]
        case_name = os.path.basename(img_rel).replace(".gz", "")
        
        # 5. Determinăm dacă pacientul aparține setului de train sau val
        mode = "train" if case_name in split["train"] else "val"
        
        try:
            # Rezolvăm căile fișierelor NIfTI
            img_path = resolve_nifti_path(img_rel)
            lbl_path = resolve_nifti_path(lbl_rel)
            
            # Încărcăm obiectele NIfTI
            img = nib.load(img_path)
            lbl = nib.load(lbl_path)
            
            # Extragem matricele 3D ca numpy arrays
            img_data = np.asanyarray(img.dataobj)
            lbl_data = np.asanyarray(lbl.dataobj)
            
            # Aplicăm ferestruirea pulmonară Hounsfield [-1000, 400] HU
            normalized_img = apply_lung_window(img_data)
            num_slices = normalized_img.shape[2]
            
            # Colectăm indecșii feliilor cu tumoare și ai celor fără tumoare
            positive_slice_indices = []
            negative_slice_indices = []
            
            for s in range(num_slices):
                if np.sum(lbl_data[:, :, s] == 1) > 0:
                    # Felia conține cel puțin 1 pixel de tumoare
                    positive_slice_indices.append(s)
                else:
                    # Felia nu are tumoare, dar filtrăm feliile goale din afara corpului (aer complet)
                    if np.mean(normalized_img[:, :, s]) > 0.1:
                        negative_slice_indices.append(s)
            
            # 6. Egalizăm numărul de felii negative pentru a echilibra setul de date (Balancing 1:1)
            if mode == "train":
                num_negatives_to_keep = len(positive_slice_indices)
            else:
                num_negatives_to_keep = len(positive_slice_indices) * 2
                
            sampled_negatives = []
            if negative_slice_indices and num_negatives_to_keep > 0:
                state = random.getstate()
                random.seed(config.CONFIG["seed"])
                sampled_negatives = random.sample(
                    negative_slice_indices, 
                    min(len(negative_slice_indices), num_negatives_to_keep)
                )
                random.setstate(state)
                
            # Combinăm feliile pozitive cu cele negative eșantionate
            slices_to_save = positive_slice_indices + sampled_negatives
            
            # 7. Salvăm feliile individuale ca fișiere .npy
            for s in slices_to_save:
                slice_img = normalized_img[:, :, s]
                slice_lbl = lbl_data[:, :, s]
                
                # Redimensionăm la 192x192
                resized_img, resized_lbl = crop_and_resize_slice(slice_img, slice_lbl)
                
                img_save_path = os.path.join(preprocessed_dir, mode, "images", f"{case_name}_slice_{s}.npy")
                lbl_save_path = os.path.join(preprocessed_dir, mode, "labels", f"{case_name}_slice_{s}.npy")
                
                # Salvăm matricele pe disc
                np.save(img_save_path, resized_img.astype(np.float32))
                np.save(lbl_save_path, resized_lbl.astype(np.uint8))
                
        except Exception as e:
            print(f"Error preprocessing case {case_name}: {e}")
            
    print(f"Preprocessed slices saved successfully to: {preprocessed_dir}")

# Punctul de intrare pentru rulare din linia de comandă
if __name__ == "__main__":
    preprocess_and_slice_all()
