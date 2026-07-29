import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandZoomd,
    EnsureChannelFirstd,
    ToTensord
)
import src.config as config

class LungSlicesDataset(Dataset):
    """
    Clasă custom PyTorch Dataset pentru încărcarea eficientă a feliilor 2D preprocesate 
    și a măștilor de tumoare salvare ca fișiere .npy.
    """
    def __init__(self, mode="train", transforms=None):
        # 1. Stocăm modul ("train" sau "val") și lanțul de transformări MONAI
        self.mode = mode
        self.transforms = transforms
        
        # 2. Definim calea folderului de bază pentru imagini și etichete
        self.base_dir = os.path.join(config.OUTPUT_DIR, "preprocessed", mode)
        
        # 3. Colectăm toate căile fișierelor de imagini .npy sortate alfabetic
        self.image_paths = sorted(glob.glob(os.path.join(self.base_dir, "images", "*.npy")))
        
        # 4. Potrivim fiecare imagine cu masca sa corespunzătoare
        self.label_paths = []
        for img_path in self.image_paths:
            base_name = os.path.basename(img_path)
            lbl_path = os.path.join(self.base_dir, "labels", base_name)
            
            # Verificăm dacă masca există; dacă nu, aruncăm o eroare explicită
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"Missing corresponding label for image: {img_path}")
            self.label_paths.append(lbl_path)
            
        print(f"Loaded {len(self.image_paths)} slices for mode={mode}")

    def __len__(self):
        """
        Returnează numărul total de felii din dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Încarcă o felie 2D și masca sa la indexul specificat, aplică transformările 
        și le returnează sub formă de tensori PyTorch.
        """
        # 1. Încărcăm matricea feliei CT și a măștii cu conversie explicită la float32
        image = np.load(self.image_paths[idx]).astype(np.float32)
        label = np.load(self.label_paths[idx]).astype(np.float32)
        
        # 2. Structurăm datele într-un dicționar cerut de transformările MONAI bazate pe chei
        data_dict = {
            "image": image,
            "label": label
        }
        
        # 3. Aplicăm lanțul de transformări (augmentări, formatare canal, conversie la tensor)
        if self.transforms:
            data_dict = self.transforms(data_dict)
            
        # 4. Returnăm imaginea și eticheta gata pentru rețeaua neuronală
        return data_dict["image"], data_dict["label"]

def get_train_transforms():
    """
    Construiește și returnează lanțul de transformări MONAI pentru ANTRENARE,
    incluzând formatarea canalului, oglindirile aleatorii, rotațiile și zoom-ul.
    """
    return Compose([
        # Adaugă dimensiunea de canal [1, H, W] necesară rețelei neuronale
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        
        # Oglindire aleatorie pe axa verticală (probabilitate 50%)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        
        # Oglindire aleatorie pe axa orizontală (probabilitate 50%)
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        
        # Rotație aleatorie între -0.3 și +0.3 radiani (~17 grade)
        RandRotated(keys=["image", "label"], range_x=0.3, prob=0.5, mode=["bilinear", "nearest"]),
        
        # Zoom/Scalare aleatorie între 80% și 120% din dimensiune
        RandZoomd(keys=["image", "label"], min_zoom=0.8, max_zoom=1.2, prob=0.3, mode=["bilinear", "nearest"]),
        
        # Convertește array-urile NumPy în tensori PyTorch
        ToTensord(keys=["image", "label"])
    ])

def get_val_transforms():
    """
    Construiește lanțul de transformări pentru VALIDARE (fără augmentări aleatorii, doar canal și conversie tensor).
    """
    return Compose([
        # Adaugă dimensiunea de canal [1, H, W]
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        
        # Convertește la tensori PyTorch
        ToTensord(keys=["image", "label"])
    ])

def get_dataloaders(batch_size=None):
    """
    Construiește și returnează obiectele PyTorch DataLoader pentru setul de Antrenare și Validare.
    """
    # 1. Dacă nu este specificată o dimensiune de batch, o luăm din configurația globală
    if batch_size is None:
        batch_size = config.CONFIG["batch_size"]
        
    # 2. Instanțiem dataset-ul de train cu augmentări și cel de val fără augmentări
    train_dataset = LungSlicesDataset(mode="train", transforms=get_train_transforms())
    val_dataset = LungSlicesDataset(mode="val", transforms=get_val_transforms())
    
    # 3. Creăm DataLoader-ul de antrenare (cu amestecare aleatorie a datelor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # 4. Creăm DataLoader-ul de validare (fără amestecare)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Punct de testare pentru verificarea funcționării corecte a DataLoader-ului
if __name__ == "__main__":
    print("=== Testing Dataset & DataLoader ===")
    train_loader, val_loader = get_dataloaders(batch_size=4)
    images, labels = next(iter(train_loader))
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
