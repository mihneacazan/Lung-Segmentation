import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.config import resolve_nifti_path, OUTPUT_DIR, DATA_DIR

def run_eda():
    """
    Rulează Analiza Exploratorie a Datelor (EDA) pe toate tomografiile (CT) 3D și măștile de tumoare.
    Extrage statistici geometrice, de intensitate HU, volum de tumoare și generează grafice.
    """
    print("=== STARTING FULL EXPLORATORY DATA ANALYSIS (EDA) ===")
    
    # 1. Construim calea către fișierul dataset.json care conține metadatele setului de date
    dataset_json_path = os.path.join(DATA_DIR, "dataset.json")
    
    # Deschidem și citim fișierul JSON cu structura cazurilor medicale
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
        
    # Extragem lista cazurilor de antrenare (imagini CT + măști)
    training_cases = dataset_info["training"]
    
    # Afișăm numărul total de cazuri găsite în dataset
    print(f"Total training cases listed: {len(training_cases)}")
    print(f"Total test cases listed: {len(dataset_info['test'])}")
    
    # Lista în care vom colecta dicționarele de statistici pentru fiecare pacient
    stats_list = []
    
    # Construim folderul în care vom salva graficele generate de EDA
    figures_dir = os.path.join(OUTPUT_DIR, "eda_figures")
    
    # Asigurăm crearea folderului eda_figures pe disc dacă nu există deja
    os.makedirs(figures_dir, exist_ok=True)
    
    # Iterăm prin fiecare caz de antrenare din lista training_cases folosind o bară de progres tqdm
    for case in tqdm(training_cases, desc="Analyzing all CT volumes & masks"):
        # Extragem căile relative pentru imaginea CT și masca de tumoare
        image_rel = case["image"]
        label_rel = case["label"]
        
        try:
            # Rezolvăm calea absolută pe disc pentru fișierele NIfTI (.nii sau .nii.gz)
            image_path = resolve_nifti_path(image_rel)
            label_path = resolve_nifti_path(label_rel)
            
            # Încărcăm antetul și obiectul NIfTI în memorie folosind nibabel (fără a încărca încă tot matricea pixelilor)
            img = nib.load(image_path)
            lbl = nib.load(label_path)
            
            # Obținem dimensiunile 3D ale volumului CT (Lățime, Înălțime, Număr de felii)
            shape = img.shape
            
            # Extragem voxel spacing-ul (dimensiunea fizică în milimetri a fiecărui voxel pe axele X, Y, Z)
            spacing = img.header.get_zooms()
            
            # Calculăm volumul fizic al unui singur voxel în milimetri cubi (mm^3)
            voxel_vol = np.prod(spacing)
            
            # Încărcăm datele reale de pixeli/voxeli ca array-uri NumPy (fără conversie inutilă la float64 pentru eficiență de RAM)
            img_data = np.asanyarray(img.dataobj)
            lbl_data = np.asanyarray(lbl.dataobj)
            
            # Calculăm valorile minime, maxime, media și deviația standard a intensităților Hounsfield (HU)
            min_val = float(np.min(img_data))
            max_val = float(np.max(img_data))
            mean_val = float(np.mean(img_data))
            std_val = float(np.std(img_data))
            
            # Calculăm percentilele de 5% și 95% pentru a elimina valorile extreme/artefactele CT
            p5 = float(np.percentile(img_data, 5))
            p95 = float(np.percentile(img_data, 95))
            
            # Numărăm câți voxeli din mască au valoarea 1 (reprezentând tumoarea)
            tumor_voxels = int(np.sum(lbl_data == 1))
            
            # Calculăm volumul total al tumorii în mm^3 (număr voxeli × volumul unui voxel)
            tumor_volume_mm3 = tumor_voxels * voxel_vol
            
            # Calculăm numărul total de voxeli ai volumului 3D
            total_voxels = np.prod(shape)
            
            # Calculăm procentul ocupat de tumoare din întregul volum CT
            tumor_ratio = (tumor_voxels / total_voxels) * 100
            
            # Extragem numărul total de felii axiale 2D (axa Z)
            num_slices = shape[2]
            
            # Numărăm feliile 2D care conțin cel puțin un pixel de tumoare
            positive_slices = int(np.sum(np.any(lbl_data == 1, axis=(0, 1))))
            
            # Numărăm feliile 2D care nu conțin tumoare (sănătoase/background)
            negative_slices = num_slices - positive_slices
            
            # Adăugăm toate statisticile colectate ale pacientului curent în listă
            stats_list.append({
                "case_name": os.path.basename(image_path),
                "width": shape[0],
                "height": shape[1],
                "num_slices": shape[2],
                "spacing_x": spacing[0],
                "spacing_y": spacing[1],
                "spacing_z": spacing[2],
                "img_min_HU": min_val,
                "img_max_HU": max_val,
                "img_mean_HU": mean_val,
                "img_std_HU": std_val,
                "img_p5_HU": p5,
                "img_p95_HU": p95,
                "tumor_voxels": tumor_voxels,
                "tumor_volume_mm3": tumor_volume_mm3,
                "tumor_ratio_percent": tumor_ratio,
                "positive_slices": positive_slices,
                "negative_slices": negative_slices
            })
            
        except Exception as e:
            # Dacă un fișier este corupt sau lipsește, afișăm eroarea fără a opri procesarea restului de cazuri
            print(f"Error processing {image_rel}: {e}")
            
    # Convertim lista de dicționare într-un DataFrame Pandas
    df = pd.DataFrame(stats_list)
    
    # Construim calea fișierului CSV de ieșire
    csv_path = os.path.join(OUTPUT_DIR, "eda_statistics.csv")
    
    # Salvăm datele tabelare în eda_statistics.csv
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset statistics to: {csv_path}")
    
    # Afișăm în consolă un rezumat sintetic al întregului set de date
    print("\n=== FULL DATASET SUMMARY ===")
    print(f"Total analyzed cases: {len(df)}")
    print(f"Average image shape: {df['width'].mean():.1f} x {df['height'].mean():.1f} x {df['num_slices'].mean():.1f}")
    print(f"Average voxel spacing (mm): {df['spacing_x'].mean():.2f} x {df['spacing_y'].mean():.2f} x {df['spacing_z'].mean():.2f}")
    print(f"Average tumor volume: {df['tumor_volume_mm3'].mean():.2f} mm^3 (Min: {df['tumor_volume_mm3'].min():.2f}, Max: {df['tumor_volume_mm3'].max():.2f})")
    print(f"Total slices across all cases: {df['num_slices'].sum()}")
    print(f"Total tumor-positive slices: {df['positive_slices'].sum()} ({df['positive_slices'].sum() / df['num_slices'].sum() * 100:.2f}%)")
    print(f"Total tumor-negative slices: {df['negative_slices'].sum()} ({df['negative_slices'].sum() / df['num_slices'].sum() * 100:.2f}%)")
    
    # Resetăm stilul grafic Matplotlib la cel implicit
    plt.style.use('default')
    
    # ---------------------------------------------------------
    # Graficul 1: Histograma Distribuției Volumelor Tumorilor
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.hist(df['tumor_volume_mm3'], bins=15, color='#e74c3c', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Lung Tumor Volumes ($mm^3$) - All Patients", fontsize=14, fontweight='bold')
    plt.xlabel("Tumor Volume ($mm^3$)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    # Salvăm histograma ca imagine PNG
    plt.savefig(os.path.join(figures_dir, "tumor_volume_distribution.png"), dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # Graficul 2: Comparație Felii Totale vs. Felii cu Tumoare per Pacient
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 6))
    x = np.arange(len(df))
    plt.bar(x - 0.2, df['num_slices'], width=0.4, label='Total Slices', color='#3498db', alpha=0.8)
    plt.bar(x + 0.2, df['positive_slices'], width=0.4, label='Tumor Slices', color='#e74c3c', alpha=0.8)
    plt.title("Total Slices vs. Tumor-Positive Slices per Case - All Patients", fontsize=14, fontweight='bold')
    plt.xlabel("Case Index", fontsize=12)
    plt.ylabel("Slice Count", fontsize=12)
    plt.xticks(x, df['case_name'].apply(lambda n: n.split('.')[0]), rotation=90, fontsize=8)
    plt.legend(fontsize=11)
    plt.tight_layout()
    # Salvăm bar chart-ul ca imagine PNG
    plt.savefig(os.path.join(figures_dir, "slices_comparison.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # Graficul 3: Vizualizarea unui caz reprezentativ (Caz cu tumoare mare)
    # ---------------------------------------------------------
    if len(df) > 0:
        # Găsim pacientul cu cel mai mare număr de voxeli de tumoare
        best_case_row = df.loc[df['tumor_voxels'].idxmax()]
        best_case_name = best_case_row['case_name']
        
        # Identificăm dicționarul cazului corespunzător din dataset_info
        best_case = next(c for c in training_cases if os.path.basename(c["image"]).startswith(best_case_name.split('.')[0]))
        
        # Rezolvăm căile fișierelor NIfTI pentru acest caz
        img_path = resolve_nifti_path(best_case["image"])
        lbl_path = resolve_nifti_path(best_case["label"])
        
        # Încărcăm matricele 3D
        img_data = np.asanyarray(nib.load(img_path).dataobj)
        lbl_data = np.asanyarray(nib.load(lbl_path).dataobj)
        
        # Găsim felia axială 2D cu cea mai mare suprafață de tumoare
        lbl_slice_sums = [np.sum(lbl_data[:, :, s] == 1) for s in range(lbl_data.shape[2])]
        max_slice_idx = np.argmax(lbl_slice_sums)
        
        # Extragem felia CT și masca corespunzătoare
        ct_slice = img_data[:, :, max_slice_idx]
        mask_slice = lbl_data[:, :, max_slice_idx]
        
        # Creăm o figură cu 3 subplot-uri (CT, Mască, Overlay)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Subplot 1: Imaginea CT în nuanțe de gri
        axes[0].imshow(ct_slice.T, cmap='gray', origin='lower')
        axes[0].set_title("CT Slice", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Subplot 2: Masca de tumoare (roșu)
        axes[1].imshow(mask_slice.T, cmap='Reds', origin='lower', alpha=0.8)
        axes[1].set_title("Tumor Mask (Ground Truth)", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Subplot 3: Overlay (Imagine CT + Masca peste ea)
        axes[2].imshow(ct_slice.T, cmap='gray', origin='lower')
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[2].imshow(masked.T, cmap='Set1', origin='lower', alpha=0.5)
        axes[2].set_title("Overlay (CT + Mask)", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Titlu general și salvarea imaginii de vizualizare
        plt.suptitle(f"Sample Case Visualization: {os.path.basename(img_path)} (Slice {max_slice_idx})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "sample_overlay.png"), dpi=150)
        plt.close()

    print("=== FULL EDA COMPLETED SUCCESSFULLY ===")

# Punctul de intrare standard pentru rularea directă a scriptului din linia de comandă
if __name__ == "__main__":
    run_eda()
