import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete

from src.config import CONFIG, OUTPUT_DIR
from src.preprocessing.dataset import get_dataloaders
from src.models.unet_2d import build_unet_2d

def train_model():
    print("=== STARTING BASELINE 2D U-NET TRAINING PIPELINE ===")
    
    # 1. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if device.type == "cpu":
        print("Note: Running on CPU. Epochs are optimized to run efficiently.")
        
    # 2. Get DataLoaders from preprocessing module
    train_loader, val_loader = get_dataloaders(batch_size=CONFIG["batch_size"])
    
    # 3. Model, Loss, Optimizer, Scheduler from models module
    model = build_unet_2d(in_channels=1, out_channels=1).to(device)
    
    # DiceCELoss combines Dice Loss + Cross Entropy for strong class imbalance resilience
    loss_function = DiceCELoss(sigmoid=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    epochs = CONFIG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Validation Metric
    # NOTE: For single-channel binary tensors [B, 1, H, W], include_background MUST be True,
    # because Channel 0 IS the tumor binary mask! (False skips channel 0).
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = AsDiscrete(threshold=0.5)
    
    best_val_dice = -1.0
    best_epoch = -1
    
    metrics_history = []
    
    start_time = time.time()
    
    # 4. Main Training Loop
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (Learning Rate: {scheduler.get_last_lr()[0]:.6f}) ---")
        
        # ---------- TRAINING PHASE ----------
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
        scheduler.step()
        avg_train_loss = train_loss / train_batches
        
        # ---------- VALIDATION PHASE ----------
        model.eval()
        val_loss = 0.0
        val_batches = 0
        dice_metric.reset()
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                
                # Apply Sigmoid + Binarize for Dice Metric
                probs = torch.sigmoid(outputs)
                preds = post_trans(probs)
                dice_metric(y_pred=preds, y=labels)
                
            avg_val_loss = val_loss / val_batches
            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            
        print(f"Epoch {epoch} Results:")
        print(f"  - Train Loss: {avg_train_loss:.4f}")
        print(f"  - Val Loss:   {avg_val_loss:.4f}")
        print(f"  - Val Dice:   {val_dice:.4f}")
        
        # Save Best Model Checkpoint
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            checkpoint_path = os.path.join(OUTPUT_DIR, "best_metric_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "config": CONFIG
            }, checkpoint_path)
            print(f"  [BEST] New best model saved! (Val Dice: {val_dice:.4f})")
            
        metrics_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": val_dice,
            "lr": scheduler.get_last_lr()[0]
        })
        
    total_time = time.time() - start_time
    print("\n=== TRAINING COMPLETED ===")
    print(f"Total time elapsed: {total_time / 60:.2f} minutes")
    print(f"Best Validation Dice Score: {best_val_dice:.4f} at Epoch {best_epoch}")
    
    # 5. Save Training Metrics CSV
    df_metrics = pd.DataFrame(metrics_history)
    csv_metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    df_metrics.to_csv(csv_metrics_path, index=False)
    print(f"Saved training history to: {csv_metrics_path}")
    
    # 6. Plot Training Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_metrics["epoch"], df_metrics["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    plt.plot(df_metrics["epoch"], df_metrics["val_loss"], label="Val Loss", color="#3498db", linewidth=2)
    plt.title("Training & Validation Loss (DiceCELoss)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df_metrics["epoch"], df_metrics["val_dice"], label="Val Dice Score", color="#2ecc71", linewidth=2)
    plt.axhline(y=best_val_dice, color="r", linestyle="--", alpha=0.7, label=f"Best Dice ({best_val_dice:.4f})")
    plt.title("Validation Dice Score per Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Dice Score", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved training curves to: {curves_path}")

if __name__ == "__main__":
    train_model()
