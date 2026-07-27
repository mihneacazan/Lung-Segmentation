"""
Prediction Viewer: Interactive Streamlit app to compare
model predictions vs ground truth masks on validation slices.

Usage (local):
    streamlit run src/stage6_visualization/prediction_viewer.py

Requirements:
    - Trained model checkpoint at output/best_metric_model.pth
    - Preprocessed validation slices at output/preprocessed/val/
"""
import os
import sys
import glob
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.config as config
from src.stage3_models.unet_2d import build_unet_2d

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prediction Viewer - Lung Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #1a1d23; }
    h1 { color: #00d4ff; }
    h2, h3 { color: #e0e0e0; }
    .dice-good { color: #2ecc71; font-weight: bold; font-size: 1.3em; }
    .dice-medium { color: #f39c12; font-weight: bold; font-size: 1.3em; }
    .dice-bad { color: #e74c3c; font-weight: bold; font-size: 1.3em; }
    .metric-box {
        background: linear-gradient(135deg, #1a1d23, #2d3039);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Model Loading (cached for performance)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model(checkpoint_path):
    """Load trained model from checkpoint."""
    model = build_unet_2d(in_channels=1, out_channels=1)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        val_dice = checkpoint.get("val_dice", "?")
        st.sidebar.success(f"Model loaded from Epoch {epoch} (Val Dice: {val_dice})")
    else:
        model.load_state_dict(checkpoint)
        st.sidebar.success("Model loaded (raw state dict).")
    
    model.eval()
    return model

@st.cache_data
def load_validation_slices():
    """Load all validation image/label pairs from preprocessed directory."""
    val_img_dir = os.path.join(config.OUTPUT_DIR, "preprocessed", "val", "images")
    val_lbl_dir = os.path.join(config.OUTPUT_DIR, "preprocessed", "val", "labels")
    
    img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.npy")))
    
    slices = []
    for img_path in img_paths:
        base = os.path.basename(img_path)
        lbl_path = os.path.join(val_lbl_dir, base)
        if os.path.exists(lbl_path):
            slices.append({
                "name": base.replace(".npy", ""),
                "image": np.load(img_path).astype(np.float32),
                "label": np.load(lbl_path).astype(np.float32),
            })
    return slices

def compute_dice(pred, target):
    """Compute Dice coefficient between two binary masks."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    if union == 0:
        return 1.0 if np.sum(target_flat) == 0 else 0.0
    return (2.0 * intersection) / union

def run_inference(model, image_np):
    """Run model inference on a single 2D slice."""
    with torch.no_grad():
        tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().numpy()
    return prob

# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
st.title("🔬 Prediction Viewer: Predicted vs Ground Truth")
st.markdown("Compare the model's segmentation predictions against ground truth masks for **validation slices**.")

# Check for model checkpoint
checkpoint_path = os.path.join(config.OUTPUT_DIR, "best_metric_model.pth")
if not os.path.exists(checkpoint_path):
    st.error(f"No trained model found at `{checkpoint_path}`. Please run training first or download the checkpoint from Kaggle.")
    st.stop()

# Check for preprocessed validation data
val_img_dir = os.path.join(config.OUTPUT_DIR, "preprocessed", "val", "images")
if not os.path.exists(val_img_dir):
    st.error(f"No preprocessed validation data found at `{val_img_dir}`. Please run preprocessing first.")
    st.stop()

# Load model and data
model = load_model(checkpoint_path)
all_slices = load_validation_slices()

if len(all_slices) == 0:
    st.error("No validation slices found.")
    st.stop()

# ──────────────────────────────────────────────
# Sidebar: Filters & Controls
# ──────────────────────────────────────────────
st.sidebar.header("🎛️ Controls")

# Filter mode
filter_mode = st.sidebar.radio(
    "Show slices:",
    ["All Slices", "Only Tumor-Positive (GT has tumor)", "Only Tumor-Negative (GT is empty)"]
)

positive_indices = [i for i, s in enumerate(all_slices) if np.sum(s["label"]) > 0]
negative_indices = [i for i, s in enumerate(all_slices) if np.sum(s["label"]) == 0]

if filter_mode == "Only Tumor-Positive (GT has tumor)":
    visible_indices = positive_indices
elif filter_mode == "Only Tumor-Negative (GT is empty)":
    visible_indices = negative_indices
else:
    visible_indices = list(range(len(all_slices)))

if len(visible_indices) == 0:
    st.warning("No slices match the current filter.")
    st.stop()

st.sidebar.markdown(f"""
<div class="metric-box">
    <strong>Total Val Slices:</strong> {len(all_slices)}<br/>
    <strong>Tumor-Positive:</strong> {len(positive_indices)}<br/>
    <strong>Tumor-Negative:</strong> {len(negative_indices)}<br/>
    <strong>Currently Showing:</strong> {len(visible_indices)}
</div>
""", unsafe_allow_html=True)

# Threshold control
threshold = st.sidebar.slider("Binarization Threshold:", 0.1, 0.9, 0.5, 0.05)

# Slice navigator
slice_pos = st.sidebar.slider("Navigate Slices:", 0, len(visible_indices) - 1, 0)
current_idx = visible_indices[slice_pos]
current_slice = all_slices[current_idx]

st.sidebar.markdown(f"**Current:** `{current_slice['name']}`")

# ──────────────────────────────────────────────
# Run Inference & Display
# ──────────────────────────────────────────────
image_np = current_slice["image"]
label_np = current_slice["label"]

# Run model prediction
prob_map = run_inference(model, image_np)
pred_mask = (prob_map >= threshold).astype(np.float32)

# Compute Dice
dice_score = compute_dice(pred_mask, label_np)

# Color-code Dice
if dice_score >= 0.7:
    dice_class = "dice-good"
elif dice_score >= 0.3:
    dice_class = "dice-medium"
else:
    dice_class = "dice-bad"

has_tumor_gt = np.sum(label_np) > 0
has_tumor_pred = np.sum(pred_mask) > 0

# Header metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"<div class='metric-box'><small>Dice Score</small><br/><span class='{dice_class}'>{dice_score:.4f}</span></div>", unsafe_allow_html=True)
with col_m2:
    gt_status = "🔴 TUMOR" if has_tumor_gt else "🟢 CLEAN"
    st.markdown(f"<div class='metric-box'><small>Ground Truth</small><br/><b>{gt_status}</b></div>", unsafe_allow_html=True)
with col_m3:
    pred_status = "🔴 DETECTED" if has_tumor_pred else "🟢 NONE"
    st.markdown(f"<div class='metric-box'><small>Prediction</small><br/><b>{pred_status}</b></div>", unsafe_allow_html=True)
with col_m4:
    max_prob = float(np.max(prob_map))
    st.markdown(f"<div class='metric-box'><small>Max Probability</small><br/><b>{max_prob:.4f}</b></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Visualization: 4-Panel View
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.patch.set_facecolor('#0e1117')

titles = ["CT Slice (Input)", "Ground Truth Mask", "Predicted Mask", "Overlay Comparison"]
colors_title = ["#ffffff", "#2ecc71", "#e74c3c", "#00d4ff"]

for ax, title, color in zip(axes, titles, colors_title):
    ax.set_facecolor('#0e1117')
    ax.set_title(title, fontsize=12, fontweight='bold', color=color, pad=10)
    ax.axis('off')

# Panel 1: CT Slice
axes[0].imshow(image_np.T, cmap='gray', origin='lower')

# Panel 2: Ground Truth Mask (green)
axes[1].imshow(image_np.T, cmap='gray', origin='lower', alpha=0.4)
if has_tumor_gt:
    gt_masked = np.ma.masked_where(label_np == 0, label_np)
    axes[1].imshow(gt_masked.T, cmap='Greens', origin='lower', alpha=0.8, vmin=0, vmax=1)
    axes[1].contour(label_np.T, colors='lime', linewidths=1.5, origin='lower', levels=[0.5])

# Panel 3: Predicted Mask (red)
axes[2].imshow(image_np.T, cmap='gray', origin='lower', alpha=0.4)
if has_tumor_pred:
    pred_masked = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[2].imshow(pred_masked.T, cmap='Reds', origin='lower', alpha=0.8, vmin=0, vmax=1)
    axes[2].contour(pred_mask.T, colors='red', linewidths=1.5, origin='lower', levels=[0.5])

# Panel 4: Overlay (Green=GT, Red=Pred, Yellow=Overlap)
axes[3].imshow(image_np.T, cmap='gray', origin='lower', alpha=0.5)
if has_tumor_gt or has_tumor_pred:
    # Create RGB overlay: Green=GT only, Red=Pred only, Yellow=Both
    overlay = np.zeros((*image_np.shape, 3))
    overlay[:, :, 1] = label_np * 0.7      # Green channel = Ground Truth
    overlay[:, :, 0] = pred_mask * 0.7      # Red channel = Prediction
    # Where both overlap, both R and G are active → Yellow
    overlay_masked = np.ma.masked_where(
        np.repeat((label_np + pred_mask)[:, :, np.newaxis] == 0, 3, axis=2),
        overlay
    )
    axes[3].imshow(np.transpose(overlay_masked, (1, 0, 2)), origin='lower', alpha=0.8)
    # Legend text
    axes[3].text(5, 10, "Green=GT  Red=Pred  Yellow=Both", fontsize=8,
                 color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                 transform=axes[3].transData)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ──────────────────────────────────────────────
# Probability Heatmap
# ──────────────────────────────────────────────
with st.expander("🔥 View Raw Probability Heatmap", expanded=False):
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')
    im = ax2.imshow(prob_map.T, cmap='hot', origin='lower', vmin=0, vmax=1)
    ax2.set_title("Model Output Probability Map", fontsize=12, fontweight='bold', color='white')
    ax2.axis('off')
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Tumor Probability', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ──────────────────────────────────────────────
# Batch Dice Summary
# ──────────────────────────────────────────────
with st.expander("📊 Dice Score Summary (All Validation Slices)", expanded=False):
    st.markdown("Computing Dice scores for all validation slices...")
    
    all_dice_scores = []
    for s in all_slices:
        prob = run_inference(model, s["image"])
        pred = (prob >= threshold).astype(np.float32)
        d = compute_dice(pred, s["label"])
        all_dice_scores.append({"name": s["name"], "dice": d, "has_tumor": np.sum(s["label"]) > 0})
    
    import pandas as pd
    df_dice = pd.DataFrame(all_dice_scores)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Mean Dice (All)", f"{df_dice['dice'].mean():.4f}")
    with col_s2:
        tumor_df = df_dice[df_dice['has_tumor'] == True]
        st.metric("Mean Dice (Tumor Slices)", f"{tumor_df['dice'].mean():.4f}" if len(tumor_df) > 0 else "N/A")
    with col_s3:
        clean_df = df_dice[df_dice['has_tumor'] == False]
        st.metric("Mean Dice (Clean Slices)", f"{clean_df['dice'].mean():.4f}" if len(clean_df) > 0 else "N/A")
    
    st.dataframe(df_dice.sort_values("dice", ascending=True).head(20), use_container_width=True)
