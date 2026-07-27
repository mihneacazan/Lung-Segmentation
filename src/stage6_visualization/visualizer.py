import os
import json
import numpy as np
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
import src.config as config
from src.config import resolve_nifti_path

st.set_page_config(
    page_title="Lung CT Visualizer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1e222b;
    }
    h1 {
        color: #ff4b4b;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background-color: #1e222b;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 Interactive Lung CT & Tumor Mask Visualizer")
st.write("Browse through patient volumes, adjust HU window settings, and inspect segmentation masks interactively.")

dataset_json_path = os.path.join(config.DATA_DIR, "dataset.json")
if not os.path.exists(dataset_json_path):
    st.error(f"Could not find dataset.json at {dataset_json_path}. Please verify the data directory.")
    st.stop()

with open(dataset_json_path, 'r') as f:
    dataset_info = json.load(f)

training_cases = dataset_info["training"]
case_names = [os.path.basename(c["image"]).replace(".gz", "") for c in training_cases]

st.sidebar.header("📂 Select Patient & Slice")
selected_case_name = st.sidebar.selectbox("Choose Patient Case:", case_names)

case_idx = case_names.index(selected_case_name)
selected_case = training_cases[case_idx]

@st.cache_resource
def load_nifti_volume(img_rel, lbl_rel):
    img_path = resolve_nifti_path(img_rel)
    lbl_path = resolve_nifti_path(lbl_rel)
    
    img = nib.load(img_path)
    lbl = nib.load(lbl_path)
    
    img_data = np.asanyarray(img.dataobj)
    lbl_data = np.asanyarray(lbl.dataobj)
    spacing = img.header.get_zooms()
    
    return img_data, lbl_data, spacing

try:
    img_data, lbl_data, spacing = load_nifti_volume(selected_case["image"], selected_case["label"])
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

shape = img_data.shape
num_slices = shape[2]

st.sidebar.markdown("### 📊 Case Metadata")
st.sidebar.markdown(f"""
<div class="metric-card">
    <strong>Resolution:</strong> {shape[0]} x {shape[1]}<br/>
    <strong>Slices:</strong> {num_slices}<br/>
    <strong>Voxel Spacing (mm):</strong><br/>
    • X: {spacing[0]:.2f} mm<br/>
    • Y: {spacing[1]:.2f} mm<br/>
    • Z: {spacing[2]:.2f} mm
</div>
""", unsafe_allow_html=True)

tumor_voxel_sums = [np.sum(lbl_data[:, :, s] == 1) for s in range(num_slices)]
positive_slices = [s for s, count in enumerate(tumor_voxel_sums) if count > 0]

st.sidebar.markdown("### 🎯 Tumor Detection")
if positive_slices:
    st.sidebar.success(f"Tumor found in {len(positive_slices)} slices!")
    default_slice = int(np.argmax(tumor_voxel_sums))
else:
    st.sidebar.warning("No tumor pixels found in this case.")
    default_slice = num_slices // 2

slice_idx = st.sidebar.slider("Select Slice Number:", 0, num_slices - 1, default_slice)

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Visualization Options")
    
    window_type = st.radio(
        "CT Intensity Window (Hounsfield Units):",
        ["Lung Window (-1000 to 400 HU)", "Mediastinum / Soft Tissue (-150 to 250 HU)", "Full Raw Range"]
    )
    
    if window_type.startswith("Lung"):
        hu_min, hu_max = -1000.0, 400.0
    elif window_type.startswith("Mediastinum"):
        hu_min, hu_max = -150.0, 250.0
    else:
        hu_min, hu_max = float(img_data.min()), float(img_data.max())
        
    st.info(f"Using HU Range: [{hu_min}, {hu_max}]")
    
    show_mask = st.checkbox("Overlay Tumor Mask (Red)", value=True)
    mask_opacity = st.slider("Mask Opacity:", 0.1, 1.0, 0.4, step=0.1) if show_mask else 0.0
    
    if positive_slices:
        st.write("---")
        st.write("**Quick jump to tumor slices:**")
        jump_slice = st.selectbox("Select Tumor Slice:", positive_slices, index=positive_slices.index(default_slice) if default_slice in positive_slices else 0)
        if st.button("Jump to selected slice"):
            slice_idx = jump_slice
            st.info(f"Jumped to slice {slice_idx}")

with col1:
    ct_slice = img_data[:, :, slice_idx]
    mask_slice = lbl_data[:, :, slice_idx]
    
    windowed_slice = np.clip(ct_slice, hu_min, hu_max)
    
    has_tumor = slice_idx in positive_slices
    status_color = "#ff4b4b" if has_tumor else "#2ecc71"
    status_text = "TUMOR POSITIVE" if has_tumor else "TUMOR NEGATIVE"
    
    st.markdown(f"### Current Slice: **{slice_idx} / {num_slices - 1}** | <span style='color:{status_color}; font-weight:bold;'>{status_text}</span>", unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    ax.imshow(windowed_slice.T, cmap='gray', origin='lower')
    
    if show_mask and has_tumor:
        masked_tumor = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked_tumor.T, cmap='Reds', origin='lower', alpha=mask_opacity, vmin=0, vmax=1)
        ax.contour(mask_slice.T, colors='red', linewidths=1.2, origin='lower', levels=[0.5])
        
    ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with st.expander("🔍 View All Tumor-Positive Slices for this Patient"):
    if positive_slices:
        st.write(f"This patient has a tumor spanning **{len(positive_slices)} slices**:")
        st.write(", ".join([str(s) for s in positive_slices]))
    else:
        st.write("No tumor slices detected in this patient.")
