# Study of U-Net Architectures and Medical Segmentation Metrics
**Project:** Automatic Lung Tumor Segmentation from CT Images (MSD Lung - Task06)

This document covers the theoretical background required to configure semantic segmentation experiments in medical imaging.

---

## 1. U-Net Architecture: Principles and Inner Workings

The **U-Net** architecture (proposed by Ronneberger et al. in 2015) is the gold standard for biomedical image segmentation due to its symmetric U-shaped structure, composed of two main paths connected by skip connections:

```text
Encoder (Contracting Path)                     Decoder (Expanding Path)
[Conv 3x3 + ReLU] x 2  ─────────────────────►  Copy & Concatenate  ──► [Conv 3x3 + ReLU] x 2
       │ (Downsampling: MaxPool 2x2)                  ▲ (Upsampling: ConvTranspose 2x2)
       ▼                                              │
[Conv 3x3 + ReLU] x 2  ─────────────────────►  Copy & Concatenate
       │                                              ▲
       ▼                                              │
                         [ Bottleneck ]
```

### A. Contracting Path (Encoder)
* **Role:** Extracts semantic features (context) by repeatedly applying $3\times3$ convolutions followed by ReLU activations and downsampling operations ($2\times2$ Max Pooling).
* **Effect:** With each level descended, the spatial resolution of the image is halved, and the number of feature channels doubles (e.g. 32 ➔ 64 ➔ 128 ➔ 256).

### B. Bottleneck
* Connects the encoder to the decoder through a compact latent representation where abstract features are most heavily condensed.

### C. Expanding Path (Decoder)
* **Role:** Reconstructs the spatial resolution of the output image through upsampling operations ($2\times2$ transposed convolutions or bilinear interpolations) and gradually reduces feature channels.
* **Effect:** Abstract feature maps are decoded back to the original resolution of the CT image.

### D. Skip Connections
* **Why they are critical:** During the downsampling steps in the Encoder, high-resolution spatial localization details (precisely where the tumor boundaries are) are lost.
* Skip connections **copy** high-resolution feature maps directly from the Encoder and **concatenate** them with the corresponding feature maps in the Decoder. This allows the model to reconstruct the exact and fine shape of the lung nodule.

---

## 2. Advanced U-Net Variants

To overcome the limitations of the classic U-Net, several advanced variants will be evaluated in this project:

1. **Attention U-Net:**
   * Introduces **Attention Gates** on skip connections.
   * These gates filter the signals transmitted from the Encoder, amplifying activations in target areas (tumors) and suppressing activations from irrelevant backgrounds (such as air or other organs).
2. **SegResNet (NVIDIA):**
   * Replaces standard convolutional blocks in the Encoder with **Residual Blocks** (ResNet style).
   * Prevents the vanishing gradient problem and enables stable training of deeper networks, yielding state-of-the-art results in medical challenges.
3. **2.5D Model:**
   * Instead of a single 2D CT slice (single gray channel), the model receives **3 consecutive adjacent slices** (slice $t-1$, central slice $t$, and slice $t+1$) as RGB-like input channels.
   * This allows the model to capture volumetric 3D context along the Z-axis without the extreme computational cost of 3D convolutions.

---

## 3. Evaluation Metrics for Segmentation

The performance of the model on the test set will be evaluated using the following domain-specific metrics:

### A. Dice Similarity Coefficient (DSC)
Measures the global overlap between the ground truth mask ($Y$) and the model's prediction ($X$):
$$Dice = \frac{2 \times |X \cap Y|}{|X| + |Y|}$$
* A value of $1.0$ indicates perfect overlap, while $0.0$ indicates no overlap. This is the primary metric in medical segmentation.

### B. Intersection over Union (IoU / Jaccard Index)
Similar to the Dice score, it is the ratio of the intersection area to the union area of the two masks:
$$IoU = \frac{|X \cap Y|}{|X \cup Y|}$$
* Mathematical relationship: $Dice = \frac{2 \times IoU}{1 + IoU}$. The IoU score is always less than or equal to the Dice score.

### C. Sensitivity (Recall / True Positive Rate)
Measures the model's ability to detect all real tumor voxels:
$$Sensitivity = \frac{TP}{TP + FN}$$
* *TP = True Positives, FN = False Negatives.* A high sensitivity ensures that no part of the tumor is missed (medically critical).

### D. Precision (Positive Predictive Value)
Measures the model's ability to avoid false alarms in healthy tissue:
$$Precision = \frac{TP}{TP + FP}$$
* *FP = False Positives.* A high precision indicates that the areas colored as tumor are indeed tumorous.

### E. Hausdorff Distance (HD95)
Measures the maximum distance in millimeters between the boundary of the predicted tumor and the ground truth boundary:
* **HD95** represents the 95th percentile of these distances. It is used to eliminate noise (such as a single isolated stray pixel that would skew the absolute maximum distance). It measures the clinical quality of the segmentation boundaries.
