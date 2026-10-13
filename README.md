# LB-UNet: Lightweight Boundary-Guided U-Net for Skin Lesion Segmentation

A PyTorch implementation and enhancement of **LB-UNet** (Lightweight Boundary-Guided U-Net) for automated medical image segmentation on dermoscopic datasets (ISIC 2018, ISIC 2017, and BUSI).

This repository features a lightweight architecture (~0.099M parameters) designed for accurate skin lesion boundary delineation, combining depthwise separable convolutions, boundary-aware feature aggregation, composite loss supervision, Test-Time Augmentation (TTA), and early stopping.

---

## 📊 Evaluation Results (ISIC 2018)

Evaluated on the ISIC 2018 validation/test benchmark with Test-Time Augmentation (TTA):

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Pixel Accuracy (Acc)** | **94.49%** (0.9449) | Overall pixel-level classification accuracy |
| **Dice / F1 Score (DSC)** | **88.12%** (0.8812) | Harmonic overlap between prediction and ground truth |
| **mIoU (Jaccard Index)** | **80.51%** (0.8051) | Intersection over Union overlap metric |
| **Specificity (Spec)** | **97.89%** (0.9789) | Healthy skin background detection rate |
| **Sensitivity (Sens)** | **83.93%** (0.8393) | True lesion pixel detection rate (Recall) |
| **Model Size** | **0.099M** | Ultra-lightweight (~99K trainable parameters) |

---

## 📂 Repository File Structure & Descriptions

Below is a detailed breakdown of each file and folder in this repository:

| File / Folder | Role & Detailed Description |
| :--- | :--- |
| [`config_setting.py`](config_setting.py) | **Configuration & Hyperparameters**: Centralizes all training and model hyperparameters, including network selection (`lbunet`), channel widths (`c_list = [16, 24, 32, 48, 64, 96]`), input resolution ($256 \times 256$), batch size (`8`), optimizer (`AdamW`), learning rate scheduler (`WP_CosineLR`), total epochs (`30`), and early stopping patience (`10`). |
| [`lbunet.py`](lbunet.py) | **Model Architecture**: Defines the lightweight U-Net model (`LBUNet`). It utilizes Depthwise Separable Convolutions (`DSConv`) for parameter efficiency, Boundary-Guided Bridge modules (`BGB`) for edge feature fusion, and multi-scale deep supervision decoders. |
| [`boundary.py`](boundary.py) | **Boundary Extraction Module**: Implements edge detection (Sobel filter), distance transform calculations, non-maximum suppression, and boundary keypoint sampling. Generates ground-truth boundary maps to supervise edge learning. |
| [`dataset.py`](dataset.py) | **Data Loader**: Implements the PyTorch `Dataset` (`NPY_datasets`) for loading preprocessed `.npy` dermoscopic images and corresponding binary masks. Connects data augmentation pipelines defined in `config_setting.py`. |
| [`engine.py`](engine.py) | **Execution Engines**: Contains the core training (`train_one_epoch`), validation (`val_one_epoch`), and evaluation (`test_one_epoch`) routines. Implements Test-Time Augmentation (TTA: original, horizontal flip, vertical flip, combined flip) and automated visualization saving. |
| [`train.py`](train.py) | **Main Training Pipeline**: Entry point for model training. Sets up GPU device, sets random seeds for reproducibility, manages mixed-precision training (`torch.amp.GradScaler`), TensorBoard logging, checkpoint saving (`best.pth`), **Early Stopping** (`patience = 10`), and triggers automated testing on completion. |
| [`test.py`](test.py) | **Independent Testing & Inference**: Standalone script to evaluate saved models. Automatically detects the latest `best.pth` checkpoint in `results/`, runs full evaluation with Test-Time Augmentation, saves prediction masks, and prints performance metrics. |
| [`utils.py`](utils.py) | **Utility Functions & Losses**: Implements the composite loss `GT_BceDiceLoss` (combining Binary Cross-Entropy with Dice Loss for both mask and boundary outputs), optimizer factories, Warm-up Cosine Schedulers (`WP_CosineLR`), evaluation metric formulas, and custom data transformations. |
| [`2135_paper.pdf`](2135_paper.pdf) | **Reference Paper**: The foundational research publication on Lightweight Boundary-guided U-Net for Skin Lesion Segmentation. |
| `data/` | **Dataset Directory**: Contains dataset splits (`isic2018`, `isic2017`, `Dataset_BUSI_with_GT`) with input `.npy` images and ground-truth labels. |
| `results/` | **Output & Checkpoints**: Generated timestamped experiment directories containing trained model checkpoints (`checkpoints/best.pth`), execution logs (`log/train.info.log`), TensorBoard curves (`summary/`), and visual segmentations (`outputs/`). |

---

## 📐 Evaluation Metrics Explained

In medical image segmentation, standard accuracy alone is insufficient due to severe class imbalance (the lesion typically occupies only 10%–30% of the image, while healthy skin covers the rest). The following 5 metrics provide a balanced, clinically meaningful evaluation:

### 1. Dice Similarity Coefficient (DSC / F1-Score)
* **Mathematical Formula**:
  $$\text{Dice} = \frac{2 \cdot |X \cap Y|}{|X| + |Y|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
* **Clinical Meaning**: Measures the spatial overlap between predicted mask $X$ and ground truth mask $Y$. Values range from $0$ (no overlap) to $1$ (perfect overlap). It is the primary metric for medical segmentation because it directly measures how closely the contour matches the real lesion.

### 2. Mean Intersection over Union (mIoU / Jaccard Index)
* **Mathematical Formula**:
  $$\text{mIoU} = \frac{|X \cap Y|}{|X \cup Y|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
* **Clinical Meaning**: The ratio of the intersection area to the total combined union area. It penalizes false positives and false negatives more strictly than Dice ($\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$).

### 3. Pixel Accuracy (Acc)
* **Mathematical Formula**:
  $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
* **Clinical Meaning**: The percentage of all pixels correctly classified as either lesion or background. While high ($>94\%$), it must always be interpreted alongside Dice and mIoU due to background pixel dominance.

### 4. Sensitivity (Recall / True Positive Rate)
* **Mathematical Formula**:
  $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **Clinical Meaning**: Measures the proportion of actual lesion pixels that the model correctly identified. In clinical dermatology, **high sensitivity is critical** to prevent under-segmenting lesions and missing malignant melanoma borders.

### 5. Specificity (True Negative Rate)
* **Mathematical Formula**:
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
* **Clinical Meaning**: Measures the proportion of normal skin pixels correctly classified as background. High specificity ($>97\%$) ensures the model does not produce excessive false alarms on healthy skin.

---

## 🚀 How to Run

### 1. Training with Early Stopping
To train the model from scratch for 30 epochs with early stopping (`patience = 10`):
```bash
python train.py
```
* Best weights are saved automatically to `results/<experiment_name>/checkpoints/best.pth`.
* If validation score does not improve for 10 epochs, training stops early and automatically evaluates the test set.

### 2. Testing / Evaluation
To evaluate the latest trained checkpoint with Test-Time Augmentation (TTA):
```bash
python test.py
```
* Automatically discovers the latest `best.pth` checkpoint.
* Outputs the quantitative metrics table and saves predicted mask images in `results/<experiment_name>/outputs/`.

---

## ⚙️ Key Hyperparameters (`config_setting.py`)

* **Network**: `lbunet`
* **Input Resolution**: $256 \times 256 \times 3$
* **Batch Size**: `8`
* **Optimizer**: `AdamW` (lr = 0.001, weight decay = $10^{-4}$)
* **Scheduler**: Warm-Up Cosine LR (`WP_CosineLR`, 5 warm-up epochs)
* **Loss Function**: `GT_BceDiceLoss` (BCE Loss + Dice Loss)
* **Epochs**: `30`
* **Early Stopping Patience**: `10`
* **Augmentation**: Horizontal Flip, Vertical Flip, Random Rotation ($0^\circ$–$360^\circ$), Min-Max Normalization.
