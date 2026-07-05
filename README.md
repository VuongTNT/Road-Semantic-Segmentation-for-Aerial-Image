# Road-Semantic-Segmentation-for-Aerial-Image
## 🔗 Dataset & Checkpoints
* **Link to dataset & checkpoints:** [Link](https://drive.google.com/drive/folders/1hVQOvkIHN43y4tuWpF1NaHFWVcMmIwMr?usp=sharing)

## 📌 Project Introduction
This repository contains the codebase for our Capstone Project focusing on **Binary Road Segmentation from Aerial Imagery**. Extracting road networks from high-resolution satellite remote-sensing images plays a critical role in urban planning, autonomous navigation mapping, and emergency disaster response systems. 

Road features present unique challenges such as narrow structures, shadows from trees or buildings, and visual similarities to other concrete surfaces. To address these, our project implements, evaluates, and directly compares three state-of-the-art Deep Learning segmentation architectures:
1. **DLinkNet (ResNet34 Backbone)**: Built specifically for linear structures like roads, leveraging dilated convolutions to maintain spatial resolution without losing global context.
2. **DeepLabV3+ (ResNet34 Backbone)**: Utilizes Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale features effectively.
3. **SegFormer (B2 Variant)**: A transformer-based semantic segmentation framework that avoids positional encodings to efficiently merge localized features with global context.

Our system supports dynamic inference sizes depending on model constraints ($1024 \times 1024$ for DLinkNet34 and $512 \times 512$ for DeepLabV3+/SegFormer) and automatically generates comparative analytical error overlays (TP, FP, FN highlights) for visualization.

---

## 📁 Repository Structure

```text
.
│   infer.py                 # Core multi-model inference script for test images
│   README.md                # Project documentation and guide
│   requirements.txt         # Required Python packages and dependencies
│   
├───infer_images             # Testbed data directory for evaluation
│   ├───input                # Raw aerial/satellite input images (.jpg)
│   └───output               # Auto-generated visualization results partitioned by model
│       ├───DeepLabV3Plus_ResNet34
│       ├───DLinkNet34
│       └───SegFormer_B2
│               
├───models                   # Native PyTorch neural network architecture declarations
│   │   deeplabv3.py         # DeepLabV3+ model definition script
│   │   dlinknet.py          # DLinkNet34 model definition script
│   └───__pycache__
│           
└───notebooks                # Step-by-step Jupyter development environments
    ├───eval                 # Performance evaluations & validation visualizations
    │       evaluate.ipynb
    │       visualize.ipynb
    │       
    ├───preprocess           # Data preparation pipelines (e.g., train/val/test splitting)
    │       data_division.ipynb
    │       
    └───train                # Model-specific training scripts and hyperparameters
            deeplabv3+.ipynb
            dlinknet.ipynb
            segformer.ipynb
```
## 🚀 Execution Guidelines

Follow these steps to configure your local setup and execute predictions using our unified multi-model script:

### Step 1: Clone the Repository
Open your terminal environment and clone this repository down to your workspace:

```bash
git clone https://github.com/VuongTNT/Road-Semantic-Segmentation-for-Aerial-Image.git
cd Road-Semantic-Segmentation-for-Aerial-Image
```

Install the package requirements specified in the environment manifest file:
```bash
pip install -r requirements.txt
```

### Step 2: Download Processed Data and Model Weights
Before initiating your run, ensure you have downloaded:
- The target processed evaluation datasets (images and target comparison masks).
- The model weight checkpoints (best_dlinknet.pth, best_deeplabv3+.pth, best_segformer_b2.pth).

### Step 3: Configure Project Path Directions
Open infer.py in your preferred code editor and edit the configuration block at the top of the script. Modify the string paths to match your local directory paths where your images, truth masks, and weights are placed:

```python
# --- Configurations ---
image_dir = r"D:\Your\Local\Path\infer_images\input"
mask_dir = r"D:\Your\Local\Path\data\masks"
output_base_dir = r"D:\Your\Local\Path\infer_images\output"

# --- Model Paths Mapping ---
model_checkpoints = {
    "SegFormer_B2": r"D:\Your\Local\Path\checkpoint\best_segformer_b2.pth",
    "DeepLabV3Plus_ResNet34": r"D:\Your\Local\Path\checkpoint\best_deeplabv3+.pth",
    "DLinkNet34": r"D:\Your\Local\Path\checkpoint\best_dlinknet.pth"
}
```

### Step 4: Run Inference Script
Execute the script using Python. The architecture script handles processing parameters automatically, scales inputs correctly, and groups the diagnostic outputs under infer_images/output:
```bash
python infer.py
```

