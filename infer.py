import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from models.dlinknet import DLinkNet34
from models.deeplabv3 import DeepLabV3PlusResNet34


# --- Configurations ---
image_dir = r"D:\Downloads\Computer_Vision\Capstone_Project\Road-Semantic-Segmentation-for-Aerial-Image\infer_images\input"
mask_dir = r"D:\Downloads\Computer_Vision\Capstone_Project\data\combined_dataset\masks"
output_base_dir = r"D:\Downloads\Computer_Vision\Capstone_Project\Road-Semantic-Segmentation-for-Aerial-Image\infer_images\output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.5
max_num_images = 10

model_target_sizes = {
    "DLinkNet34": (1024, 1024),
    "SegFormer_B2": (512, 512),
    "DeepLabV3Plus_ResNet34": (512, 512)
}

# --- Model Paths Mapping ---
# Update these paths to point to the respective weights for each model
model_checkpoints = {
    "SegFormer_B2": r"D:\Downloads\Computer_Vision\Capstone_Project\checkpoint\best_segformer_b2.pth",
    "DeepLabV3Plus_ResNet34": r"D:\Downloads\Computer_Vision\Capstone_Project\checkpoint\best_deeplabv3+.pth",
    "DLinkNet34": r"D:\Downloads\Computer_Vision\Capstone_Project\checkpoint\best_dlinknet.pth"
}

# --- 1. Initialize Models & Load Weights ---
models = {}

for model_name, path in model_checkpoints.items():
    if not os.path.exists(path):
        print(f"Warning: Checkpoint not found for {model_name} at {path}. Skipping.")
        continue
        
    print(f"Loading {model_name}...")
    
    # Initialize the correct architecture
    if model_name == "SegFormer_B2":
        config = SegformerConfig(
            num_labels=1, widths=[64, 128, 320, 512], depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512], num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], decoder_hidden_size=768
        )
        model = SegformerForSemanticSegmentation(config)
    elif model_name == "DeepLabV3Plus_ResNet34":
        model = DeepLabV3PlusResNet34(num_classes=1)
    elif model_name == "DLinkNet34":
        model = DLinkNet34(num_classes=1)
        
    # Load and clean state dict
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        clean_name = k[7:] if k.startswith('module.') else k
        clean_state_dict[clean_name] = v
        
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device).eval()
    models[model_name] = model

# --- 2. Get Target Files ---
image_filenames = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))][:max_num_images]

# --- 3. Multi-Model Processing Loop ---
for model_name, model in models.items():
    print(f"Running inference with {model_name}...")

    current_target_size = model_target_sizes.get(model_name, (512, 512))
    print(f"Target size set to: {current_target_size}")
    
    # Create model-specific output directory to avoid overwriting
    model_output_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    for idx, filename in enumerate(image_filenames):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + '.png')
        
        image_bgr = cv2.imread(img_path)
        if image_bgr is None: continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Preprocessing
        img_resized = cv2.resize(image_rgb, current_target_size, interpolation=cv2.INTER_LINEAR)
        
        # Note: If DeepLabV3/DLinkNet used standard ImageNet normalization during training, 
        # you might want to adjust this block per model. Keeping your original Segformer scale for now.
        input_tensor = (img_resized.astype(np.float32) / 255.0) * 3.2 - 1.6
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            # Handle HuggingFace SegFormer output object vs Standard PyTorch Tensor outputs
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            if logits.shape[-2:] != input_tensor.shape[-2:]:
                logits = F.interpolate(
                    logits, 
                    size=input_tensor.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            pred = logits.squeeze().cpu().numpy()
            pred_mask = (pred > threshold).astype(np.uint8)

        # Scale back to original resolution
        pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # --- Plotting & Visualizing Results ---
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(image_rgb); axes[0, 0].set_title("Image"); axes[0, 0].axis("off")
        
        if mask is not None:
            axes[0, 1].imshow(mask, cmap='gray'); axes[0, 1].set_title("Ground Truth"); axes[0, 1].axis("off")
            gt_bin = mask > 0
        else:
            axes[0, 1].axis("off")
            gt_bin = np.zeros_like(pred_mask_resized, dtype=bool)

        axes[1, 0].imshow(pred_mask_resized, cmap='gray'); axes[1, 0].set_title(f"{model_name} Prediction"); axes[1, 0].axis("off")

        # Error Analysis Overlay Creation
        pred_bin = pred_mask_resized > 0
        overlay = image_rgb.copy()
        color_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        
        color_mask[pred_bin & gt_bin] = [0, 255, 0]   # True Positives (Green)
        color_mask[pred_bin & ~gt_bin] = [255, 255, 0] # False Positives (Yellow)
        color_mask[~pred_bin & gt_bin] = [255, 0, 0]   # False Negatives (Red)
        
        cv2.addWeighted(color_mask, 0.4, image_rgb, 0.6, 0, overlay)
        axes[1, 1].imshow(overlay); axes[1, 1].set_title("Overlay (Green:TP, Yellow:FP, Red:FN)"); axes[1, 1].axis("off")

        plt.tight_layout()
        
        # Save output plot directly to disk
        save_path = os.path.join(model_output_dir, f"{model_name}_{os.path.splitext(filename)[0]}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)  # Free system memory resources immediately

print(f"Completed! Saved evaluation images to subdirectories under: {output_base_dir}")