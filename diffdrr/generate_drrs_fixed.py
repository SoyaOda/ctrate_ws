#!/usr/bin/env python3
"""
Fixed DRR generation with proper axis order
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.data import read


def generate_chest_xray():
    """Generate chest X-ray style DRR"""
    
    # Input paths
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    output_dir = Path("./chest_xray_output")
    output_dir.mkdir(exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CT with DiffDRR's read function
    print("Loading CT data...")
    subject = read(ct_path, bone_attenuation_multiplier=3.0)
    
    # Check data
    print(f"Volume shape: {subject.volume.data.shape}")
    print(f"Volume range: [{subject.volume.data.min():.3f}, {subject.volume.data.max():.3f}]")
    
    # Create DRR generator
    drr = DRR(
        subject=subject,
        sdd=1800.0,
        height=512,
        width=512,
        delx=0.7,
        dely=0.7,
    ).to(device)
    
    # Generate multiple views
    poses = [
        ("frontal", [0, 0, 0], [0, -50, 0]),
        ("frontal_high", [0, 0, 0], [0, -100, 0]),
        ("slight_rotation", [5, 0, 0], [0, -75, 0]),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, rot, trans) in enumerate(poses):
        print(f"\nGenerating {name} view...")
        
        # Create pose
        rotations = torch.tensor([rot], dtype=torch.float32)
        translations = torch.tensor([trans], dtype=torch.float32)
        
        rotations_rad = torch.deg2rad(rotations).to(device)
        translations = translations.to(device)
        
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(device)
        
        # Generate DRR
        img = drr(transform)
        img_np = img.squeeze().cpu().numpy()
        
        print(f"  Image range: [{img_np.min():.3f}, {img_np.max():.3f}]")
        
        # Process for display
        img_np = np.clip(img_np, 0, None)
        if img_np.max() > 0:
            img_display = -np.log(img_np + 0.01)
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        else:
            img_display = img_np
        
        axes[idx].imshow(img_display, cmap='gray')
        axes[idx].set_title(f'{name}\nRot:{rot} Trans:{trans}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "chest_xray_views.png", dpi=150)
    plt.show()
    print(f"\nSaved to {output_dir}/chest_xray_views.png")


if __name__ == "__main__":
    generate_chest_xray()
