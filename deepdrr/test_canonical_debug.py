#!/usr/bin/env python3
"""
デバッグ用スクリプト：ToCanonicalとLAC変換の効果を確認
"""

import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tempfile
from pathlib import Path
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.data import read


def debug_volume_conversion():
    """
    ボリューム変換のデバッグ
    """
    CT_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    
    print("="*60)
    print("Debug: Volume Conversion Process")
    print("="*60)
    
    # 1. Original data
    print("\n[1] Loading original NIfTI...")
    subject_orig = tio.Subject(ct=tio.ScalarImage(CT_PATH))
    data_orig = subject_orig.ct.data.squeeze()
    print(f"Original shape: {data_orig.shape}")
    print(f"Original HU range: [{data_orig.min():.1f}, {data_orig.max():.1f}]")
    
    # 2. After ToCanonical
    print("\n[2] Applying ToCanonical...")
    to_canonical = tio.ToCanonical()
    subject_canonical = to_canonical(subject_orig)
    data_canonical = subject_canonical.ct.data.squeeze()
    print(f"Canonical shape: {data_canonical.shape}")
    print(f"Canonical HU range: [{data_canonical.min():.1f}, {data_canonical.max():.1f}]")
    
    # 3. HU to LAC conversion
    print("\n[3] Converting HU to LAC...")
    mu_water = 0.02  # mm^-1
    bone_scaling = 3.0
    
    volume_clipped = torch.clamp(data_canonical, -1024, 3071)
    
    # Bilinear conversion
    lac_soft = mu_water * (1.0 + volume_clipped / 1000.0)
    lac_bone = mu_water + (volume_clipped / 1000.0) * (mu_water * (bone_scaling - 1.0))
    volume_lac = torch.where(volume_clipped > 0, lac_bone, lac_soft)
    volume_lac[volume_clipped < -1000] = 0.0
    
    print(f"LAC shape: {volume_lac.shape}")
    print(f"LAC range: [{volume_lac.min():.4f}, {volume_lac.max():.4f}]")
    print(f"LAC mean: {volume_lac.mean():.4f}")
    print(f"LAC std: {volume_lac.std():.4f}")
    print(f"Non-zero voxels: {(volume_lac > 0).sum().item():,} / {volume_lac.numel():,}")
    
    # 4. Save as temp file and reload with DiffDRR
    print("\n[4] Saving LAC volume and testing DiffDRR read()...")
    temp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nifti_lac = nib.Nifti1Image(
        volume_lac.cpu().numpy(),
        subject_canonical.ct.affine
    )
    nib.save(nifti_lac, temp_file.name)
    print(f"Saved to: {temp_file.name}")
    
    # Try reading with DiffDRR
    try:
        diffdrr_subject = read(temp_file.name)
        print("✓ DiffDRR read successful")
        print(f"  Subject volume shape: {diffdrr_subject.volume.data.shape}")
        vol_data = diffdrr_subject.volume.data
        print(f"  Subject volume range: [{vol_data.min():.4f}, {vol_data.max():.4f}]")
        
        # Test if DiffDRR is applying its own transformation
        print("\n[5] Checking DiffDRR's internal processing...")
        print(f"  Has density? {hasattr(diffdrr_subject, 'density')}")
        if hasattr(diffdrr_subject, 'density'):
            dens_data = diffdrr_subject.density.data
            print(f"  Density range: [{dens_data.min():.4f}, {dens_data.max():.4f}]")
            
    except Exception as e:
        print(f"✗ DiffDRR read failed: {e}")
    
    # 5. Try with original HU file
    print("\n[6] Testing DiffDRR with original HU file...")
    try:
        subject_hu = read(CT_PATH, bone_attenuation_multiplier=3.0)
        print("✓ Original HU read successful")
        vol_hu = subject_hu.volume.data
        print(f"  Volume range: [{vol_hu.min():.4f}, {vol_hu.max():.4f}]")
        if hasattr(subject_hu, 'density'):
            dens_hu = subject_hu.density.data
            print(f"  Density range: [{dens_hu.min():.4f}, {dens_hu.max():.4f}]")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # 6. Create visualization
    print("\n[7] Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original HU
    slice_idx = data_orig.shape[2] // 2
    axes[0, 0].imshow(data_orig[:, :, slice_idx], cmap='gray')
    axes[0, 0].set_title(f'Original HU (slice {slice_idx})')
    axes[0, 0].axis('off')
    
    # Canonical HU
    axes[0, 1].imshow(data_canonical[:, :, slice_idx], cmap='gray')
    axes[0, 1].set_title(f'After ToCanonical (slice {slice_idx})')
    axes[0, 1].axis('off')
    
    # LAC
    axes[0, 2].imshow(volume_lac[:, :, slice_idx], cmap='gray')
    axes[0, 2].set_title(f'LAC conversion (slice {slice_idx})')
    axes[0, 2].axis('off')
    
    # Histograms
    axes[1, 0].hist(data_orig.flatten()[::100], bins=50, alpha=0.7)
    axes[1, 0].set_title('Original HU histogram')
    axes[1, 0].set_xlabel('HU')
    
    axes[1, 1].hist(data_canonical.flatten()[::100], bins=50, alpha=0.7)
    axes[1, 1].set_title('Canonical HU histogram')
    axes[1, 1].set_xlabel('HU')
    
    axes[1, 2].hist(volume_lac.flatten()[::100], bins=50, alpha=0.7)
    axes[1, 2].set_title('LAC histogram')
    axes[1, 2].set_xlabel('LAC (mm^-1)')
    
    plt.tight_layout()
    plt.savefig('debug_volume_conversion.png', dpi=150)
    print("Saved visualization to: debug_volume_conversion.png")
    
    return temp_file.name


def test_simple_drr(lac_file_path=None):
    """
    シンプルなDRRテスト
    """
    print("\n" + "="*60)
    print("Simple DRR Test")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use original file with DiffDRR's conversion
    print("\nUsing original HU file with bone_attenuation_multiplier...")
    CT_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    
    subject = read(CT_PATH, bone_attenuation_multiplier=3.0)
    
    drr = DRR(
        subject=subject,
        sdd=1800.0,
        height=256,  # Smaller for testing
        width=256,
        delx=1.0,
        dely=1.0,
    ).to(device)
    
    # Simple frontal view
    rotations = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    translations = torch.tensor([[0.0, -100.0, 0.0]], dtype=torch.float32)
    
    rotations_rad = torch.deg2rad(rotations).to(device)
    translations = translations.to(device)
    
    transform = convert(
        rotations_rad,
        translations,
        parameterization="euler_angles",
        convention="XYZ",
    ).to(device)
    
    img = drr(transform)
    img_np = img.squeeze().cpu().numpy()
    
    print(f"\nDRR output stats:")
    print(f"  Range: [{img_np.min():.6f}, {img_np.max():.6f}]")
    print(f"  Mean: {img_np.mean():.6f}")
    print(f"  Std: {img_np.std():.6f}")
    
    # Try different post-processing
    # 1. Log transform
    img_log = -np.log(np.maximum(img_np, 1e-6))
    
    # 2. Direct normalization
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f'Raw DRR\n[{img_np.min():.3f}, {img_np.max():.3f}]')
    axes[0].axis('off')
    
    axes[1].imshow(img_log, cmap='gray')
    axes[1].set_title(f'Log transform\n[{img_log.min():.3f}, {img_log.max():.3f}]')
    axes[1].axis('off')
    
    axes[2].imshow(img_norm, cmap='gray')
    axes[2].set_title(f'Normalized\n[{img_norm.min():.3f}, {img_norm.max():.3f}]')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_drr_output.png', dpi=150)
    print("\nSaved DRR output to: debug_drr_output.png")


if __name__ == "__main__":
    # First debug the volume conversion
    lac_file = debug_volume_conversion()
    
    # Then test DRR generation
    test_simple_drr(lac_file)