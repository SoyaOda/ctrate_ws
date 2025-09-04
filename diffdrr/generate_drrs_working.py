#!/usr/bin/env python3
"""
動作確認済みコードに基づくDRR生成
ToCanonicalを使わず、直接read()で読み込み
"""

import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.visualization import plot_drr
from pathlib import Path
import numpy as np


def generate_drr_simple():
    """
    シンプルで動作確認済みの方法でDRR生成
    """
    # Paths
    filepath = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    output_dir = Path("./drr_working_output")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Simple DRR Generation (Based on Working Code)")
    print("="*60)
    
    # Read CT volume directly (no ToCanonical preprocessing)
    print(f"\nLoading CT from: {filepath}")
    subject = read(filepath)  # API change: filename parameter removed
    
    print(f"Volume shape: {subject.volume.data.shape}")
    print(f"Volume range: [{subject.volume.data.min():.1f}, {subject.volume.data.max():.1f}]")
    
    # Initialize the DRR module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    drr = DRR(
        subject,
        sdd=1020.0,   # Same as working example
        height=200,   # Same as working example  
        width=200,    # Square image
        delx=2.0,     # Same pixel spacing as working example
        dely=2.0,
    ).to(device)
    
    # Test different camera poses
    poses = [
        {
            'name': 'frontal_850',
            'rotations': [0.0, 0.0, 0.0],
            'translations': [0.0, 850.0, 0.0],  # Same as working example
            'description': 'Frontal view (Y=850)'
        },
        {
            'name': 'frontal_500',
            'rotations': [0.0, 0.0, 0.0],
            'translations': [0.0, 500.0, 0.0],
            'description': 'Frontal view (Y=500)'
        },
        {
            'name': 'frontal_200',
            'rotations': [0.0, 0.0, 0.0],
            'translations': [0.0, 200.0, 0.0],
            'description': 'Frontal view (Y=200)'
        },
        {
            'name': 'rotated_10deg',
            'rotations': [10.0, 0.0, 0.0],
            'translations': [0.0, 850.0, 0.0],
            'description': '10° rotation'
        },
    ]
    
    # Generate DRRs for each pose
    results = []
    
    for pose in poses:
        print(f"\nGenerating {pose['name']}...")
        
        # Set camera pose
        rotations = torch.tensor([pose['rotations']], device=device)
        translations = torch.tensor([pose['translations']], device=device)
        
        # Generate DRR using direct call (same as working example)
        img = drr(rotations, translations, 
                 parameterization="euler_angles", 
                 convention="ZXY")  # Same convention as working example
        
        # Convert to numpy
        img_np = img.squeeze().cpu().numpy()
        print(f"  Raw range: [{img_np.min():.4f}, {img_np.max():.4f}]")
        
        # Save raw DRR
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np, cmap='gray')
        plt.title(f"{pose['description']}\nRaw: [{img_np.min():.3f}, {img_np.max():.3f}]")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"{pose['name']}_raw.png", dpi=150)
        plt.close()
        
        # Apply log transform for better visualization
        img_log = -np.log(np.maximum(img_np, 1e-6))
        img_log_norm = (img_log - img_log.min()) / (img_log.max() - img_log.min() + 1e-6)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_log_norm, cmap='gray')
        plt.title(f"{pose['description']}\nLog transformed")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / f"{pose['name']}_log.png", dpi=150)
        plt.close()
        
        results.append({
            'name': pose['name'],
            'description': pose['description'],
            'raw': img_np,
            'processed': img_log_norm
        })
        
        print(f"  Saved to: {output_dir}/{pose['name']}_*.png")
    
    # Create comparison figure
    print("\nCreating comparison figure...")
    fig, axes = plt.subplots(2, len(results), figsize=(5*len(results), 10))
    
    for idx, result in enumerate(results):
        # Raw images
        axes[0, idx].imshow(result['raw'], cmap='gray')
        axes[0, idx].set_title(f"{result['name']}\n(Raw)")
        axes[0, idx].axis('off')
        
        # Processed images
        axes[1, idx].imshow(result['processed'], cmap='gray')
        axes[1, idx].set_title(f"{result['name']}\n(Processed)")
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    print(f"Comparison saved to: {output_dir}/comparison.png")
    
    # Also try with plot_drr utility
    print("\nUsing plot_drr utility...")
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    img = drr(rotations, translations, 
             parameterization="euler_angles", 
             convention="ZXY")
    
    plt.figure(figsize=(8, 8))
    plot_drr(img, ticks=False)
    plt.title("Using plot_drr utility")
    plt.savefig(output_dir / "plot_drr_result.png", dpi=150)
    plt.close()
    print(f"plot_drr result saved to: {output_dir}/plot_drr_result.png")
    
    print("\n" + "="*60)
    print("✓ DRR generation completed!")
    print(f"✓ Results saved in: {output_dir}")
    print("="*60)


def test_with_higher_resolution():
    """
    高解像度でテスト
    """
    filepath = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    output_dir = Path("./drr_working_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("High Resolution Test")
    print("="*60)
    
    # Read CT
    subject = read(filepath)
    
    # Initialize with higher resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    drr = DRR(
        subject,
        sdd=1800.0,   # Standard chest X-ray distance
        height=512,   # Higher resolution
        width=512,
        delx=0.7,     # Finer pixel spacing
        dely=0.7,
    ).to(device)
    
    # Generate with optimal pose
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    img = drr(rotations, translations,
             parameterization="euler_angles",
             convention="ZXY")
    
    # Process and save
    img_np = img.squeeze().cpu().numpy()
    print(f"High-res range: [{img_np.min():.4f}, {img_np.max():.4f}]")
    
    # Log transform
    img_log = -np.log(np.maximum(img_np, 1e-6))
    img_norm = (img_log - img_log.min()) / (img_log.max() - img_log.min())
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_norm, cmap='gray')
    plt.title(f"High Resolution (512x512)\nPixel spacing: 0.7mm")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "high_resolution.png", dpi=150)
    plt.close()
    
    print(f"High-res image saved to: {output_dir}/high_resolution.png")


if __name__ == "__main__":
    # Run simple test first
    generate_drr_simple()
    
    # Then test with higher resolution
    test_with_higher_resolution()