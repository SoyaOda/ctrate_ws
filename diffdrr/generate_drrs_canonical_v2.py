#!/usr/bin/env python3
"""
高忠実度DRR生成 v2
ToCanonicalで標準化したHUボリュームをDiffDRRのread()で処理
"""

import torch
import torchio as tio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffdrr.drr import DRR
from diffdrr.pose import convert
from diffdrr.data import read
import tempfile
import nibabel as nib


def preprocess_and_save_canonical_ct(ct_path: str) -> str:
    """
    CTをToCanonicalで標準化し、HUのまま一時ファイルに保存
    """
    print("Loading and standardizing CT volume...")
    
    # Load original CT
    subject = tio.Subject(
        ct=tio.ScalarImage(ct_path)
    )
    
    print(f"Original shape: {subject.ct.shape}")
    print(f"Original orientation: {subject.ct.orientation}")
    
    # Apply ToCanonical to standardize to RAS+
    to_canonical = tio.ToCanonical()
    subject_canonical = to_canonical(subject)
    
    print(f"Canonical shape: {subject_canonical.ct.shape}")
    print(f"Canonical orientation: {subject_canonical.ct.orientation}")
    
    # Get HU data and affine
    hu_data = subject_canonical.ct.data.squeeze()
    affine = subject_canonical.ct.affine
    
    if torch.is_tensor(hu_data):
        hu_data = hu_data.cpu().numpy()
    if torch.is_tensor(affine):
        affine = affine.cpu().numpy()
    
    print(f"HU range: [{hu_data.min():.1f}, {hu_data.max():.1f}]")
    
    # Save canonical HU volume
    temp_file = tempfile.NamedTemporaryFile(suffix='_canonical.nii.gz', delete=False)
    nifti_image = nib.Nifti1Image(hu_data, affine)
    nib.save(nifti_image, temp_file.name)
    
    print(f"Saved canonical HU volume to: {temp_file.name}")
    return temp_file.name


def generate_chest_xray_drr(canonical_ct_path: str, output_dir: Path):
    """
    標準化されたCTからDRRを生成
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load with DiffDRR's read function
    # This will apply its own HU to density conversion
    print("\nLoading canonical CT with DiffDRR...")
    subject = read(
        canonical_ct_path,
        bone_attenuation_multiplier=2.5  # Moderate bone emphasis
    )
    
    # Check loaded data
    print(f"Volume shape: {subject.volume.data.shape}")
    vol_range = [subject.volume.data.min().item(), subject.volume.data.max().item()]
    print(f"Volume range: {vol_range}")
    
    if hasattr(subject, 'density'):
        dens_range = [subject.density.data.min().item(), subject.density.data.max().item()]
        print(f"Density range: {dens_range}")
    
    # Create DRR generator
    print("\nCreating DRR generator...")
    drr = DRR(
        subject=subject,
        sdd=1800.0,  # Standard PA chest X-ray distance
        height=512,
        width=512,
        delx=0.7,  # 0.7mm pixel spacing
        dely=0.7,
    ).to(device)
    
    # Generate multiple views
    views = [
        {
            'name': 'PA_frontal',
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, -50.0, 0.0],  # Adjusted Y position
            'description': 'Standard PA view'
        },
        {
            'name': 'PA_elevated',
            'rotation': [0.0, 0.0, 0.0],
            'translation': [0.0, -100.0, 0.0],  # Higher Y position
            'description': 'PA view (elevated)'
        },
        {
            'name': 'PA_rotated',
            'rotation': [0.0, 0.0, 10.0],  # 10 degree rotation
            'translation': [0.0, -75.0, 0.0],
            'description': 'PA with rotation'
        },
        {
            'name': 'Lateral',
            'rotation': [0.0, 90.0, 0.0],  # 90 degree Y rotation
            'translation': [-100.0, 0.0, 0.0],  # X translation for lateral
            'description': 'Lateral view'
        },
    ]
    
    results = []
    
    for view in views:
        print(f"\nGenerating {view['name']}...")
        
        # Setup pose
        rotations = torch.tensor([view['rotation']], dtype=torch.float32)
        translations = torch.tensor([view['translation']], dtype=torch.float32)
        
        rotations_rad = torch.deg2rad(rotations).to(device)
        translations = translations.to(device)
        
        transform = convert(
            rotations_rad,
            translations,
            parameterization="euler_angles",
            convention="XYZ",
        ).to(device)
        
        # Generate DRR
        img_raw = drr(transform)
        img_np = img_raw.squeeze().cpu().numpy()
        
        print(f"  Raw range: [{img_np.min():.4f}, {img_np.max():.4f}]")
        
        # Enhanced post-processing for chest X-ray appearance
        # 1. Ensure positive values
        img_np = np.maximum(img_np, 1e-6)
        
        # 2. Log transform (Beer-Lambert law)
        img_log = -np.log(img_np / img_np.max())
        
        # 3. Enhance contrast with power law
        img_power = np.power(img_log / img_log.max(), 0.8)
        
        # 4. Histogram equalization
        from skimage import exposure
        img_eq = exposure.equalize_adapthist(
            img_power,
            clip_limit=0.03
        )
        
        # 5. Final normalization
        img_final = (img_eq - img_eq.min()) / (img_eq.max() - img_eq.min())
        
        # Save result
        img_path = output_dir / f"{view['name']}.png"
        plt.imsave(img_path, img_final, cmap='gray')
        print(f"  Saved to: {img_path}")
        
        results.append({
            'name': view['name'],
            'description': view['description'],
            'image': img_final,
            'raw_range': [img_np.min(), img_np.max()],
            'rotation': view['rotation'],
            'translation': view['translation']
        })
    
    return results


def create_visualization(results, output_dir):
    """
    Create comprehensive visualization
    """
    n_views = len(results)
    fig = plt.figure(figsize=(5*n_views, 6))
    
    for idx, result in enumerate(results):
        ax = plt.subplot(1, n_views, idx+1)
        ax.imshow(result['image'], cmap='gray')
        ax.set_title(f"{result['name']}\n{result['description']}", fontsize=10)
        ax.axis('off')
        
        # Add technical info
        info_text = (f"Raw: [{result['raw_range'][0]:.3f}, {result['raw_range'][1]:.3f}]\n"
                    f"R: {result['rotation']}\n"
                    f"T: {result['translation']}")
        ax.text(0.5, -0.15, info_text, 
                transform=ax.transAxes, ha='center', fontsize=8,
                verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / "all_views_canonical_v2.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_path}")
    plt.show()


def main():
    """
    Main execution
    """
    # Setup paths
    CT_PATH = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    OUTPUT_DIR = Path("./drr_canonical_v2_output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("High-Fidelity DRR Generation v2")
    print("ToCanonical + DiffDRR read() approach")
    print("="*60)
    
    # Install scikit-image if needed
    try:
        from skimage import exposure
    except ImportError:
        print("\nInstalling scikit-image...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-image"])
    
    # Step 1: Preprocess and standardize CT
    print("\n[Step 1] Preprocessing CT with ToCanonical...")
    canonical_ct_path = preprocess_and_save_canonical_ct(CT_PATH)
    
    # Step 2: Generate DRRs
    print("\n[Step 2] Generating DRRs...")
    results = generate_chest_xray_drr(canonical_ct_path, OUTPUT_DIR)
    
    # Step 3: Create visualization
    print("\n[Step 3] Creating visualization...")
    create_visualization(results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ DRR generation completed!")
    print(f"✓ Results saved in: {OUTPUT_DIR}")
    print("="*60)
    
    # Cleanup temp file (optional)
    # import os
    # os.unlink(canonical_ct_path)


if __name__ == "__main__":
    main()