#!/usr/bin/env python3
"""
EAT+PAT v5.3可視化スクリプト（接触領域表示版）
CT、Shell、EAT+PAT、ILAM追加領域、肺接触成分、胃接触成分を可視化
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import argparse
import json

def load_all_masks(masks_dir, ct_img):
    """
    すべてのマスクを読み込み（接触分類マスク含む）
    """
    masks = {}
    mask_files = {
        'shell': 'shell.nii.gz',
        'eat_pat': 'eat_pat.nii.gz',
        'ilam_addition': 'ilam_addition.nii.gz',
        'visceral_fat': 'visceral_fat.nii.gz',
        'lung_contact': 'lung_contact_components.nii.gz',
        'stomach_contact': 'stomach_contact_components.nii.gz'
    }
    
    for name, filename in mask_files.items():
        mask_path = masks_dir / filename
        if mask_path.exists():
            img = nib.load(str(mask_path))
            masks[name] = img.get_fdata() > 0
            print(f"  Loaded {name}: {np.sum(masks[name]):,} voxels")
        else:
            print(f"  Warning: {name} mask not found")
            masks[name] = np.zeros(ct_img.shape, dtype=bool)
    
    return masks

def create_slice_visualization(ct_data, masks, slice_idx, spacing, output_path):
    """
    単一スライスの可視化（6パネル - 接触分類表示）
    """
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.15, wspace=0.15)
    
    ct_slice = ct_data[:, :, slice_idx]
    
    # Window level for soft tissue
    window_center = 40
    window_width = 400
    ct_min = window_center - window_width / 2
    ct_max = window_center + window_width / 2
    ct_windowed = np.clip(ct_slice, ct_min, ct_max)
    ct_windowed = (ct_windowed - ct_min) / (ct_max - ct_min)
    
    # Panel 1: CT Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_windowed, cmap='gray')
    ax1.set_title(f'CT Image (Slice {slice_idx})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Shell Visualization
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_windowed, cmap='gray', alpha=0.7)
    shell_slice = masks['shell'][:, :, slice_idx]
    ax2.contour(shell_slice, levels=[0.5], colors='cyan', linewidths=2)
    ax2.set_title('Shell (15mm Dilation)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: EAT+PAT
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_windowed, cmap='gray', alpha=0.7)
    eat_pat_slice = masks['eat_pat'][:, :, slice_idx]
    ax3.contour(eat_pat_slice, levels=[0.5], colors='red', linewidths=2)
    ax3.set_title('EAT+PAT (Final)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: ILAM Addition
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(ct_windowed, cmap='gray', alpha=0.7)
    ilam_slice = masks['ilam_addition'][:, :, slice_idx]
    if np.any(ilam_slice):
        ax4.contour(ilam_slice, levels=[0.5], colors='yellow', linewidths=2)
    ax4.contour(eat_pat_slice, levels=[0.5], colors='red', linewidths=1, alpha=0.5)
    ax4.set_title('ILAM Addition (Yellow)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Panel 5: Lung Contact Components
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(ct_windowed, cmap='gray', alpha=0.7)
    lung_contact_slice = masks['lung_contact'][:, :, slice_idx]
    if np.any(lung_contact_slice):
        ax5.contour(lung_contact_slice, levels=[0.5], colors='blue', linewidths=2)
    ax5.contour(shell_slice, levels=[0.5], colors='cyan', linewidths=1, alpha=0.3)
    ax5.set_title('Lung Contact Components', fontsize=14, fontweight='bold', color='blue')
    ax5.axis('off')
    
    # Panel 6: Stomach Contact Components
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(ct_windowed, cmap='gray', alpha=0.7)
    stomach_contact_slice = masks['stomach_contact'][:, :, slice_idx]
    if np.any(stomach_contact_slice):
        ax6.contour(stomach_contact_slice, levels=[0.5], colors='orange', linewidths=2)
    ax6.contour(shell_slice, levels=[0.5], colors='cyan', linewidths=1, alpha=0.3)
    ax6.set_title('Stomach Contact Components', fontsize=14, fontweight='bold', color='orange')
    ax6.axis('off')
    
    # Add legends with volume info
    shell_voxels = np.sum(shell_slice)
    eat_pat_voxels = np.sum(eat_pat_slice)
    ilam_voxels = np.sum(ilam_slice)
    lung_contact_voxels = np.sum(lung_contact_slice)
    stomach_contact_voxels = np.sum(stomach_contact_slice)
    
    voxel_volume = np.prod(spacing) / 1000.0  # ml
    
    info_text = (
        f"Slice {slice_idx} Statistics:\n"
        f"Shell: {shell_voxels:,} voxels ({shell_voxels * voxel_volume:.2f} ml)\n"
        f"EAT+PAT: {eat_pat_voxels:,} voxels ({eat_pat_voxels * voxel_volume:.2f} ml)\n"
        f"ILAM add: {ilam_voxels:,} voxels ({ilam_voxels * voxel_volume:.2f} ml)\n"
        f"Lung contact: {lung_contact_voxels:,} voxels ({lung_contact_voxels * voxel_volume:.2f} ml)\n"
        f"Stomach contact: {stomach_contact_voxels:,} voxels ({stomach_contact_voxels * voxel_volume:.2f} ml)"
    )
    
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'EAT+PAT Analysis v5.3 - Slice {slice_idx}', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'shell_voxels': int(shell_voxels),
        'eat_pat_voxels': int(eat_pat_voxels),
        'ilam_voxels': int(ilam_voxels),
        'lung_contact_voxels': int(lung_contact_voxels),
        'stomach_contact_voxels': int(stomach_contact_voxels)
    }

def create_mip_visualization(ct_data, masks, spacing, output_path):
    """
    MIP (Maximum Intensity Projection) 可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create MIPs
    ct_mip = np.max(ct_data, axis=1)  # Coronal MIP
    
    # Window level for soft tissue
    window_center = 40
    window_width = 400
    ct_min = window_center - window_width / 2
    ct_max = window_center + window_width / 2
    ct_windowed = np.clip(ct_mip, ct_min, ct_max)
    ct_windowed = (ct_windowed - ct_min) / (ct_max - ct_min)
    
    # MIPs for masks
    shell_mip = np.max(masks['shell'], axis=1)
    eat_pat_mip = np.max(masks['eat_pat'], axis=1)
    ilam_mip = np.max(masks['ilam_addition'], axis=1)
    lung_contact_mip = np.max(masks['lung_contact'], axis=1)
    stomach_contact_mip = np.max(masks['stomach_contact'], axis=1)
    
    # Plot CT
    axes[0, 0].imshow(ct_windowed.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('CT MIP (Coronal)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Plot Shell
    axes[0, 1].imshow(ct_windowed.T, cmap='gray', alpha=0.7, origin='lower')
    axes[0, 1].contour(shell_mip.T, levels=[0.5], colors='cyan', linewidths=2)
    axes[0, 1].set_title('Shell', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Plot EAT+PAT
    axes[0, 2].imshow(ct_windowed.T, cmap='gray', alpha=0.7, origin='lower')
    axes[0, 2].contour(eat_pat_mip.T, levels=[0.5], colors='red', linewidths=2)
    axes[0, 2].set_title('EAT+PAT', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Plot ILAM
    axes[1, 0].imshow(ct_windowed.T, cmap='gray', alpha=0.7, origin='lower')
    if np.any(ilam_mip):
        axes[1, 0].contour(ilam_mip.T, levels=[0.5], colors='yellow', linewidths=2)
    axes[1, 0].set_title('ILAM Addition', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot Lung Contact
    axes[1, 1].imshow(ct_windowed.T, cmap='gray', alpha=0.7, origin='lower')
    if np.any(lung_contact_mip):
        axes[1, 1].contour(lung_contact_mip.T, levels=[0.5], colors='blue', linewidths=2)
    axes[1, 1].set_title('Lung Contact', fontsize=12, fontweight='bold', color='blue')
    axes[1, 1].axis('off')
    
    # Plot Stomach Contact
    axes[1, 2].imshow(ct_windowed.T, cmap='gray', alpha=0.7, origin='lower')
    if np.any(stomach_contact_mip):
        axes[1, 2].contour(stomach_contact_mip.T, levels=[0.5], colors='orange', linewidths=2)
    axes[1, 2].set_title('Stomach Contact', fontsize=12, fontweight='bold', color='orange')
    axes[1, 2].axis('off')
    
    plt.suptitle('EAT+PAT Analysis v5.3 - MIP Views', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize EAT+PAT v5.3 analysis with contact regions')
    parser.add_argument('ct_path', help='Path to CT NIfTI file')
    parser.add_argument('masks_dir', help='Path to masks directory')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('--slices', nargs='+', type=int, help='Specific slice indices to visualize')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading CT data...")
    ct_img = nib.load(args.ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    print("\nLoading masks...")
    masks_dir = Path(args.masks_dir)
    masks = load_all_masks(masks_dir, ct_img)
    
    # Find relevant slices if not specified
    if args.slices:
        slice_indices = args.slices
    else:
        # Find slices with significant EAT+PAT content
        eat_pat_sum = np.sum(masks['eat_pat'], axis=(0, 1))
        nonzero_slices = np.where(eat_pat_sum > 100)[0]
        
        if len(nonzero_slices) > 0:
            # Select evenly distributed slices
            n_slices = min(10, len(nonzero_slices))
            slice_indices = nonzero_slices[::max(1, len(nonzero_slices)//n_slices)][:n_slices]
        else:
            # Fallback to center slices
            center = ct_data.shape[2] // 2
            slice_indices = [center - 10, center, center + 10]
    
    print(f"\nVisualizing {len(slice_indices)} slices...")
    all_stats = []
    
    for i, slice_idx in enumerate(slice_indices):
        if 0 <= slice_idx < ct_data.shape[2]:
            output_path = output_dir / f"slice_{slice_idx:04d}.png"
            stats = create_slice_visualization(ct_data, masks, slice_idx, spacing, output_path)
            all_stats.append({'slice': int(slice_idx), **stats})
            if args.verbose:
                print(f"  Slice {slice_idx}: saved to {output_path}")
    
    # Create MIP visualization
    print("\nCreating MIP visualization...")
    mip_path = output_dir / "mip_comparison.png"
    create_mip_visualization(ct_data, masks, spacing, mip_path)
    print(f"  MIP saved to {mip_path}")
    
    # Create summary statistics
    print("\nCalculating summary statistics...")
    total_stats = {
        'total_shell_voxels': int(np.sum(masks['shell'])),
        'total_eat_pat_voxels': int(np.sum(masks['eat_pat'])),
        'total_ilam_voxels': int(np.sum(masks['ilam_addition'])),
        'total_lung_contact_voxels': int(np.sum(masks['lung_contact'])),
        'total_stomach_contact_voxels': int(np.sum(masks['stomach_contact'])),
        'slice_statistics': all_stats
    }
    
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    total_stats['volumes_ml'] = {
        'shell': float(round(total_stats['total_shell_voxels'] * voxel_volume_ml, 2)),
        'eat_pat': float(round(total_stats['total_eat_pat_voxels'] * voxel_volume_ml, 2)),
        'ilam_addition': float(round(total_stats['total_ilam_voxels'] * voxel_volume_ml, 2)),
        'lung_contact': float(round(total_stats['total_lung_contact_voxels'] * voxel_volume_ml, 2)),
        'stomach_contact': float(round(total_stats['total_stomach_contact_voxels'] * voxel_volume_ml, 2))
    }
    
    stats_path = output_dir / "visualization_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(total_stats, f, indent=2)
    print(f"  Statistics saved to {stats_path}")
    
    print("\nVisualization complete!")
    print(f"  Shell volume: {total_stats['volumes_ml']['shell']:.2f} ml")
    print(f"  EAT+PAT volume: {total_stats['volumes_ml']['eat_pat']:.2f} ml")
    print(f"  ILAM addition: {total_stats['volumes_ml']['ilam_addition']:.2f} ml")
    print(f"  Lung contact: {total_stats['volumes_ml']['lung_contact']:.2f} ml")
    print(f"  Stomach contact: {total_stats['volumes_ml']['stomach_contact']:.2f} ml")

if __name__ == "__main__":
    main()