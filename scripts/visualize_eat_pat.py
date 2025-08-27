#!/usr/bin/env python3
"""
EAT+PAT抽出結果の可視化
心臓、膨張領域、脂肪マスクを表示
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def create_visualization(ct_path, eat_pat_mask_path, heart_dilated_path, 
                        output_dir, stats_path=None, num_slices=15):
    """
    EAT+PAT結果を可視化
    """
    print("[INFO] Loading data...")
    
    # CTデータ読み込み
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # EAT+PATマスク読み込み
    eat_pat_img = nib.load(eat_pat_mask_path)
    eat_pat_mask = eat_pat_img.get_fdata() > 0
    
    # 心臓膨張マスク読み込み
    heart_dilated_img = nib.load(heart_dilated_path)
    heart_dilated = heart_dilated_img.get_fdata() > 0
    
    # 統計情報読み込み
    if stats_path and Path(stats_path).exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"\n[Statistics]")
        print(f"  EAT+PAT Volume: {stats['eat_pat_volume_ml']:.2f} ml")
        print(f"  Voxels: {stats['eat_pat_voxels']:,}")
        if stats['eat_pat_mean_hu']:
            print(f"  Mean HU: {stats['eat_pat_mean_hu']:.1f} ± {stats['eat_pat_std_hu']:.1f}")
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ
    (output_dir / "slices").mkdir(exist_ok=True)
    (output_dir / "overlay").mkdir(exist_ok=True)
    (output_dir / "combined").mkdir(exist_ok=True)
    
    # スライス選択（EAT+PATが存在する範囲）
    eat_pat_slices = np.where(np.any(eat_pat_mask, axis=(0, 1)))[0]
    
    if len(eat_pat_slices) == 0:
        print("[WARNING] No EAT+PAT found in the mask")
        eat_pat_slices = np.array([ct_data.shape[2] // 2])
    
    z_start = eat_pat_slices[0]
    z_end = eat_pat_slices[-1]
    z_range = z_end - z_start + 1
    
    print(f"\n[INFO] EAT+PAT found in slices {z_start} to {z_end} ({z_range} slices)")
    
    # 等間隔でスライスを選択
    if z_range < num_slices:
        slice_indices = eat_pat_slices[::max(1, len(eat_pat_slices) // num_slices)][:num_slices]
    else:
        slice_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
    
    print(f"[INFO] Creating visualizations for {len(slice_indices)} slices...")
    
    # 脂肪のHU範囲
    fat_hu_min = -190
    fat_hu_max = -30
    fat_in_hu_range = (ct_data >= fat_hu_min) & (ct_data <= fat_hu_max)
    
    for idx, z in enumerate(slice_indices):
        print(f"  Processing slice {z}/{ct_data.shape[2]}...")
        
        # スライス抽出
        ct_slice = ct_data[:, :, z]
        eat_pat_slice = eat_pat_mask[:, :, z]
        heart_dilated_slice = heart_dilated[:, :, z]
        fat_hu_slice = fat_in_hu_range[:, :, z]
        
        # Window/Level調整（胸部CT用）
        window_center = 40
        window_width = 400
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        
        # 1. 基本画像（CT + EAT+PAT）
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        masked_eat_pat = np.ma.masked_where(eat_pat_slice < 0.5, eat_pat_slice)
        ax.imshow(masked_eat_pat, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'EAT+PAT - Slice {z}')
        plt.savefig(output_dir / "slices" / f"slice_{z:04d}_eat_pat.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. オーバーレイ（心臓領域、脂肪、EAT+PAT）
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        
        # 心臓膨張領域の輪郭
        ax.contour(heart_dilated_slice, levels=[0.5], colors='blue', linewidths=1, alpha=0.5)
        
        # すべての脂肪（HU範囲）
        masked_fat = np.ma.masked_where(fat_hu_slice < 0.5, fat_hu_slice)
        ax.imshow(masked_fat, cmap='YlOrBr', alpha=0.2, vmin=0, vmax=1)
        
        # EAT+PAT
        ax.imshow(masked_eat_pat, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        
        ax.axis('off')
        ax.set_title(f'Overlay - Slice {z}')
        plt.savefig(output_dir / "overlay" / f"slice_{z:04d}_overlay.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. 3パネル表示
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: CT
        axes[0].imshow(ct_windowed, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('CT Image')
        
        # Panel 2: 心臓膨張領域
        axes[1].imshow(ct_windowed, cmap='gray', alpha=0.7)
        masked_heart = np.ma.masked_where(heart_dilated_slice < 0.5, heart_dilated_slice)
        axes[1].imshow(masked_heart, cmap='Blues', alpha=0.3)
        axes[1].axis('off')
        axes[1].set_title('Heart ROI (15mm dilated)')
        
        # Panel 3: EAT+PAT
        axes[2].imshow(ct_windowed, cmap='gray', alpha=0.7)
        axes[2].imshow(masked_eat_pat, cmap='hot', alpha=0.6)
        axes[2].axis('off')
        axes[2].set_title('EAT+PAT')
        
        plt.suptitle(f'Slice {z}', fontsize=12)
        plt.savefig(output_dir / "combined" / f"slice_{z:04d}_combined.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
    
    # サマリー画像（代表的なスライス）
    print("\n[INFO] Creating summary image...")
    
    middle_idx = len(slice_indices) // 2
    summary_indices = slice_indices[max(0, middle_idx-2):min(len(slice_indices), middle_idx+3)][:5]
    
    fig, axes = plt.subplots(2, len(summary_indices), figsize=(20, 8))
    
    for i, z in enumerate(summary_indices):
        ct_slice = ct_data[:, :, z]
        eat_pat_slice = eat_pat_mask[:, :, z]
        heart_dilated_slice = heart_dilated[:, :, z]
        
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        
        # Top row: CT + Heart ROI
        axes[0, i].imshow(ct_windowed, cmap='gray')
        masked_heart = np.ma.masked_where(heart_dilated_slice < 0.5, heart_dilated_slice)
        axes[0, i].imshow(masked_heart, cmap='Blues', alpha=0.3)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Slice {z}')
        
        # Bottom row: CT + EAT+PAT
        axes[1, i].imshow(ct_windowed, cmap='gray')
        masked_eat_pat = np.ma.masked_where(eat_pat_slice < 0.5, eat_pat_slice)
        axes[1, i].imshow(masked_eat_pat, cmap='hot', alpha=0.6)
        axes[1, i].axis('off')
    
    # 左側にラベル
    axes[0, 0].set_ylabel('Heart ROI\n(15mm dilated)', fontsize=12)
    axes[1, 0].set_ylabel('EAT+PAT', fontsize=12)
    
    # タイトルと統計情報
    if stats_path and Path(stats_path).exists():
        title = f"EAT+PAT Extraction Results\n"
        title += f"Volume: {stats['eat_pat_volume_ml']:.2f} ml | "
        title += f"Voxels: {stats['eat_pat_voxels']:,}"
        if stats['eat_pat_mean_hu']:
            title += f" | Mean HU: {stats['eat_pat_mean_hu']:.1f}±{stats['eat_pat_std_hu']:.1f}"
    else:
        title = "EAT+PAT Extraction Results"
    
    plt.suptitle(title, fontsize=14)
    plt.savefig(output_dir / "summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # 3D投影図（オプション）
    print("[INFO] Creating 3D projection...")
    
    # 軸ごとの最大値投影（MIP）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial MIP (Z軸方向)
    mip_axial = np.max(eat_pat_mask, axis=2)
    axes[0].imshow(mip_axial.T, cmap='hot', origin='lower')
    axes[0].set_title('Axial MIP')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axis('equal')
    
    # Coronal MIP (Y軸方向)
    mip_coronal = np.max(eat_pat_mask, axis=1)
    axes[1].imshow(mip_coronal.T, cmap='hot', origin='lower', aspect=spacing[2]/spacing[0])
    axes[1].set_title('Coronal MIP')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    # Sagittal MIP (X軸方向)
    mip_sagittal = np.max(eat_pat_mask, axis=0)
    axes[2].imshow(mip_sagittal.T, cmap='hot', origin='lower', aspect=spacing[2]/spacing[1])
    axes[2].set_title('Sagittal MIP')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    
    plt.suptitle('EAT+PAT Maximum Intensity Projection', fontsize=14)
    plt.savefig(output_dir / "mip_3views.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"\n[INFO] Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize EAT+PAT extraction results"
    )
    
    parser.add_argument(
        "ct_path",
        help="Path to CT NIfTI file"
    )
    
    parser.add_argument(
        "eat_pat_mask",
        help="Path to EAT+PAT mask NIfTI file"
    )
    
    parser.add_argument(
        "heart_dilated",
        help="Path to dilated heart mask NIfTI file"
    )
    
    parser.add_argument(
        "output_dir",
        help="Path to output directory for visualizations"
    )
    
    parser.add_argument(
        "--stats",
        help="Path to statistics JSON file"
    )
    
    parser.add_argument(
        "--num-slices",
        type=int,
        default=15,
        help="Number of slices to visualize (default: 15)"
    )
    
    args = parser.parse_args()
    
    create_visualization(
        ct_path=args.ct_path,
        eat_pat_mask_path=args.eat_pat_mask,
        heart_dilated_path=args.heart_dilated,
        output_dir=args.output_dir,
        stats_path=args.stats,
        num_slices=args.num_slices
    )
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()