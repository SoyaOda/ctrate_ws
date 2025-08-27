#!/usr/bin/env python3
"""
改善版DeepHeartSegの結果を可視化
心臓マスクと心膜マスクを表示
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.ndimage import distance_transform_edt

def create_visualization(ct_path, heart_path, peri_path, output_dir, num_slices=15):
    """
    DeepHeartSeg改善版の結果を可視化
    """
    print("[INFO] Loading data...")
    
    # CTデータ読み込み
    ct_img = sitk.ReadImage(ct_path)
    ct_data = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  CT spacing: {spacing}")
    
    # 心臓マスク読み込み
    heart_img = sitk.ReadImage(heart_path)
    # リサンプリングが必要な場合
    if heart_img.GetSize() != ct_img.GetSize():
        print("[INFO] Resampling heart mask to match CT...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        heart_img = resampler.Execute(heart_img)
    heart_data = sitk.GetArrayFromImage(heart_img)
    
    # 心膜マスク読み込み
    peri_img = sitk.ReadImage(peri_path)
    # リサンプリングが必要な場合
    if peri_img.GetSize() != ct_img.GetSize():
        print("[INFO] Resampling pericardium mask to match CT...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        peri_img = resampler.Execute(peri_img)
    peri_data = sitk.GetArrayFromImage(peri_img)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ
    (output_dir / "heart").mkdir(exist_ok=True)
    (output_dir / "pericardium").mkdir(exist_ok=True)
    (output_dir / "combined").mkdir(exist_ok=True)
    (output_dir / "eat_region").mkdir(exist_ok=True)
    
    # スライス選択（心臓が存在する範囲）
    heart_slices = np.where(np.any(heart_data > 0, axis=(1, 2)))[0]
    
    if len(heart_slices) == 0:
        print("[ERROR] No heart segmentation found")
        return
    
    z_start = heart_slices[0]
    z_end = heart_slices[-1]
    z_range = z_end - z_start + 1
    
    print(f"\n[INFO] Heart found in slices {z_start} to {z_end} ({z_range} slices)")
    
    # 選択するスライス
    if z_range < num_slices:
        slice_indices = heart_slices[::max(1, len(heart_slices) // num_slices)][:num_slices]
    else:
        slice_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
    
    print(f"[INFO] Creating visualizations for {len(slice_indices)} slices...")
    
    # EAT候補領域を計算（心膜内の脂肪）
    fat_hu_min = -190
    fat_hu_max = -30
    fat_mask = (ct_data >= fat_hu_min) & (ct_data <= fat_hu_max)
    
    # 心膜内かつ心臓外の領域
    eat_candidate = peri_data & (~heart_data) & fat_mask
    
    for idx, z in enumerate(slice_indices):
        print(f"  Processing slice {z}/{ct_data.shape[0]}...")
        
        # CTスライス
        ct_slice = ct_data[z, :, :]
        
        # Window/Level調整
        window_center = 40
        window_width = 400
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        
        # マスクスライス
        heart_slice = heart_data[z, :, :].astype(float)
        peri_slice = peri_data[z, :, :].astype(float)
        eat_slice = eat_candidate[z, :, :].astype(float)
        
        # 1. 心臓マスク
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        masked_heart = np.ma.masked_where(heart_slice < 0.5, heart_slice)
        ax.imshow(masked_heart, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Heart Mask - Slice {z}')
        plt.savefig(output_dir / "heart" / f"slice_{z:04d}_heart.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. 心膜マスク
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        masked_peri = np.ma.masked_where(peri_slice < 0.5, peri_slice)
        ax.imshow(masked_peri, cmap='Blues', alpha=0.4, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Pericardium Mask - Slice {z}')
        plt.savefig(output_dir / "pericardium" / f"slice_{z:04d}_peri.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. 結合表示
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        ax.imshow(masked_heart, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
        ax.imshow(masked_peri, cmap='Blues', alpha=0.2, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Heart + Pericardium - Slice {z}')
        plt.savefig(output_dir / "combined" / f"slice_{z:04d}_combined.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 4. EAT候補領域
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        masked_eat = np.ma.masked_where(eat_slice < 0.5, eat_slice)
        ax.imshow(masked_eat, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        ax.contour(heart_slice, levels=[0.5], colors='red', linewidths=1)
        ax.contour(peri_slice, levels=[0.5], colors='blue', linewidths=1)
        ax.axis('off')
        ax.set_title(f'EAT Candidate Region - Slice {z}')
        plt.savefig(output_dir / "eat_region" / f"slice_{z:04d}_eat.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
    
    # サマリー画像（4パネル表示）
    print("[INFO] Creating summary image...")
    middle_idx = len(slice_indices) // 2
    z_middle = slice_indices[middle_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # CTスライス準備
    ct_slice = ct_data[z_middle, :, :]
    ct_windowed = np.clip(
        (ct_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    
    heart_slice = heart_data[z_middle, :, :].astype(float)
    peri_slice = peri_data[z_middle, :, :].astype(float)
    eat_slice = eat_candidate[z_middle, :, :].astype(float)
    
    # Panel 1: CT Only
    axes[0, 0].imshow(ct_windowed, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original CT', fontsize=14)
    
    # Panel 2: Heart
    axes[0, 1].imshow(ct_windowed, cmap='gray')
    masked_heart = np.ma.masked_where(heart_slice < 0.5, heart_slice)
    axes[0, 1].imshow(masked_heart, cmap='Reds', alpha=0.4)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Heart Segmentation', fontsize=14)
    
    # Panel 3: Pericardium
    axes[1, 0].imshow(ct_windowed, cmap='gray')
    masked_peri = np.ma.masked_where(peri_slice < 0.5, peri_slice)
    axes[1, 0].imshow(masked_peri, cmap='Blues', alpha=0.4)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Pericardium Region', fontsize=14)
    
    # Panel 4: EAT Candidate
    axes[1, 1].imshow(ct_windowed, cmap='gray')
    masked_eat = np.ma.masked_where(eat_slice < 0.5, eat_slice)
    axes[1, 1].imshow(masked_eat, cmap='Greens', alpha=0.5)
    axes[1, 1].contour(heart_slice, levels=[0.5], colors='red', linewidths=1)
    axes[1, 1].contour(peri_slice, levels=[0.5], colors='blue', linewidths=1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('EAT Candidate (Green)', fontsize=14)
    
    plt.suptitle(f'DeepHeartSeg Improved Results - Slice {z_middle}', fontsize=16)
    plt.savefig(output_dir / "summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[INFO] Visualizations saved to {output_dir}")
    
    # 統計情報
    heart_voxels = np.sum(heart_data > 0)
    peri_voxels = np.sum(peri_data > 0)
    eat_voxels = np.sum(eat_candidate)
    
    voxel_volume = np.prod(spacing[::-1]) / 1000.0  # mm^3 to ml
    heart_volume = heart_voxels * voxel_volume
    peri_volume = peri_voxels * voxel_volume
    eat_volume = eat_voxels * voxel_volume
    
    print("\n[STATISTICS]")
    print(f"  Heart: {heart_voxels:,} voxels ({heart_volume:.2f} ml)")
    print(f"  Pericardium: {peri_voxels:,} voxels ({peri_volume:.2f} ml)")
    print(f"  EAT candidate: {eat_voxels:,} voxels ({eat_volume:.2f} ml)")
    
    # 統計を保存
    stats = {
        "heart": {
            "voxels": int(heart_voxels),
            "volume_ml": float(heart_volume)
        },
        "pericardium": {
            "voxels": int(peri_voxels),
            "volume_ml": float(peri_volume)
        },
        "eat_candidate": {
            "voxels": int(eat_voxels),
            "volume_ml": float(eat_volume),
            "hu_range": [fat_hu_min, fat_hu_max]
        },
        "ct_shape": ct_data.shape,
        "ct_spacing": spacing
    }
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  Statistics saved to {stats_path}")

def main():
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    heart_path = "/home/soya/ctrate_ws/outputs/deepheartseg_improved/heart_mask_improved.nii.gz"
    peri_path = "/home/soya/ctrate_ws/outputs/deepheartseg_improved/pseudo_pericardium.nii.gz"
    output_dir = "/home/soya/ctrate_ws/outputs/deepheartseg_improved_visualization"
    
    print("="*60)
    print("DeepHeartSeg Improved Visualization")
    print("="*60)
    print(f"CT: {ct_path}")
    print(f"Heart mask: {heart_path}")
    print(f"Pericardium mask: {peri_path}")
    print(f"Output: {output_dir}")
    print()
    
    create_visualization(ct_path, heart_path, peri_path, output_dir)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)

if __name__ == "__main__":
    main()