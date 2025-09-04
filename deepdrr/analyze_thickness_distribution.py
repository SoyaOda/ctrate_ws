#!/usr/bin/env python3
"""
厚みマップの分布を詳細に分析
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats
import os


def analyze_thickness_distribution():
    """厚みマップの分布を詳細に分析"""
    
    # データを読み込み
    output_dir = "/home/soya/ctrate_ws/deepdrr/final_output"
    thickness_calibrated = np.load(os.path.join(output_dir, "thickness_calibrated.npy"))
    thickness_raw = np.load(os.path.join(output_dir, "thickness_raw.npy"))
    
    # 元の3Dマスクも読み込み
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    
    # Z方向の実際の厚み分布
    actual_z_thickness = np.sum(mask_binary, axis=2) * voxel_spacing[2]  # mm単位
    
    print("="*60)
    print("Thickness Distribution Analysis")
    print("="*60)
    
    # 1. キャリブレート後の厚み分析
    print("\n1. Calibrated Thickness Map:")
    print("-"*40)
    non_zero_calibrated = thickness_calibrated[thickness_calibrated > 0]
    print(f"Non-zero pixels: {len(non_zero_calibrated)}")
    print(f"Range: [{non_zero_calibrated.min():.3f}, {non_zero_calibrated.max():.3f}] mm")
    print(f"Mean: {non_zero_calibrated.mean():.3f} mm")
    print(f"Std: {non_zero_calibrated.std():.3f} mm")
    print(f"Coefficient of Variation: {non_zero_calibrated.std() / non_zero_calibrated.mean() * 100:.1f}%")
    
    # ユニークな値の数
    unique_values = np.unique(non_zero_calibrated)
    print(f"Number of unique values: {len(unique_values)}")
    if len(unique_values) < 20:
        print(f"Unique values: {unique_values}")
    
    # 2. 生の厚み分析
    print("\n2. Raw Thickness Map:")
    print("-"*40)
    non_zero_raw = thickness_raw[thickness_raw > 0]
    print(f"Non-zero pixels: {len(non_zero_raw)}")
    print(f"Range: [{non_zero_raw.min():.3f}, {non_zero_raw.max():.3f}] mm")
    print(f"Mean: {non_zero_raw.mean():.3f} mm")
    print(f"Std: {non_zero_raw.std():.3f} mm")
    print(f"Coefficient of Variation: {non_zero_raw.std() / non_zero_raw.mean() * 100:.1f}%")
    
    # 3. 実際のZ方向厚み分析
    print("\n3. Actual Z-direction Thickness (Ground Truth):")
    print("-"*40)
    actual_non_zero = actual_z_thickness[actual_z_thickness > 0]
    print(f"Non-zero pixels: {len(actual_non_zero)}")
    print(f"Range: [{actual_non_zero.min():.3f}, {actual_non_zero.max():.3f}] mm")
    print(f"Mean: {actual_non_zero.mean():.3f} mm")
    print(f"Std: {actual_non_zero.std():.3f} mm")
    print(f"Coefficient of Variation: {actual_non_zero.std() / actual_non_zero.mean() * 100:.1f}%")
    
    unique_actual = np.unique(actual_non_zero)
    print(f"Number of unique values in actual: {len(unique_actual)}")
    
    # 4. 可視化
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: 厚みマップ
    im0 = axes[0, 0].imshow(thickness_calibrated, cmap='hot', vmin=0, vmax=2)
    axes[0, 0].set_title('Calibrated Thickness')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[0, 1].imshow(thickness_raw, cmap='hot')
    axes[0, 1].set_title('Raw Thickness')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(actual_z_thickness, cmap='hot')
    axes[0, 2].set_title('Actual Z-thickness')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # Row 2: ヒストグラム
    axes[1, 0].hist(non_zero_calibrated, bins=50, edgecolor='black')
    axes[1, 0].set_title('Calibrated Thickness Distribution')
    axes[1, 0].set_xlabel('Thickness (mm)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(non_zero_raw, bins=50, edgecolor='black')
    axes[1, 1].set_title('Raw Thickness Distribution')
    axes[1, 1].set_xlabel('Thickness (mm)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(actual_non_zero, bins=50, edgecolor='black')
    axes[1, 2].set_title('Actual Z-thickness Distribution')
    axes[1, 2].set_xlabel('Thickness (mm)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: 詳細分析
    # カラーマップを拡大表示（中央部分）
    h, w = thickness_calibrated.shape
    center_region = thickness_calibrated[h//2-50:h//2+50, w//2-50:w//2+50]
    im3 = axes[2, 0].imshow(center_region, cmap='hot', vmin=0, vmax=2)
    axes[2, 0].set_title('Calibrated (Center 100x100)')
    axes[2, 0].axis('off')
    plt.colorbar(im3, ax=axes[2, 0], fraction=0.046)
    
    # 差分マップ
    # 実際のZ厚みをリサイズして比較
    h_actual, w_actual = actual_z_thickness.shape
    h_target, w_target = thickness_calibrated.shape
    
    # 中央部分を切り出し
    h_start = max(0, (h_actual - h_target) // 2)
    w_start = max(0, (w_actual - w_target) // 2)
    actual_cropped = actual_z_thickness[h_start:h_start+h_target, w_start:w_start+w_target]
    
    # パディングが必要な場合
    if actual_cropped.shape != thickness_calibrated.shape:
        pad_h = (h_target - actual_cropped.shape[0]) // 2
        pad_w = (w_target - actual_cropped.shape[1]) // 2
        actual_cropped = np.pad(actual_cropped, ((pad_h, h_target-actual_cropped.shape[0]-pad_h), 
                                                  (pad_w, w_target-actual_cropped.shape[1]-pad_w)), 
                                mode='constant')
    
    # スケールを合わせるため正規化
    scale_factor = np.sum(thickness_calibrated) / np.sum(actual_cropped) if np.sum(actual_cropped) > 0 else 1
    actual_scaled = actual_cropped * scale_factor
    
    diff = thickness_calibrated - actual_scaled
    im4 = axes[2, 1].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[2, 1].set_title('Difference (Calibrated - Actual scaled)')
    axes[2, 1].axis('off')
    plt.colorbar(im4, ax=axes[2, 1], fraction=0.046)
    
    # 散布図
    mask_both = (thickness_calibrated > 0) & (actual_scaled > 0)
    if np.any(mask_both):
        axes[2, 2].scatter(actual_scaled[mask_both], thickness_calibrated[mask_both], 
                          alpha=0.1, s=1)
        axes[2, 2].plot([0, 70], [0, 70], 'r--', alpha=0.5)  # 理想的な線
        axes[2, 2].set_xlabel('Actual scaled thickness (mm)')
        axes[2, 2].set_ylabel('Calibrated thickness (mm)')
        axes[2, 2].set_title('Correlation')
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_xlim([0, max(actual_scaled[mask_both].max(), thickness_calibrated[mask_both].max())])
        axes[2, 2].set_ylim([0, max(actual_scaled[mask_both].max(), thickness_calibrated[mask_both].max())])
    
    plt.tight_layout()
    plt.savefig('/home/soya/ctrate_ws/deepdrr/thickness_analysis.png', dpi=150)
    print(f"\nAnalysis saved to: /home/soya/ctrate_ws/deepdrr/thickness_analysis.png")
    
    # 5. 問題の診断
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if non_zero_calibrated.std() / non_zero_calibrated.mean() < 0.1:
        print("⚠️ WARNING: Very low variation in calibrated thickness!")
        print("   This suggests the projection is too uniform.")
        print("   Possible causes:")
        print("   1. Beer-Lambert compression is still too strong")
        print("   2. Calibration factor is compressing the dynamic range")
        print("   3. The attenuation scale (0.001) might be too small")
    
    compression_ratio = (actual_non_zero.max() - actual_non_zero.min()) / (non_zero_calibrated.max() - non_zero_calibrated.min())
    print(f"\nDynamic range compression ratio: {compression_ratio:.1f}x")
    print(f"  Actual range: {actual_non_zero.max() - actual_non_zero.min():.1f} mm")
    print(f"  Calibrated range: {non_zero_calibrated.max() - non_zero_calibrated.min():.3f} mm")
    
    if compression_ratio > 10:
        print("⚠️ Severe dynamic range compression detected!")
    
    return thickness_calibrated, thickness_raw, actual_z_thickness


if __name__ == "__main__":
    analyze_thickness_distribution()