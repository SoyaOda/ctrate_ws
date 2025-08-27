#!/usr/bin/env python3
"""
DeepHeartSeg心臓セグメンテーションの可視化
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def create_visualization(ct_path, heart_mask_path, output_dir, num_slices=15):
    """
    DeepHeartSegの心臓セグメンテーションを可視化
    """
    print("[INFO] Loading data...")
    
    # CTデータ読み込み
    ct_img = sitk.ReadImage(ct_path)
    ct_data = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  CT spacing: {spacing}")
    
    # 心臓マスク読み込み
    heart_img = sitk.ReadImage(heart_mask_path)
    
    # リサンプリングが必要な場合
    if heart_img.GetSize() != ct_img.GetSize():
        print("[INFO] Resampling heart mask to match CT...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        heart_img = resampler.Execute(heart_img)
    
    heart_data = sitk.GetArrayFromImage(heart_img)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    (output_dir / "ct_only").mkdir(exist_ok=True)
    (output_dir / "heart_overlay").mkdir(exist_ok=True)
    (output_dir / "heart_contour").mkdir(exist_ok=True)
    (output_dir / "side_by_side").mkdir(exist_ok=True)
    
    # スライス選択（心臓が存在する範囲）
    heart_slices = np.where(np.any(heart_data > 0, axis=(1, 2)))[0]
    
    if len(heart_slices) == 0:
        print("[ERROR] No heart segmentation found")
        return
    
    z_start = heart_slices[0]
    z_end = heart_slices[-1]
    z_range = z_end - z_start + 1
    
    print(f"[INFO] Heart found in slices {z_start} to {z_end} ({z_range} slices)")
    
    # 選択するスライス
    if z_range < num_slices:
        slice_indices = heart_slices
    else:
        slice_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
    
    print(f"[INFO] Creating visualizations for {len(slice_indices)} slices...")
    
    # カラーマップ設定
    heart_cmap = plt.cm.Reds
    heart_cmap.set_under('none')
    
    for idx, z in enumerate(slice_indices):
        print(f"  Processing slice {z}/{ct_data.shape[0]}...")
        
        # CTスライス（Z軸は最初の次元）
        ct_slice = ct_data[z, :, :]
        
        # Window/Level調整（胸部CT用）
        window_center = 40
        window_width = 400
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        
        # 心臓マスクスライス
        heart_slice = heart_data[z, :, :].astype(float)
        
        # 1. CT画像のみ
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        ax.axis('off')
        ax.set_title(f'CT Slice {z}')
        plt.savefig(output_dir / "ct_only" / f"slice_{z:04d}_ct.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. 心臓セグメンテーション（オーバーレイ）
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        masked_heart = np.ma.masked_where(heart_slice < 0.5, heart_slice)
        ax.imshow(masked_heart, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'DeepHeartSeg Segmentation - Slice {z}')
        plt.savefig(output_dir / "heart_overlay" / f"slice_{z:04d}_overlay.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. 心臓セグメンテーション（輪郭）
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        ax.contour(heart_slice, levels=[0.5], colors='red', linewidths=2)
        ax.axis('off')
        ax.set_title(f'DeepHeartSeg Contour - Slice {z}')
        plt.savefig(output_dir / "heart_contour" / f"slice_{z:04d}_contour.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 4. 並列表示
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # CT画像
        axes[0].imshow(ct_windowed, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('CT Image')
        
        # 心臓マスク
        axes[1].imshow(ct_windowed, cmap='gray', alpha=0.7)
        masked_heart = np.ma.masked_where(heart_slice < 0.5, heart_slice)
        axes[1].imshow(masked_heart, cmap='Reds', alpha=0.5)
        axes[1].axis('off')
        axes[1].set_title('Heart Segmentation')
        
        # 輪郭
        axes[2].imshow(ct_windowed, cmap='gray')
        axes[2].contour(heart_slice, levels=[0.5], colors='red', linewidths=2)
        axes[2].axis('off')
        axes[2].set_title('Heart Contour')
        
        plt.suptitle(f'DeepHeartSeg Results - Slice {z}', fontsize=14)
        plt.savefig(output_dir / "side_by_side" / f"slice_{z:04d}_combined.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
    
    # サマリー画像作成（中央5スライス）
    print("[INFO] Creating summary image...")
    middle_idx = len(slice_indices) // 2
    summary_indices = slice_indices[max(0, middle_idx-2):min(len(slice_indices), middle_idx+3)]
    
    fig, axes = plt.subplots(1, len(summary_indices), figsize=(20, 4))
    if len(summary_indices) == 1:
        axes = [axes]
    
    for i, z in enumerate(summary_indices):
        ct_slice = ct_data[z, :, :]
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        heart_slice = heart_data[z, :, :].astype(float)
        
        axes[i].imshow(ct_windowed, cmap='gray')
        masked_heart = np.ma.masked_where(heart_slice < 0.5, heart_slice)
        axes[i].imshow(masked_heart, cmap='Reds', alpha=0.4)
        axes[i].axis('off')
        axes[i].set_title(f'Slice {z}')
    
    plt.suptitle('DeepHeartSeg Heart Segmentation Summary', fontsize=14)
    plt.savefig(output_dir / "summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[INFO] Visualizations saved to {output_dir}")
    
    # 統計情報を出力
    heart_voxels = np.sum(heart_data > 0)
    voxel_volume = np.prod(spacing[::-1]) / 1000.0  # mm^3 to ml
    heart_volume = heart_voxels * voxel_volume
    
    print("\n[STATISTICS]")
    print(f"  Heart voxels: {heart_voxels:,}")
    print(f"  Heart volume: {heart_volume:.2f} ml")
    
    # 統計情報を保存
    stats = {
        "heart_voxels": int(heart_voxels),
        "heart_volume_ml": float(heart_volume),
        "ct_shape": ct_data.shape,
        "ct_spacing": spacing,
        "num_slices": len(slice_indices)
    }
    
    import json
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Statistics saved to {stats_path}")

def main():
    # デフォルトパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    heart_mask_path = "/home/soya/ctrate_ws/outputs/deepheartseg_test/step2_heartseg.nii.gz"
    output_dir = "/home/soya/ctrate_ws/outputs/deepheartseg_visualization"
    
    # 引数がある場合は上書き
    if len(sys.argv) > 1:
        ct_path = sys.argv[1]
    if len(sys.argv) > 2:
        heart_mask_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    print("="*60)
    print("DeepHeartSeg Visualization Tool")
    print("="*60)
    print(f"CT: {ct_path}")
    print(f"Heart mask: {heart_mask_path}")
    print(f"Output: {output_dir}")
    print()
    
    create_visualization(ct_path, heart_mask_path, output_dir)
    
    print("\n" + "="*60)
    print("Visualization complete!")

if __name__ == "__main__":
    main()