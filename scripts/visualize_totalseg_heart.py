#!/usr/bin/env python3
"""
TotalSegmentator心臓セグメンテーションの可視化
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

def create_visualization(ct_path, totalseg_dir, output_dir, num_slices=15):
    """
    TotalSegmentatorの心臓セグメンテーションを可視化
    """
    print("[INFO] Loading data...")
    
    # CTデータ読み込み
    ct_img = sitk.ReadImage(ct_path)
    ct_data = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  CT spacing: {spacing}")
    
    # 心臓構造を読み込み
    heart_components = {
        "heart_myocardium": "Myocardium",
        "heart_ventricle_left": "Left Ventricle",
        "heart_ventricle_right": "Right Ventricle",
        "heart_atrium_left": "Left Atrium",
        "heart_atrium_right": "Right Atrium"
    }
    
    heart_masks = {}
    heart_all = np.zeros(ct_data.shape, dtype=bool)
    
    for comp, label in heart_components.items():
        comp_path = Path(totalseg_dir) / f"{comp}.nii.gz"
        if comp_path.exists():
            print(f"  Loading {label}...")
            img = sitk.ReadImage(str(comp_path))
            mask = sitk.GetArrayFromImage(img) > 0
            heart_masks[comp] = mask
            heart_all |= mask
            voxels = np.sum(mask)
            volume = voxels * np.prod(spacing[::-1]) / 1000.0
            print(f"    {voxels:,} voxels ({volume:.1f} ml)")
        else:
            print(f"  {label}: NOT FOUND")
    
    if not heart_masks:
        print("[ERROR] No heart segmentation found")
        return
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    (output_dir / "combined_heart").mkdir(exist_ok=True)
    (output_dir / "individual_components").mkdir(exist_ok=True)
    (output_dir / "multi_view").mkdir(exist_ok=True)
    
    # スライス選択（心臓が存在する範囲）
    heart_slices = np.where(np.any(heart_all, axis=(1, 2)))[0]
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
    
    # カラーマップ設定
    colors = {
        "heart_myocardium": 'Reds',
        "heart_ventricle_left": 'Blues',
        "heart_ventricle_right": 'Greens',
        "heart_atrium_left": 'Purples',
        "heart_atrium_right": 'Oranges'
    }
    
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
        
        # 1. 全心臓の結合マスク
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ct_windowed, cmap='gray')
        
        # 各コンポーネントを異なる色で表示
        for comp, mask in heart_masks.items():
            if comp in colors:
                comp_slice = mask[z, :, :].astype(float)
                masked_comp = np.ma.masked_where(comp_slice < 0.5, comp_slice)
                ax.imshow(masked_comp, cmap=colors[comp], alpha=0.3, vmin=0, vmax=1)
        
        ax.axis('off')
        ax.set_title(f'TotalSegmentator Heart - Slice {z}')
        plt.savefig(output_dir / "combined_heart" / f"slice_{z:04d}_combined.png", 
                   bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. 個別コンポーネント表示
        n_components = len(heart_masks)
        if n_components > 0:
            fig, axes = plt.subplots(1, n_components + 1, figsize=(4 * (n_components + 1), 4))
            
            # CT画像
            axes[0].imshow(ct_windowed, cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('CT')
            
            # 各コンポーネント
            for i, (comp, mask) in enumerate(heart_masks.items(), 1):
                axes[i].imshow(ct_windowed, cmap='gray', alpha=0.7)
                comp_slice = mask[z, :, :].astype(float)
                masked_comp = np.ma.masked_where(comp_slice < 0.5, comp_slice)
                axes[i].imshow(masked_comp, cmap=colors.get(comp, 'hot'), alpha=0.5)
                axes[i].axis('off')
                axes[i].set_title(heart_components.get(comp, comp))
            
            plt.suptitle(f'Heart Components - Slice {z}', fontsize=12)
            plt.savefig(output_dir / "individual_components" / f"slice_{z:04d}_components.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
    
    # 3. 多断面表示（冠状断、矢状断、軸断）
    print("[INFO] Creating multi-view visualization...")
    
    # 中心座標を取得
    center_coords = np.array(np.where(heart_all)).mean(axis=1).astype(int)
    z_center, y_center, x_center = center_coords
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 軸断（Axial）
    ax_slice = ct_data[z_center, :, :]
    ax_windowed = np.clip(
        (ax_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    axes[0].imshow(ax_windowed, cmap='gray')
    for comp, mask in heart_masks.items():
        comp_slice = mask[z_center, :, :].astype(float)
        masked_comp = np.ma.masked_where(comp_slice < 0.5, comp_slice)
        axes[0].imshow(masked_comp, cmap=colors.get(comp, 'hot'), alpha=0.3)
    axes[0].axis('off')
    axes[0].set_title(f'Axial (z={z_center})')
    
    # 冠状断（Coronal）
    cor_slice = ct_data[:, y_center, :]
    cor_windowed = np.clip(
        (cor_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    axes[1].imshow(cor_windowed, cmap='gray', aspect=spacing[2]/spacing[0])
    for comp, mask in heart_masks.items():
        comp_slice = mask[:, y_center, :].astype(float)
        masked_comp = np.ma.masked_where(comp_slice < 0.5, comp_slice)
        axes[1].imshow(masked_comp, cmap=colors.get(comp, 'hot'), alpha=0.3, aspect=spacing[2]/spacing[0])
    axes[1].axis('off')
    axes[1].set_title(f'Coronal (y={y_center})')
    
    # 矢状断（Sagittal）
    sag_slice = ct_data[:, :, x_center]
    sag_windowed = np.clip(
        (sag_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    axes[2].imshow(sag_windowed, cmap='gray', aspect=spacing[2]/spacing[1])
    for comp, mask in heart_masks.items():
        comp_slice = mask[:, :, x_center].astype(float)
        masked_comp = np.ma.masked_where(comp_slice < 0.5, comp_slice)
        axes[2].imshow(masked_comp, cmap=colors.get(comp, 'hot'), alpha=0.3, aspect=spacing[2]/spacing[1])
    axes[2].axis('off')
    axes[2].set_title(f'Sagittal (x={x_center})')
    
    plt.suptitle('TotalSegmentator Heart - Multi-View', fontsize=14)
    plt.savefig(output_dir / "multi_view.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # サマリー画像
    print("[INFO] Creating summary image...")
    middle_idx = len(slice_indices) // 2
    summary_indices = slice_indices[max(0, middle_idx-2):min(len(slice_indices), middle_idx+3)][:5]
    
    fig, axes = plt.subplots(1, len(summary_indices), figsize=(20, 4))
    if len(summary_indices) == 1:
        axes = [axes]
    
    for i, z in enumerate(summary_indices):
        ct_slice = ct_data[z, :, :]
        ct_windowed = np.clip(
            (ct_slice - (window_center - window_width/2)) / window_width * 255,
            0, 255
        ).astype(np.uint8)
        
        axes[i].imshow(ct_windowed, cmap='gray')
        for comp, mask in heart_masks.items():
            comp_slice = mask[z, :, :].astype(float)
            masked_comp = np.ma.masked_where(comp_slice < 0.5, comp_slice)
            axes[i].imshow(masked_comp, cmap=colors.get(comp, 'hot'), alpha=0.3)
        axes[i].axis('off')
        axes[i].set_title(f'Slice {z}')
    
    plt.suptitle('TotalSegmentator Heart Segmentation Summary', fontsize=14)
    plt.savefig(output_dir / "summary.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"[INFO] Visualizations saved to {output_dir}")
    
    # 統計情報
    total_voxels = np.sum(heart_all)
    total_volume = total_voxels * np.prod(spacing[::-1]) / 1000.0
    
    print("\n[STATISTICS]")
    print(f"  Total heart voxels: {total_voxels:,}")
    print(f"  Total heart volume: {total_volume:.2f} ml")
    
    # 統計を保存
    stats = {
        "total_heart_voxels": int(total_voxels),
        "total_heart_volume_ml": float(total_volume),
        "components": {},
        "ct_shape": ct_data.shape,
        "ct_spacing": spacing
    }
    
    for comp, mask in heart_masks.items():
        voxels = np.sum(mask)
        volume = voxels * np.prod(spacing[::-1]) / 1000.0
        stats["components"][comp] = {
            "voxels": int(voxels),
            "volume_ml": float(volume)
        }
    
    stats_path = output_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Statistics saved to {stats_path}")

def main():
    # デフォルトパス
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    totalseg_dir = "/home/soya/ctrate_ws/outputs/batch_results/valid_2_valid_2_a_1/segmentation"
    output_dir = "/home/soya/ctrate_ws/outputs/totalseg_heart_visualization"
    
    # 引数がある場合は上書き
    if len(sys.argv) > 1:
        ct_path = sys.argv[1]
    if len(sys.argv) > 2:
        totalseg_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    print("="*60)
    print("TotalSegmentator Heart Visualization Tool")
    print("="*60)
    print(f"CT: {ct_path}")
    print(f"TotalSeg dir: {totalseg_dir}")
    print(f"Output: {output_dir}")
    print()
    
    create_visualization(ct_path, totalseg_dir, output_dir)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("\n[COMPARISON WITH DEEPHEARTSEG]")
    print("  DeepHeartSeg: 151,732 voxels (binary mask only)")
    print("  TotalSegmentator: Multi-label detailed segmentation")
    print("  -> TotalSegmentator provides more detailed and reliable results")

if __name__ == "__main__":
    main()