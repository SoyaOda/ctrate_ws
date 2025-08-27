#!/usr/bin/env python3
"""
内臓脂肪（torso_fat）の可視化スクリプト
TotalSegmentatorのtissue_typesタスク出力を可視化
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import json

def create_visualization(ct_path, visceral_fat_path, eat_pat_path, output_dir, 
                        slice_indices=None, verbose=False):
    """
    内臓脂肪とEAT+PATの可視化
    
    Args:
        ct_path: CT画像のパス
        visceral_fat_path: 内臓脂肪マスクのパス
        eat_pat_path: EAT+PATマスクのパス（オプション）
        output_dir: 出力ディレクトリ
        slice_indices: 可視化するスライスのインデックス（Noneの場合自動選択）
        verbose: 詳細出力
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading data...")
    
    # データの読み込み
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    visceral_fat_img = nib.load(visceral_fat_path)
    visceral_fat_mask = visceral_fat_img.get_fdata() > 0
    
    # EAT+PATマスクがあれば読み込み
    eat_pat_mask = None
    if eat_pat_path and Path(eat_pat_path).exists():
        eat_pat_img = nib.load(eat_pat_path)
        eat_pat_mask = eat_pat_img.get_fdata() > 0
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    print(f"  Visceral fat voxels: {np.sum(visceral_fat_mask):,}")
    if eat_pat_mask is not None:
        print(f"  EAT+PAT voxels: {np.sum(eat_pat_mask):,}")
    
    # スライスの選択（内臓脂肪が多いスライス）
    if slice_indices is None:
        # 各スライスの内臓脂肪量を計算
        fat_per_slice = np.sum(visceral_fat_mask, axis=(0, 1))
        
        # 上位5スライスを選択
        top_slices = np.argsort(fat_per_slice)[::-1][:5]
        
        # 心臓レベルのスライスも追加（EAT+PATがある場合）
        if eat_pat_mask is not None:
            eat_pat_per_slice = np.sum(eat_pat_mask, axis=(0, 1))
            heart_slices = np.where(eat_pat_per_slice > 0)[0]
            if len(heart_slices) > 0:
                # 心臓の中央スライスを選択
                mid_heart = heart_slices[len(heart_slices) // 2]
                slice_indices = sorted(set(list(top_slices) + [mid_heart]))[:7]
            else:
                slice_indices = top_slices
        else:
            slice_indices = top_slices
    
    print(f"\n[INFO] Creating visualization for {len(slice_indices)} slices...")
    
    # 各スライスの可視化
    for i, slice_idx in enumerate(slice_indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Slice {slice_idx} (Z = {slice_idx * spacing[2]:.1f} mm)', fontsize=14)
        
        # CTスライス
        ct_slice = ct_data[:, :, slice_idx]
        visceral_slice = visceral_fat_mask[:, :, slice_idx]
        
        # 1. CT画像のみ
        ax = axes[0]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        ax.set_title('CT Image')
        ax.axis('off')
        
        # 2. 内臓脂肪オーバーレイ
        ax = axes[1]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        
        # 内臓脂肪をオーバーレイ
        masked_visceral = np.ma.masked_where(visceral_slice < 0.5, visceral_slice)
        ax.imshow(masked_visceral, cmap='YlOrRd', alpha=0.5, vmin=0, vmax=1)
        
        visceral_pixels = np.sum(visceral_slice)
        visceral_area_cm2 = visceral_pixels * spacing[0] * spacing[1] / 100
        ax.set_title(f'Visceral Fat\n({visceral_area_cm2:.1f} cm²)')
        ax.axis('off')
        
        # 3. EAT+PAT比較（あれば）
        ax = axes[2]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        
        if eat_pat_mask is not None:
            eat_pat_slice = eat_pat_mask[:, :, slice_idx]
            
            # 内臓脂肪全体を薄く表示
            masked_visceral_light = np.ma.masked_where(visceral_slice < 0.5, visceral_slice)
            ax.imshow(masked_visceral_light, cmap='YlOrBr', alpha=0.3, vmin=0, vmax=1)
            
            # EAT+PATを強調表示
            masked_eat_pat = np.ma.masked_where(eat_pat_slice < 0.5, eat_pat_slice)
            ax.imshow(masked_eat_pat, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
            
            eat_pat_pixels = np.sum(eat_pat_slice)
            eat_pat_area_cm2 = eat_pat_pixels * spacing[0] * spacing[1] / 100
            ax.set_title(f'EAT+PAT (Red) vs Visceral (Yellow)\nEAT+PAT: {eat_pat_area_cm2:.1f} cm²')
        else:
            # 内臓脂肪のみ
            ax.imshow(masked_visceral, cmap='YlOrRd', alpha=0.5, vmin=0, vmax=1)
            ax.set_title('Visceral Fat')
        
        ax.axis('off')
        
        # 保存
        output_path = output_dir / f'visceral_fat_slice_{slice_idx:03d}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"  Saved: {output_path}")
    
    # サマリー画像の作成
    create_summary_image(ct_data, visceral_fat_mask, eat_pat_mask, spacing, output_dir)
    
    print(f"\n[INFO] Visualization saved to: {output_dir}")

def create_summary_image(ct_data, visceral_fat_mask, eat_pat_mask, spacing, output_dir):
    """
    サマリー画像の作成（MIPと統計情報）
    """
    fig = plt.figure(figsize=(16, 10))
    
    # MIP（最大値投影）の作成
    visceral_mip_axial = np.max(visceral_fat_mask, axis=2)
    visceral_mip_coronal = np.max(visceral_fat_mask, axis=1)
    visceral_mip_sagittal = np.max(visceral_fat_mask, axis=0)
    
    ct_mip_axial = np.max(ct_data, axis=2)
    ct_mip_coronal = np.max(ct_data, axis=1)
    ct_mip_sagittal = np.max(ct_data, axis=0)
    
    # 1. Axial MIP
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(ct_mip_axial, cmap='gray', vmin=-200, vmax=200)
    masked_visceral = np.ma.masked_where(visceral_mip_axial < 0.5, visceral_mip_axial)
    ax1.imshow(masked_visceral, cmap='YlOrRd', alpha=0.5)
    ax1.set_title('Axial MIP - Visceral Fat')
    ax1.axis('off')
    
    # 2. Coronal MIP
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(ct_mip_coronal.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
    masked_visceral = np.ma.masked_where(visceral_mip_coronal.T < 0.5, visceral_mip_coronal.T)
    ax2.imshow(masked_visceral, cmap='YlOrRd', alpha=0.5, origin='lower')
    ax2.set_title('Coronal MIP - Visceral Fat')
    ax2.axis('off')
    
    # 3. Sagittal MIP
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(ct_mip_sagittal.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
    masked_visceral = np.ma.masked_where(visceral_mip_sagittal.T < 0.5, visceral_mip_sagittal.T)
    ax3.imshow(masked_visceral, cmap='YlOrRd', alpha=0.5, origin='lower')
    ax3.set_title('Sagittal MIP - Visceral Fat')
    ax3.axis('off')
    
    # EAT+PATがある場合の比較
    if eat_pat_mask is not None:
        eat_pat_mip_axial = np.max(eat_pat_mask, axis=2)
        eat_pat_mip_coronal = np.max(eat_pat_mask, axis=1)
        eat_pat_mip_sagittal = np.max(eat_pat_mask, axis=0)
        
        # 4. Axial比較
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(ct_mip_axial, cmap='gray', vmin=-200, vmax=200)
        ax4.imshow(masked_visceral, cmap='YlOrBr', alpha=0.3)
        masked_eat_pat = np.ma.masked_where(eat_pat_mip_axial < 0.5, eat_pat_mip_axial)
        ax4.imshow(masked_eat_pat, cmap='Reds', alpha=0.7)
        ax4.set_title('Axial - EAT+PAT (Red) vs Visceral (Yellow)')
        ax4.axis('off')
        
        # 5. Coronal比較
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(ct_mip_coronal.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
        masked_visceral_c = np.ma.masked_where(visceral_mip_coronal.T < 0.5, visceral_mip_coronal.T)
        ax5.imshow(masked_visceral_c, cmap='YlOrBr', alpha=0.3, origin='lower')
        masked_eat_pat_c = np.ma.masked_where(eat_pat_mip_coronal.T < 0.5, eat_pat_mip_coronal.T)
        ax5.imshow(masked_eat_pat_c, cmap='Reds', alpha=0.7, origin='lower')
        ax5.set_title('Coronal - EAT+PAT vs Visceral')
        ax5.axis('off')
        
        # 6. 統計情報
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # 体積計算
        voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000
        visceral_volume_ml = np.sum(visceral_fat_mask) * voxel_vol_ml
        eat_pat_volume_ml = np.sum(eat_pat_mask) * voxel_vol_ml
        ratio = (eat_pat_volume_ml / visceral_volume_ml * 100) if visceral_volume_ml > 0 else 0
        
        stats_text = f"""Fat Volume Statistics
        
Visceral Fat: {visceral_volume_ml:.1f} ml
EAT+PAT: {eat_pat_volume_ml:.1f} ml
EAT+PAT/Visceral: {ratio:.1f}%

Normal Ranges:
• EAT+PAT: 50-200 ml
• Visceral Fat: varies by BMI
• Ratio: typically 5-15%"""
        
        ax6.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # 統計情報のみ
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        voxel_vol_ml = spacing[0] * spacing[1] * spacing[2] / 1000
        visceral_volume_ml = np.sum(visceral_fat_mask) * voxel_vol_ml
        
        stats_text = f"""Fat Volume Statistics
        
Visceral Fat: {visceral_volume_ml:.1f} ml

Note: Run EAT+PAT extraction
for cardiac fat analysis"""
        
        ax6.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Visceral Fat Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'visceral_fat_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Summary saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize visceral fat (torso_fat) from TotalSegmentator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "ct_path",
        help="Path to CT NIfTI file"
    )
    
    parser.add_argument(
        "visceral_fat_path",
        help="Path to visceral fat mask (torso_fat.nii.gz or extracted mask)"
    )
    
    parser.add_argument(
        "output_dir",
        help="Path to output directory for visualizations"
    )
    
    parser.add_argument(
        "--eat-pat",
        help="Path to EAT+PAT mask for comparison (optional)"
    )
    
    parser.add_argument(
        "--slices",
        nargs="+",
        type=int,
        help="Specific slice indices to visualize"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # 実行
    try:
        create_visualization(
            ct_path=args.ct_path,
            visceral_fat_path=args.visceral_fat_path,
            eat_pat_path=args.eat_pat,
            output_dir=args.output_dir,
            slice_indices=args.slices,
            verbose=args.verbose
        )
        
        print("\n[SUCCESS] Visceral fat visualization completed!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()