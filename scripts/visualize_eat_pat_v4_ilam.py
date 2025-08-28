#!/usr/bin/env python3
"""
EAT+PAT統合可視化スクリプト
Heart、Shell、EAT+PAT、Visceral Fatをすべて可視化
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
    すべてのマスクを読み込み
    """
    masks = {}
    mask_files = {
        'heart': 'heart.nii.gz',
        'shell': 'shell.nii.gz',
        'eat_pat': 'eat_pat.nii.gz',
        'visceral_fat': 'visceral_fat.nii.gz'
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
    単一スライスの可視化（4パネル）
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.15)
    
    ct_slice = ct_data[:, :, slice_idx]
    
    # 1. CT + Heart (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    if np.any(masks['heart'][:, :, slice_idx]):
        heart_masked = np.ma.masked_where(masks['heart'][:, :, slice_idx] < 0.5, 
                                         masks['heart'][:, :, slice_idx])
        ax1.imshow(heart_masked, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    
    heart_pixels = np.sum(masks['heart'][:, :, slice_idx])
    heart_area = heart_pixels * spacing[0] * spacing[1] / 100
    ax1.set_title(f'Heart\nArea: {heart_area:.1f} cm²', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. CT + Shell (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    if np.any(masks['shell'][:, :, slice_idx]):
        shell_masked = np.ma.masked_where(masks['shell'][:, :, slice_idx] < 0.5,
                                         masks['shell'][:, :, slice_idx])
        ax2.imshow(shell_masked, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    
    shell_pixels = np.sum(masks['shell'][:, :, slice_idx])
    shell_area = shell_pixels * spacing[0] * spacing[1] / 100
    ax2.set_title(f'Shell (15mm dilation)\nArea: {shell_area:.1f} cm²', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. CT + Visceral Fat (左下)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    if np.any(masks['visceral_fat'][:, :, slice_idx]):
        vf_masked = np.ma.masked_where(masks['visceral_fat'][:, :, slice_idx] < 0.5,
                                      masks['visceral_fat'][:, :, slice_idx])
        ax3.imshow(vf_masked, cmap='YlOrBr', alpha=0.5, vmin=0, vmax=1)
    
    vf_pixels = np.sum(masks['visceral_fat'][:, :, slice_idx])
    vf_area = vf_pixels * spacing[0] * spacing[1] / 100
    ax3.set_title(f'Visceral Fat (torso_fat)\nArea: {vf_area:.1f} cm²', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Combined view (右下)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # Heart (赤)
    if np.any(masks['heart'][:, :, slice_idx]):
        heart_masked = np.ma.masked_where(masks['heart'][:, :, slice_idx] < 0.5,
                                         masks['heart'][:, :, slice_idx])
        ax4.imshow(heart_masked, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    
    # Shell outline (青い輪郭)
    if np.any(masks['shell'][:, :, slice_idx]):
        shell_outline = masks['shell'][:, :, slice_idx] & ~masks['heart'][:, :, slice_idx]
        shell_masked = np.ma.masked_where(shell_outline < 0.5, shell_outline)
        ax4.imshow(shell_masked, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
    
    # EAT+PAT (黄色、強調)
    if np.any(masks['eat_pat'][:, :, slice_idx]):
        eat_pat_masked = np.ma.masked_where(masks['eat_pat'][:, :, slice_idx] < 0.5,
                                           masks['eat_pat'][:, :, slice_idx])
        ax4.imshow(eat_pat_masked, cmap='YlOrRd', alpha=0.7, vmin=0, vmax=1)
    
    eat_pat_pixels = np.sum(masks['eat_pat'][:, :, slice_idx])
    eat_pat_area = eat_pat_pixels * spacing[0] * spacing[1] / 100
    fat_fraction = (eat_pat_pixels / shell_pixels * 100) if shell_pixels > 0 else 0
    
    ax4.set_title(f'EAT+PAT (Yellow) in Shell\nEAT+PAT: {eat_pat_area:.1f} cm² ({fat_fraction:.1f}% of shell)',
                 fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 全体のタイトル
    fig.suptitle(f'Slice {slice_idx} (Z = {slice_idx * spacing[2]:.1f} mm)', 
                fontsize=14, fontweight='bold')
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_mip_visualization(ct_data, masks, spacing, output_path):
    """
    MIP（最大値投影）による3方向表示
    """
    fig = plt.figure(figsize=(18, 12))
    
    # MIPの作成
    views = {
        'Axial': (2, ct_data.max(axis=2), [m.max(axis=2) for m in masks.values()]),
        'Coronal': (1, ct_data.max(axis=1), [m.max(axis=1) for m in masks.values()]),
        'Sagittal': (0, ct_data.max(axis=0), [m.max(axis=0) for m in masks.values()])
    }
    
    mask_names = list(masks.keys())
    colors = {'heart': 'Reds', 'shell': 'Blues', 'eat_pat': 'YlOrBr', 'visceral_fat': 'Greens'}
    alphas = {'heart': 0.4, 'shell': 0.3, 'eat_pat': 0.7, 'visceral_fat': 0.2}
    
    for idx, (view_name, (axis, ct_mip, mask_mips)) in enumerate(views.items()):
        ax = plt.subplot(2, 3, idx + 1)
        
        # CT背景
        if axis == 1:  # Coronal
            ax.imshow(ct_mip.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
        elif axis == 0:  # Sagittal
            ax.imshow(ct_mip.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
        else:  # Axial
            ax.imshow(ct_mip, cmap='gray', vmin=-200, vmax=200)
        
        # 各マスクをオーバーレイ
        for mask_name, mask_mip in zip(mask_names, mask_mips):
            if np.any(mask_mip):
                if axis == 1:  # Coronal
                    masked = np.ma.masked_where(mask_mip.T < 0.5, mask_mip.T)
                    ax.imshow(masked, cmap=colors[mask_name], alpha=alphas[mask_name], origin='lower')
                elif axis == 0:  # Sagittal
                    masked = np.ma.masked_where(mask_mip.T < 0.5, mask_mip.T)
                    ax.imshow(masked, cmap=colors[mask_name], alpha=alphas[mask_name], origin='lower')
                else:  # Axial
                    masked = np.ma.masked_where(mask_mip < 0.5, mask_mip)
                    ax.imshow(masked, cmap=colors[mask_name], alpha=alphas[mask_name])
        
        ax.set_title(f'{view_name} MIP', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 統計パネル（右側）
    ax_stats = plt.subplot(2, 3, 4)
    ax_stats.axis('off')
    
    # 凡例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.4, label='Heart'),
        plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.3, label='Shell'),
        plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.7, label='EAT+PAT'),
        plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.2, label='Visceral Fat')
    ]
    ax_stats.legend(handles=legend_elements, loc='center', fontsize=12)
    ax_stats.set_title('Components', fontsize=12, fontweight='bold')
    
    # 体積統計（下段）
    ax_vol = plt.subplot(2, 1, 2)
    ax_vol.axis('off')
    
    # 体積計算とテキスト表示
    voxel_vol = spacing[0] * spacing[1] * spacing[2] / 1000  # ml
    volumes = {name: np.sum(mask) * voxel_vol for name, mask in masks.items()}
    
    stats_text = "Volume Statistics\n" + "="*50 + "\n\n"
    stats_text += f"Heart:        {volumes['heart']:8.1f} ml\n"
    stats_text += f"Shell:        {volumes['shell']:8.1f} ml\n"
    stats_text += f"EAT+PAT:      {volumes['eat_pat']:8.1f} ml\n"
    stats_text += f"Visceral Fat: {volumes['visceral_fat']:8.1f} ml\n\n"
    
    # 比率計算
    fat_fraction = (volumes['eat_pat'] / volumes['shell'] * 100) if volumes['shell'] > 0 else 0
    eat_to_visceral = (volumes['eat_pat'] / volumes['visceral_fat'] * 100) if volumes['visceral_fat'] > 0 else 0
    
    stats_text += "Ratios\n" + "-"*50 + "\n"
    stats_text += f"Fat fraction in shell:    {fat_fraction:5.1f}%\n"
    stats_text += f"EAT+PAT / Visceral fat:   {eat_to_visceral:5.1f}%\n\n"
    
    # 正常範囲
    stats_text += "Reference Ranges\n" + "-"*50 + "\n"
    stats_text += "EAT+PAT:          50-200 ml (normal)\n"
    stats_text += "Fat fraction:     <30% (normal shell)\n"
    
    ax_vol.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('EAT+PAT Analysis - MIP Views', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_visceral_fat_analysis(ct_data, masks, spacing, output_path):
    """
    内臓脂肪の全体分布を確認するための可視化
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 内臓脂肪が存在するスライス範囲
    vf_slices = np.where(np.any(masks['visceral_fat'], axis=(0, 1)))[0]
    heart_slices = np.where(np.any(masks['heart'], axis=(0, 1)))[0]
    
    if len(vf_slices) == 0:
        print("  Warning: No visceral fat found")
        plt.close()
        return
    
    # 6つの代表的なスライスを選択（内臓脂肪全体から）
    n_samples = min(6, len(vf_slices))
    sample_indices = np.linspace(0, len(vf_slices)-1, n_samples, dtype=int)
    sample_slices = vf_slices[sample_indices]
    
    for idx, slice_idx in enumerate(sample_slices):
        ax = plt.subplot(2, 3, idx + 1)
        
        # CTスライス
        ct_slice = ct_data[:, :, slice_idx]
        ax.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
        
        # 内臓脂肪（黄色）
        if np.any(masks['visceral_fat'][:, :, slice_idx]):
            vf_masked = np.ma.masked_where(masks['visceral_fat'][:, :, slice_idx] < 0.5,
                                          masks['visceral_fat'][:, :, slice_idx])
            ax.imshow(vf_masked, cmap='YlOrBr', alpha=0.5)
        
        # 心臓（赤、もし存在すれば）
        if slice_idx in heart_slices and np.any(masks['heart'][:, :, slice_idx]):
            heart_masked = np.ma.masked_where(masks['heart'][:, :, slice_idx] < 0.5,
                                             masks['heart'][:, :, slice_idx])
            ax.imshow(heart_masked, cmap='Reds', alpha=0.3)
        
        # EAT+PAT（明るい黄色、もし存在すれば）
        if np.any(masks['eat_pat'][:, :, slice_idx]):
            eat_masked = np.ma.masked_where(masks['eat_pat'][:, :, slice_idx] < 0.5,
                                           masks['eat_pat'][:, :, slice_idx])
            ax.imshow(eat_masked, cmap='YlOrRd', alpha=0.7)
        
        vf_pixels = np.sum(masks['visceral_fat'][:, :, slice_idx])
        vf_area = vf_pixels * spacing[0] * spacing[1] / 100
        
        # 心臓レベルかどうかを表示
        location = "Heart level" if slice_idx in heart_slices else "Non-heart level"
        ax.set_title(f'Z={slice_idx} ({slice_idx * spacing[2]:.1f} mm)\n'
                    f'VF Area: {vf_area:.1f} cm² - {location}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Visceral Fat Distribution Analysis\n'
                f'Total range: Z={vf_slices[0]}-{vf_slices[-1]} '
                f'({vf_slices[0]*spacing[2]:.1f}-{vf_slices[-1]*spacing[2]:.1f} mm)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visceral fat distribution: {output_path}")

def visualize_eat_pat(ct_path, masks_dir, output_dir, slice_indices=None, verbose=False):
    """
    EAT+PAT統合可視化のメイン関数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(exist_ok=True)
    
    print("[INFO] Loading data...")
    
    # CT画像読み込み
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    spacing = tuple(float(s) for s in spacing)
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # マスク読み込み
    masks_dir = Path(masks_dir)
    masks = load_all_masks(masks_dir, ct_data)
    
    # スライス選択
    if slice_indices is None:
        # 心臓が存在するスライス範囲を取得
        heart_slices = np.where(np.any(masks['heart'], axis=(0, 1)))[0]
        
        # 内臓脂肪が存在するスライス範囲も取得（全体を確認するため）
        visceral_slices = np.where(np.any(masks['visceral_fat'], axis=(0, 1)))[0]
        
        if len(heart_slices) > 0:
            # 心臓レベルから15スライス選択（密に）
            heart_start = heart_slices[0]
            heart_end = heart_slices[-1]
            n_heart_slices = min(15, len(heart_slices))
            heart_indices = np.linspace(heart_start, heart_end, n_heart_slices, dtype=int)
            
            # 内臓脂肪全体を見るために心臓範囲外のスライスも追加
            # 心臓より上部（胸部）のスライス
            if len(visceral_slices) > 0 and visceral_slices[0] < heart_start:
                upper_indices = np.linspace(visceral_slices[0], heart_start - 5, 
                                           min(5, heart_start - visceral_slices[0]), 
                                           dtype=int, endpoint=False)
            else:
                upper_indices = np.array([], dtype=int)
            
            # 心臓より下部（腹部）のスライス
            if len(visceral_slices) > 0 and visceral_slices[-1] > heart_end:
                lower_indices = np.linspace(heart_end + 5, visceral_slices[-1], 
                                           min(5, visceral_slices[-1] - heart_end), 
                                           dtype=int, endpoint=False)
            else:
                lower_indices = np.array([], dtype=int)
            
            # すべてのスライスを結合
            slice_indices = np.concatenate([upper_indices, heart_indices, lower_indices])
            slice_indices = np.unique(slice_indices)  # 重複を削除
            
            print(f"  Heart range: Z={heart_start}-{heart_end}")
            if len(visceral_slices) > 0:
                print(f"  Visceral fat range: Z={visceral_slices[0]}-{visceral_slices[-1]}")
            print(f"  Selected {len(slice_indices)} slices covering both heart and visceral fat regions")
        else:
            # 心臓が見つからない場合は中央付近を密にサンプリング
            mid = ct_data.shape[2] // 2
            slice_indices = range(mid-20, mid+20, 2)
    
    print(f"\n[INFO] Creating slice visualizations for {len(slice_indices)} slices...")
    
    # 各スライスの可視化
    for i, slice_idx in enumerate(slice_indices):
        output_path = slices_dir / f'slice_{slice_idx:03d}.png'
        create_slice_visualization(ct_data, masks, slice_idx, spacing, output_path)
        if verbose:
            print(f"  Saved: {output_path}")
        elif (i+1) % 5 == 0:
            print(f"  Processed {i+1}/{len(slice_indices)} slices...")
    
    # MIP可視化
    print("\n[INFO] Creating MIP visualization...")
    mip_path = output_dir / 'mip_analysis.png'
    create_mip_visualization(ct_data, masks, spacing, mip_path)
    print(f"  Saved: {mip_path}")
    
    # 内臓脂肪の分布確認用の追加可視化
    print("\n[INFO] Creating visceral fat distribution analysis...")
    create_visceral_fat_analysis(ct_data, masks, spacing, output_dir / 'visceral_fat_distribution.png')
    
    print(f"\n[SUCCESS] Visualization completed!")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize EAT+PAT analysis (integrated version)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "ct_path",
        help="Path to CT NIfTI file"
    )
    
    parser.add_argument(
        "masks_dir",
        help="Path to masks directory containing heart.nii.gz, shell.nii.gz, etc."
    )
    
    parser.add_argument(
        "output_dir",
        help="Path to output directory for visualizations"
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
        visualize_eat_pat(
            ct_path=args.ct_path,
            masks_dir=args.masks_dir,
            output_dir=args.output_dir,
            slice_indices=args.slices,
            verbose=args.verbose
        )
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()