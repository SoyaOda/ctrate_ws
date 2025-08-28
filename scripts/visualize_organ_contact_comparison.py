#!/usr/bin/env python3
"""
臓器接触脂肪パターンの比較可視化スクリプト
4つのパターンを並列表示して違いを明確に可視化

1. 心臓Shell内の脂肪
2. 肺接触脂肪塊
3. 胃接触脂肪塊  
4. 肝臓接触脂肪塊
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import argparse
import json

def load_masks(masks_dir, ct_shape):
    """
    4つのパターンマスクを読み込み
    """
    masks = {}
    mask_files = {
        'shell_fat': 'shell_fat.nii.gz',
        'lung_fat': 'lung_fat.nii.gz',
        'stomach_fat': 'stomach_fat.nii.gz',
        'liver_fat': 'liver_fat.nii.gz'
    }
    
    for name, filename in mask_files.items():
        mask_path = masks_dir / filename
        if mask_path.exists():
            img = nib.load(str(mask_path))
            masks[name] = img.get_fdata() > 0
            print(f"  Loaded {name}: {np.sum(masks[name]):,} voxels")
        else:
            print(f"  Warning: {name} not found")
            masks[name] = np.zeros(ct_shape, dtype=bool)
    
    return masks

def load_organs(seg_dir, ct_shape):
    """
    臓器マスクを読み込み（参照用）
    """
    organs = {}
    
    # 心臓
    heart_path = seg_dir / "heart_myocardium.nii.gz"
    if not heart_path.exists():
        heart_path = seg_dir / "heart.nii.gz"
    
    if heart_path.exists():
        img = nib.load(str(heart_path))
        organs['heart'] = img.get_fdata() > 0
    else:
        organs['heart'] = np.zeros(ct_shape, dtype=bool)
    
    # 心臓チャンバー追加
    for chamber in ["heart_atrium_left", "heart_atrium_right", 
                   "heart_ventricle_left", "heart_ventricle_right"]:
        chamber_path = seg_dir / f"{chamber}.nii.gz"
        if chamber_path.exists():
            img = nib.load(str(chamber_path))
            mask = img.get_fdata() > 0
            organs['heart'] |= mask
    
    # 肺
    left_lung_path = seg_dir / "lung_left.nii.gz"
    right_lung_path = seg_dir / "lung_right.nii.gz"
    
    if left_lung_path.exists() and right_lung_path.exists():
        left = nib.load(str(left_lung_path)).get_fdata() > 0
        right = nib.load(str(right_lung_path)).get_fdata() > 0
        organs['lungs'] = left | right
    else:
        organs['lungs'] = np.zeros(ct_shape, dtype=bool)
    
    # 胃
    stomach_path = seg_dir / "stomach.nii.gz"
    if stomach_path.exists():
        img = nib.load(str(stomach_path))
        organs['stomach'] = img.get_fdata() > 0
    else:
        organs['stomach'] = np.zeros(ct_shape, dtype=bool)
    
    # 肝臓
    liver_path = seg_dir / "liver.nii.gz"
    if liver_path.exists():
        img = nib.load(str(liver_path))
        organs['liver'] = img.get_fdata() > 0
    else:
        organs['liver'] = np.zeros(ct_shape, dtype=bool)
    
    return organs

def create_detailed_slice_visualization(ct_data, masks, organs, slice_idx, spacing, output_path):
    """
    詳細な単一スライス可視化（4パネル + オーバーラップ解析）
    """
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.2, wspace=0.2,
                          height_ratios=[1, 1, 0.8])
    
    ct_slice = ct_data[:, :, slice_idx]
    
    # パネル1: Shell脂肪（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    if np.any(organs['heart'][:, :, slice_idx]):
        heart_masked = np.ma.masked_where(organs['heart'][:, :, slice_idx] < 0.5,
                                         organs['heart'][:, :, slice_idx])
        ax1.imshow(heart_masked, cmap='Reds', alpha=0.3)
    
    if np.any(masks['shell_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(masks['shell_fat'][:, :, slice_idx] < 0.5,
                                       masks['shell_fat'][:, :, slice_idx])
        ax1.imshow(fat_masked, cmap='YlOrBr', alpha=0.7)
    
    shell_area = np.sum(masks['shell_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax1.set_title(f'Shell-based Fat\nArea: {shell_area:.1f} cm²', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # パネル2: 肺接触脂肪（中央上）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    if np.any(organs['lungs'][:, :, slice_idx]):
        lung_masked = np.ma.masked_where(organs['lungs'][:, :, slice_idx] < 0.5,
                                        organs['lungs'][:, :, slice_idx])
        ax2.imshow(lung_masked, cmap='Blues', alpha=0.2)
    
    if np.any(masks['lung_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(masks['lung_fat'][:, :, slice_idx] < 0.5,
                                       masks['lung_fat'][:, :, slice_idx])
        ax2.imshow(fat_masked, cmap='YlOrBr', alpha=0.7)
    
    lung_area = np.sum(masks['lung_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax2.set_title(f'Lung-Contact Fat (All Components)\nArea: {lung_area:.1f} cm²', 
                 fontsize=12, fontweight='bold', color='blue')
    ax2.axis('off')
    
    # パネル3: Shell vs Lung比較（右上）
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # 心臓（薄く）
    if np.any(organs['heart'][:, :, slice_idx]):
        heart_masked = np.ma.masked_where(organs['heart'][:, :, slice_idx] < 0.5,
                                         organs['heart'][:, :, slice_idx])
        ax3.imshow(heart_masked, cmap='Reds', alpha=0.2)
    
    # Shell脂肪（黄色）
    shell_only = masks['shell_fat'][:, :, slice_idx] & ~masks['lung_fat'][:, :, slice_idx]
    if np.any(shell_only):
        shell_masked = np.ma.masked_where(shell_only < 0.5, shell_only)
        ax3.imshow(shell_masked, cmap='YlOrBr', alpha=0.7, vmin=0, vmax=1)
    
    # 肺脂肪のみ（青）
    lung_only = masks['lung_fat'][:, :, slice_idx] & ~masks['shell_fat'][:, :, slice_idx]
    if np.any(lung_only):
        lung_masked = np.ma.masked_where(lung_only < 0.5, lung_only)
        ax3.imshow(lung_masked, cmap='Blues', alpha=0.7, vmin=0, vmax=1)
    
    # オーバーラップ（緑）
    overlap = masks['shell_fat'][:, :, slice_idx] & masks['lung_fat'][:, :, slice_idx]
    if np.any(overlap):
        overlap_masked = np.ma.masked_where(overlap < 0.5, overlap)
        ax3.imshow(overlap_masked, cmap='Greens', alpha=0.8, vmin=0, vmax=1)
    
    shell_only_area = np.sum(shell_only) * spacing[0] * spacing[1] / 100
    lung_only_area = np.sum(lung_only) * spacing[0] * spacing[1] / 100
    overlap_area = np.sum(overlap) * spacing[0] * spacing[1] / 100
    
    ax3.set_title(f'Shell vs Lung Comparison\n'
                 f'Yellow: Shell only ({shell_only_area:.1f} cm²)\n'
                 f'Blue: Lung only ({lung_only_area:.1f} cm²)\n'
                 f'Green: Overlap ({overlap_area:.1f} cm²)',
                 fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # パネル4: 胃接触脂肪（左中）
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    if np.any(organs['stomach'][:, :, slice_idx]):
        stomach_masked = np.ma.masked_where(organs['stomach'][:, :, slice_idx] < 0.5,
                                           organs['stomach'][:, :, slice_idx])
        ax4.imshow(stomach_masked, cmap='Greens', alpha=0.2)
    
    if np.any(masks['stomach_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(masks['stomach_fat'][:, :, slice_idx] < 0.5,
                                       masks['stomach_fat'][:, :, slice_idx])
        ax4.imshow(fat_masked, cmap='YlOrBr', alpha=0.7)
    
    stomach_area = np.sum(masks['stomach_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax4.set_title(f'Stomach-Contact Fat (All Components)\nArea: {stomach_area:.1f} cm²',
                 fontsize=12, fontweight='bold', color='green')
    ax4.axis('off')
    
    # パネル5: 肝臓接触脂肪（中央中）
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    if np.any(organs['liver'][:, :, slice_idx]):
        liver_masked = np.ma.masked_where(organs['liver'][:, :, slice_idx] < 0.5,
                                         organs['liver'][:, :, slice_idx])
        ax5.imshow(liver_masked, cmap='Purples', alpha=0.2)
    
    if np.any(masks['liver_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(masks['liver_fat'][:, :, slice_idx] < 0.5,
                                       masks['liver_fat'][:, :, slice_idx])
        ax5.imshow(fat_masked, cmap='YlOrBr', alpha=0.7)
    
    liver_area = np.sum(masks['liver_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax5.set_title(f'Liver-Contact Fat (All Components)\nArea: {liver_area:.1f} cm²',
                 fontsize=12, fontweight='bold', color='purple')
    ax5.axis('off')
    
    # パネル6: 全パターン統合（右中）
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # 各臓器を薄く表示
    if np.any(organs['heart'][:, :, slice_idx]):
        heart_masked = np.ma.masked_where(organs['heart'][:, :, slice_idx] < 0.5,
                                         organs['heart'][:, :, slice_idx])
        ax6.imshow(heart_masked, cmap='Reds', alpha=0.15)
    
    if np.any(organs['lungs'][:, :, slice_idx]):
        lung_masked = np.ma.masked_where(organs['lungs'][:, :, slice_idx] < 0.5,
                                        organs['lungs'][:, :, slice_idx])
        ax6.imshow(lung_masked, cmap='Blues', alpha=0.1)
    
    # 全ての脂肪パターンの和集合
    all_fat = (masks['shell_fat'][:, :, slice_idx] | 
               masks['lung_fat'][:, :, slice_idx] |
               masks['stomach_fat'][:, :, slice_idx] |
               masks['liver_fat'][:, :, slice_idx])
    
    if np.any(all_fat):
        # Shell脂肪を強調
        shell_fat_display = masks['shell_fat'][:, :, slice_idx].astype(float)
        lung_extra = masks['lung_fat'][:, :, slice_idx] & ~masks['shell_fat'][:, :, slice_idx]
        
        # カラーコーディング
        combined = np.zeros((*ct_slice.shape, 3))
        
        # Shell脂肪（黄色）
        combined[:, :, 0] += shell_fat_display * 1.0  # R
        combined[:, :, 1] += shell_fat_display * 0.7  # G
        combined[:, :, 2] += shell_fat_display * 0.2  # B
        
        # 肺脂肪のShell外部分（シアン）
        combined[:, :, 0] += lung_extra * 0.2  # R
        combined[:, :, 1] += lung_extra * 0.7  # G
        combined[:, :, 2] += lung_extra * 1.0  # B
        
        combined = np.clip(combined, 0, 1)
        ax6.imshow(combined, alpha=0.7)
    
    total_area = np.sum(all_fat) * spacing[0] * spacing[1] / 100
    ax6.set_title(f'All Patterns Combined\n'
                 f'Total unique fat: {total_area:.1f} cm²\n'
                 f'Yellow: Shell, Cyan: Lung-extra',
                 fontsize=10, fontweight='bold')
    ax6.axis('off')
    
    # パネル7-9: 統計表示（下段）
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # 統計情報計算
    shell_voxels = np.sum(masks['shell_fat'][:, :, slice_idx])
    lung_voxels = np.sum(masks['lung_fat'][:, :, slice_idx])
    stomach_voxels = np.sum(masks['stomach_fat'][:, :, slice_idx])
    liver_voxels = np.sum(masks['liver_fat'][:, :, slice_idx])
    
    # オーバーラップ計算
    shell_lung_overlap = np.sum(masks['shell_fat'][:, :, slice_idx] & 
                               masks['lung_fat'][:, :, slice_idx])
    lung_stomach_overlap = np.sum(masks['lung_fat'][:, :, slice_idx] & 
                                 masks['stomach_fat'][:, :, slice_idx])
    lung_liver_overlap = np.sum(masks['lung_fat'][:, :, slice_idx] & 
                               masks['liver_fat'][:, :, slice_idx])
    
    # 統計テキスト
    stats_text = f"""
Slice {slice_idx} Statistics (Z = {slice_idx * spacing[2]:.1f} mm)
{'='*80}

Pattern Areas:
  • Shell-based:     {shell_area:6.1f} cm² ({shell_voxels:6,} voxels)
  • Lung-contact:    {lung_area:6.1f} cm² ({lung_voxels:6,} voxels)
  • Stomach-contact: {stomach_area:6.1f} cm² ({stomach_voxels:6,} voxels)  
  • Liver-contact:   {liver_area:6.1f} cm² ({liver_voxels:6,} voxels)

Key Comparisons:
  • Shell ∩ Lung:     {shell_lung_overlap * spacing[0] * spacing[1] / 100:6.1f} cm² 
    (Shell coverage: {(shell_lung_overlap/shell_voxels*100 if shell_voxels > 0 else 0):5.1f}%)
  • Lung \\ Shell:     {(lung_voxels - shell_lung_overlap) * spacing[0] * spacing[1] / 100:6.1f} cm² 
    (Extra beyond shell: {((lung_voxels - shell_lung_overlap)/lung_voxels*100 if lung_voxels > 0 else 0):5.1f}%)
  • Lung ∩ Stomach:   {lung_stomach_overlap * spacing[0] * spacing[1] / 100:6.1f} cm²
  • Lung ∩ Liver:     {lung_liver_overlap * spacing[0] * spacing[1] / 100:6.1f} cm²

Legend:
  🟨 Shell-based fat (traditional method)
  🟦 Lung-contact fat components (beyond shell)
  🟩 Overlap regions
"""
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 全体タイトル
    fig.suptitle(f'Organ Contact Fat Patterns - Slice {slice_idx}', 
                fontsize=16, fontweight='bold')
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_3d_mip_visualization(ct_data, masks, organs, spacing, output_path):
    """
    3方向MIP表示での比較
    """
    fig = plt.figure(figsize=(24, 20))
    
    # 3方向のMIPを作成
    views = {
        'Axial': (2, ct_data.max(axis=2)),
        'Coronal': (1, ct_data.max(axis=1)),
        'Sagittal': (0, ct_data.max(axis=0))
    }
    
    # 各マスクのMIP作成
    mask_mips = {}
    for view_name, (axis, _) in views.items():
        mask_mips[view_name] = {}
        for mask_name, mask in masks.items():
            mask_mips[view_name][mask_name] = mask.max(axis=axis)
    
    organ_mips = {}
    for view_name, (axis, _) in views.items():
        organ_mips[view_name] = {}
        for organ_name, organ in organs.items():
            organ_mips[view_name][organ_name] = organ.max(axis=axis)
    
    # 4x3グリッド（4パターン × 3方向）
    for col_idx, (view_name, (axis, ct_mip)) in enumerate(views.items()):
        
        # 各パターンを行に配置
        patterns = [
            ('Shell-based Fat', 'shell_fat', 'heart', 'Reds'),
            ('Lung-contact Fat', 'lung_fat', 'lungs', 'Blues'),
            ('Stomach-contact Fat', 'stomach_fat', 'stomach', 'Greens'),
            ('Liver-contact Fat', 'liver_fat', 'liver', 'Purples')
        ]
        
        for row_idx, (title, mask_key, organ_key, organ_color) in enumerate(patterns):
            ax = plt.subplot(4, 3, row_idx * 3 + col_idx + 1)
            
            # CT背景
            if axis == 1:  # Coronal
                ax.imshow(ct_mip.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
            elif axis == 0:  # Sagittal
                ax.imshow(ct_mip.T, cmap='gray', vmin=-200, vmax=200, origin='lower')
            else:  # Axial
                ax.imshow(ct_mip, cmap='gray', vmin=-200, vmax=200)
            
            # 臓器表示
            organ_mip = organ_mips[view_name][organ_key]
            if np.any(organ_mip):
                if axis == 1:  # Coronal
                    organ_masked = np.ma.masked_where(organ_mip.T < 0.5, organ_mip.T)
                    ax.imshow(organ_masked, cmap=organ_color, alpha=0.2, origin='lower')
                elif axis == 0:  # Sagittal
                    organ_masked = np.ma.masked_where(organ_mip.T < 0.5, organ_mip.T)
                    ax.imshow(organ_masked, cmap=organ_color, alpha=0.2, origin='lower')
                else:  # Axial
                    organ_masked = np.ma.masked_where(organ_mip < 0.5, organ_mip)
                    ax.imshow(organ_masked, cmap=organ_color, alpha=0.2)
            
            # 脂肪マスク表示
            fat_mip = mask_mips[view_name][mask_key]
            if np.any(fat_mip):
                if axis == 1:  # Coronal
                    fat_masked = np.ma.masked_where(fat_mip.T < 0.5, fat_mip.T)
                    ax.imshow(fat_masked, cmap='YlOrBr', alpha=0.7, origin='lower')
                elif axis == 0:  # Sagittal
                    fat_masked = np.ma.masked_where(fat_mip.T < 0.5, fat_mip.T)
                    ax.imshow(fat_masked, cmap='YlOrBr', alpha=0.7, origin='lower')
                else:  # Axial
                    fat_masked = np.ma.masked_where(fat_mip < 0.5, fat_mip)
                    ax.imshow(fat_masked, cmap='YlOrBr', alpha=0.7)
            
            # タイトル
            if col_idx == 1:  # 中央列のみパターン名表示
                ax.set_title(f'{title}\n{view_name} MIP', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'{view_name} MIP', fontsize=10)
            
            ax.axis('off')
    
    # 全体タイトル
    plt.suptitle('Organ Contact Fat Patterns - 3D MIP Comparison', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 3D MIP comparison: {output_path}")

def create_volume_chart(masks, spacing, output_path):
    """
    体積比較の棒グラフ
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    voxel_vol_ml = np.prod(spacing) / 1000.0
    
    # 体積計算
    volumes = {}
    for name, mask in masks.items():
        volumes[name] = np.sum(mask) * voxel_vol_ml
    
    # 棒グラフ1: 絶対体積
    names = ['Shell', 'Lung', 'Stomach', 'Liver']
    values = [volumes['shell_fat'], volumes['lung_fat'], 
              volumes['stomach_fat'], volumes['liver_fat']]
    colors = ['gold', 'skyblue', 'lightgreen', 'plum']
    
    bars1 = ax1.bar(names, values, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Volume (ml)', fontsize=12)
    ax1.set_title('Absolute Fat Volumes by Contact Pattern', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 値を棒の上に表示
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} ml', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 棒グラフ2: Shellに対する比率
    if volumes['shell_fat'] > 0:
        ratios = [100,  # Shell自身
                  volumes['lung_fat'] / volumes['shell_fat'] * 100,
                  volumes['stomach_fat'] / volumes['shell_fat'] * 100,
                  volumes['liver_fat'] / volumes['shell_fat'] * 100]
        
        bars2 = ax2.bar(names, ratios, color=colors, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Ratio to Shell (%)', fontsize=12)
        ax2.set_title('Fat Volume Ratios (Shell = 100%)', fontsize=14, fontweight='bold')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Shell baseline')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # 値を棒の上に表示
        for bar, val in zip(bars2, ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Fat Volume Analysis by Organ Contact Pattern', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved volume chart: {output_path}")

def visualize_organ_contact_comparison(ct_path, masks_dir, seg_dir, output_dir, 
                                       slice_indices=None, verbose=False):
    """
    臓器接触パターン比較可視化のメイン関数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(exist_ok=True)
    
    print("\n[INFO] Organ Contact Pattern Comparison Visualization")
    print("="*60)
    
    # 1. CT画像読み込み
    print("\n1. Loading CT image...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    spacing = tuple(float(s) for s in spacing)
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # 2. マスク読み込み
    print("\n2. Loading fat pattern masks...")
    masks_dir = Path(masks_dir)
    masks = load_masks(masks_dir, ct_data.shape)
    
    print("\n3. Loading organ masks...")
    seg_dir = Path(seg_dir)
    organs = load_organs(seg_dir, ct_data.shape)
    
    # 4. スライス選択
    if slice_indices is None:
        # 心臓レベルを中心に選択
        if np.any(organs['heart']):
            heart_z = np.where(np.any(organs['heart'], axis=(0, 1)))[0]
            if len(heart_z) > 0:
                # 心臓の全範囲から均等に選択
                n_slices = min(20, len(heart_z))
                slice_indices = np.linspace(heart_z[0], heart_z[-1], n_slices, dtype=int)
                print(f"\n4. Selected {len(slice_indices)} slices from heart region")
                print(f"   Z range: {heart_z[0]} to {heart_z[-1]}")
        else:
            # 心臓が見つからない場合
            mid = ct_data.shape[2] // 2
            slice_indices = range(mid-30, mid+30, 3)
            print(f"\n4. No heart found, using central slices")
    
    # 5. 体積統計
    print("\n5. Volume statistics...")
    voxel_vol = np.prod(spacing) / 1000.0
    
    for name, mask in masks.items():
        volume = np.sum(mask) * voxel_vol
        print(f"  {name}: {volume:.1f} ml ({np.sum(mask):,} voxels)")
    
    # オーバーラップ統計
    print("\n6. Overlap analysis...")
    shell_lung_overlap = np.sum(masks['shell_fat'] & masks['lung_fat'])
    shell_lung_overlap_ml = shell_lung_overlap * voxel_vol
    
    shell_only = np.sum(masks['shell_fat'] & ~masks['lung_fat'])
    lung_only = np.sum(masks['lung_fat'] & ~masks['shell_fat'])
    
    print(f"  Shell ∩ Lung: {shell_lung_overlap_ml:.1f} ml")
    print(f"  Shell \\ Lung: {shell_only * voxel_vol:.1f} ml (Shell only)")
    print(f"  Lung \\ Shell: {lung_only * voxel_vol:.1f} ml (Lung beyond shell)")
    
    if lung_only > 0:
        excess_ratio = (lung_only / np.sum(masks['shell_fat'])) * 100
        print(f"  → Lung fat exceeds shell by {excess_ratio:.1f}% of shell volume")
    
    # 7. スライス可視化
    print(f"\n7. Creating slice visualizations...")
    
    for i, slice_idx in enumerate(slice_indices):
        output_path = slices_dir / f'slice_{slice_idx:03d}.png'
        create_detailed_slice_visualization(ct_data, masks, organs, 
                                           slice_idx, spacing, output_path)
        
        if verbose:
            print(f"  Saved: {output_path}")
        elif (i+1) % 5 == 0:
            print(f"  Processed {i+1}/{len(slice_indices)} slices...")
    
    # 8. 3D MIP可視化
    print("\n8. Creating 3D MIP visualization...")
    mip_path = output_dir / 'mip_3d_comparison.png'
    create_3d_mip_visualization(ct_data, masks, organs, spacing, mip_path)
    
    # 9. 体積比較チャート
    print("\n9. Creating volume comparison chart...")
    chart_path = output_dir / 'volume_comparison.png'
    create_volume_chart(masks, spacing, chart_path)
    
    print(f"\n[SUCCESS] Visualization complete!")
    print(f"  Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize organ contact fat pattern comparison",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "ct_path",
        help="Path to CT NIfTI file"
    )
    
    parser.add_argument(
        "masks_dir",
        help="Path to masks directory with 4 pattern masks"
    )
    
    parser.add_argument(
        "seg_dir",
        help="Path to segmentation directory with organ masks"
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
        visualize_organ_contact_comparison(
            ct_path=args.ct_path,
            masks_dir=args.masks_dir,
            seg_dir=args.seg_dir,
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