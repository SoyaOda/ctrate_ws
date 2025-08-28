#!/usr/bin/env python3
"""
臓器接触脂肪の比較テストスクリプト
Z制限なしで純粋に各臓器に接するtorso_fat塊を可視化

1. 心臓Shellに基づくtorso_fat
2. 肺に接するtorso_fat塊（全連結成分）
3. 胃に接するtorso_fat塊（全連結成分）
4. 肝臓に接するtorso_fat塊（全連結成分）
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure
from scipy.ndimage import binary_dilation

def iso_dilate(mask, radius_mm, spacing):
    """
    等方的な膨張をEDTを使って実現
    
    Args:
        mask: 入力マスク
        radius_mm: 膨張半径（mm）
        spacing: ボクセルスペーシング（mm）
    
    Returns:
        膨張後のマスク
    """
    if radius_mm <= 0:
        return mask.copy()
    
    # 距離変換で外側への距離を計算
    dist = distance_transform_edt(~mask, sampling=spacing)
    
    # 指定距離以内を含める
    return dist <= radius_mm

def get_connected_components_touching(fat_mask, organ_mask, contact_mm=1.5, spacing=(1, 1, 1)):
    """
    臓器に接触するfat_maskの連結成分を全て取得
    
    Args:
        fat_mask: torso_fatマスク
        organ_mask: 臓器マスク
        contact_mm: 接触判定の許容距離（mm）
        spacing: ボクセルスペーシング
    
    Returns:
        接触する連結成分の合成マスク
    """
    if organ_mask is None or not np.any(organ_mask):
        return np.zeros_like(fat_mask)
    
    # 臓器を少し膨張させて接触帯を作成
    organ_touch = iso_dilate(organ_mask, contact_mm, spacing)
    
    # fatの連結成分ラベリング（26近傍）
    conn = generate_binary_structure(3, 2)
    lbl, n_components = label(fat_mask, structure=conn)
    
    # 臓器接触帯と重なるfat
    seed = fat_mask & organ_touch
    
    # シードに触れるラベルIDを取得
    touch_ids = np.unique(lbl[seed])
    touch_ids = touch_ids[touch_ids > 0]  # 背景（0）を除外
    
    # 触れた連結成分を全て採用
    result = np.isin(lbl, touch_ids)
    
    print(f"  Found {n_components} fat components, {len(touch_ids)} touching organ")
    print(f"  Total voxels: {np.sum(result):,}")
    
    return result

def analyze_organ_contact_fat(ct_path, segmentation_dir, output_dir, 
                              shell_radius_mm=15.0, contact_mm=1.5, verbose=False):
    """
    各臓器に接触する脂肪塊を分析・可視化
    
    Args:
        ct_path: CTファイルパス
        segmentation_dir: セグメンテーションディレクトリ
        output_dir: 出力ディレクトリ
        shell_radius_mm: 心臓Shell半径（mm）
        contact_mm: 接触判定距離（mm）
        verbose: 詳細出力
    """
    
    print("\n[INFO] Organ Contact Fat Analysis")
    print("="*60)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)
    
    # 1. CT画像読み込み
    print("\n1. Loading CT image...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    spacing = tuple(float(s) for s in spacing)
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # 2. セグメンテーションマスク読み込み
    print("\n2. Loading segmentation masks...")
    seg_dir = Path(segmentation_dir)
    
    # torso_fat（必須）
    fat_path = seg_dir / "torso_fat.nii.gz"
    if not fat_path.exists():
        print("[ERROR] torso_fat.nii.gz not found!")
        return None
    
    fat_img = nib.load(str(fat_path))
    fat_mask = fat_img.get_fdata() > 0
    print(f"  Loaded torso_fat: {np.sum(fat_mask):,} voxels")
    
    # 心臓マスク
    heart_mask = None
    heart_path = seg_dir / "heart_myocardium.nii.gz"
    if not heart_path.exists():
        heart_path = seg_dir / "heart.nii.gz"
    
    if heart_path.exists():
        heart_img = nib.load(str(heart_path))
        heart_mask = heart_img.get_fdata() > 0
        print(f"  Loaded heart: {np.sum(heart_mask):,} voxels")
    
    # 心臓チャンバー追加
    for chamber in ["heart_atrium_left", "heart_atrium_right", 
                   "heart_ventricle_left", "heart_ventricle_right"]:
        chamber_path = seg_dir / f"{chamber}.nii.gz"
        if chamber_path.exists():
            chamber_img = nib.load(str(chamber_path))
            chamber_mask = chamber_img.get_fdata() > 0
            if heart_mask is None:
                heart_mask = chamber_mask
            else:
                heart_mask |= chamber_mask
            if verbose:
                print(f"    Added {chamber}")
    
    # 肺マスク
    lungs = None
    left_lung_path = seg_dir / "lung_left.nii.gz"
    right_lung_path = seg_dir / "lung_right.nii.gz"
    
    if left_lung_path.exists() and right_lung_path.exists():
        left_lung = nib.load(str(left_lung_path)).get_fdata() > 0
        right_lung = nib.load(str(right_lung_path)).get_fdata() > 0
        lungs = left_lung | right_lung
        print(f"  Loaded lungs: {np.sum(lungs):,} voxels")
    
    # 胃マスク
    stomach = None
    stomach_path = seg_dir / "stomach.nii.gz"
    if stomach_path.exists():
        stomach_img = nib.load(str(stomach_path))
        stomach = stomach_img.get_fdata() > 0
        print(f"  Loaded stomach: {np.sum(stomach):,} voxels")
    
    # 肝臓マスク
    liver = None
    liver_path = seg_dir / "liver.nii.gz"
    if liver_path.exists():
        liver_img = nib.load(str(liver_path))
        liver = liver_img.get_fdata() > 0
        print(f"  Loaded liver: {np.sum(liver):,} voxels")
    
    # 3. 各パターンの脂肪抽出
    print(f"\n3. Extracting fat patterns (contact tolerance: {contact_mm} mm)...")
    
    results = {}
    
    # パターン1: 心臓Shellに基づくtorso_fat
    print("\n  Pattern 1: Heart Shell-based fat...")
    if heart_mask is not None:
        shell = iso_dilate(heart_mask, shell_radius_mm, spacing)
        shell_fat = shell & fat_mask
        results['shell_fat'] = shell_fat
        print(f"    Shell fat: {np.sum(shell_fat):,} voxels")
        
        # 保存
        shell_fat_img = nib.Nifti1Image(shell_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(shell_fat_img, str(masks_dir / "shell_fat.nii.gz"))
    else:
        results['shell_fat'] = np.zeros_like(fat_mask)
        print("    No heart mask available")
    
    # パターン2: 肺に接するtorso_fat塊
    print("\n  Pattern 2: Lung-touching fat components...")
    if lungs is not None:
        lung_fat = get_connected_components_touching(fat_mask, lungs, contact_mm, spacing)
        results['lung_fat'] = lung_fat
        
        # 保存
        lung_fat_img = nib.Nifti1Image(lung_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(lung_fat_img, str(masks_dir / "lung_fat.nii.gz"))
    else:
        results['lung_fat'] = np.zeros_like(fat_mask)
        print("    No lung mask available")
    
    # パターン3: 胃に接するtorso_fat塊
    print("\n  Pattern 3: Stomach-touching fat components...")
    if stomach is not None:
        stomach_fat = get_connected_components_touching(fat_mask, stomach, contact_mm, spacing)
        results['stomach_fat'] = stomach_fat
        
        # 保存
        stomach_fat_img = nib.Nifti1Image(stomach_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(stomach_fat_img, str(masks_dir / "stomach_fat.nii.gz"))
    else:
        results['stomach_fat'] = np.zeros_like(fat_mask)
        print("    No stomach mask available")
    
    # パターン4: 肝臓に接するtorso_fat塊
    print("\n  Pattern 4: Liver-touching fat components...")
    if liver is not None:
        liver_fat = get_connected_components_touching(fat_mask, liver, contact_mm, spacing)
        results['liver_fat'] = liver_fat
        
        # 保存
        liver_fat_img = nib.Nifti1Image(liver_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(liver_fat_img, str(masks_dir / "liver_fat.nii.gz"))
    else:
        results['liver_fat'] = np.zeros_like(fat_mask)
        print("    No liver mask available")
    
    # 4. 統計情報
    print("\n4. Volume statistics...")
    voxel_vol_ml = np.prod(spacing) / 1000.0
    
    volumes = {}
    for name, mask in results.items():
        vol_ml = np.sum(mask) * voxel_vol_ml
        volumes[name] = vol_ml
        pattern_name = name.replace('_', ' ').title()
        print(f"  {pattern_name}: {vol_ml:.1f} ml ({np.sum(mask):,} voxels)")
    
    # 5. オーバーラップ分析
    print("\n5. Overlap analysis...")
    
    # Shell vs Lung
    if np.any(results['shell_fat']) and np.any(results['lung_fat']):
        overlap = results['shell_fat'] & results['lung_fat']
        overlap_ml = np.sum(overlap) * voxel_vol_ml
        shell_only = np.sum(results['shell_fat'] & ~results['lung_fat']) * voxel_vol_ml
        lung_only = np.sum(results['lung_fat'] & ~results['shell_fat']) * voxel_vol_ml
        
        print(f"  Shell ∩ Lung: {overlap_ml:.1f} ml")
        print(f"  Shell \\ Lung: {shell_only:.1f} ml (Shell only)")
        print(f"  Lung \\ Shell: {lung_only:.1f} ml (Lung only)")
    
    # Lung vs Abdominal organs
    if np.any(results['lung_fat']):
        if np.any(results['stomach_fat']):
            lung_stomach = np.sum(results['lung_fat'] & results['stomach_fat']) * voxel_vol_ml
            print(f"  Lung ∩ Stomach: {lung_stomach:.1f} ml")
        
        if np.any(results['liver_fat']):
            lung_liver = np.sum(results['lung_fat'] & results['liver_fat']) * voxel_vol_ml
            print(f"  Lung ∩ Liver: {lung_liver:.1f} ml")
    
    # 6. 可視化
    print("\n6. Creating visualizations...")
    
    # 代表的なスライスを選択（心臓レベル）
    if heart_mask is not None:
        heart_z = np.where(np.any(heart_mask, axis=(0, 1)))[0]
        if len(heart_z) > 0:
            slice_indices = np.linspace(heart_z[0], heart_z[-1], min(12, len(heart_z)), dtype=int)
        else:
            mid = ct_data.shape[2] // 2
            slice_indices = range(mid-20, mid+20, 3)
    else:
        mid = ct_data.shape[2] // 2
        slice_indices = range(mid-20, mid+20, 3)
    
    # スライスごとの可視化
    for slice_idx in slice_indices:
        create_comparison_slice(ct_data, results, 
                              {'heart': heart_mask, 'lungs': lungs, 
                               'stomach': stomach, 'liver': liver},
                              slice_idx, spacing, vis_dir)
    
    # MIP可視化
    create_mip_comparison(ct_data, results, 
                         {'heart': heart_mask, 'lungs': lungs, 
                          'stomach': stomach, 'liver': liver},
                         spacing, vis_dir / "mip_comparison.png")
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"  Output directory: {output_dir}")
    
    return volumes

def create_comparison_slice(ct_data, fat_masks, organ_masks, slice_idx, spacing, output_dir):
    """
    単一スライスでの4パターン比較可視化
    """
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.15)
    
    ct_slice = ct_data[:, :, slice_idx]
    
    # カラーマップと透過度
    organ_colors = {'heart': 'Reds', 'lungs': 'Blues', 
                   'stomach': 'Greens', 'liver': 'Purples'}
    fat_color = 'YlOrBr'
    
    # 1. Shell-based fat（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # 心臓表示
    if organ_masks['heart'] is not None and np.any(organ_masks['heart'][:, :, slice_idx]):
        heart_masked = np.ma.masked_where(organ_masks['heart'][:, :, slice_idx] < 0.5,
                                         organ_masks['heart'][:, :, slice_idx])
        ax1.imshow(heart_masked, cmap='Reds', alpha=0.3)
    
    # Shell fat表示
    if np.any(fat_masks['shell_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(fat_masks['shell_fat'][:, :, slice_idx] < 0.5,
                                       fat_masks['shell_fat'][:, :, slice_idx])
        ax1.imshow(fat_masked, cmap=fat_color, alpha=0.7)
    
    fat_area = np.sum(fat_masks['shell_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax1.set_title(f'Heart Shell Fat\nArea: {fat_area:.1f} cm²', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Lung-touching fat（右上）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # 肺表示
    if organ_masks['lungs'] is not None and np.any(organ_masks['lungs'][:, :, slice_idx]):
        lung_masked = np.ma.masked_where(organ_masks['lungs'][:, :, slice_idx] < 0.5,
                                        organ_masks['lungs'][:, :, slice_idx])
        ax2.imshow(lung_masked, cmap='Blues', alpha=0.3)
    
    # Lung fat表示
    if np.any(fat_masks['lung_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(fat_masks['lung_fat'][:, :, slice_idx] < 0.5,
                                       fat_masks['lung_fat'][:, :, slice_idx])
        ax2.imshow(fat_masked, cmap=fat_color, alpha=0.7)
    
    fat_area = np.sum(fat_masks['lung_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax2.set_title(f'Lung-Touching Fat (All Components)\nArea: {fat_area:.1f} cm²', 
                 fontsize=12, fontweight='bold', color='blue')
    ax2.axis('off')
    
    # 3. Stomach-touching fat（左下）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # 胃表示
    if organ_masks['stomach'] is not None and np.any(organ_masks['stomach'][:, :, slice_idx]):
        stomach_masked = np.ma.masked_where(organ_masks['stomach'][:, :, slice_idx] < 0.5,
                                           organ_masks['stomach'][:, :, slice_idx])
        ax3.imshow(stomach_masked, cmap='Greens', alpha=0.3)
    
    # Stomach fat表示
    if np.any(fat_masks['stomach_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(fat_masks['stomach_fat'][:, :, slice_idx] < 0.5,
                                       fat_masks['stomach_fat'][:, :, slice_idx])
        ax3.imshow(fat_masked, cmap=fat_color, alpha=0.7)
    
    fat_area = np.sum(fat_masks['stomach_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax3.set_title(f'Stomach-Touching Fat (All Components)\nArea: {fat_area:.1f} cm²',
                 fontsize=12, fontweight='bold', color='green')
    ax3.axis('off')
    
    # 4. Liver-touching fat（右下）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    
    # 肝臓表示
    if organ_masks['liver'] is not None and np.any(organ_masks['liver'][:, :, slice_idx]):
        liver_masked = np.ma.masked_where(organ_masks['liver'][:, :, slice_idx] < 0.5,
                                         organ_masks['liver'][:, :, slice_idx])
        ax4.imshow(liver_masked, cmap='Purples', alpha=0.3)
    
    # Liver fat表示
    if np.any(fat_masks['liver_fat'][:, :, slice_idx]):
        fat_masked = np.ma.masked_where(fat_masks['liver_fat'][:, :, slice_idx] < 0.5,
                                       fat_masks['liver_fat'][:, :, slice_idx])
        ax4.imshow(fat_masked, cmap=fat_color, alpha=0.7)
    
    fat_area = np.sum(fat_masks['liver_fat'][:, :, slice_idx]) * spacing[0] * spacing[1] / 100
    ax4.set_title(f'Liver-Touching Fat (All Components)\nArea: {fat_area:.1f} cm²',
                 fontsize=12, fontweight='bold', color='purple')
    ax4.axis('off')
    
    # 全体タイトル
    fig.suptitle(f'Organ Contact Fat Comparison - Slice {slice_idx} (Z = {slice_idx * spacing[2]:.1f} mm)',
                fontsize=14, fontweight='bold')
    
    # 保存
    output_path = output_dir / f'slice_{slice_idx:03d}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_mip_comparison(ct_data, fat_masks, organ_masks, spacing, output_path):
    """
    MIPによる3方向比較表示
    """
    fig = plt.figure(figsize=(24, 16))
    
    # MIP作成（Axial）
    ct_mip = ct_data.max(axis=2)
    
    # 各マスクのMIP
    mips = {}
    for name, mask in fat_masks.items():
        mips[name] = mask.max(axis=2)
    
    organ_mips = {}
    for name, mask in organ_masks.items():
        if mask is not None:
            organ_mips[name] = mask.max(axis=2)
    
    # 2x2グリッドで表示
    titles = ['Heart Shell Fat', 'Lung-Touching Fat', 
              'Stomach-Touching Fat', 'Liver-Touching Fat']
    mask_keys = ['shell_fat', 'lung_fat', 'stomach_fat', 'liver_fat']
    organ_keys = ['heart', 'lungs', 'stomach', 'liver']
    organ_colors = ['Reds', 'Blues', 'Greens', 'Purples']
    
    for idx in range(4):
        ax = plt.subplot(2, 2, idx + 1)
        
        # CT背景
        ax.imshow(ct_mip, cmap='gray', vmin=-200, vmax=200)
        
        # 対応する臓器
        if organ_keys[idx] in organ_mips:
            organ_masked = np.ma.masked_where(organ_mips[organ_keys[idx]] < 0.5,
                                             organ_mips[organ_keys[idx]])
            ax.imshow(organ_masked, cmap=organ_colors[idx], alpha=0.3)
        
        # 脂肪マスク
        if np.any(mips[mask_keys[idx]]):
            fat_masked = np.ma.masked_where(mips[mask_keys[idx]] < 0.5,
                                           mips[mask_keys[idx]])
            ax.imshow(fat_masked, cmap='YlOrBr', alpha=0.7)
        
        # 体積計算
        voxel_vol = np.prod(spacing) / 1000.0
        volume = np.sum(fat_masks[mask_keys[idx]]) * voxel_vol
        
        ax.set_title(f'{titles[idx]}\nVolume: {volume:.1f} ml', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Organ Contact Fat - MIP Comparison (Axial View)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved MIP comparison: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Test organ contact fat patterns",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "ct_path",
        help="Path to CT NIfTI file"
    )
    
    parser.add_argument(
        "segmentation_dir", 
        help="Path to segmentation directory"
    )
    
    parser.add_argument(
        "output_dir",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--shell-radius",
        type=float,
        default=15.0,
        help="Heart shell radius in mm (default: 15.0)"
    )
    
    parser.add_argument(
        "--contact-mm",
        type=float,
        default=1.5,
        help="Contact tolerance in mm (default: 1.5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # 実行
    try:
        volumes = analyze_organ_contact_fat(
            ct_path=args.ct_path,
            segmentation_dir=args.segmentation_dir,
            output_dir=args.output_dir,
            shell_radius_mm=args.shell_radius,
            contact_mm=args.contact_mm,
            verbose=args.verbose
        )
        
        if volumes:
            print("\n[Summary]")
            for name, vol in volumes.items():
                print(f"  {name}: {vol:.1f} ml")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()