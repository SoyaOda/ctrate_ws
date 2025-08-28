#!/usr/bin/env python3
"""
心臓レベルに限定した臓器接触脂肪の比較テストスクリプト
巨大連結成分問題を回避するため、Z範囲を制限
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
from scipy.ndimage import binary_opening

def iso_dilate(mask, radius_mm, spacing):
    """等方的な膨張をEDTを使って実現"""
    if radius_mm <= 0:
        return mask.copy()
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm

def get_connected_components_touching(fat_mask, organ_mask, contact_mm=1.5, spacing=(1, 1, 1)):
    """臓器に接触するfat_maskの連結成分を全て取得"""
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
    touch_ids = touch_ids[touch_ids > 0]
    
    # 触れた連結成分を全て採用
    result = np.isin(lbl, touch_ids)
    
    print(f"  Found {n_components} fat components, {len(touch_ids)} touching organ")
    print(f"  Total voxels: {np.sum(result):,}")
    
    return result

def analyze_cardiac_level_fat(ct_path, segmentation_dir, output_dir, 
                              shell_radius_mm=15.0, contact_mm=1.5, 
                              cardiac_margin_mm=30.0, verbose=False):
    """
    心臓レベルに限定して各臓器に接触する脂肪塊を分析
    
    Args:
        cardiac_margin_mm: 心臓の上下に追加するマージン（mm）
    """
    
    print("\n[INFO] Cardiac-Level Organ Contact Fat Analysis")
    print("="*60)
    
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
    
    # 必要なマスクを読み込み
    fat_path = seg_dir / "torso_fat.nii.gz"
    if not fat_path.exists():
        print("[ERROR] torso_fat.nii.gz not found!")
        return None
    
    fat_img = nib.load(str(fat_path))
    fat_mask_full = fat_img.get_fdata() > 0
    print(f"  Loaded torso_fat: {np.sum(fat_mask_full):,} voxels (full volume)")
    
    # 心臓マスク
    heart_mask = None
    heart_path = seg_dir / "heart_myocardium.nii.gz"
    if not heart_path.exists():
        heart_path = seg_dir / "heart.nii.gz"
    
    if heart_path.exists():
        heart_img = nib.load(str(heart_path))
        heart_mask = heart_img.get_fdata() > 0
        
        # 心臓チャンバー追加
        for chamber in ["heart_atrium_left", "heart_atrium_right", 
                       "heart_ventricle_left", "heart_ventricle_right"]:
            chamber_path = seg_dir / f"{chamber}.nii.gz"
            if chamber_path.exists():
                chamber_img = nib.load(str(chamber_path))
                heart_mask |= chamber_img.get_fdata() > 0
        
        print(f"  Loaded heart: {np.sum(heart_mask):,} voxels")
    else:
        print("[ERROR] Heart mask not found!")
        return None
    
    # 3. 心臓レベルのZ範囲を決定
    print("\n3. Determining cardiac Z-range...")
    heart_z = np.where(np.any(heart_mask, axis=(0, 1)))[0]
    z_min = int(heart_z[0])
    z_max = int(heart_z[-1])
    
    # マージンを追加
    margin_slices = int(np.ceil(cardiac_margin_mm / spacing[2]))
    z_min_margin = max(0, z_min - margin_slices)
    z_max_margin = min(ct_data.shape[2] - 1, z_max + margin_slices)
    
    print(f"  Heart Z-range: {z_min} to {z_max}")
    print(f"  With {cardiac_margin_mm}mm margin: {z_min_margin} to {z_max_margin}")
    print(f"  Total slices: {z_max_margin - z_min_margin + 1}")
    
    # 4. 脂肪マスクを心臓レベルに制限
    print("\n4. Restricting fat to cardiac level...")
    fat_mask = fat_mask_full.copy()
    fat_mask[:, :, :z_min_margin] = False
    fat_mask[:, :, z_max_margin+1:] = False
    
    original_voxels = np.sum(fat_mask_full)
    cardiac_voxels = np.sum(fat_mask)
    removed_percentage = (1 - cardiac_voxels/original_voxels) * 100
    
    print(f"  Original fat: {original_voxels:,} voxels")
    print(f"  Cardiac-level fat: {cardiac_voxels:,} voxels")
    print(f"  Removed: {removed_percentage:.1f}% (outside cardiac region)")
    
    # 連結成分の再分析
    conn = generate_binary_structure(3, 2)
    lbl, n_components = label(fat_mask, structure=conn)
    
    # サイズ分布
    component_sizes = []
    for i in range(1, n_components + 1):
        size = np.sum(lbl == i)
        component_sizes.append((i, size))
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    voxel_vol_ml = np.prod(spacing) / 1000.0
    
    print(f"\n  Connected components in cardiac-level fat:")
    print(f"  Total components: {n_components}")
    for i, (comp_id, size) in enumerate(component_sizes[:5]):
        volume_ml = size * voxel_vol_ml
        percentage = (size / cardiac_voxels) * 100
        print(f"    Component {comp_id}: {size:,} voxels ({volume_ml:.1f} ml, {percentage:.1f}%)")
    
    # 最大成分の割合確認
    if component_sizes:
        largest_percent = (component_sizes[0][1] / cardiac_voxels) * 100
        if largest_percent > 80:
            print(f"\n  Note: Largest component is still {largest_percent:.1f}% of cardiac fat")
            
            # Opening操作で細い接続を切断
            print("  Applying morphological opening to separate weakly connected regions...")
            struct = generate_binary_structure(3, 1)  # 6-connectivity for opening
            fat_mask_opened = binary_opening(fat_mask, structure=struct, iterations=1)
            
            # 再度ラベリング
            lbl_opened, n_comp_opened = label(fat_mask_opened, structure=conn)
            print(f"  After opening: {n_comp_opened} components")
            
            if n_comp_opened > n_components:
                print("  Successfully separated some components!")
                fat_mask = fat_mask_opened
                lbl = lbl_opened
                n_components = n_comp_opened
    
    # 5. 他の臓器マスクを読み込み（心臓レベルに制限）
    print("\n5. Loading other organ masks (cardiac level)...")
    
    # 肺
    lungs = None
    left_lung_path = seg_dir / "lung_left.nii.gz"
    right_lung_path = seg_dir / "lung_right.nii.gz"
    
    if left_lung_path.exists() and right_lung_path.exists():
        left_lung = nib.load(str(left_lung_path)).get_fdata() > 0
        right_lung = nib.load(str(right_lung_path)).get_fdata() > 0
        lungs = left_lung | right_lung
        # 心臓レベルに制限
        lungs[:, :, :z_min_margin] = False
        lungs[:, :, z_max_margin+1:] = False
        print(f"  Loaded lungs (cardiac level): {np.sum(lungs):,} voxels")
    
    # 胃（心臓レベルでは一部のみ）
    stomach = None
    stomach_path = seg_dir / "stomach.nii.gz"
    if stomach_path.exists():
        stomach = nib.load(str(stomach_path)).get_fdata() > 0
        stomach[:, :, :z_min_margin] = False
        stomach[:, :, z_max_margin+1:] = False
        if np.any(stomach):
            print(f"  Loaded stomach (cardiac level): {np.sum(stomach):,} voxels")
        else:
            print("  Stomach: not present at cardiac level")
            stomach = None
    
    # 肝臓（心臓レベルの下部）
    liver = None
    liver_path = seg_dir / "liver.nii.gz"
    if liver_path.exists():
        liver = nib.load(str(liver_path)).get_fdata() > 0
        liver[:, :, :z_min_margin] = False
        liver[:, :, z_max_margin+1:] = False
        if np.any(liver):
            print(f"  Loaded liver (cardiac level): {np.sum(liver):,} voxels")
        else:
            print("  Liver: not present at cardiac level")
            liver = None
    
    # 6. 各パターンの脂肪抽出
    print(f"\n6. Extracting fat patterns (contact tolerance: {contact_mm} mm)...")
    
    results = {}
    
    # パターン1: 心臓Shellに基づくtorso_fat
    print("\n  Pattern 1: Heart Shell-based fat...")
    shell = iso_dilate(heart_mask, shell_radius_mm, spacing)
    shell_fat = shell & fat_mask
    results['shell_fat'] = shell_fat
    print(f"    Shell fat: {np.sum(shell_fat):,} voxels")
    
    # 保存
    shell_fat_img = nib.Nifti1Image(shell_fat.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(shell_fat_img, str(masks_dir / "shell_fat_cardiac.nii.gz"))
    
    # パターン2: 肺に接するtorso_fat塊
    print("\n  Pattern 2: Lung-touching fat components...")
    if lungs is not None and np.any(lungs):
        lung_fat = get_connected_components_touching(fat_mask, lungs, contact_mm, spacing)
        results['lung_fat'] = lung_fat
        
        lung_fat_img = nib.Nifti1Image(lung_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(lung_fat_img, str(masks_dir / "lung_fat_cardiac.nii.gz"))
    else:
        results['lung_fat'] = np.zeros_like(fat_mask)
        print("    No lung available at this level")
    
    # パターン3: 胃に接するtorso_fat塊（心臓レベルでは限定的）
    print("\n  Pattern 3: Stomach-touching fat components...")
    if stomach is not None and np.any(stomach):
        stomach_fat = get_connected_components_touching(fat_mask, stomach, contact_mm, spacing)
        results['stomach_fat'] = stomach_fat
        
        stomach_fat_img = nib.Nifti1Image(stomach_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(stomach_fat_img, str(masks_dir / "stomach_fat_cardiac.nii.gz"))
    else:
        results['stomach_fat'] = np.zeros_like(fat_mask)
        print("    No stomach available at cardiac level")
    
    # パターン4: 肝臓に接するtorso_fat塊（心臓レベル下部）
    print("\n  Pattern 4: Liver-touching fat components...")
    if liver is not None and np.any(liver):
        liver_fat = get_connected_components_touching(fat_mask, liver, contact_mm, spacing)
        results['liver_fat'] = liver_fat
        
        liver_fat_img = nib.Nifti1Image(liver_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(liver_fat_img, str(masks_dir / "liver_fat_cardiac.nii.gz"))
    else:
        results['liver_fat'] = np.zeros_like(fat_mask)
        print("    No liver available at cardiac level")
    
    # 7. 統計情報
    print("\n7. Volume statistics (cardiac level only)...")
    
    volumes = {}
    for name, mask in results.items():
        vol_ml = np.sum(mask) * voxel_vol_ml
        volumes[name] = vol_ml
        pattern_name = name.replace('_', ' ').title()
        print(f"  {pattern_name}: {vol_ml:.1f} ml ({np.sum(mask):,} voxels)")
    
    # 8. 比較分析
    print("\n8. Comparison analysis...")
    
    if np.any(results['shell_fat']) and np.any(results['lung_fat']):
        overlap = results['shell_fat'] & results['lung_fat']
        overlap_ml = np.sum(overlap) * voxel_vol_ml
        shell_only = np.sum(results['shell_fat'] & ~results['lung_fat']) * voxel_vol_ml
        lung_only = np.sum(results['lung_fat'] & ~results['shell_fat']) * voxel_vol_ml
        
        print(f"  Shell ∩ Lung: {overlap_ml:.1f} ml")
        print(f"  Shell \\ Lung: {shell_only:.1f} ml (Shell only)")
        print(f"  Lung \\ Shell: {lung_only:.1f} ml (Lung only)")
        
        if lung_only > shell_only:
            excess_ratio = ((lung_only - shell_only) / (shell_only + overlap_ml)) * 100
            print(f"  → Lung fat exceeds shell by {excess_ratio:.1f}% of shell volume")
    
    # 9. 簡易可視化（中央スライス）
    print("\n9. Creating quick visualization...")
    
    mid_z = (z_min + z_max) // 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    ct_slice = ct_data[:, :, mid_z]
    
    # Shell fat
    axes[0, 0].imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    axes[0, 0].imshow(np.ma.masked_where(results['shell_fat'][:, :, mid_z] < 0.5,
                                         results['shell_fat'][:, :, mid_z]),
                     cmap='YlOrBr', alpha=0.7)
    axes[0, 0].set_title(f"Shell Fat: {volumes['shell_fat']:.1f} ml")
    axes[0, 0].axis('off')
    
    # Lung fat
    axes[0, 1].imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    axes[0, 1].imshow(np.ma.masked_where(results['lung_fat'][:, :, mid_z] < 0.5,
                                         results['lung_fat'][:, :, mid_z]),
                     cmap='YlOrBr', alpha=0.7)
    axes[0, 1].set_title(f"Lung Fat: {volumes['lung_fat']:.1f} ml")
    axes[0, 1].axis('off')
    
    # Stomach fat
    axes[1, 0].imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    axes[1, 0].imshow(np.ma.masked_where(results['stomach_fat'][:, :, mid_z] < 0.5,
                                         results['stomach_fat'][:, :, mid_z]),
                     cmap='YlOrBr', alpha=0.7)
    axes[1, 0].set_title(f"Stomach Fat: {volumes['stomach_fat']:.1f} ml")
    axes[1, 0].axis('off')
    
    # Liver fat
    axes[1, 1].imshow(ct_slice, cmap='gray', vmin=-200, vmax=200)
    axes[1, 1].imshow(np.ma.masked_where(results['liver_fat'][:, :, mid_z] < 0.5,
                                         results['liver_fat'][:, :, mid_z]),
                     cmap='YlOrBr', alpha=0.7)
    axes[1, 1].set_title(f"Liver Fat: {volumes['liver_fat']:.1f} ml")
    axes[1, 1].axis('off')
    
    plt.suptitle(f'Cardiac-Level Fat Patterns (Slice {mid_z})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(vis_dir / 'cardiac_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {vis_dir / 'cardiac_comparison.png'}")
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"  Output directory: {output_dir}")
    
    return volumes

def main():
    parser = argparse.ArgumentParser(
        description="Test cardiac-level organ contact fat patterns",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("ct_path", help="Path to CT NIfTI file")
    parser.add_argument("segmentation_dir", help="Path to segmentation directory")
    parser.add_argument("output_dir", help="Path to output directory")
    
    parser.add_argument("--shell-radius", type=float, default=15.0,
                       help="Heart shell radius in mm (default: 15.0)")
    parser.add_argument("--contact-mm", type=float, default=1.5,
                       help="Contact tolerance in mm (default: 1.5)")
    parser.add_argument("--cardiac-margin", type=float, default=30.0,
                       help="Margin around heart in mm (default: 30.0)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        volumes = analyze_cardiac_level_fat(
            ct_path=args.ct_path,
            segmentation_dir=args.segmentation_dir,
            output_dir=args.output_dir,
            shell_radius_mm=args.shell_radius,
            contact_mm=args.contact_mm,
            cardiac_margin_mm=args.cardiac_margin,
            verbose=args.verbose
        )
        
        if volumes:
            print("\n[Summary - Cardiac Level Only]")
            for name, vol in volumes.items():
                print(f"  {name}: {vol:.1f} ml")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()