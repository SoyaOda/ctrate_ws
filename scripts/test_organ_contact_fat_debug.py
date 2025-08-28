#!/usr/bin/env python3
"""
臓器接触脂肪の比較テストスクリプト（デバッグ版）
連結成分の詳細分析を追加
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
    """
    if radius_mm <= 0:
        return mask.copy()
    
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm

def analyze_connected_components(fat_mask, spacing):
    """
    連結成分の詳細分析
    """
    print("\n[Connected Component Analysis]")
    
    # 26近傍で連結成分ラベリング
    conn = generate_binary_structure(3, 2)
    lbl, n_components = label(fat_mask, structure=conn)
    
    print(f"  Total components: {n_components}")
    
    # 各成分のサイズを計算
    component_sizes = []
    for i in range(1, n_components + 1):
        size = np.sum(lbl == i)
        component_sizes.append((i, size))
    
    # サイズでソート
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # 上位10個を表示
    voxel_vol_ml = np.prod(spacing) / 1000.0
    total_voxels = np.sum(fat_mask)
    
    print(f"\n  Top 10 largest components:")
    for i, (comp_id, size) in enumerate(component_sizes[:10]):
        volume_ml = size * voxel_vol_ml
        percentage = (size / total_voxels) * 100
        print(f"    Component {comp_id}: {size:,} voxels ({volume_ml:.1f} ml, {percentage:.1f}% of total)")
    
    # 最大成分の割合
    if component_sizes:
        largest_size = component_sizes[0][1]
        largest_percent = (largest_size / total_voxels) * 100
        print(f"\n  Largest component contains {largest_percent:.1f}% of all fat voxels")
        
        if largest_percent > 90:
            print("  ⚠️ WARNING: Over 90% of fat is in a single connected component!")
            print("     This explains why different organs select similar volumes.")
    
    return lbl, component_sizes

def get_connected_components_touching_detailed(fat_mask, organ_mask, contact_mm, spacing, lbl=None):
    """
    臓器に接触する連結成分を取得（詳細情報付き）
    """
    if organ_mask is None or not np.any(organ_mask):
        return np.zeros_like(fat_mask), []
    
    # 連結成分ラベルが提供されていない場合は計算
    if lbl is None:
        conn = generate_binary_structure(3, 2)
        lbl, n_components = label(fat_mask, structure=conn)
    else:
        n_components = lbl.max()
    
    # 臓器を少し膨張させて接触帯を作成
    organ_touch = iso_dilate(organ_mask, contact_mm, spacing)
    
    # 臓器接触帯と重なるfat
    seed = fat_mask & organ_touch
    
    # シードに触れるラベルIDを取得
    touch_ids = np.unique(lbl[seed])
    touch_ids = touch_ids[touch_ids > 0]  # 背景（0）を除外
    
    # 各接触成分のサイズ
    touch_component_sizes = []
    voxel_vol_ml = np.prod(spacing) / 1000.0
    
    print(f"\n  Components touching this organ:")
    for comp_id in touch_ids[:5]:  # 上位5個を表示
        size = np.sum(lbl == comp_id)
        volume_ml = size * voxel_vol_ml
        touch_component_sizes.append((comp_id, size))
        print(f"    Component {comp_id}: {size:,} voxels ({volume_ml:.1f} ml)")
    
    # 触れた連結成分を全て採用
    result = np.isin(lbl, touch_ids)
    
    print(f"  Total: {len(touch_ids)} components touching, {np.sum(result):,} voxels")
    
    return result, touch_component_sizes

def analyze_organ_contact_fat_debug(ct_path, segmentation_dir, output_dir, 
                                   shell_radius_mm=15.0, contact_mm=1.5, verbose=False):
    """
    各臓器に接触する脂肪塊を分析・可視化（デバッグ版）
    """
    
    print("\n[INFO] Organ Contact Fat Analysis (Debug Version)")
    print("="*60)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
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
    
    # 3. 連結成分分析
    print("\n3. Analyzing fat connected components...")
    lbl, component_sizes = analyze_connected_components(fat_mask, spacing)
    
    # 4. 各パターンの脂肪抽出
    print(f"\n4. Extracting fat patterns (contact tolerance: {contact_mm} mm)...")
    
    results = {}
    
    # パターン1: 心臓Shellに基づくtorso_fat
    print("\n  Pattern 1: Heart Shell-based fat...")
    if heart_mask is not None:
        shell = iso_dilate(heart_mask, shell_radius_mm, spacing)
        shell_fat = shell & fat_mask
        results['shell_fat'] = shell_fat
        print(f"    Shell fat: {np.sum(shell_fat):,} voxels")
        
        # Shellに含まれる連結成分を分析
        shell_components = np.unique(lbl[shell_fat])
        shell_components = shell_components[shell_components > 0]
        print(f"    Components in shell: {len(shell_components)}")
        
        # 保存
        shell_fat_img = nib.Nifti1Image(shell_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(shell_fat_img, str(masks_dir / "shell_fat.nii.gz"))
    else:
        results['shell_fat'] = np.zeros_like(fat_mask)
    
    # パターン2: 肺に接するtorso_fat塊
    print("\n  Pattern 2: Lung-touching fat components...")
    if lungs is not None:
        lung_fat, lung_components = get_connected_components_touching_detailed(
            fat_mask, lungs, contact_mm, spacing, lbl)
        results['lung_fat'] = lung_fat
        
        # 保存
        lung_fat_img = nib.Nifti1Image(lung_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(lung_fat_img, str(masks_dir / "lung_fat.nii.gz"))
    else:
        results['lung_fat'] = np.zeros_like(fat_mask)
    
    # パターン3: 胃に接するtorso_fat塊
    print("\n  Pattern 3: Stomach-touching fat components...")
    if stomach is not None:
        stomach_fat, stomach_components = get_connected_components_touching_detailed(
            fat_mask, stomach, contact_mm, spacing, lbl)
        results['stomach_fat'] = stomach_fat
        
        # 保存
        stomach_fat_img = nib.Nifti1Image(stomach_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(stomach_fat_img, str(masks_dir / "stomach_fat.nii.gz"))
    else:
        results['stomach_fat'] = np.zeros_like(fat_mask)
    
    # パターン4: 肝臓に接するtorso_fat塊
    print("\n  Pattern 4: Liver-touching fat components...")
    if liver is not None:
        liver_fat, liver_components = get_connected_components_touching_detailed(
            fat_mask, liver, contact_mm, spacing, lbl)
        results['liver_fat'] = liver_fat
        
        # 保存
        liver_fat_img = nib.Nifti1Image(liver_fat.astype(np.uint8), ct_img.affine, ct_img.header)
        nib.save(liver_fat_img, str(masks_dir / "liver_fat.nii.gz"))
    else:
        results['liver_fat'] = np.zeros_like(fat_mask)
    
    # 5. 成分の重複分析
    print("\n5. Component overlap analysis...")
    
    if 'lung_components' in locals() and 'stomach_components' in locals():
        lung_comp_ids = set([c[0] for c in lung_components])
        stomach_comp_ids = set([c[0] for c in stomach_components])
        liver_comp_ids = set([c[0] for c in liver_components]) if 'liver_components' in locals() else set()
        
        # 共通成分を見つける
        lung_stomach_common = lung_comp_ids & stomach_comp_ids
        lung_liver_common = lung_comp_ids & liver_comp_ids
        all_common = lung_comp_ids & stomach_comp_ids & liver_comp_ids
        
        print(f"  Components touching both lung and stomach: {lung_stomach_common}")
        print(f"  Components touching both lung and liver: {lung_liver_common}")
        print(f"  Components touching all three organs: {all_common}")
        
        if all_common:
            print("\n  ⚠️ WARNING: Some fat components touch multiple organs!")
            print("     This indicates a large interconnected fat mass.")
            
            # 最大共通成分のサイズを計算
            for comp_id in all_common:
                size = np.sum(lbl == comp_id)
                voxel_vol_ml = np.prod(spacing) / 1000.0
                volume_ml = size * voxel_vol_ml
                print(f"     Component {comp_id}: {volume_ml:.1f} ml")
    
    # 6. 統計情報
    print("\n6. Volume statistics...")
    voxel_vol_ml = np.prod(spacing) / 1000.0
    
    volumes = {}
    for name, mask in results.items():
        vol_ml = np.sum(mask) * voxel_vol_ml
        volumes[name] = vol_ml
        pattern_name = name.replace('_', ' ').title()
        print(f"  {pattern_name}: {vol_ml:.1f} ml ({np.sum(mask):,} voxels)")
    
    # 7. 推奨事項
    print("\n7. Recommendations:")
    
    if component_sizes and component_sizes[0][1] / np.sum(fat_mask) > 0.9:
        print("  ⚠️ The torso_fat is mostly one giant connected component.")
        print("     Consider:")
        print("     1. Using Z-range restrictions to separate upper/lower fat")
        print("     2. Using anatomical landmarks to divide regions")
        print("     3. Applying morphological operations to separate weakly connected parts")
    
    print(f"\n[SUCCESS] Analysis complete!")
    print(f"  Output directory: {output_dir}")
    
    return volumes

def main():
    parser = argparse.ArgumentParser(
        description="Debug organ contact fat patterns",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("ct_path", help="Path to CT NIfTI file")
    parser.add_argument("segmentation_dir", help="Path to segmentation directory")
    parser.add_argument("output_dir", help="Path to output directory")
    
    parser.add_argument("--shell-radius", type=float, default=15.0,
                       help="Heart shell radius in mm (default: 15.0)")
    parser.add_argument("--contact-mm", type=float, default=1.5,
                       help="Contact tolerance in mm (default: 1.5)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # 実行
    try:
        volumes = analyze_organ_contact_fat_debug(
            ct_path=args.ct_path,
            segmentation_dir=args.segmentation_dir,
            output_dir=args.output_dir,
            shell_radius_mm=args.shell_radius,
            contact_mm=args.contact_mm,
            verbose=args.verbose
        )
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()