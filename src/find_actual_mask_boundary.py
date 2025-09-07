#!/usr/bin/env python3
"""
マスクデータの実際の境界位置を正確に特定

275.2999mm付近ですべて0になっていることが判明。
実際のマスクデータの範囲と急激境界を詳細調査。
"""

import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from diffdrr.data import read

def analyze_actual_mask_range(mask_path):
    """マスクデータの実際の範囲を詳細解析"""
    print("🔍 マスクデータの実際の範囲解析")
    
    # 生のNIfTIファイルを直接読み込み
    nii = nib.load(mask_path)
    raw_data = nii.get_fdata()
    affine = nii.affine
    
    print(f"📐 生データ形状: {raw_data.shape}")
    print(f"🔢 アフィン行列:\n{affine}")
    
    # DiffDRRでの読み込みと比較
    subject_centered = read(str(mask_path), orientation="PA", center_volume=True)
    subject_original = read(str(mask_path), orientation="PA", center_volume=False)
    
    print(f"📊 DiffDRR Centered: {subject_centered['volume'].data.shape}")
    print(f"📊 DiffDRR Original: {subject_original['volume'].data.shape}")
    
    # 非ゼロ領域の特定
    nonzero_coords = np.where(raw_data > 0.5)
    
    if len(nonzero_coords[0]) > 0:
        print(f"\n✅ 非ゼロ領域発見: {len(nonzero_coords[0])} voxels")
        
        # 各軸での範囲
        for axis, name in enumerate(['X', 'Y', 'Z']):
            coords = nonzero_coords[axis]
            min_idx, max_idx = coords.min(), coords.max()
            print(f"📍 {name}軸: インデックス {min_idx} - {max_idx}")
            
            # 物理座標での変換
            if axis == 2:  # Z軸の場合
                # アフィン行列から物理座標計算
                origin_z = affine[2, 3]
                spacing_z = affine[2, 2]
                min_phys = origin_z + min_idx * spacing_z
                max_phys = origin_z + max_idx * spacing_z
                print(f"📏 {name}軸物理座標: {min_phys:.3f} - {max_phys:.3f} mm")
        
        # Z軸プロファイル詳細解析
        print(f"\n🔬 Z軸プロファイル解析")
        z_profile = []
        for z_idx in range(raw_data.shape[2]):
            slice_data = raw_data[:, :, z_idx]
            nonzero_count = np.sum(slice_data > 0.5)
            z_physical = affine[2, 3] + z_idx * affine[2, 2]
            z_profile.append((z_idx, z_physical, nonzero_count))
        
        # 非ゼロスライスのみ表示
        nonzero_slices = [(idx, phys, count) for idx, phys, count in z_profile if count > 0]
        
        if nonzero_slices:
            print(f"📈 非ゼロスライス数: {len(nonzero_slices)}")
            print(f"🎯 最初の非ゼロ: インデックス{nonzero_slices[0][0]}, {nonzero_slices[0][1]:.3f}mm, {nonzero_slices[0][2]}voxels")
            print(f"🎯 最後の非ゼロ: インデックス{nonzero_slices[-1][0]}, {nonzero_slices[-1][1]:.3f}mm, {nonzero_slices[-1][2]}voxels")
            
            # 急激な変化の検出
            print(f"\n💥 急激変化検出:")
            for i in range(len(nonzero_slices)-1):
                curr_idx, curr_phys, curr_count = nonzero_slices[i]
                next_idx, next_phys, next_count = nonzero_slices[i+1]
                
                # インデックスが連続していない = 間にゼロスライスが存在
                if next_idx - curr_idx > 1:
                    gap_slices = next_idx - curr_idx - 1
                    print(f"   ⚡ {curr_phys:.3f}mm → {next_phys:.3f}mm: {gap_slices}スライスの空白")
                
                # 密度の大きな変化
                density_change = abs(next_count - curr_count)
                if density_change > 1000:  # 閾値設定
                    print(f"   📊 {curr_phys:.3f}mm → {next_phys:.3f}mm: 密度変化 {density_change}")
            
            # 末尾の急激途切れ検出
            last_nonzero_idx = nonzero_slices[-1][0]
            last_nonzero_phys = nonzero_slices[-1][1]
            total_slices = raw_data.shape[2]
            
            if last_nonzero_idx < total_slices - 1:
                next_phys = affine[2, 3] + (last_nonzero_idx + 1) * affine[2, 2]
                print(f"   ⚡ 末尾急激途切れ: {last_nonzero_phys:.3f}mm → {next_phys:.3f}mm (完全ゼロ化)")
        
    else:
        print("❌ 非ゼロ領域が見つかりません")
        
    return raw_data, nonzero_slices if len(nonzero_coords[0]) > 0 else []

def compare_with_275mm_claim(nonzero_slices):
    """275.2999mmでの途切れ主張と実データの比較"""
    print(f"\n🎯 275.2999mm途切れ主張との比較")
    
    if not nonzero_slices:
        print("❌ 非ゼロデータが存在しないため比較不可")
        return
    
    target_z = 275.2999
    last_nonzero_phys = nonzero_slices[-1][1]
    
    print(f"📊 主張される途切れ位置: {target_z:.3f}mm")
    print(f"📊 実際の最終非ゼロ位置: {last_nonzero_phys:.3f}mm")
    print(f"📊 差分: {abs(last_nonzero_phys - target_z):.3f}mm")
    
    if abs(last_nonzero_phys - target_z) < 2.0:  # 2mm以内
        print("✅ 主張と実データが概ね一致")
        return True
    else:
        print("❌ 主張と実データに大きな差異")
        return False

def main():
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(mask_path).exists():
        print(f"❌ マスクファイルが見つかりません: {mask_path}")
        return
    
    print("="*60)
    print("🔍 マスクデータ実際の境界位置調査")
    print("="*60)
    
    raw_data, nonzero_slices = analyze_actual_mask_range(mask_path)
    
    if nonzero_slices:
        matches_claim = compare_with_275mm_claim(nonzero_slices)
        
        if matches_claim:
            print(f"\n🎯 結論: マスクデータは確実に急激途切れを持っている")
            print(f"🔍 問題: DiffDRRでの座標変換または投射処理に問題がある可能性")
        else:
            print(f"\n🎯 結論: 275.2999mm途切れ主張に問題がある可能性")
            print(f"🔍 実際の境界位置を基準に再調査が必要")
    
    print("🏁 実際境界調査完了")

if __name__ == "__main__":
    main()