#!/usr/bin/env python3
"""
STEP5: EAT/PAT (Epicardial/Pericardial Adipose Tissue) 抽出スクリプト

心臓セグメンテーション結果からEAT/PATを抽出します。
- EAT: 心筋から5mm以内の脂肪組織
- PAT: 心筋から10mm以内の脂肪組織
- 脂肪のHU範囲: -190 ～ -30 HU
"""

import os
import sys
import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path


def load_nifti(filepath):
    """NIfTIファイルを読み込み"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    
    img = nib.load(filepath)
    data = img.get_fdata()
    return img, data


def calculate_dilation_iterations(spacing, distance_mm):
    """
    ボクセルスペーシングと距離(mm)から必要な膨張回数を計算
    
    Args:
        spacing: ボクセルスペーシング (x, y, z) in mm
        distance_mm: 膨張させたい距離 in mm
    
    Returns:
        膨張回数
    """
    min_spacing = min(spacing)
    iterations = int(np.ceil(distance_mm / min_spacing))
    return iterations


def extract_fat_tissue(ct_data, hu_min=-190, hu_max=-30):
    """
    HU値の範囲から脂肪組織を抽出
    
    Args:
        ct_data: CT画像データ (HU値)
        hu_min: 脂肪のHU最小値
        hu_max: 脂肪のHU最大値
    
    Returns:
        脂肪組織のバイナリマスク
    """
    fat_mask = (ct_data >= hu_min) & (ct_data <= hu_max)
    return fat_mask


def dilate_mask(mask, iterations, structure=None):
    """
    マスクを指定回数膨張させる
    
    Args:
        mask: 入力バイナリマスク
        iterations: 膨張回数
        structure: 構造要素（Noneの場合は3D 26-connected）
    
    Returns:
        膨張したマスク
    """
    if structure is None:
        structure = generate_binary_structure(3, 2)  # 26-connected
    
    dilated = mask.copy()
    for _ in range(iterations):
        dilated = binary_dilation(dilated, structure)
    
    return dilated


def calculate_eat_pat(ct_img, ct_data, myocardium_mask, 
                      eat_distance_mm=5.0, pat_distance_mm=10.0,
                      hu_min=-190, hu_max=-30):
    """
    EATとPATを計算
    
    Args:
        ct_img: NIfTI画像オブジェクト
        ct_data: CT画像データ (HU値)
        myocardium_mask: 心筋のバイナリマスク
        eat_distance_mm: EATの距離 (mm)
        pat_distance_mm: PATの距離 (mm)
        hu_min: 脂肪のHU最小値
        hu_max: 脂肪のHU最大値
    
    Returns:
        (EATマスク, PATマスク, EAT体積(ml), PAT体積(ml))
    """
    # ボクセルスペーシングを取得
    spacing = ct_img.header.get_zooms()[:3]
    voxel_volume_ml = np.prod(spacing) / 1000.0  # mm³ to ml
    
    print(f"ボクセルスペーシング: {spacing} mm")
    print(f"ボクセル体積: {voxel_volume_ml:.6f} ml")
    
    # 脂肪組織を抽出
    fat_mask = extract_fat_tissue(ct_data, hu_min, hu_max)
    fat_voxels = fat_mask.sum()
    print(f"脂肪ボクセル数: {fat_voxels}")
    print(f"脂肪体積: {fat_voxels * voxel_volume_ml:.2f} ml")
    
    # 膨張回数を計算
    eat_iterations = calculate_dilation_iterations(spacing, eat_distance_mm)
    pat_iterations = calculate_dilation_iterations(spacing, pat_distance_mm)
    
    print(f"EAT膨張回数 ({eat_distance_mm}mm): {eat_iterations}")
    print(f"PAT膨張回数 ({pat_distance_mm}mm): {pat_iterations}")
    
    # 心筋マスクを膨張
    structure = generate_binary_structure(3, 2)  # 26-connected
    
    # EAT領域（心筋から5mm）
    eat_roi = dilate_mask(myocardium_mask, eat_iterations, structure)
    eat_mask = fat_mask & eat_roi & ~myocardium_mask
    
    # PAT領域（心筋から10mm）
    pat_roi = dilate_mask(myocardium_mask, pat_iterations, structure)
    pat_mask = fat_mask & pat_roi & ~myocardium_mask
    
    # 体積計算
    eat_volume_ml = eat_mask.sum() * voxel_volume_ml
    pat_volume_ml = pat_mask.sum() * voxel_volume_ml
    
    return eat_mask, pat_mask, eat_volume_ml, pat_volume_ml


def save_mask(mask, reference_img, output_path):
    """
    マスクをNIfTIファイルとして保存
    
    Args:
        mask: 保存するバイナリマスク
        reference_img: 参照NIfTI画像（アフィン変換とヘッダー用）
        output_path: 出力ファイルパス
    """
    mask_img = nib.Nifti1Image(
        mask.astype(np.uint8), 
        reference_img.affine, 
        reference_img.header
    )
    nib.save(mask_img, output_path)
    print(f"保存: {output_path}")


def process_single_case(ct_path, myocardium_path, output_dir,
                       eat_distance_mm=5.0, pat_distance_mm=10.0):
    """
    単一症例を処理
    
    Args:
        ct_path: CT画像のパス
        myocardium_path: 心筋セグメンテーションのパス
        output_dir: 出力ディレクトリ
        eat_distance_mm: EATの距離 (mm)
        pat_distance_mm: PATの距離 (mm)
    
    Returns:
        (EAT体積, PAT体積) in ml
    """
    print(f"\n処理中: {ct_path}")
    print(f"心筋マスク: {myocardium_path}")
    
    # データ読み込み
    ct_img, ct_data = load_nifti(ct_path)
    myo_img, myo_data = load_nifti(myocardium_path)
    
    # 心筋マスクをバイナリ化
    myocardium_mask = myo_data > 0
    
    # データの統計情報
    print(f"\nCTデータ統計:")
    print(f"  形状: {ct_data.shape}")
    print(f"  HU範囲: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    print(f"  心筋ボクセル数: {myocardium_mask.sum()}")
    
    # EAT/PAT計算
    eat_mask, pat_mask, eat_volume, pat_volume = calculate_eat_pat(
        ct_img, ct_data, myocardium_mask,
        eat_distance_mm, pat_distance_mm
    )
    
    # 結果を保存
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名のベースを取得
    case_name = Path(ct_path).stem.replace('.nii', '')
    
    eat_path = os.path.join(output_dir, f"{case_name}_eat_{int(eat_distance_mm)}mm.nii.gz")
    pat_path = os.path.join(output_dir, f"{case_name}_pat_{int(pat_distance_mm)}mm.nii.gz")
    
    save_mask(eat_mask, ct_img, eat_path)
    save_mask(pat_mask, ct_img, pat_path)
    
    # 結果サマリー
    print(f"\n結果:")
    print(f"  EAT体積 ({eat_distance_mm}mm): {eat_volume:.2f} ml")
    print(f"  PAT体積 ({pat_distance_mm}mm): {pat_volume:.2f} ml")
    print(f"  EAT/PAT比: {eat_volume/pat_volume:.3f}" if pat_volume > 0 else "  EAT/PAT比: N/A")
    
    # 結果をテキストファイルに保存
    result_file = os.path.join(output_dir, f"{case_name}_volumes.txt")
    with open(result_file, 'w') as f:
        f.write(f"CT File: {ct_path}\n")
        f.write(f"Myocardium File: {myocardium_path}\n")
        f.write(f"EAT Volume ({eat_distance_mm}mm): {eat_volume:.2f} ml\n")
        f.write(f"PAT Volume ({pat_distance_mm}mm): {pat_volume:.2f} ml\n")
        f.write(f"EAT/PAT Ratio: {eat_volume/pat_volume:.3f}\n" if pat_volume > 0 else "EAT/PAT Ratio: N/A\n")
    
    return eat_volume, pat_volume


def main():
    parser = argparse.ArgumentParser(
        description='EAT/PAT (Epicardial/Pericardial Adipose Tissue) 抽出'
    )
    parser.add_argument('ct_path', help='CT画像ファイル (NIfTI)')
    parser.add_argument('myocardium_path', help='心筋セグメンテーションファイル (NIfTI)')
    parser.add_argument('-o', '--output', default='./eat_pat_output',
                       help='出力ディレクトリ (デフォルト: ./eat_pat_output)')
    parser.add_argument('--eat-distance', type=float, default=5.0,
                       help='EAT距離 in mm (デフォルト: 5.0)')
    parser.add_argument('--pat-distance', type=float, default=10.0,
                       help='PAT距離 in mm (デフォルト: 10.0)')
    parser.add_argument('--hu-min', type=float, default=-190,
                       help='脂肪のHU最小値 (デフォルト: -190)')
    parser.add_argument('--hu-max', type=float, default=-30,
                       help='脂肪のHU最大値 (デフォルト: -30)')
    
    args = parser.parse_args()
    
    # 処理実行
    try:
        eat_vol, pat_vol = process_single_case(
            args.ct_path,
            args.myocardium_path,
            args.output,
            args.eat_distance,
            args.pat_distance
        )
        
        print(f"\n処理完了!")
        print(f"出力ディレクトリ: {args.output}")
        
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()