#!/usr/bin/env python3
# STEP5: EAT/PAT抽出スクリプト
# Epicardial Adipose Tissue (EAT) と Pericardial Adipose Tissue (PAT) の算出

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
import os
import sys
import argparse

def extract_eat_pat(ct_path, myocardium_path, output_dir, verbose=False):
    """
    CTデータと心筋マスクからEAT/PATを抽出
    
    Parameters:
    - ct_path: CTデータのパス (NIfTI)
    - myocardium_path: 心筋マスクのパス (NIfTI)
    - output_dir: 出力ディレクトリ
    - verbose: 詳細出力フラグ
    """
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データの読み込み
    if verbose:
        print(f"CTデータを読み込み中: {ct_path}")
    ct = nib.load(ct_path)
    ct_data = ct.get_fdata()
    
    # 心筋マスクの読み込み
    if os.path.exists(myocardium_path):
        if verbose:
            print(f"心筋マスクを読み込み中: {myocardium_path}")
        myo = nib.load(myocardium_path).get_fdata() > 0
    else:
        # 心筋マスクがない場合は代替処理
        print(f"警告: 心筋マスクが見つかりません: {myocardium_path}")
        print("代替として心臓領域を推定します...")
        # 簡易的な心臓領域の推定（HU値ベース）
        myo = (ct_data > 30) & (ct_data < 150)  # 軟部組織のHU範囲
    
    # ボクセルサイズの取得
    spacing = ct.header.get_zooms()[:3]
    voxel_volume = np.prod(spacing) / 1000.0  # mLに変換
    
    if verbose:
        print(f"ボクセルサイズ: {spacing} mm")
        print(f"ボクセル体積: {voxel_volume} mL")
    
    # 脂肪組織の抽出（HU値: -190 〜 -30）
    fat_mask = (ct_data >= -190) & (ct_data <= -30)
    
    # 膨張用の構造要素
    struct_elem = generate_binary_structure(3, 2)
    
    # 膨張距離の計算（ボクセル単位）
    min_spacing = min(spacing)
    dilation_5mm = int(5.0 / min_spacing)
    dilation_10mm = int(10.0 / min_spacing)
    
    if verbose:
        print(f"5mm膨張: {dilation_5mm} ボクセル")
        print(f"10mm膨張: {dilation_10mm} ボクセル")
    
    # 5mm膨張領域（Epicardial用）
    print("心筋から5mm膨張中...")
    roi_5mm = myo.copy()
    for i in range(dilation_5mm):
        roi_5mm = binary_dilation(roi_5mm, struct_elem)
        if verbose and i % 5 == 0:
            print(f"  膨張 {i+1}/{dilation_5mm}")
    
    # 10mm膨張領域（Pericardial用）
    print("心筋から10mm膨張中...")
    roi_10mm = myo.copy()
    for i in range(dilation_10mm):
        roi_10mm = binary_dilation(roi_10mm, struct_elem)
        if verbose and i % 5 == 0:
            print(f"  膨張 {i+1}/{dilation_10mm}")
    
    # EATとPATの計算
    # EAT: 心筋から5mm以内の脂肪（心筋自体は除外）
    eat_mask = fat_mask & roi_5mm & ~myo
    
    # PAT: 心筋から10mm以内の脂肪（心筋自体は除外）
    pat_mask = fat_mask & roi_10mm & ~myo
    
    # 体積計算
    eat_volume = eat_mask.sum() * voxel_volume
    pat_volume = pat_mask.sum() * voxel_volume
    
    # マスクの保存
    def save_mask(mask, filename):
        path = os.path.join(output_dir, filename)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), ct.affine, ct.header), path)
        return path
    
    eat_path = save_mask(eat_mask, "epicardial_adipose_5mm.nii.gz")
    pat_path = save_mask(pat_mask, "pericardial_adipose_10mm.nii.gz")
    
    # 結果の表示
    print("\n========== 結果 ==========")
    print(f"EAT (Epicardial Adipose Tissue):")
    print(f"  体積: {eat_volume:.2f} mL")
    print(f"  ボクセル数: {eat_mask.sum()}")
    print(f"  保存先: {eat_path}")
    print(f"\nPAT (Pericardial Adipose Tissue):")
    print(f"  体積: {pat_volume:.2f} mL")
    print(f"  ボクセル数: {pat_mask.sum()}")
    print(f"  保存先: {pat_path}")
    print("==========================")
    
    # 統計情報をテキストファイルに保存
    stats_path = os.path.join(output_dir, "eat_pat_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("EAT/PAT Statistics\n")
        f.write("==================\n\n")
        f.write(f"Input CT: {ct_path}\n")
        f.write(f"Myocardium mask: {myocardium_path}\n")
        f.write(f"Voxel spacing: {spacing} mm\n")
        f.write(f"Voxel volume: {voxel_volume:.6f} mL\n\n")
        f.write(f"EAT (Epicardial Adipose Tissue - 5mm):\n")
        f.write(f"  Volume: {eat_volume:.2f} mL\n")
        f.write(f"  Voxel count: {eat_mask.sum()}\n\n")
        f.write(f"PAT (Pericardial Adipose Tissue - 10mm):\n")
        f.write(f"  Volume: {pat_volume:.2f} mL\n")
        f.write(f"  Voxel count: {pat_mask.sum()}\n")
    
    print(f"\n統計情報を保存: {stats_path}")
    
    return eat_volume, pat_volume


def main():
    parser = argparse.ArgumentParser(description='EAT/PAT抽出スクリプト')
    parser.add_argument('--ct', required=True, help='CTデータのパス (NIfTI)')
    parser.add_argument('--myocardium', help='心筋マスクのパス (NIfTI)')
    parser.add_argument('--output', default='outputs/eat_pat', help='出力ディレクトリ')
    parser.add_argument('--verbose', action='store_true', help='詳細出力')
    
    args = parser.parse_args()
    
    # デフォルトの心筋マスクパスを設定
    if not args.myocardium:
        # TotalSegmentatorの出力から推定
        ct_dir = os.path.dirname(args.ct)
        possible_paths = [
            os.path.join(ct_dir, "../outputs/step3_output/total/heart_myocardium.nii.gz"),
            os.path.join(ct_dir, "../outputs/debug_test/total/heart_myocardium.nii.gz"),
            os.path.join(ct_dir, "heart_myocardium.nii.gz"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.myocardium = path
                print(f"心筋マスクを自動検出: {path}")
                break
    
    if not args.myocardium:
        print("警告: 心筋マスクが指定されていません。簡易推定を使用します。")
        args.myocardium = "dummy"  # ダミーパス
    
    # 実行
    extract_eat_pat(args.ct, args.myocardium, args.output, args.verbose)


if __name__ == "__main__":
    main()