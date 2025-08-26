#!/usr/bin/env python3
"""
HARTA出力の詳細分析ツール
セグメンテーション結果の問題を特定
"""
import numpy as np
from PIL import Image
import pydicom
import nibabel as nib
import os
import sys
from pathlib import Path

def analyze_harta_png_output(harta_dir, sample_indices=[0, 50, 100, 150, 200]):
    """HARTA PNG出力の詳細分析"""
    print("\n" + "="*80)
    print("HARTA PNG出力分析")
    print("="*80)
    
    # 各サブディレクトリを分析
    subdirs = ['slices', 'fat', 'contours', 'combined']
    
    for subdir in subdirs:
        subdir_path = Path(harta_dir) / subdir
        if not subdir_path.exists():
            print(f"\n{subdir}/: ディレクトリが存在しません")
            continue
            
        print(f"\n{subdir}/ の分析:")
        print("-"*40)
        
        png_files = sorted(list(subdir_path.glob("*.png")))
        print(f"ファイル数: {len(png_files)}")
        
        if png_files:
            # サンプルファイルを分析
            for idx in sample_indices:
                if idx < len(png_files):
                    png_path = png_files[idx]
                    img = Image.open(png_path)
                    arr = np.array(img)
                    
                    print(f"\n  {png_path.name}:")
                    print(f"    モード: {img.mode}")
                    print(f"    サイズ: {img.size}")
                    print(f"    配列形状: {arr.shape}")
                    print(f"    データ型: {arr.dtype}")
                    print(f"    値の範囲: [{np.min(arr)}, {np.max(arr)}]")
                    
                    if len(arr.shape) == 3:  # カラー/RGBA画像
                        for ch in range(arr.shape[2]):
                            ch_data = arr[:, :, ch]
                            unique_vals = np.unique(ch_data)
                            print(f"    チャンネル{ch}: ユニーク値数={len(unique_vals)}, "
                                  f"非ゼロ={np.count_nonzero(ch_data)}, "
                                  f"範囲=[{np.min(ch_data)}, {np.max(ch_data)}]")
                    else:  # グレースケール
                        unique_vals = np.unique(arr)
                        print(f"    ユニーク値: {unique_vals[:10] if len(unique_vals) > 10 else unique_vals}")
                        print(f"    非ゼロピクセル: {np.count_nonzero(arr)} / {arr.size} "
                              f"({100*np.count_nonzero(arr)/arr.size:.2f}%)")
                    
                    # fatディレクトリの場合、マスクの統計を詳しく分析
                    if subdir == 'fat':
                        if len(arr.shape) == 3 and arr.shape[2] >= 3:
                            # RGBチャンネルから2値マスクを作成
                            mask = (arr[:, :, 0] > 0) | (arr[:, :, 1] > 0) | (arr[:, :, 2] > 0)
                        else:
                            mask = arr > 0
                        
                        print(f"    マスク統計:")
                        print(f"      EATピクセル数: {np.sum(mask)}")
                        print(f"      占有率: {100*np.sum(mask)/(512*512):.2f}%")

def compare_with_original_dicom(dicom_dir, harta_fat_dir, sample_index=100):
    """元のDICOMとHARTAマスクの比較"""
    print("\n" + "="*80)
    print("DICOM vs HARTAマスク比較")
    print("="*80)
    
    # DICOMファイル読み込み
    dicom_files = sorted(list(Path(dicom_dir).glob("*.dcm")))
    if sample_index >= len(dicom_files):
        sample_index = len(dicom_files) // 2
    
    dcm = pydicom.dcmread(str(dicom_files[sample_index]))
    dicom_pixels = dcm.pixel_array
    
    # 対応するHARTAマスク読み込み
    fat_files = sorted(list(Path(harta_fat_dir).glob("*.png")))
    if sample_index < len(fat_files):
        fat_img = Image.open(fat_files[sample_index])
        fat_arr = np.array(fat_img)
        
        print(f"\nスライス {sample_index} の比較:")
        print(f"  DICOM:")
        print(f"    形状: {dicom_pixels.shape}")
        print(f"    HU値範囲: [{np.min(dicom_pixels)}, {np.max(dicom_pixels)}]")
        print(f"    脂肪HU範囲(-190~-30)のピクセル数: {np.sum((dicom_pixels >= -190) & (dicom_pixels <= -30))}")
        
        print(f"  HARTAマスク:")
        print(f"    形状: {fat_arr.shape}")
        if len(fat_arr.shape) == 3:
            mask = (fat_arr[:, :, 0] > 0) | (fat_arr[:, :, 1] > 0) | (fat_arr[:, :, 2] > 0)
        else:
            mask = fat_arr > 0
        print(f"    マスクピクセル数: {np.sum(mask)}")
        print(f"    占有率: {100*np.sum(mask)/(512*512):.2f}%")
        
        # マスクがCT画像そのものになっていないか確認
        if len(fat_arr.shape) == 3 and fat_arr.shape[2] >= 3:
            # RGBの平均値を計算
            rgb_mean = np.mean(fat_arr[:, :, :3], axis=2)
            correlation = np.corrcoef(dicom_pixels.flatten(), rgb_mean.flatten())[0, 1]
            print(f"    DICOMとの相関係数: {correlation:.4f}")
            if abs(correlation) > 0.5:
                print("    ⚠️ 警告: マスクがCT画像と高い相関を示しています！")

def analyze_final_nifti(nifti_path, original_nifti_path):
    """最終的なNIfTIセグメンテーションの分析"""
    print("\n" + "="*80)
    print("最終NIfTIセグメンテーション分析")
    print("="*80)
    
    # セグメンテーションNIfTI読み込み
    seg_nii = nib.load(nifti_path)
    seg_data = seg_nii.get_fdata()
    
    # オリジナルNIfTI読み込み  
    orig_nii = nib.load(original_nifti_path)
    orig_data = orig_nii.get_fdata()
    
    print(f"\nセグメンテーション統計:")
    print(f"  形状: {seg_data.shape}")
    print(f"  ユニーク値: {np.unique(seg_data)}")
    print(f"  非ゼロボクセル: {np.count_nonzero(seg_data)} / {seg_data.size}")
    print(f"  占有率: {100*np.count_nonzero(seg_data)/seg_data.size:.2f}%")
    
    # ボリューム計算
    spacing = seg_nii.header.get_zooms()[:3]
    voxel_volume_ml = np.prod(spacing) / 1000.0
    volume_ml = np.sum(seg_data > 0) * voxel_volume_ml
    print(f"  総ボリューム: {volume_ml:.2f} ml")
    
    # オリジナルとの相関チェック
    correlation = np.corrcoef(orig_data.flatten(), seg_data.flatten())[0, 1]
    print(f"\nオリジナルCTとの相関係数: {correlation:.4f}")
    if abs(correlation) > 0.3:
        print("⚠️ 警告: セグメンテーションがオリジナルCTと異常に高い相関を示しています！")
    
    # スライスごとの統計
    print(f"\nスライスごとの統計（サンプル）:")
    for z in [0, 50, 100, 150, 200]:
        if z < seg_data.shape[2]:
            slice_data = seg_data[:, :, z]
            slice_orig = orig_data[:, :, z]
            nonzero = np.count_nonzero(slice_data)
            if nonzero > 0:
                slice_corr = np.corrcoef(slice_orig.flatten(), slice_data.flatten())[0, 1]
                print(f"  スライス {z}: 非ゼロ={nonzero}, 占有率={100*nonzero/(512*512):.1f}%, "
                      f"相関={slice_corr:.4f}")

def main():
    # パスの設定
    base_dir = Path("/home/soya/ctrate_ws")
    test_dir = base_dir / "outputs/test_harta_fixed"
    
    harta_dir = test_dir / "harta_output/test_valid_2_a_1_fixed"
    dicom_dir = test_dir / "dicom/test_valid_2_a_1_fixed"
    nifti_path = test_dir / "nifti/test_valid_2_a_1_fixed/eat_harta_fixed.nii.gz"
    original_nifti = base_dir / "data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    
    # 1. HARTA PNG出力の分析
    if harta_dir.exists():
        analyze_harta_png_output(harta_dir)
    
    # 2. DICOMとの比較
    if dicom_dir.exists() and (harta_dir / "fat").exists():
        compare_with_original_dicom(dicom_dir, harta_dir / "fat")
    
    # 3. 最終NIfTIの分析
    if nifti_path.exists() and original_nifti.exists():
        analyze_final_nifti(nifti_path, original_nifti)
    
    print("\n" + "="*80)
    print("分析完了")
    print("="*80)

if __name__ == "__main__":
    main()