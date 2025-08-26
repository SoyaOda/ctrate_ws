#!/usr/bin/env python3
"""
DICOM変換後のファイル検証とHARTA出力の検証
"""
import pydicom
import numpy as np
import glob
import os
from PIL import Image
import sys

def check_dicom_series(dicom_dir):
    """DICOMシリーズの情報を確認"""
    print(f"\nDICOMディレクトリ: {dicom_dir}")
    print("="*60)
    
    dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
    if not dcm_files:
        print("DICOMファイルが見つかりません")
        return
    
    print(f"DICOMファイル数: {len(dcm_files)}")
    
    # 最初、中間、最後のファイルをチェック
    for idx in [0, len(dcm_files)//2, -1]:
        dcm = pydicom.dcmread(dcm_files[idx])
        print(f"\nファイル {idx}: {os.path.basename(dcm_files[idx])}")
        print(f"  InstanceNumber: {dcm.InstanceNumber if hasattr(dcm, 'InstanceNumber') else 'N/A'}")
        print(f"  ImagePositionPatient: {dcm.ImagePositionPatient if hasattr(dcm, 'ImagePositionPatient') else 'N/A'}")
        print(f"  ImageOrientationPatient: {dcm.ImageOrientationPatient if hasattr(dcm, 'ImageOrientationPatient') else 'N/A'}")
        print(f"  PixelSpacing: {dcm.PixelSpacing if hasattr(dcm, 'PixelSpacing') else 'N/A'}")
        print(f"  SliceThickness: {dcm.SliceThickness if hasattr(dcm, 'SliceThickness') else 'N/A'}")
        print(f"  Rows x Columns: {dcm.Rows} x {dcm.Columns}")
        
        # ピクセルデータの確認
        pixel_array = dcm.pixel_array
        print(f"  ピクセル値範囲: [{np.min(pixel_array)}, {np.max(pixel_array)}]")

def check_harta_output(harta_dir, patient_id="CTRATE001"):
    """HARTA出力の構造を確認"""
    print(f"\nHARTA出力ディレクトリ: {harta_dir}")
    print("="*60)
    
    # 各サブディレクトリを確認
    subdirs = ['slices', 'fat', 'contours', 'combined']
    for subdir in subdirs:
        subdir_path = os.path.join(harta_dir, subdir)
        if os.path.exists(subdir_path):
            files = sorted(glob.glob(os.path.join(subdir_path, "*.png")))
            print(f"\n{subdir}/: {len(files)} files")
            
            if files and subdir == 'fat':
                # fatマスクのサンプルをチェック
                for idx in [0, len(files)//2, -1]:
                    if idx < len(files):
                        img = Image.open(files[idx])
                        arr = np.array(img)
                        print(f"  {os.path.basename(files[idx])}: shape={arr.shape}, dtype={arr.dtype}, range=[{np.min(arr)}, {np.max(arr)}], nonzero={np.count_nonzero(arr)}")

def analyze_conversion_pipeline():
    """変換パイプライン全体の分析"""
    test_case = "test_valid_2_a_1"
    base_dir = "/home/soya/ctrate_ws"
    
    print("\n" + "="*80)
    print("HARTA変換パイプライン分析")
    print("="*80)
    
    # 1. DICOM変換結果の確認
    dicom_dir = f"{base_dir}/outputs/test_harta/dicom/{test_case}"
    check_dicom_series(dicom_dir)
    
    # 2. HARTA出力の確認
    harta_dir = f"{base_dir}/outputs/test_harta/harta_output/{test_case}"
    check_harta_output(harta_dir)
    
    print("\n" + "="*80)
    print("問題の可能性:")
    print("="*80)
    print("1. DICOMのImageOrientationPatientが固定値 [1,0,0,0,1,0] になっている")
    print("2. ImagePositionPatientのZ座標計算が不正確")
    print("3. HARTAの2Dマスクから3D NIfTIへの変換時の座標系不整合")
    print("4. アフィン変換行列の再構築が不適切")

if __name__ == "__main__":
    analyze_conversion_pipeline()