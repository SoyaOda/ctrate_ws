#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image
from pathlib import Path

def read_dicom_series(dicom_dir):
    """DICOMシリーズを読み込んで幾何情報を取得"""
    # DICOMファイルをInstanceNumber順にソート
    dcm_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dcm_files:
        raise RuntimeError(f"No DICOM files found in {dicom_dir}")
    
    # 最初のファイルで属性確認後、全体をソート
    first_dcm = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    
    # InstanceNumberでソート
    files = sorted(dcm_files, key=lambda x: pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber)
    refs = [pydicom.dcmread(fp, stop_before_pixels=True) for fp in files]
    
    # 画素配列のサイズを確認
    rr, cc = int(refs[0].Rows), int(refs[0].Columns)
    
    # スペーシング（mm）
    ps = [float(x) for x in refs[0].PixelSpacing]  # [row, col] mm
    try:
        st = float(refs[0].SliceThickness)
    except Exception:
        # Slice間隔から推定
        zs = [float(ds.ImagePositionPatient[2]) for ds in refs]
        zs_sorted = sorted(zs)
        diffs = [abs(zs_sorted[i+1]-zs_sorted[i]) for i in range(len(zs_sorted)-1)]
        st = np.median(diffs) if diffs else 1.0
    
    spacing = (ps[1], ps[0], st)  # (x, y, z)
    
    # Image Position Patientを取得
    origin = [float(x) for x in refs[0].ImagePositionPatient]
    
    # Direction cosines
    orientation = [float(x) for x in refs[0].ImageOrientationPatient]
    
    return refs, (rr, cc), spacing, origin, orientation, files

def load_harta_slice_masks(mask_dir, num_slices, patient_id):
    """HARTAの出力マスクを読み込む"""
    # HARTAの出力パターン: fat/{patient_id}_{slice_num}_fat.png
    mask_files = []
    
    for i in range(num_slices):
        # まずfatディレクトリのマスクを探す
        mask_path = os.path.join(mask_dir, "fat", f"{patient_id}_{i}_fat.png")
        if os.path.exists(mask_path):
            mask_files.append((i, mask_path))
        else:
            # 見つからない場合は0で埋める
            mask_files.append((i, None))
    
    # 3Dボリューム作成
    first_valid = next((path for _, path in mask_files if path is not None), None)
    if first_valid is None:
        raise RuntimeError(f"No mask images found in {mask_dir}/fat/")
    
    # 最初の有効なマスクでサイズを確認
    sample_img = Image.open(first_valid).convert("L")
    shape = np.array(sample_img).shape
    
    # ボリューム作成
    vol = np.zeros((shape[0], shape[1], num_slices), dtype=np.uint8)
    
    for idx, mask_path in mask_files:
        if mask_path is not None:
            im = Image.open(mask_path).convert("L")
            arr = np.array(im, dtype=np.uint8)
            # バイナリマスクに変換（0 or 1）
            arr = (arr > 0).astype(np.uint8)
            vol[:, :, idx] = arr
    
    return vol

def save_as_nifti(mask_vol, spacing, origin, out_nifti):
    """3DマスクをNIfTIとして保存"""
    # SimpleITKでNIfTI保存
    img = sitk.GetImageFromArray(mask_vol.astype(np.uint8).transpose(2, 0, 1))  # Z,Y,X
    img.SetSpacing((spacing[0], spacing[1], spacing[2]))
    img.SetOrigin(origin)
    
    # ディレクトリ作成
    Path(os.path.dirname(out_nifti)).mkdir(parents=True, exist_ok=True)
    
    sitk.WriteImage(img, out_nifti)

def main():
    ap = argparse.ArgumentParser(description="Convert HARTA 2D masks to 3D NIfTI")
    ap.add_argument("--dicom_dir", required=True, help="Original DICOM directory")
    ap.add_argument("--harta_output_dir", required=True, help="HARTA output directory")
    ap.add_argument("--out_nifti", required=True, help="Output NIfTI file path")
    args = ap.parse_args()
    
    try:
        # DICOMシリーズ読み込み
        print(f"[INFO] Reading DICOM series from: {args.dicom_dir}")
        refs, shape, spacing, origin, orientation, _ = read_dicom_series(args.dicom_dir)
        num_slices = len(refs)
        patient_id = refs[0].PatientID if hasattr(refs[0], 'PatientID') else 'CTRATE001'
        
        print(f"[INFO] Found {num_slices} DICOM slices")
        print(f"[INFO] Image size: {shape}, Spacing: {spacing}")
        
        # HARTAマスク読み込み
        print(f"[INFO] Loading HARTA masks from: {args.harta_output_dir}")
        mask_vol = load_harta_slice_masks(args.harta_output_dir, num_slices, patient_id)
        
        # NIfTI保存
        print(f"[INFO] Saving NIfTI to: {args.out_nifti}")
        save_as_nifti(mask_vol, spacing, origin, args.out_nifti)
        
        # ボリューム計算
        voxel_volume_ml = np.prod(spacing) / 1000.0
        eat_volume_ml = np.sum(mask_vol) * voxel_volume_ml
        
        print(f"[OK] Saved NIfTI: {args.out_nifti}")
        print(f"[INFO] EAT Volume: {eat_volume_ml:.2f} ml")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()