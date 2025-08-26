#!/usr/bin/env bash
set -euo pipefail

# 使い方: ./nifti_to_dicom.sh /path/to/case.nii.gz /out/dicom/CASE_ID

if [ $# -ne 2 ]; then
    echo "Usage: $0 <nifti_file> <output_dicom_dir>"
    echo "Example: $0 /path/to/case.nii.gz /output/dicom/case_001"
    exit 1
fi

NII="$1"
OUT="$2"

# 入力ファイル存在確認
if [ ! -f "$NII" ]; then
    echo "[ERROR] Input file not found: $NII"
    exit 1
fi

# 出力ディレクトリ作成
mkdir -p "$OUT"

# Plastimatchで変換（DockerでTotalSegmentatorイメージ内のツールを利用）
echo "[INFO] Converting NIfTI to DICOM: $NII -> $OUT"
docker run --rm \
    -v "$HOME:/home/soya" \
    wasserth/totalsegmentator:2.10.0 \
    bash -c "
        # PlastimatchのようなNIfTI→DICOM変換をPythonで実装
        python3 -c \"
import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pathlib import Path
import datetime
import os

# 入力読み込み
nii_path = '$NII'
out_dir = '$OUT'
Path(out_dir).mkdir(parents=True, exist_ok=True)

# NIfTI読み込み
nii = nib.load(nii_path)
data = nii.get_fdata()
affine = nii.affine
spacing = nii.header.get_zooms()[:3]

# データ型を調整（CTスケール）
data = np.clip(data, -1024, 3071)  # CT値の範囲
data = data.astype(np.int16)

# DICOM基本情報
study_uid = pydicom.uid.generate_uid()
series_uid = pydicom.uid.generate_uid()
frame_of_ref_uid = pydicom.uid.generate_uid()

# スライスごとにDICOM保存
for i in range(data.shape[2]):
    # ファイル名
    filename = os.path.join(out_dir, f'slice_{i:04d}.dcm')
    
    # DICOM Dataset作成
    ds = FileDataset(filename, {}, file_meta=None, preamble=b'\\0' * 128)
    
    # File Meta Information
    ds.file_meta = pydicom.Dataset()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    
    # Patient Module
    ds.PatientName = 'Anonymous'
    ds.PatientID = 'CTRATE001'
    ds.PatientBirthDate = ''
    ds.PatientSex = ''
    
    # General Study Module
    ds.StudyInstanceUID = study_uid
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    ds.StudyID = '1'
    ds.AccessionNumber = ''
    
    # General Series Module
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = 1
    ds.Modality = 'CT'
    
    # Frame of Reference Module
    ds.FrameOfReferenceUID = frame_of_ref_uid
    ds.PositionReferenceIndicator = ''
    
    # General Image Module
    ds.InstanceNumber = i + 1
    ds.ImagePositionPatient = [float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3] + i * spacing[2])]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # 軸方向
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = data.shape[0]
    ds.Columns = data.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [row, col]
    ds.SliceThickness = float(spacing[2])
    
    # Image Pixel Module
    ds.PixelData = data[:, :, i].tobytes()
    
    # SOPCommon Module
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    
    # CT Image Module
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    
    # 保存
    ds.save_as(filename)

print(f'[OK] Converted {data.shape[2]} slices to DICOM')
\"
    "

if [ $? -eq 0 ]; then
    echo "[OK] Conversion completed: $NII -> $OUT"
else
    echo "[ERROR] Conversion failed"
    exit 1
fi