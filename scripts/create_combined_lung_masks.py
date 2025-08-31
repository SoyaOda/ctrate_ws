#!/usr/bin/env python3
"""
肺の葉（lobe）マスクを結合してlung_left.nii.gzとlung_right.nii.gzを作成
"""

import os
import sys
import nibabel as nib
import numpy as np
from pathlib import Path

def combine_lung_masks(segmentation_dir):
    """
    肺の葉マスクを結合して左右の肺マスクを作成
    """
    segmentation_dir = Path(segmentation_dir)
    
    # 左肺の葉を結合
    left_lobes = [
        "lung_upper_lobe_left.nii.gz",
        "lung_lower_lobe_left.nii.gz"
    ]
    
    # 右肺の葉を結合
    right_lobes = [
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz"
    ]
    
    # 左肺を結合
    lung_left = None
    affine = None
    header = None
    for lobe in left_lobes:
        lobe_path = segmentation_dir / lobe
        if lobe_path.exists():
            img = nib.load(str(lobe_path))
            data = img.get_fdata() > 0
            if lung_left is None:
                lung_left = data
                affine = img.affine
                header = img.header
            else:
                lung_left |= data
            print(f"  Added {lobe}")
    
    if lung_left is not None:
        output_path = segmentation_dir / "lung_left.nii.gz"
        img = nib.Nifti1Image(lung_left.astype(np.uint8), affine, header)
        nib.save(img, str(output_path))
        print(f"  Created: {output_path}")
    else:
        print("  Warning: No left lung lobes found")
    
    # 右肺を結合
    lung_right = None
    for lobe in right_lobes:
        lobe_path = segmentation_dir / lobe
        if lobe_path.exists():
            img = nib.load(str(lobe_path))
            data = img.get_fdata() > 0
            if lung_right is None:
                lung_right = data
                affine = img.affine
                header = img.header
            else:
                lung_right |= data
            print(f"  Added {lobe}")
    
    if lung_right is not None:
        output_path = segmentation_dir / "lung_right.nii.gz"
        img = nib.Nifti1Image(lung_right.astype(np.uint8), affine, header)
        nib.save(img, str(output_path))
        print(f"  Created: {output_path}")
    else:
        print("  Warning: No right lung lobes found")

def main():
    # 処理する症例
    cases = [
        "valid_1_valid_1_a_1",
        "valid_1_valid_1_a_2",
        "valid_2_valid_2_a_1"
    ]
    
    for case in cases:
        segmentation_dir = f"/home/soya/ctrate_ws/outputs/batch_results/{case}/segmentation_merged"
        print(f"\nProcessing {case}...")
        
        if not Path(segmentation_dir).exists():
            print(f"  Directory not found: {segmentation_dir}")
            continue
        
        # lung_left.nii.gzとlung_right.nii.gzが既に存在するか確認
        left_exists = (Path(segmentation_dir) / "lung_left.nii.gz").exists()
        right_exists = (Path(segmentation_dir) / "lung_right.nii.gz").exists()
        
        if left_exists and right_exists:
            print(f"  lung_left.nii.gz and lung_right.nii.gz already exist")
        else:
            combine_lung_masks(segmentation_dir)

if __name__ == "__main__":
    main()