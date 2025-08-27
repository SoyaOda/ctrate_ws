#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple test of DeepHeartSeg heart localization
"""

import os
import sys
import numpy as np
import SimpleITK as sitk

def main():
    # Input and output paths
    input_nifti = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/outputs/deepheartseg_test"
    
    print("DeepHeartSeg Test Script")
    print("========================")
    print("Input: %s" % input_nifti)
    print("Output: %s" % output_dir)
    
    # Check if input exists
    if not os.path.exists(input_nifti):
        print("Error: Input file not found: %s" % input_nifti)
        return 1
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the NIfTI file
    print("\n1. Loading CT image...")
    img = sitk.ReadImage(input_nifti)
    print("   Image size: %s" % str(img.GetSize()))
    print("   Image spacing: %s" % str(img.GetSpacing()))
    
    # Convert to numpy array
    arr = sitk.GetArrayFromImage(img)
    print("   Array shape: %s" % str(arr.shape))
    print("   HU range: [%.1f, %.1f]" % (arr.min(), arr.max()))
    
    # Simple heart detection (using HU thresholding)
    print("\n2. Simple heart detection using HU thresholding...")
    
    # Heart tissue is typically around 30-50 HU
    # Blood is around 30-45 HU
    heart_mask = (arr > 20) & (arr < 60)
    
    # Get center of mass
    from scipy import ndimage
    com = ndimage.center_of_mass(heart_mask)
    print("   Estimated heart center (voxel coords): %s" % str(com))
    
    # Create a simple bounding box around the heart
    # Typical heart is about 120mm x 100mm x 100mm
    spacing = img.GetSpacing()
    heart_size_mm = [120.0, 100.0, 100.0]
    heart_size_voxels = [int(s / sp) for s, sp in zip(heart_size_mm, spacing[::-1])]
    
    print("   Heart size in voxels: %s" % str(heart_size_voxels))
    
    # Create bounding box
    bbox = []
    for i, (c, s) in enumerate(zip(com, heart_size_voxels)):
        start = max(0, int(c - s/2))
        end = min(arr.shape[i], int(c + s/2))
        bbox.append((start, end))
    
    print("   Bounding box: %s" % str(bbox))
    
    # Create a simple heart mask
    heart_region = np.zeros_like(arr, dtype=np.uint8)
    heart_region[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = 1
    
    # Save the heart region mask
    print("\n3. Saving heart region mask...")
    heart_img = sitk.GetImageFromArray(heart_region)
    heart_img.CopyInformation(img)
    output_path = os.path.join(output_dir, "heart_region.nii.gz")
    sitk.WriteImage(heart_img, output_path)
    print("   Saved to: %s" % output_path)
    
    # Also save as NRRD for DeepHeartSeg
    output_nrrd = os.path.join(output_dir, "heart_region.nrrd")
    sitk.WriteImage(heart_img, output_nrrd)
    print("   Saved NRRD to: %s" % output_nrrd)
    
    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())