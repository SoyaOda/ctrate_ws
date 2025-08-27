#!/usr/bin/env python3
"""
Extract EAT using DeepHeartSeg heart mask and morphological operations
"""
import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from pathlib import Path

def create_pericardium_estimate(heart_mask, dilation_mm=6.0, spacing=(1.0, 1.0, 1.0)):
    """
    Create pericardium estimation by dilating heart mask
    """
    # Convert mm to voxels for dilation
    dilation_voxels = [int(dilation_mm / s) for s in spacing]
    
    # Create structuring element
    struct_elem = ndimage.generate_binary_structure(3, 2)
    
    # Dilate heart mask to approximate pericardium
    dilated = ndimage.binary_dilation(heart_mask, structure=struct_elem, iterations=max(dilation_voxels))
    
    # Create shell (pericardium region)
    shell = dilated & ~heart_mask
    
    return dilated, shell

def extract_eat(ct_path, heart_mask_path, output_dir):
    """
    Extract EAT using heart mask from DeepHeartSeg
    """
    print(f"Extracting EAT from {ct_path}")
    
    # Read CT image
    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    spacing = ct_img.GetSpacing()
    
    print(f"CT shape: {ct_arr.shape}")
    print(f"CT spacing: {spacing}")
    
    # Read heart mask
    heart_img = sitk.ReadImage(heart_mask_path)
    
    # Resample heart mask to match CT if needed
    if heart_img.GetSize() != ct_img.GetSize():
        print("Resampling heart mask to match CT...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        heart_img = resampler.Execute(heart_img)
    
    heart_arr = sitk.GetArrayFromImage(heart_img)
    
    # Create pericardium estimate
    print("Creating pericardium estimate...")
    peri_region, peri_shell = create_pericardium_estimate(heart_arr > 0, dilation_mm=8.0, spacing=spacing[::-1])
    
    # Define HU range for fat
    fat_min_hu = -190
    fat_max_hu = -30
    
    # Create fat mask
    fat_mask = (ct_arr >= fat_min_hu) & (ct_arr <= fat_max_hu)
    
    # EAT = fat within pericardium but outside heart
    eat_mask = fat_mask & peri_region & ~(heart_arr > 0)
    
    # Remove small components
    eat_mask = ndimage.binary_opening(eat_mask, iterations=1)
    
    # Calculate volume
    voxel_volume = np.prod(spacing) / 1000.0  # Convert to ml
    eat_volume = np.sum(eat_mask) * voxel_volume
    
    print(f"EAT volume: {eat_volume:.2f} ml")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save EAT mask
    eat_img = sitk.GetImageFromArray(eat_mask.astype(np.uint8))
    eat_img.CopyInformation(ct_img)
    eat_path = os.path.join(output_dir, "eat_mask_deepheartseg.nii.gz")
    sitk.WriteImage(eat_img, eat_path)
    print(f"EAT mask saved to: {eat_path}")
    
    # Save pericardium region for debugging
    peri_img = sitk.GetImageFromArray(peri_region.astype(np.uint8))
    peri_img.CopyInformation(ct_img)
    peri_path = os.path.join(output_dir, "pericardium_region.nii.gz")
    sitk.WriteImage(peri_img, peri_path)
    print(f"Pericardium region saved to: {peri_path}")
    
    # Save volume info
    info_path = os.path.join(output_dir, "eat_volume_deepheartseg.txt")
    with open(info_path, "w") as f:
        f.write(f"EAT Volume: {eat_volume:.2f} ml\n")
        f.write(f"Method: DeepHeartSeg heart mask with morphological dilation\n")
        f.write(f"Dilation: 8mm\n")
        f.write(f"HU range: [{fat_min_hu}, {fat_max_hu}]\n")
    
    return eat_volume

def main():
    # Input paths
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz"
    heart_mask_path = "/home/soya/ctrate_ws/outputs/deepheartseg_test/step2_heartseg.nii.gz"
    output_dir = "/home/soya/ctrate_ws/outputs/eat_deepheartseg"
    
    # Extract EAT
    eat_volume = extract_eat(ct_path, heart_mask_path, output_dir)
    
    print("\n" + "="*50)
    print("EAT extraction complete!")
    print(f"Total EAT volume: {eat_volume:.2f} ml")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()