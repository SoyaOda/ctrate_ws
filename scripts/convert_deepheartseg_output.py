#!/usr/bin/env python3
"""
Convert DeepHeartSeg output from NRRD to NIfTI format
"""
import os
import sys
import SimpleITK as sitk
import numpy as np

def convert_nrrd_to_nifti(nrrd_path, output_dir):
    """Convert NRRD file to NIfTI format"""
    print(f"Converting {nrrd_path}")
    
    # Read NRRD file
    img = sitk.ReadImage(nrrd_path)
    
    # Get info
    print(f"  Size: {img.GetSize()}")
    print(f"  Spacing: {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")
    
    # Get array for analysis
    arr = sitk.GetArrayFromImage(img)
    unique_labels = np.unique(arr)
    print(f"  Unique labels: {unique_labels}")
    
    # Save as NIfTI
    basename = os.path.basename(nrrd_path).replace('.nrrd', '')
    output_path = os.path.join(output_dir, f"{basename}.nii.gz")
    sitk.WriteImage(img, output_path)
    print(f"  Saved to: {output_path}")
    
    return output_path

def main():
    # DeepHeartSeg output directory
    dhs_output = "/home/soya/ctrate_ws/tools/DeepHeartSeg/data/test_data/step1_heartloc/model_output_nrrd"
    output_dir = "/home/soya/ctrate_ws/outputs/deepheartseg_test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting DeepHeartSeg outputs to NIfTI")
    print("="*50)
    
    # Convert all NRRD files
    for fname in os.listdir(dhs_output):
        if fname.endswith('.nrrd'):
            nrrd_path = os.path.join(dhs_output, fname)
            convert_nrrd_to_nifti(nrrd_path, output_dir)
    
    print("\nConversion completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()