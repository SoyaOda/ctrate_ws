#!/usr/bin/env python3
"""
Analyze DeepHeartSeg Step2 output and convert to NIfTI
"""
import os
import SimpleITK as sitk
import numpy as np

def main():
    # DeepHeartSeg Step2 output
    step2_output = "/home/soya/ctrate_ws/tools/DeepHeartSeg/data/test_data/step2_heartseg/model_output_nrrd/patient001_pred.nrrd"
    output_dir = "/home/soya/ctrate_ws/outputs/deepheartseg_test"
    
    print("Analyzing DeepHeartSeg Step2 Output")
    print("="*50)
    
    # Read the segmentation
    img = sitk.ReadImage(step2_output)
    arr = sitk.GetArrayFromImage(img)
    
    print(f"Image shape: {arr.shape}")
    print(f"Image spacing: {img.GetSpacing()}")
    print(f"Image origin: {img.GetOrigin()}")
    
    # Analyze labels
    unique_labels = np.unique(arr)
    print(f"\nUnique labels: {unique_labels}")
    
    for label in unique_labels:
        if label == 0:
            continue
        count = np.sum(arr == label)
        print(f"  Label {label}: {count} voxels")
    
    # Save as NIfTI for easier visualization
    output_path = os.path.join(output_dir, "step2_heartseg.nii.gz")
    sitk.WriteImage(img, output_path)
    print(f"\nSaved to: {output_path}")
    
    # Create individual masks for each label
    print("\nCreating individual label masks:")
    for label in unique_labels:
        if label == 0:
            continue
        
        # Create binary mask for this label
        mask = (arr == label).astype(np.uint8)
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(img)
        
        # Save
        label_path = os.path.join(output_dir, f"step2_label_{int(label)}.nii.gz")
        sitk.WriteImage(mask_img, label_path)
        print(f"  Label {label} mask saved to: {label_path}")
    
    print("\nStep2 analysis complete!")
    
    # Label mapping (based on DeepHeartSeg documentation)
    print("\nExpected label mapping:")
    print("  0: Background")
    print("  1: Left Atrium")
    print("  2: Left Ventricle Blood Cavity") 
    print("  3: Left Ventricle Myocardium")
    print("  4: Right Atrium")
    print("  5: Right Ventricle Blood Cavity")
    print("  6: Right Ventricle Myocardium")
    print("  7: Ascending Aorta")
    print("  8: Pulmonary Artery")
    print("  9: Pericardium (if included)")

if __name__ == "__main__":
    main()