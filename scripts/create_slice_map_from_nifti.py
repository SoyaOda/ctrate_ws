#!/usr/bin/env python3
"""
NIfTIファイルから幾何情報を抽出してslice_map.jsonを作成
"""
import json
import nibabel as nib
import numpy as np
import argparse
from pathlib import Path

def create_slice_map_from_nifti(nifti_path, output_json):
    """
    Create slice_map.json from NIfTI file geometry
    """
    print(f"[INFO] Loading NIfTI: {nifti_path}")
    
    # Load NIfTI
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    
    # Get geometry
    affine = nii.affine
    shape = data.shape
    spacing = nii.header.get_zooms()[:3]
    
    # Extract direction and origin from affine
    direction = affine[:3, :3].copy()
    origin = affine[:3, 3].copy()
    
    # Normalize direction vectors
    for i in range(3):
        norm = np.linalg.norm(direction[:, i])
        if norm > 0:
            direction[:, i] /= norm
    
    # Create slice map structure
    slice_map = {
        "geometry": {
            "size": [int(shape[0]), int(shape[1]), int(shape[2])],
            "spacing": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
            "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
            "direction": direction.flatten().tolist()
        },
        "meta": []
    }
    
    # Add metadata for each slice
    for z in range(shape[2]):
        slice_meta = {
            "slice_idx": z,
            "slice_location": float(origin[2] + z * spacing[2])
        }
        slice_map["meta"].append(slice_meta)
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(slice_map, f, indent=2)
    
    print(f"[INFO] Slice map saved to: {output_json}")
    print(f"[INFO] Geometry - Size: {slice_map['geometry']['size']}")
    print(f"[INFO] Geometry - Spacing: {slice_map['geometry']['spacing']}")
    print(f"[INFO] Geometry - Origin: {slice_map['geometry']['origin']}")
    print(f"[INFO] Number of slices: {len(slice_map['meta'])}")
    
    return slice_map

def main():
    parser = argparse.ArgumentParser(
        description="Create slice_map.json from NIfTI file"
    )
    parser.add_argument("--nifti", required=True,
                       help="Path to NIfTI file")
    parser.add_argument("--output", required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    create_slice_map_from_nifti(args.nifti, args.output)

if __name__ == "__main__":
    main()