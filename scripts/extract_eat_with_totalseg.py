#!/usr/bin/env python3
"""
TotalSegmentatorの心臓マスクとEDT（ユークリッド距離変換）を使用して
HARTAの脂肪セグメンテーションから正確なEAT（心外膜脂肪組織）を抽出
"""
import os
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
import argparse
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
import subprocess

def load_mask_from_png(png_path, rgb_threshold=127):
    """
    HARTA PNGからバイナリマスクを正しく抽出
    RGB値を使用（アルファチャンネルは常に255なので使わない）
    """
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im)  # (H, W, 4)
    rgb = arr[..., :3].astype(np.int16)
    # R=G=B=255がEAT、0が背景。閾値を超えたら1
    mask = (rgb > rgb_threshold).any(axis=-1).astype(np.uint8)
    return mask

def reconstruct_3d_from_harta(harta_png_dir, slice_map_json, patient_id="CTRATE001"):
    """
    HARTA PNGから3Dボリュームを再構成
    """
    print(f"[INFO] Reconstructing 3D from HARTA PNGs...")
    
    # Load geometry information
    with open(slice_map_json, 'r') as f:
        smap = json.load(f)
    
    geometry = smap["geometry"]
    size = tuple(geometry["size"])
    meta = smap["meta"]
    
    # Find PNG files from HARTA fat output
    hdir = Path(harta_png_dir) / "fat"
    
    # Collect masks in slice order
    masks_dict = {}
    for p in hdir.glob("*.png"):
        fname = p.stem
        parts = fname.split('_')
        if len(parts) >= 2:
            try:
                slice_idx = int(parts[1])
                if slice_idx < len(meta):
                    masks_dict[slice_idx] = load_mask_from_png(str(p))
            except ValueError:
                continue
    
    # Stack masks in z order
    stacks = []
    for z in range(size[2]):
        if z in masks_dict:
            stacks.append(masks_dict[z])
        else:
            # Empty slice if missing
            stacks.append(np.zeros((size[1], size[0]), dtype=np.uint8))
    
    # Create 3D volume (X, Y, Z)
    volume = np.stack(stacks, axis=2).astype(np.uint8)
    
    print(f"[INFO] Reconstructed volume shape: {volume.shape}")
    return volume, geometry

def run_totalsegmentator(ct_nifti_path, output_dir):
    """
    TotalSegmentatorを実行して心臓構造を取得
    """
    print(f"[INFO] Running TotalSegmentator for heart structures...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run TotalSegmentator with heart chambers task
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{Path(ct_nifti_path).parent}:/input",
        "-v", f"{output_dir}:/output",
        "wasserth/totalsegmentator:2.10.0",
        "TotalSegmentator",
        "-i", f"/input/{Path(ct_nifti_path).name}",
        "-o", "/output",
        "--task", "heartchambers_highres",
        "--device", "cpu",  # Use GPU if available
        "--fast",  # Speed optimization
        "--body_seg"  # Include body segmentation
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] TotalSegmentator failed: {result.stderr}")
        return False
    
    print(f"[INFO] TotalSegmentator completed successfully")
    return True

def create_heart_roi_with_shell(totalseg_dir, ct_nifti_path, shell_thickness_mm=6.0):
    """
    TotalSegmentatorの心臓マスクからEDTを使って心膜周囲の薄い殻を作成
    修正版：心臓全体（心筋+4腔）の外側のみにシェルを生成
    """
    print(f"[INFO] Creating heart ROI with {shell_thickness_mm}mm shell (outer only)...")
    
    # Load CT for spacing information
    ct_img = nib.load(ct_nifti_path)
    spacing = ct_img.header.get_zooms()[:3]  # (sx, sy, sz) in mm
    print(f"[INFO] CT spacing: {spacing} mm")
    
    # Load heart structures from TotalSegmentator
    heart_parts = []
    
    # Try to load heart components (heartchambers_highres task names)
    components = [
        "heart_myocardium",     # 心筋
        "heart_ventricle_left", # 左心室
        "heart_ventricle_right",# 右心室
        "heart_atrium_left",    # 左心房
        "heart_atrium_right"    # 右心房
    ]
    
    for comp in components:
        comp_path = Path(totalseg_dir) / f"{comp}.nii.gz"
        if comp_path.exists():
            print(f"[INFO] Loading {comp}...")
            mask = nib.load(str(comp_path)).get_fdata() > 0
            heart_parts.append(mask)
        else:
            print(f"[WARNING] {comp} not found")
    
    if not heart_parts:
        print(f"[ERROR] No heart components found!")
        return None, None
    
    # Combine all heart parts (心筋+4腔の和集合)
    heart = np.logical_or.reduce(heart_parts)
    print(f"[INFO] Combined heart mask shape: {heart.shape}")
    
    # Get myocardium specifically (if available)
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
    else:
        print(f"[WARNING] Myocardium not found, using heart mask")
        myo = heart
    
    # Create shell using Euclidean Distance Transform
    # 重要：心臓全体の外側のみを計算領域にする（血液腔を除外）
    outside = ~heart  # 修正：~myo → ~heart に変更
    dist_mm = distance_transform_edt(outside, sampling=spacing)
    
    # EAT shell: 0-6mm from heart surface (outer only)
    shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness_mm)
    
    # Constrain to dilated heart region for safety
    struct = generate_binary_structure(3, 2)
    shell_eat &= binary_dilation(heart, structure=struct, iterations=2)
    
    print(f"[INFO] Shell created with thickness {shell_thickness_mm}mm")
    print(f"[INFO] Shell voxels: {np.sum(shell_eat)}")
    
    return shell_eat, myo

def extract_eat(ct_nifti_path, harta_volume, shell_eat, myo, 
                 hu_min=-190, hu_max=-30):
    """
    最終的なEATを抽出: HARTA ∧ HU ∧ Shell ∧ ¬Myocardium
    """
    print(f"[INFO] Extracting EAT with HU range [{hu_min}, {hu_max}]...")
    
    # Load CT data
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()
    
    # HU thresholding for fat
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    print(f"[INFO] HU fat voxels: {np.sum(hu_fat)}")
    
    # HARTA fat
    harta_fat = harta_volume > 0
    print(f"[INFO] HARTA fat voxels: {np.sum(harta_fat)}")
    
    # Final EAT = HARTA ∧ HU ∧ Shell ∧ ¬Myocardium
    eat = harta_fat & hu_fat & shell_eat & (~myo)
    
    print(f"[INFO] Final EAT voxels: {np.sum(eat)}")
    
    # Calculate volume
    spacing = ct_img.header.get_zooms()[:3]
    voxel_volume_ml = np.prod(spacing) / 1000.0
    eat_volume_ml = np.sum(eat) * voxel_volume_ml
    
    print(f"[INFO] EAT Volume: {eat_volume_ml:.2f} ml")
    
    # Quality check
    if eat_volume_ml < 50:
        print(f"[WARNING] EAT volume unusually low ({eat_volume_ml:.2f} ml)")
    elif eat_volume_ml > 300:
        print(f"[WARNING] EAT volume unusually high ({eat_volume_ml:.2f} ml)")
    else:
        print(f"[INFO] EAT volume within expected range (50-300 ml)")
    
    return eat, eat_volume_ml

def save_eat_nifti(eat_mask, geometry, ct_nifti_path, output_path):
    """
    EATマスクをNIfTI形式で保存
    """
    print(f"[INFO] Saving EAT mask to {output_path}...")
    
    # Load reference CT for header information
    ct_img = nib.load(ct_nifti_path)
    
    # Create NIfTI image with same geometry as CT
    eat_img = nib.Nifti1Image(eat_mask.astype(np.uint8), 
                              affine=ct_img.affine,
                              header=ct_img.header)
    
    # Save
    nib.save(eat_img, output_path)
    print(f"[INFO] EAT mask saved")

def main():
    parser = argparse.ArgumentParser(
        description="Extract accurate EAT using TotalSegmentator and HARTA"
    )
    parser.add_argument("--ct_nifti", required=True,
                       help="Path to CT NIfTI file")
    parser.add_argument("--harta_png_dir", required=True,
                       help="HARTA output directory containing fat/ subdirectory")
    parser.add_argument("--slice_map_json", required=True,
                       help="Slice map JSON with geometry information")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--shell_thickness", type=float, default=6.0,
                       help="Shell thickness in mm (default: 6.0)")
    parser.add_argument("--hu_min", type=float, default=-190,
                       help="Minimum HU for fat (default: -190)")
    parser.add_argument("--hu_max", type=float, default=-30,
                       help="Maximum HU for fat (default: -30)")
    parser.add_argument("--skip_totalseg", action="store_true",
                       help="Skip TotalSegmentator if already run")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Reconstruct 3D from HARTA PNGs
    harta_volume, geometry = reconstruct_3d_from_harta(
        args.harta_png_dir, 
        args.slice_map_json
    )
    
    # Step 2: Run TotalSegmentator (unless skipped)
    totalseg_dir = output_dir / "totalseg"
    if not args.skip_totalseg:
        success = run_totalsegmentator(args.ct_nifti, str(totalseg_dir))
        if not success:
            print("[ERROR] TotalSegmentator failed, exiting")
            return 1
    else:
        print("[INFO] Skipping TotalSegmentator (using existing results)")
    
    # Step 3: Create heart ROI with shell
    shell_eat, myo = create_heart_roi_with_shell(
        totalseg_dir, 
        args.ct_nifti,
        args.shell_thickness
    )
    
    if shell_eat is None:
        print("[ERROR] Failed to create heart ROI")
        return 1
    
    # Step 4: Extract EAT
    eat_mask, eat_volume = extract_eat(
        args.ct_nifti,
        harta_volume,
        shell_eat,
        myo,
        args.hu_min,
        args.hu_max
    )
    
    # Step 5: Save results
    eat_output = output_dir / "eat_final.nii.gz"
    save_eat_nifti(eat_mask, geometry, args.ct_nifti, str(eat_output))
    
    # Save summary
    summary = {
        "eat_volume_ml": float(eat_volume),
        "shell_thickness_mm": args.shell_thickness,
        "hu_range": [args.hu_min, args.hu_max],
        "eat_voxels": int(np.sum(eat_mask)),
        "harta_voxels": int(np.sum(harta_volume > 0))
    }
    
    summary_path = output_dir / "eat_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SUMMARY]")
    print(f"EAT Volume: {eat_volume:.2f} ml")
    print(f"Output: {eat_output}")
    print(f"Summary: {summary_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())