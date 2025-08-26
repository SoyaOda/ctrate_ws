#!/bin/bash
# TotalSegmentatorのDockerコンテナ内でEAT抽出を実行

# 引数を受け取る
CT_NIFTI="$1"
HARTA_PNG_DIR="$2"
SLICE_MAP_JSON="$3"
OUTPUT_DIR="$4"
SHELL_THICKNESS="${5:-6.0}"
HU_MIN="${6:--190}"
HU_MAX="${7:--30}"
SKIP_TOTALSEG="${8:-}"

# Pythonスクリプトの内容をヒアドキュメントで定義
PYTHON_SCRIPT='
import os
import json
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
import subprocess
import sys

def load_mask_from_png(png_path, rgb_threshold=127):
    """HARTA PNGからバイナリマスクを正しく抽出"""
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im)
    rgb = arr[..., :3].astype(np.int16)
    mask = (rgb > rgb_threshold).any(axis=-1).astype(np.uint8)
    return mask

def reconstruct_3d_from_harta(harta_png_dir, slice_map_json, patient_id="CTRATE001"):
    """HARTA PNGから3Dボリュームを再構成"""
    print(f"[INFO] Reconstructing 3D from HARTA PNGs...")
    
    with open(slice_map_json, "r") as f:
        smap = json.load(f)
    
    geometry = smap["geometry"]
    size = tuple(geometry["size"])
    meta = smap["meta"]
    
    hdir = Path(harta_png_dir) / "fat"
    
    masks_dict = {}
    for p in hdir.glob("*.png"):
        fname = p.stem
        parts = fname.split("_")
        if len(parts) >= 2:
            try:
                slice_idx = int(parts[1])
                if slice_idx < len(meta):
                    masks_dict[slice_idx] = load_mask_from_png(str(p))
            except ValueError:
                continue
    
    stacks = []
    for z in range(size[2]):
        if z in masks_dict:
            stacks.append(masks_dict[z])
        else:
            stacks.append(np.zeros((size[1], size[0]), dtype=np.uint8))
    
    volume = np.stack(stacks, axis=2).astype(np.uint8)
    print(f"[INFO] Reconstructed volume shape: {volume.shape}")
    return volume, geometry

def create_heart_roi_with_shell(totalseg_dir, ct_nifti_path, shell_thickness_mm=6.0):
    """TotalSegmentatorの心臓マスクからEDTを使って心膜周囲の薄い殻を作成（外側のみ）"""
    print(f"[INFO] Creating heart ROI with {shell_thickness_mm}mm shell (outer only)...")
    
    ct_img = nib.load(ct_nifti_path)
    spacing = ct_img.header.get_zooms()[:3]
    print(f"[INFO] CT spacing: {spacing} mm")
    
    heart_parts = []
    components = [
        "heart_myocardium",
        "heart_ventricle_left", 
        "heart_ventricle_right",
        "heart_atrium_left",
        "heart_atrium_right"
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
    
    heart = np.logical_or.reduce(heart_parts)
    print(f"[INFO] Combined heart mask shape: {heart.shape}")
    
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
    else:
        print(f"[WARNING] Myocardium not found, using heart mask")
        myo = heart
    
    # 重要な修正：心臓全体の外側のみを計算領域にする
    outside = ~heart  # 修正：~myo → ~heart
    dist_mm = distance_transform_edt(outside, sampling=spacing)
    shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness_mm)
    
    struct = generate_binary_structure(3, 2)
    shell_eat &= binary_dilation(heart, structure=struct, iterations=2)
    
    print(f"[INFO] Shell created with thickness {shell_thickness_mm}mm")
    print(f"[INFO] Shell voxels: {np.sum(shell_eat)}")
    
    return shell_eat, myo

def extract_eat(ct_nifti_path, harta_volume, shell_eat, myo, hu_min=-190, hu_max=-30):
    """最終的なEATを抽出"""
    print(f"[INFO] Extracting EAT with HU range [{hu_min}, {hu_max}]...")
    
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()
    
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    print(f"[INFO] HU fat voxels: {np.sum(hu_fat)}")
    
    harta_fat = harta_volume > 0
    print(f"[INFO] HARTA fat voxels: {np.sum(harta_fat)}")
    
    eat = harta_fat & hu_fat & shell_eat & (~myo)
    print(f"[INFO] Final EAT voxels: {np.sum(eat)}")
    
    spacing = ct_img.header.get_zooms()[:3]
    voxel_volume_ml = np.prod(spacing) / 1000.0
    eat_volume_ml = np.sum(eat) * voxel_volume_ml
    
    print(f"[INFO] EAT Volume: {eat_volume_ml:.2f} ml")
    
    if eat_volume_ml < 50:
        print(f"[WARNING] EAT volume unusually low ({eat_volume_ml:.2f} ml)")
    elif eat_volume_ml > 300:
        print(f"[WARNING] EAT volume unusually high ({eat_volume_ml:.2f} ml)")
    else:
        print(f"[INFO] EAT volume within expected range (50-300 ml)")
    
    return eat, eat_volume_ml

def save_eat_nifti(eat_mask, geometry, ct_nifti_path, output_path):
    """EATマスクをNIfTI形式で保存"""
    print(f"[INFO] Saving EAT mask to {output_path}...")
    
    ct_img = nib.load(ct_nifti_path)
    eat_img = nib.Nifti1Image(eat_mask.astype(np.uint8), 
                              affine=ct_img.affine,
                              header=ct_img.header)
    nib.save(eat_img, output_path)
    print(f"[INFO] EAT mask saved")

# メイン処理
ct_nifti = sys.argv[1]
harta_png_dir = sys.argv[2]
slice_map_json = sys.argv[3]
output_dir = sys.argv[4]
shell_thickness = float(sys.argv[5])
hu_min = float(sys.argv[6])
hu_max = float(sys.argv[7])

output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Reconstruct 3D from HARTA PNGs
harta_volume, geometry = reconstruct_3d_from_harta(harta_png_dir, slice_map_json)

# Step 2: Find TotalSegmentator results
totalseg_dir = output_dir / "totalseg"
if not totalseg_dir.exists():
    # Try common location
    totalseg_dir = Path("/home/soya/ctrate_ws/outputs/eat_extraction_test/totalseg")
    if not totalseg_dir.exists():
        print("[ERROR] TotalSegmentator results not found at:", totalseg_dir)
        sys.exit(1)
print(f"[INFO] Using TotalSegmentator results from: {totalseg_dir}")

# Step 3: Create heart ROI with shell
shell_eat, myo = create_heart_roi_with_shell(totalseg_dir, ct_nifti, shell_thickness)

if shell_eat is None:
    print("[ERROR] Failed to create heart ROI")
    sys.exit(1)

# Step 4: Extract EAT
eat_mask, eat_volume = extract_eat(ct_nifti, harta_volume, shell_eat, myo, hu_min, hu_max)

# Step 5: Save results
eat_output = output_dir / "eat_final.nii.gz"
save_eat_nifti(eat_mask, geometry, ct_nifti, str(eat_output))

# Save summary
summary = {
    "eat_volume_ml": float(eat_volume),
    "shell_thickness_mm": shell_thickness,
    "hu_range": [hu_min, hu_max],
    "eat_voxels": int(np.sum(eat_mask)),
    "harta_voxels": int(np.sum(harta_volume > 0))
}

summary_path = output_dir / "eat_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\\n[SUMMARY]")
print(f"EAT Volume: {eat_volume:.2f} ml")
print(f"Output: {eat_output}")
print(f"Summary: {summary_path}")
'

# Dockerでスクリプトを実行
echo "[INFO] Running EAT extraction in Docker container..."
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 -c "$PYTHON_SCRIPT" \
    "$CT_NIFTI" "$HARTA_PNG_DIR" "$SLICE_MAP_JSON" "$OUTPUT_DIR" \
    "$SHELL_THICKNESS" "$HU_MIN" "$HU_MAX"