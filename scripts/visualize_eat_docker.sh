#!/bin/bash
# Docker内でEAT結果を可視化

CT_NIFTI="$1"
TOTALSEG_DIR="$2"
EAT_NIFTI="$3"
OUTPUT_DIR="$4"
SHELL_THICKNESS="${5:-6.0}"
NUM_SLICES="${6:-20}"

echo "[INFO] Visualizing EAT results..."
echo "  CT: $CT_NIFTI"
echo "  TotalSeg: $TOTALSEG_DIR"
echo "  EAT: $EAT_NIFTI"
echo "  Output: $OUTPUT_DIR"
echo "  Shell: ${SHELL_THICKNESS}mm"
echo "  Slices: $NUM_SLICES"

# Pythonスクリプト
PYTHON_SCRIPT='
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys

ct_nifti = sys.argv[1]
totalseg_dir = sys.argv[2]
eat_nifti = sys.argv[3]
output_dir = sys.argv[4]
shell_thickness = float(sys.argv[5])
num_slices = int(sys.argv[6])

print(f"[INFO] Loading data...")

# CTデータ読み込み
ct_img = nib.load(ct_nifti)
ct_data = ct_img.get_fdata()
spacing = ct_img.header.get_zooms()[:3]

# 心臓構造を読み込み
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

# 心臓全体のマスク
heart = np.logical_or.reduce(heart_parts) if heart_parts else np.zeros_like(ct_data, dtype=bool)

# 心筋マスク
myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
if myo_path.exists():
    myo = nib.load(str(myo_path)).get_fdata() > 0
else:
    myo = heart

# 6mm殻を生成
print(f"[INFO] Creating {shell_thickness}mm shell...")
outside = ~myo
dist_mm = distance_transform_edt(outside, sampling=spacing)
shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness)
struct = generate_binary_structure(3, 2)
shell_eat &= binary_dilation(heart, structure=struct, iterations=2)

# EATマスク読み込み
if eat_nifti and Path(eat_nifti).exists():
    eat_img = nib.load(eat_nifti)
    eat_data = eat_img.get_fdata() > 0
    print(f"[INFO] EAT mask loaded: {np.sum(eat_data)} voxels")
else:
    print(f"[WARNING] EAT file not found: {eat_nifti}")
    eat_data = np.zeros_like(ct_data, dtype=bool)

# 出力ディレクトリ作成
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# サブディレクトリ作成
(output_dir / "ct_slices").mkdir(exist_ok=True)
(output_dir / "heart").mkdir(exist_ok=True)
(output_dir / "shell").mkdir(exist_ok=True)
(output_dir / "eat").mkdir(exist_ok=True)
(output_dir / "combined").mkdir(exist_ok=True)

# スライス選択（心臓の範囲を中心に）
heart_z = np.any(heart, axis=(0, 1))
heart_z_indices = np.where(heart_z)[0]
if len(heart_z_indices) > 0:
    z_start = heart_z_indices[0]
    z_end = heart_z_indices[-1]
    slice_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
else:
    z_size = ct_data.shape[2]
    slice_indices = np.linspace(0, z_size-1, num_slices, dtype=int)

print(f"[INFO] Creating visualizations for {num_slices} slices...")
print(f"[INFO] Slice range: {slice_indices[0]} to {slice_indices[-1]}")

for idx, z in enumerate(slice_indices):
    if idx % 5 == 0:
        print(f"  Processing slice {idx+1}/{num_slices} (z={z})...")
    
    # CTスライス
    ct_slice = ct_data[:, :, z]
    
    # Window/Level調整（胸部CT用）
    window_center = 40
    window_width = 400
    ct_windowed = np.clip(
        (ct_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    
    # 各マスクのスライス
    heart_slice = heart[:, :, z].astype(np.uint8) * 255
    shell_slice = shell_eat[:, :, z].astype(np.uint8) * 255
    eat_slice = eat_data[:, :, z].astype(np.uint8) * 255
    
    # 1. CT画像
    Image.fromarray(ct_windowed).save(
        output_dir / "ct_slices" / f"slice_{z:03d}_ct.png"
    )
    
    # 2. 心臓セグメンテーション（赤）
    heart_overlay = np.stack([ct_windowed, ct_windowed, ct_windowed], axis=-1)
    heart_mask = heart_slice > 0
    heart_overlay[heart_mask, 0] = np.minimum(heart_overlay[heart_mask, 0] + 100, 255)
    heart_overlay[heart_mask, 1] = np.maximum(heart_overlay[heart_mask, 1] - 50, 0)
    heart_overlay[heart_mask, 2] = np.maximum(heart_overlay[heart_mask, 2] - 50, 0)
    Image.fromarray(heart_overlay.astype(np.uint8)).save(
        output_dir / "heart" / f"slice_{z:03d}_heart.png"
    )
    
    # 3. 6mm殻（青）
    shell_overlay = np.stack([ct_windowed, ct_windowed, ct_windowed], axis=-1)
    shell_mask = shell_slice > 0
    shell_overlay[shell_mask, 2] = np.minimum(shell_overlay[shell_mask, 2] + 100, 255)
    shell_overlay[shell_mask, 0] = np.maximum(shell_overlay[shell_mask, 0] - 50, 0)
    shell_overlay[shell_mask, 1] = np.maximum(shell_overlay[shell_mask, 1] - 50, 0)
    Image.fromarray(shell_overlay.astype(np.uint8)).save(
        output_dir / "shell" / f"slice_{z:03d}_shell.png"
    )
    
    # 4. EAT（緑）
    eat_overlay = np.stack([ct_windowed, ct_windowed, ct_windowed], axis=-1)
    eat_mask = eat_slice > 0
    eat_overlay[eat_mask, 1] = np.minimum(eat_overlay[eat_mask, 1] + 150, 255)
    eat_overlay[eat_mask, 0] = np.maximum(eat_overlay[eat_mask, 0] - 50, 0)
    eat_overlay[eat_mask, 2] = np.maximum(eat_overlay[eat_mask, 2] - 50, 0)
    Image.fromarray(eat_overlay.astype(np.uint8)).save(
        output_dir / "eat" / f"slice_{z:03d}_eat.png"
    )
    
    # 5. 統合画像（全要素）
    combined_overlay = np.stack([ct_windowed, ct_windowed, ct_windowed], axis=-1)
    
    # 心臓を薄い赤で
    if np.any(heart_mask):
        combined_overlay[heart_mask, 0] = np.minimum(combined_overlay[heart_mask, 0] + 50, 255)
    
    # 殻を薄い青で
    if np.any(shell_mask):
        combined_overlay[shell_mask, 2] = np.minimum(combined_overlay[shell_mask, 2] + 80, 255)
    
    # EATを濃い緑で（最も重要）
    if np.any(eat_mask):
        combined_overlay[eat_mask, 1] = 255
        combined_overlay[eat_mask, 0] = 0
        combined_overlay[eat_mask, 2] = 0
    
    Image.fromarray(combined_overlay.astype(np.uint8)).save(
        output_dir / "combined" / f"slice_{z:03d}_combined.png"
    )

# 統計情報を保存
stats = {
    "heart_voxels": int(np.sum(heart)),
    "shell_voxels": int(np.sum(shell_eat)),
    "eat_voxels": int(np.sum(eat_data)),
    "shell_thickness_mm": shell_thickness,
    "num_slices_visualized": num_slices,
    "slice_indices": slice_indices.tolist(),
    "heart_slices_with_data": int(np.sum(np.any(heart, axis=(0,1)))),
    "shell_slices_with_data": int(np.sum(np.any(shell_eat, axis=(0,1)))),
    "eat_slices_with_data": int(np.sum(np.any(eat_data, axis=(0,1))))
}

stats_path = output_dir / "visualization_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print("")
print(f"[SUCCESS] Visualizations saved to {output_dir}")
print(f"[STATS] Heart: {stats['heart_voxels']} voxels in {stats['heart_slices_with_data']} slices")
print(f"[STATS] Shell: {stats['shell_voxels']} voxels in {stats['shell_slices_with_data']} slices")  
print(f"[STATS] EAT: {stats['eat_voxels']} voxels in {stats['eat_slices_with_data']} slices")
'

# Dockerで実行
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 -c "$PYTHON_SCRIPT" \
    "$CT_NIFTI" "$TOTALSEG_DIR" "$EAT_NIFTI" "$OUTPUT_DIR" \
    "$SHELL_THICKNESS" "$NUM_SLICES"