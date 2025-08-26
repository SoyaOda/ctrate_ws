#!/bin/bash
# 改善された可視化スクリプト（アルファ合成と輪郭表示）

CT_NIFTI="$1"
TOTALSEG_DIR="$2"
EAT_NIFTI="$3"
OUTPUT_DIR="$4"
SHELL_THICKNESS="${5:-6.0}"
NUM_SLICES="${6:-20}"

echo "[INFO] Improved EAT visualization..."

PYTHON_SCRIPT='
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, generate_binary_structure, binary_erosion
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys

def overlay_contour(bg_uint8, mask_bool, color=(255,0,0)):
    """輪郭だけ描く"""
    b = mask_bool.astype(bool)
    boundary = b ^ binary_erosion(b)
    rgb = np.stack([bg_uint8]*3, axis=-1).astype(np.uint8)
    rgb[boundary] = color
    return rgb

def overlay_alpha(bg_uint8, mask_bool, color=(255,0,0), alpha=0.35):
    """均一に塗るオーバーレイ"""
    rgb = np.stack([bg_uint8]*3, axis=-1).astype(np.float32)
    color_arr = np.array(color, np.float32)
    rgb[mask_bool] = alpha*color_arr + (1-alpha)*rgb[mask_bool]
    return rgb.clip(0,255).astype(np.uint8)

def verify_mask_quality(mask_data, name):
    """マスクの品質チェック"""
    unique_vals = np.unique(mask_data)
    print(f"  {name} unique values: {unique_vals[:5]}")
    print(f"  {name} dtype: {mask_data.dtype}")
    voxel_count = np.sum(mask_data > 0)
    print(f"  {name} voxels: {voxel_count}")
    return voxel_count

ct_nifti = sys.argv[1]
totalseg_dir = sys.argv[2]
eat_nifti = sys.argv[3]
output_dir = sys.argv[4]
shell_thickness = float(sys.argv[5])
num_slices = int(sys.argv[6])

print("[INFO] Loading and verifying data...")

# CTデータ読み込み
ct_img = nib.load(ct_nifti)
ct_data = ct_img.get_fdata()
spacing = ct_img.header.get_zooms()[:3]

# 心臓構造を読み込みと検証
heart_parts = []
components = [
    "heart_myocardium",
    "heart_ventricle_left", 
    "heart_ventricle_right",
    "heart_atrium_left",
    "heart_atrium_right"
]

print("[INFO] Verifying segmentation masks:")
for comp in components:
    comp_path = Path(totalseg_dir) / f"{comp}.nii.gz"
    if comp_path.exists():
        img = nib.load(str(comp_path))
        data = img.get_fdata()
        verify_mask_quality(data, comp)
        heart_parts.append(data > 0)

# アフィン一致の確認
if heart_parts:
    seg_img = nib.load(Path(totalseg_dir) / f"{components[0]}.nii.gz")
    affine_match = np.allclose(ct_img.affine, seg_img.affine)
    shape_match = ct_img.shape == seg_img.shape
    print(f"[INFO] Affine match: {affine_match}, Shape match: {shape_match}")

# 心臓全体のマスク
heart = np.logical_or.reduce(heart_parts) if heart_parts else np.zeros_like(ct_data, dtype=bool)

# 心筋マスク
myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
if myo_path.exists():
    myo = nib.load(str(myo_path)).get_fdata() > 0
else:
    myo = heart

# 6mm殻を生成（外側のみ）
print(f"[INFO] Creating {shell_thickness}mm shell with EDT (outer only)...")
outside = ~heart  # 修正：心臓全体の外側のみ
dist_mm = distance_transform_edt(outside, sampling=spacing)
shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness)
struct = generate_binary_structure(3, 2)
shell_eat &= binary_dilation(heart, structure=struct, iterations=2)

# EATマスク読み込み
if eat_nifti and Path(eat_nifti).exists():
    eat_img = nib.load(eat_nifti)
    eat_data = eat_img.get_fdata() > 0
    verify_mask_quality(eat_data, "EAT")
else:
    eat_data = np.zeros_like(ct_data, dtype=bool)

# 出力ディレクトリ作成
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# サブディレクトリ作成
(output_dir / "ct_slices").mkdir(exist_ok=True)
(output_dir / "heart_alpha").mkdir(exist_ok=True)
(output_dir / "heart_contour").mkdir(exist_ok=True)
(output_dir / "shell_alpha").mkdir(exist_ok=True)
(output_dir / "shell_contour").mkdir(exist_ok=True)
(output_dir / "eat_alpha").mkdir(exist_ok=True)
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

print(f"[INFO] Creating improved visualizations for {num_slices} slices...")

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
    heart_slice = heart[:, :, z]
    shell_slice = shell_eat[:, :, z]
    eat_slice = eat_data[:, :, z]
    
    # 1. CT画像
    Image.fromarray(ct_windowed).save(
        output_dir / "ct_slices" / f"slice_{z:03d}_ct.png"
    )
    
    # 2. 心臓セグメンテーション（アルファ合成）
    heart_alpha = overlay_alpha(ct_windowed, heart_slice, color=(255,0,0), alpha=0.3)
    Image.fromarray(heart_alpha).save(
        output_dir / "heart_alpha" / f"slice_{z:03d}_heart_alpha.png"
    )
    
    # 3. 心臓セグメンテーション（輪郭）
    heart_contour = overlay_contour(ct_windowed, heart_slice, color=(255,0,0))
    Image.fromarray(heart_contour).save(
        output_dir / "heart_contour" / f"slice_{z:03d}_heart_contour.png"
    )
    
    # 4. 6mm殻（アルファ合成）
    shell_alpha = overlay_alpha(ct_windowed, shell_slice, color=(0,0,255), alpha=0.3)
    Image.fromarray(shell_alpha).save(
        output_dir / "shell_alpha" / f"slice_{z:03d}_shell_alpha.png"
    )
    
    # 5. 6mm殻（輪郭）
    shell_contour = overlay_contour(ct_windowed, shell_slice, color=(0,0,255))
    Image.fromarray(shell_contour).save(
        output_dir / "shell_contour" / f"slice_{z:03d}_shell_contour.png"
    )
    
    # 6. EAT（アルファ合成）
    eat_alpha = overlay_alpha(ct_windowed, eat_slice, color=(0,255,0), alpha=0.5)
    Image.fromarray(eat_alpha).save(
        output_dir / "eat_alpha" / f"slice_{z:03d}_eat_alpha.png"
    )
    
    # 7. 統合画像（全要素をアルファ合成）
    combined = ct_windowed.copy()
    combined_rgb = np.stack([combined]*3, axis=-1).astype(np.float32)
    
    # 心臓を薄い赤で
    if np.any(heart_slice):
        combined_rgb = overlay_alpha(ct_windowed, heart_slice, color=(255,100,100), alpha=0.2)
    
    # 殻を薄い青で重ねる
    if np.any(shell_slice):
        combined_rgb[shell_slice] = 0.3*np.array([100,100,255], dtype=np.float32) + 0.7*combined_rgb[shell_slice]
    
    # EATを濃い緑で（最重要）
    if np.any(eat_slice):
        combined_rgb[eat_slice] = np.array([0,255,0], dtype=np.float32)
    
    Image.fromarray(combined_rgb.clip(0,255).astype(np.uint8)).save(
        output_dir / "combined" / f"slice_{z:03d}_combined.png"
    )

# 統計情報を保存
from scipy import ndimage
heart_components = ndimage.label(heart)[1] if np.any(heart) else 0
shell_components = ndimage.label(shell_eat)[1] if np.any(shell_eat) else 0

stats = {
    "verification": {
        "affine_match": affine_match if heart_parts else False,
        "shape_match": shape_match if heart_parts else False,
        "heart_components": int(heart_components),
        "shell_components": int(shell_components)
    },
    "voxel_counts": {
        "heart_voxels": int(np.sum(heart)),
        "shell_voxels": int(np.sum(shell_eat)),
        "eat_voxels": int(np.sum(eat_data))
    },
    "shell_thickness_mm": shell_thickness,
    "num_slices_visualized": num_slices,
    "slice_indices": slice_indices.tolist(),
    "slices_with_data": {
        "heart": int(np.sum(np.any(heart, axis=(0,1)))),
        "shell": int(np.sum(np.any(shell_eat, axis=(0,1)))),
        "eat": int(np.sum(np.any(eat_data, axis=(0,1))))
    }
}

stats_path = output_dir / "visualization_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print("")
print("[SUCCESS] Improved visualizations saved to", output_dir)
print("[VERIFICATION] Results:")
for key, value in stats["verification"].items():
    print(f"  {key}: {value}")
print("[STATS] Voxel counts:")
for key, value in stats["voxel_counts"].items():
    print(f"  {key}: {value}")
'

# Dockerで実行
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 -c "$PYTHON_SCRIPT" \
    "$CT_NIFTI" "$TOTALSEG_DIR" "$EAT_NIFTI" "$OUTPUT_DIR" \
    "$SHELL_THICKNESS" "$NUM_SLICES"