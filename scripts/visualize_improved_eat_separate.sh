#!/bin/bash
# 改善版EATの項目別可視化（Heart, Shell, EAT等を個別に表示）

CT_NIFTI="$1"
TOTALSEG_DIR="$2"
EAT_DIR="$3"
OUTPUT_DIR="$4"
NUM_SLICES="${5:-10}"

echo "[INFO] Creating separate visualizations for improved EAT results..."

PYTHON_SCRIPT='
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys
from scipy.ndimage import binary_erosion

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

ct_nifti = sys.argv[1]
totalseg_dir = sys.argv[2]
eat_dir = sys.argv[3]
output_dir = sys.argv[4]
num_slices = int(sys.argv[5])

print("[INFO] Loading data...")

# Load CT
ct_img = nib.load(ct_nifti)
ct_data = ct_img.get_fdata()
spacing = ct_img.header.get_zooms()[:3]

# Load segmentations
heart_path = Path(totalseg_dir) / "heart.nii.gz"
heart = nib.load(str(heart_path)).get_fdata() > 0 if heart_path.exists() else np.zeros_like(ct_data, dtype=bool)

aorta_path = Path(totalseg_dir) / "aorta.nii.gz"
aorta = nib.load(str(aorta_path)).get_fdata() > 0 if aorta_path.exists() else np.zeros_like(ct_data, dtype=bool)

pulmonary_vein_path = Path(totalseg_dir) / "pulmonary_vein.nii.gz"
pulmonary_vein = nib.load(str(pulmonary_vein_path)).get_fdata() > 0 if pulmonary_vein_path.exists() else np.zeros_like(ct_data, dtype=bool)

# Load improved outputs
shell_path = Path(eat_dir) / "shell_mask.nii.gz"
shell = nib.load(str(shell_path)).get_fdata() > 0 if shell_path.exists() else np.zeros_like(ct_data, dtype=bool)

eat_path = Path(eat_dir) / "eat_mask.nii.gz"
eat = nib.load(str(eat_path)).get_fdata() > 0 if eat_path.exists() else np.zeros_like(ct_data, dtype=bool)

# Create output directories
output_dir = Path(output_dir)
dirs = ["ct_slices", "heart", "vessels", "shell_alpha", "shell_contour", 
        "eat_alpha", "eat_contour", "combined", "combined_contour"]
for d in dirs:
    (output_dir / d).mkdir(parents=True, exist_ok=True)

# Find relevant slice range
heart_or_shell = heart | shell
z_any = np.any(heart_or_shell, axis=(0, 1))
z_indices = np.where(z_any)[0]

if len(z_indices) > 0:
    z_start = z_indices[0]
    z_end = z_indices[-1]
    slice_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
else:
    slice_indices = np.linspace(0, ct_data.shape[2]-1, num_slices, dtype=int)

print(f"[INFO] Processing {num_slices} slices from z={z_start} to z={z_end}")
print(f"[INFO] Shell thickness: See {eat_dir}/eat_summary.json")

# Process each slice
for idx, z in enumerate(slice_indices):
    if idx % 3 == 0:
        print(f"  Processing slice {idx+1}/{num_slices} (z={z})...")
    
    # Get CT slice
    ct_slice = ct_data[:, :, z]
    
    # Window/Level for chest CT
    window_center = 40
    window_width = 400
    ct_windowed = np.clip(
        (ct_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    
    # Get mask slices
    heart_slice = heart[:, :, z]
    aorta_slice = aorta[:, :, z]
    pulmonary_vein_slice = pulmonary_vein[:, :, z]
    shell_slice = shell[:, :, z]
    eat_slice = eat[:, :, z]
    
    # 1. CT only
    Image.fromarray(ct_windowed).save(
        output_dir / "ct_slices" / f"slice_{z:03d}_ct.png"
    )
    
    # 2. Heart (red)
    heart_img = overlay_alpha(ct_windowed, heart_slice, color=(255,0,0), alpha=0.3)
    Image.fromarray(heart_img).save(
        output_dir / "heart" / f"slice_{z:03d}_heart.png"
    )
    
    # 3. Vessels (aorta + pulmonary_vein in darker red)
    vessels = aorta_slice | pulmonary_vein_slice
    vessels_img = overlay_alpha(ct_windowed, vessels, color=(200,50,50), alpha=0.4)
    Image.fromarray(vessels_img).save(
        output_dir / "vessels" / f"slice_{z:03d}_vessels.png"
    )
    
    # 4. Shell - alpha (blue)
    shell_alpha = overlay_alpha(ct_windowed, shell_slice, color=(0,0,255), alpha=0.3)
    Image.fromarray(shell_alpha).save(
        output_dir / "shell_alpha" / f"slice_{z:03d}_shell_alpha.png"
    )
    
    # 5. Shell - contour (blue)
    shell_contour = overlay_contour(ct_windowed, shell_slice, color=(0,0,255))
    Image.fromarray(shell_contour).save(
        output_dir / "shell_contour" / f"slice_{z:03d}_shell_contour.png"
    )
    
    # 6. EAT - alpha (green)
    eat_alpha = overlay_alpha(ct_windowed, eat_slice, color=(0,255,0), alpha=0.5)
    Image.fromarray(eat_alpha).save(
        output_dir / "eat_alpha" / f"slice_{z:03d}_eat_alpha.png"
    )
    
    # 7. EAT - contour (green)
    eat_contour = overlay_contour(ct_windowed, eat_slice, color=(0,255,0))
    Image.fromarray(eat_contour).save(
        output_dir / "eat_contour" / f"slice_{z:03d}_eat_contour.png"
    )
    
    # 8. Combined - all with alpha
    combined = ct_windowed.copy()
    combined_rgb = np.stack([combined]*3, axis=-1).astype(np.float32)
    
    # Heart in light red
    if np.any(heart_slice):
        combined_rgb[heart_slice] = 0.3*np.array([255,100,100], dtype=np.float32) + 0.7*combined_rgb[heart_slice]
    
    # Vessels in darker red
    if np.any(vessels):
        combined_rgb[vessels] = 0.4*np.array([200,50,50], dtype=np.float32) + 0.6*combined_rgb[vessels]
    
    # Shell in blue
    if np.any(shell_slice):
        combined_rgb[shell_slice] = 0.3*np.array([100,100,255], dtype=np.float32) + 0.7*combined_rgb[shell_slice]
    
    # EAT in bright green (priority)
    if np.any(eat_slice):
        combined_rgb[eat_slice] = np.array([0,255,0], dtype=np.float32)
    
    Image.fromarray(combined_rgb.clip(0,255).astype(np.uint8)).save(
        output_dir / "combined" / f"slice_{z:03d}_combined.png"
    )
    
    # 9. Combined - contours only
    contour_img = np.stack([ct_windowed]*3, axis=-1).astype(np.uint8)
    
    # Draw contours
    if np.any(heart_slice):
        boundary = heart_slice ^ binary_erosion(heart_slice)
        contour_img[boundary] = [255,100,100]
    
    if np.any(vessels):
        boundary = vessels ^ binary_erosion(vessels)
        contour_img[boundary] = [200,50,50]
    
    if np.any(shell_slice):
        boundary = shell_slice ^ binary_erosion(shell_slice)
        contour_img[boundary] = [100,100,255]
    
    if np.any(eat_slice):
        boundary = eat_slice ^ binary_erosion(eat_slice)
        contour_img[boundary] = [0,255,0]
    
    Image.fromarray(contour_img).save(
        output_dir / "combined_contour" / f"slice_{z:03d}_combined_contour.png"
    )

# Save statistics
stats = {
    "num_slices_visualized": num_slices,
    "slice_indices": slice_indices.tolist(),
    "z_range": [int(z_start), int(z_end)] if len(z_indices) > 0 else [0, ct_data.shape[2]-1],
    "voxel_counts": {
        "heart": int(np.sum(heart)),
        "aorta": int(np.sum(aorta)),
        "pulmonary_vein": int(np.sum(pulmonary_vein)),
        "shell": int(np.sum(shell)),
        "eat": int(np.sum(eat))
    },
    "slices_with_data": {
        "heart": int(np.sum(np.any(heart, axis=(0,1)))),
        "vessels": int(np.sum(np.any(aorta | pulmonary_vein, axis=(0,1)))),
        "shell": int(np.sum(np.any(shell, axis=(0,1)))),
        "eat": int(np.sum(np.any(eat, axis=(0,1))))
    }
}

stats_path = output_dir / "visualization_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

# Load and include EAT summary
eat_summary_path = Path(eat_dir) / "eat_summary.json"
if eat_summary_path.exists():
    with open(eat_summary_path, "r") as f:
        eat_summary = json.load(f)
    print("\n[EAT EXTRACTION RESULTS]")
    eat_vol = eat_summary.get("eat_volume_ml", 0)
    shell_thick = eat_summary.get("shell_thickness_mm", 0)
    hu_range = eat_summary.get("hu_range", [-190, -30])
    print(f"  EAT Volume: {eat_vol:.2f} ml")
    print(f"  Shell Thickness: {shell_thick:.2f} mm")
    print(f"  HU Range: {hu_range}")

print(f"\n[SUCCESS] Visualizations saved to {output_dir}")
print(f"[INFO] Available folders:")
for d in dirs:
    print(f"  - {d}/")
'

# Run with Docker
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 -c "$PYTHON_SCRIPT" \
    "$CT_NIFTI" "$TOTALSEG_DIR" "$EAT_DIR" "$OUTPUT_DIR" "$NUM_SLICES"

echo ""
echo "[INFO] Visualization complete. Check the following directories:"
echo "  - $OUTPUT_DIR/ct_slices/       (Original CT)"
echo "  - $OUTPUT_DIR/heart/           (Heart segmentation)"
echo "  - $OUTPUT_DIR/vessels/         (Aorta + Pulmonary vein)"
echo "  - $OUTPUT_DIR/shell_alpha/     (Shell with transparency)"
echo "  - $OUTPUT_DIR/shell_contour/   (Shell contours only)"
echo "  - $OUTPUT_DIR/eat_alpha/       (EAT with transparency)"
echo "  - $OUTPUT_DIR/eat_contour/     (EAT contours only)"
echo "  - $OUTPUT_DIR/combined/        (All components)"
echo "  - $OUTPUT_DIR/combined_contour/ (All contours)"