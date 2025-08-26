#!/bin/bash
# 改善版EATの可視化

CT_NIFTI="$1"
TOTALSEG_DIR="$2"
EAT_DIR="$3"
OUTPUT_DIR="$4"
NUM_SLICES="${5:-10}"

echo "[INFO] Visualizing improved EAT results..."

PYTHON_SCRIPT='
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import json

ct_nifti = sys.argv[1]
totalseg_dir = sys.argv[2]
eat_dir = sys.argv[3]
output_dir = sys.argv[4]
num_slices = int(sys.argv[5])

# Load data
ct_img = nib.load(ct_nifti)
ct_data = ct_img.get_fdata()
spacing = ct_img.header.get_zooms()[:3]

# Load heart
heart_path = Path(totalseg_dir) / "heart.nii.gz"
if heart_path.exists():
    heart = nib.load(str(heart_path)).get_fdata() > 0
else:
    heart = np.zeros_like(ct_data, dtype=bool)

# Load aorta
aorta_path = Path(totalseg_dir) / "aorta.nii.gz"
if aorta_path.exists():
    aorta = nib.load(str(aorta_path)).get_fdata() > 0
else:
    aorta = np.zeros_like(ct_data, dtype=bool)

# Load shell
shell_path = Path(eat_dir) / "shell_mask.nii.gz"
if shell_path.exists():
    shell = nib.load(str(shell_path)).get_fdata() > 0
else:
    shell = np.zeros_like(ct_data, dtype=bool)

# Load EAT
eat_path = Path(eat_dir) / "eat_mask.nii.gz"
if eat_path.exists():
    eat = nib.load(str(eat_path)).get_fdata() > 0
else:
    eat = np.zeros_like(ct_data, dtype=bool)

# Create output directory
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Find heart range
heart_or_shell = heart | shell
z_any = np.any(heart_or_shell, axis=(0, 1))
z_indices = np.where(z_any)[0]

if len(z_indices) > 0:
    z_start = z_indices[0]
    z_end = z_indices[-1]
    slice_indices = np.linspace(z_start, z_end, num_slices, dtype=int)
else:
    slice_indices = np.linspace(0, ct_data.shape[2]-1, num_slices, dtype=int)

print(f"[INFO] Creating visualizations for slices: {slice_indices}")

# Save combined images
for idx, z in enumerate(slice_indices):
    ct_slice = ct_data[:, :, z]
    
    # Window/Level for chest CT
    window_center = 40
    window_width = 400
    ct_windowed = np.clip(
        (ct_slice - (window_center - window_width/2)) / window_width * 255,
        0, 255
    ).astype(np.uint8)
    
    # Create RGB image
    rgb = np.stack([ct_windowed]*3, axis=-1)
    
    # Overlay components
    if np.any(heart[:, :, z]):
        rgb[heart[:, :, z]] = [255, 100, 100]  # Light red for heart
    
    if np.any(aorta[:, :, z]):
        rgb[aorta[:, :, z]] = [200, 100, 100]  # Dark red for aorta
    
    if np.any(shell[:, :, z]):
        rgb[shell[:, :, z]] = [100, 100, 255]  # Blue for shell
    
    if np.any(eat[:, :, z]):
        rgb[eat[:, :, z]] = [0, 255, 0]  # Green for EAT
    
    # Save
    Image.fromarray(rgb).save(
        output_dir / f"slice_{z:03d}_combined.png"
    )
    
    if idx == 0 or idx == num_slices//2 or idx == num_slices-1:
        print(f"  Slice {z}: Heart={np.sum(heart[:,:,z])}, Shell={np.sum(shell[:,:,z])}, EAT={np.sum(eat[:,:,z])}")

# Statistics
stats = {
    "num_slices": num_slices,
    "slice_indices": slice_indices.tolist(),
    "heart_voxels": int(np.sum(heart)),
    "aorta_voxels": int(np.sum(aorta)),
    "shell_voxels": int(np.sum(shell)),
    "eat_voxels": int(np.sum(eat))
}

with open(output_dir / "vis_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"[SUCCESS] Saved to {output_dir}")
'

# Run with Docker
docker run --rm \
    -v /home/soya:/home/soya \
    wasserth/totalsegmentator:2.10.0 \
    python3 -c "import sys; $PYTHON_SCRIPT" \
    "$CT_NIFTI" "$TOTALSEG_DIR" "$EAT_DIR" "$OUTPUT_DIR" "$NUM_SLICES"