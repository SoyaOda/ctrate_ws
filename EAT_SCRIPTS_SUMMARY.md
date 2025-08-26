# EATセグメンテーション関連スクリプト一覧

生成日時: $(date +"%Y-%m-%d %H:%M:%S")

## 1. コアスクリプト（EAT抽出）

### 1.1 extract_eat_with_totalseg.py
**パス**: `/home/soya/ctrate_ws/scripts/extract_eat_with_totalseg.py`  
**説明**: TotalSegmentatorの心臓マスクとEDTを使用したEAT抽出のメインロジック  
**重要修正**: 心臓全体の外側のみシェル生成（`outside = ~heart`）

```python
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
    exit(main())```

---

### 1.2 extract_eat_with_totalseg_docker.sh
**パス**: `/home/soya/ctrate_ws/scripts/extract_eat_with_totalseg_docker.sh`  
**説明**: Docker内でEAT抽出を実行（nibabel等の依存関係対応）  
**重要修正**: 心臓全体の外側のみシェル生成

```bash
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
    "$SHELL_THICKNESS" "$HU_MIN" "$HU_MAX"```

---

## 2. HARTA関連スクリプト

### 2.1 harta_masks_to_nifti_preserve_geom.py
**パス**: `/home/soya/ctrate_ws/scripts/harta_masks_to_nifti_preserve_geom.py`  
**説明**: HARTA PNG出力から3D NIfTIへの変換  
**重要修正**: load_mask_from_png関数（RGBチャンネル使用）

```python
#!/usr/bin/env python3
"""
Reconstruct 3D NIfTI from HARTA 2D PNG masks with preserved geometry
"""
import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
import argparse

def load_mask_from_png(png_path, threshold=127):
    """
    HARTAのPNG（RGBA/RGB/グレースケール/パレット）から
    バイナリマスクを安全に抽出して np.uint8 {0,1} を返す。

    仕様想定:
      - fat/*.png : 背景=黒(0,0,0), マスク=白/色(例: 255系), Aチャネルは多くのケースで常に255。
      - combined/*.png : オーバーレイ等は使わない（fatを使う）。

    ロバスト化ポイント:
      1) 'P' (パレット)は 'RGBA' に変換してから判定（色展開のため）
      2) 'L' (グレースケール) は閾値二値化
      3) 'RGB' は「どこかの色成分>threshold」をマスクとする
      4) 'RGBA' は、Aが全255なら **RGBの合成**で判定（Aには頼らない）
         Aに0〜255の変化がある稀ケースのみ A で二値化を許容
    """
    img = Image.open(png_path)
    mode = img.mode

    # パレット画像は RGBA に変換して色を展開
    if mode == 'P':
        img = img.convert('RGBA')
        mode = img.mode

    arr = np.array(img)

    # 2D (L/1) の場合: 閾値で二値化
    if arr.ndim == 2:
        # グレースケール (L) や1bit(1)
        return (arr > threshold).astype(np.uint8)

    # 3D の場合
    if arr.ndim == 3 and arr.shape[2] == 3:  # RGB
        # いずれかのチャネルが threshold 超え → マスク
        return (arr > threshold).any(axis=-1).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA
        rgb = arr[..., :3]
        alpha = arr[..., 3]

        # ◇ ケース1: Aが全255（=常に不透明）→ Aは使わずRGBで判定
        if np.all(alpha == 255):
            return (rgb > threshold).any(axis=-1).astype(np.uint8)

        # ◇ ケース2: Aが0/255以外も混在（半透明等）→ Aを閾値化
        #   ※ 画像が境界アンチエイリアス等で半透明を使うケースに対応
        if (alpha.min() < 255) and (alpha.max() > 0):
            return (alpha > threshold).astype(np.uint8)

        # ◇ フォールバック：RGBで判定
        return (rgb > threshold).any(axis=-1).astype(np.uint8)

    raise ValueError(f"Unsupported PNG format: shape={arr.shape}, mode={mode}")

def reconstruct_3d_from_harta(harta_png_dir, slice_map_json, out_nifti, ref_nifti, patient_id="CTRATE001"):
    """
    Reconstruct 3D volume from HARTA PNG outputs with correct geometry
    """
    print(f"[INFO] Reconstructing 3D NIfTI from HARTA outputs")
    print(f"[INFO] PNG directory: {harta_png_dir}")
    print(f"[INFO] Geometry from: {slice_map_json}")
    print(f"[INFO] Reference NIfTI: {ref_nifti}")
    
    # Load geometry information
    with open(slice_map_json, 'r') as f:
        smap = json.load(f)
    
    geometry = smap["geometry"]
    direction = tuple(geometry["direction"])
    origin = tuple(geometry["origin"])
    spacing = tuple(geometry["spacing"])
    size = tuple(geometry["size"])
    meta = smap["meta"]  # List of slice metadata
    
    print(f"[INFO] Original geometry - Size: {size}, Spacing: {spacing}")
    
    # Find PNG files from HARTA output
    hdir = Path(harta_png_dir)
    
    # HARTA output pattern: {patient_id}_{slice_num}_fat.png
    png_files = {}
    for p in hdir.glob("*.png"):
        # Extract slice number from filename
        # Pattern: CTRATE001_0_fat.png -> 0
        fname = p.stem  # filename without extension
        parts = fname.split('_')
        if len(parts) >= 2:
            try:
                slice_idx = int(parts[1])  # Get the slice number
                png_files[slice_idx] = p
            except ValueError:
                continue
    
    if not png_files:
        raise RuntimeError(f"No PNG files found in {harta_png_dir}")
    
    print(f"[INFO] Found {len(png_files)} PNG mask files")
    
    # Calculate slice ordering based on IPP and normal vector
    # IOP defines the row and column directions
    iop = np.array([
        direction[0], direction[3], direction[6],  # Row direction
        direction[1], direction[4], direction[7]   # Column direction
    ], dtype=float).reshape(2, 3)
    
    # Slice normal is cross product of row and column directions
    normal = np.cross(iop[0], iop[1])
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    # Project IPP onto normal to get slice positions
    slice_positions = []
    for m in meta:
        ipp = np.array(m["IPP"])
        z_pos = np.dot(ipp, normal)
        slice_positions.append((m["index"], z_pos, ipp))
    
    # Sort by position along normal
    slice_positions.sort(key=lambda x: x[1])
    
    # Load masks in correct order
    masks = []
    for idx, z_pos, ipp in slice_positions:
        if idx in png_files:
            mask = load_mask_from_png(png_files[idx])
            masks.append(mask)
        else:
            # If mask not found, create empty mask
            print(f"[WARN] No mask found for slice {idx}, using empty mask")
            if masks:
                # Use shape from previous mask
                mask = np.zeros_like(masks[-1], dtype=np.uint8)
            else:
                # Use first available mask to get shape
                first_mask = load_mask_from_png(next(iter(png_files.values())))
                mask = np.zeros_like(first_mask, dtype=np.uint8)
            masks.append(mask)
    
    # Stack masks into 3D volume
    vol = np.stack(masks, axis=-1)  # (H, W, Nz)
    print(f"[INFO] Stacked volume shape: {vol.shape}")
    
    # Create SimpleITK image with correct geometry
    sitk_mask = sitk.GetImageFromArray(vol.transpose(2, 0, 1).astype(np.uint8))  # (z, y, x)
    sitk_mask.SetDirection(direction)
    sitk_mask.SetOrigin(origin)
    sitk_mask.SetSpacing(spacing)
    
    # Load reference CT for size validation
    ref = sitk.ReadImage(str(ref_nifti))
    ref_size = ref.GetSize()
    mask_size = sitk_mask.GetSize()
    
    print(f"[INFO] Mask size: {mask_size}, Reference size: {ref_size}")
    
    # Resample to match reference if sizes differ
    if mask_size != ref_size:
        print(f"[INFO] Resampling mask to match reference size...")
        
        # Use the reference image's geometry exactly
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        
        sitk_mask = resampler.Execute(sitk_mask)
        print(f"[INFO] Resampled mask size: {sitk_mask.GetSize()}")
    
    # Ensure output directory exists
    out_path = Path(out_nifti)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save NIfTI with preserved geometry
    sitk.WriteImage(sitk_mask, str(out_nifti))
    
    # Calculate volume for verification
    mask_array = sitk.GetArrayFromImage(sitk_mask)
    voxel_volume_ml = np.prod(spacing) / 1000.0
    eat_volume_ml = np.sum(mask_array) * voxel_volume_ml
    
    print(f"[OK] Saved NIfTI: {out_nifti}")
    print(f"[INFO] EAT Volume: {eat_volume_ml:.2f} ml")
    print(f"[INFO] Non-zero voxels: {np.sum(mask_array)}")
    
    return out_nifti

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D NIfTI from HARTA PNG masks with preserved geometry"
    )
    parser.add_argument("--png_dir", required=True,
                        help="Directory containing HARTA PNG output (e.g., fat/ folder)")
    parser.add_argument("--slice_map", required=True,
                        help="JSON file with geometry info from nifti_to_dicom_preserve_geom.py")
    parser.add_argument("--out", required=True,
                        help="Output NIfTI file path")
    parser.add_argument("--ref", required=True,
                        help="Reference NIfTI (original CT) for geometry")
    parser.add_argument("--patient_id", default="CTRATE001",
                        help="Patient ID used in HARTA output filenames")
    
    args = parser.parse_args()
    
    try:
        reconstruct_3d_from_harta(
            args.png_dir,
            args.slice_map,
            args.out,
            args.ref,
            args.patient_id
        )
    except Exception as e:
        print(f"[ERROR] Reconstruction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()```

---

### 2.2 run_harta_auto.py
**パス**: `/home/soya/ctrate_ws/scripts/run_harta_auto.py`  
**説明**: HARTA自動セグメンテーション実行

```python
#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import glob
import shlex
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Run HARTA automatic segmentation on DICOM cases")
    ap.add_argument("--harta_repo", required=True, help="HARTA repo path (e.g., ~/ctrate_ws/tools/HARTA)")
    ap.add_argument("--dicom_case", required=True, help="DICOM case directory path")
    ap.add_argument("--out_dir", required=True, help="Output directory for HARTA results")
    ap.add_argument("--python", default=sys.executable, help="Python executable in harta env")
    args = ap.parse_args()

    # Expand paths
    harta_repo = os.path.expanduser(args.harta_repo)
    dicom_case = os.path.expanduser(args.dicom_case)
    out_dir = os.path.expanduser(args.out_dir)
    
    # Check if HARTA repo exists
    auto_py = os.path.join(harta_repo, "_automatic_.py")
    if not os.path.exists(auto_py):
        sys.exit(f"ERROR: _automatic_.py not found at {auto_py}")
    
    # Check if DICOM directory exists
    if not os.path.exists(dicom_case):
        sys.exit(f"ERROR: DICOM directory not found: {dicom_case}")
    
    # Create output directory and subdirectories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "slices"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "contours"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "fat"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "combined"), exist_ok=True)
    
    # Prepare Python script to run HARTA
    run_script = f"""
import sys
import os
sys.path.insert(0, '{harta_repo}')
os.chdir('{harta_repo}')

from _automatic_ import segmentEpicardialFat

# Run HARTA automatic segmentation
try:
    patient_id, no_slices, vol = segmentEpicardialFat(
        DICOM_DATASET='{dicom_case}',
        OUTPUT_FOLDER='{out_dir}/'
    )
    print(f"SUCCESS: Patient={{patient_id}}, Slices={{no_slices}}, Volume={{vol:.2f}}ml")
except Exception as e:
    print(f"ERROR: {{str(e)}}")
    sys.exit(1)
"""
    
    # Run the script
    cmd = [args.python, "-c", run_script]
    print(f"[RUN] HARTA automatic segmentation for: {dicom_case}")
    print(f"[INFO] Output directory: {out_dir}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        if result.stderr:
            print(f"[STDERR] {result.stderr}", file=sys.stderr)
        
        if result.returncode != 0:
            print(f"[ERROR] HARTA returned code {result.returncode}")
            sys.exit(1)
        else:
            print("[OK] HARTA segmentation completed")
            
    except Exception as e:
        print(f"[ERROR] Failed to run HARTA: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()```

---

### 2.3 nifti_to_dicom_preserve_geom.py
**パス**: `/home/soya/ctrate_ws/scripts/nifti_to_dicom_preserve_geom.py`  
**説明**: NIfTI→DICOM変換（HARTAの入力用）

```python
#!/usr/bin/env python3
"""
NIfTI to DICOM conversion with preserved geometry
Based on SimpleITK official examples for proper IOP/IPP handling
"""
import json
import os
import pathlib
import SimpleITK as sitk
import argparse
import numpy as np
from datetime import datetime
import pydicom
from pydicom.dataset import Dataset, FileDataset
import pydicom.uid

def write_dicom_series_from_nifti(nifti_path, out_dir):
    """
    Convert NIfTI to DICOM series preserving full geometry
    Using pydicom for proper DICOM file structure
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Reading NIfTI: {nifti_path}")
    img = sitk.ReadImage(str(nifti_path))
    
    # Get numpy array from SimpleITK image
    img_array = sitk.GetArrayFromImage(img)  # (z, y, x)
    
    spacing = img.GetSpacing()      # (sx, sy, sz)
    origin = img.GetOrigin()        # (ox, oy, oz)
    direction = img.GetDirection()  # 3x3 matrix flattened to length-9 tuple
    size = img.GetSize()            # (Nx, Ny, Nz)
    
    print(f"[INFO] Size: {size}, Spacing: {spacing}")
    print(f"[INFO] Origin: {origin}")
    print(f"[INFO] Direction matrix preserved from original NIfTI")
    
    # IOP: Direction cosines for row and column directions
    iop = [
        direction[0], direction[3], direction[6],  # Row direction (X)
        direction[1], direction[4], direction[7]   # Column direction (Y)
    ]
    
    # Generate consistent UIDs
    study_uid = pydicom.uid.generate_uid()
    series_uid = pydicom.uid.generate_uid()
    frame_of_ref_uid = pydicom.uid.generate_uid()
    
    file_names = []
    meta_list = []
    
    print(f"[INFO] Writing {size[2]} DICOM slices...")
    
    for k in range(size[2]):
        # Calculate IPP (Image Position Patient) for k-th slice
        # Use SimpleITK's TransformIndexToPhysicalPoint
        ipp = img.TransformIndexToPhysicalPoint([0, 0, k])
        
        # Get slice data (need to flip from ZYX to XY for DICOM)
        slice_data = img_array[k, :, :].astype(np.int16)
        
        # Create DICOM file
        filename = str(out_dir / f"slice_{k:04d}.dcm")
        
        # File meta info
        file_meta = pydicom.Dataset()
        file_meta.FileMetaInformationGroupLength = 196
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
        file_meta.ImplementationVersionName = 'PYDICOM'
        
        # Create the FileDataset
        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b'\x00' * 128)
        
        # Patient Module
        ds.PatientName = 'Anonymous'
        ds.PatientID = 'CTRATE001'
        ds.PatientBirthDate = ''
        ds.PatientSex = ''
        
        # General Study Module
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S.%f")
        ds.StudyID = '1'
        ds.AccessionNumber = ''
        ds.ReferringPhysicianName = ''
        
        # General Series Module
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = 1
        ds.Modality = 'CT'
        ds.SeriesDescription = 'NIfTI to DICOM conversion'
        ds.BodyPartExamined = ''
        
        # Frame of Reference Module
        ds.FrameOfReferenceUID = frame_of_ref_uid
        ds.PositionReferenceIndicator = ''
        
        # General Image Module
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = list(ipp)
        ds.ImageOrientationPatient = iop
        
        # Image Pixel Module
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # signed
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # [row, col]
        
        # CT Image Module
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        ds.SliceThickness = float(spacing[2])
        ds.SpacingBetweenSlices = float(spacing[2])
        
        # SOP Common Module
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        
        # Pixel Data
        ds.PixelData = slice_data.tobytes()
        
        # Save the file
        ds.save_as(filename)
        
        file_names.append(filename)
        meta_list.append({
            "index": k,
            "InstanceNumber": k+1,
            "IPP": list(ipp),
            "filename": filename
        })
    
    # Save slice mapping with full geometry information
    geometry_info = {
        "nifti_source": str(nifti_path),
        "dicom_output": str(out_dir),
        "files": file_names,
        "meta": meta_list,
        "geometry": {
            "origin": list(origin),
            "spacing": list(spacing),
            "direction": list(direction),
            "size": list(size),
            "iop": iop
        },
        "uids": {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "frame_of_ref_uid": frame_of_ref_uid
        }
    }
    
    # Save JSON in parent directory to avoid HARTA reading it
    json_path = out_dir.parent / f"{out_dir.name}_slice_map.json"
    with open(json_path, "w") as f:
        json.dump(geometry_info, f, indent=2)
    
    print(f"[OK] Converted {size[2]} slices to DICOM")
    print(f"[OK] Geometry preserved in: {json_path}")
    
    return json_path

def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI to DICOM with preserved geometry"
    )
    parser.add_argument("-i", "--input", required=True, 
                        help="Input NIfTI file path")
    parser.add_argument("-o", "--outdir", required=True,
                        help="Output DICOM directory")
    
    args = parser.parse_args()
    
    try:
        write_dicom_series_from_nifti(args.input, args.outdir)
    except Exception as e:
        print(f"[ERROR] Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()```

---

## 3. 可視化スクリプト

### 3.1 visualize_eat_improved.sh
**パス**: `/home/soya/ctrate_ws/scripts/visualize_eat_improved.sh`  
**説明**: 心臓、シェル、EATのPNG生成（改善版）  
**特徴**: アルファ合成と輪郭表示  
**重要修正**: 心臓全体の外側のみシェル生成

```bash
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
    "$SHELL_THICKNESS" "$NUM_SLICES"```

---

### 3.2 visualize_eat_docker.sh
**パス**: `/home/soya/ctrate_ws/scripts/visualize_eat_docker.sh`  
**説明**: 基本的な可視化（旧版）

```bash
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
    "$SHELL_THICKNESS" "$NUM_SLICES"```

---

## 4. 設定ファイル

### 4.1 .env
**パス**: `/home/soya/ctrate_ws/.env`  
**説明**: TotalSegmentatorライセンスキー等の設定

```bash
# TotalSegmentator環境設定
TOTALSEG_LICENSE_KEY=aca_DHDKRFJMHUX0PK```

---

## 5. バッチ処理スクリプト

### 5.1 batch_process_all.sh
**パス**: `/home/soya/ctrate_ws/scripts/batch_process_all.sh`  
**説明**: 全体のパイプライン実行（存在する場合）

```bash
#!/bin/bash
# 複数のCTデータに対してSTEP3(TotalSegmentator)とSTEP4(EAT/PAT)を実行

echo "========================================="
echo "バッチ処理: 全CTデータのEAT/PAT抽出"
echo "========================================="

# ベースディレクトリ
BASE_DIR="/home/soya/ctrate_ws"
DATA_DIR="$BASE_DIR/data/CT-RATE-v2/dataset/valid_fixed"
OUTPUT_BASE="$BASE_DIR/outputs/batch_results"

# 処理するCTファイルのリスト取得（全ての.nii.gzファイル）
CT_FILES=$(find "$DATA_DIR" -name "*.nii.gz" -type f | sort)

# 総ファイル数
TOTAL_FILES=$(echo "$CT_FILES" | wc -l)
echo "処理対象: $TOTAL_FILES 個のCTファイル"
echo ""

# 処理カウンタ
COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 結果サマリーファイル
SUMMARY_FILE="$OUTPUT_BASE/processing_summary.txt"
mkdir -p "$OUTPUT_BASE"
echo "EAT/PAT Batch Processing Summary" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Started at: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 各CTファイルを処理
for CT_FILE in $CT_FILES; do
    COUNT=$((COUNT + 1))
    
    # パスから識別子を抽出（例: valid_1_a_1）
    CASE_ID=$(echo "$CT_FILE" | sed 's/.*\/\(valid_[0-9]\)\/\(valid_[0-9]_[a-z]\)\/\(valid_[0-9]_[a-z]_[0-9]\).*/\1_\3/')
    
    echo "----------------------------------------"
    echo "[$COUNT/$TOTAL_FILES] 処理中: $CASE_ID"
    echo "CTファイル: $CT_FILE"
    
    # 出力ディレクトリ設定
    CASE_OUTPUT_DIR="$OUTPUT_BASE/$CASE_ID"
    mkdir -p "$CASE_OUTPUT_DIR"
    
    # STEP3: TotalSegmentator (heartchambers_highres)
    echo "  STEP3: 心臓セグメンテーション実行中..."
    SEGMENTATION_OUTPUT="$CASE_OUTPUT_DIR/segmentation"
    mkdir -p "$SEGMENTATION_OUTPUT"
    
    # Docker経由でTotalSegmentator実行
    timeout 600 docker run --rm --gpus all --ipc=host \
        --shm-size=8g \
        -v "$HOME:/home/soya" \
        -v "$HOME/.totalsegmentator:/root/.totalsegmentator" \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        wasserth/totalsegmentator:2.10.0 \
        bash -c "
            TotalSegmentator \
                -i '$CT_FILE' \
                -o /tmp/output \
                --task heartchambers_highres \
                --device gpu \
                --force_split \
                --robust_crop \
                --body_seg 2>/dev/null && \
            cp -r /tmp/output/* '$SEGMENTATION_OUTPUT/' 2>/dev/null
        " > /dev/null 2>&1
    
    # セグメンテーション結果確認
    if [ -f "$SEGMENTATION_OUTPUT/heart_myocardium.nii.gz" ]; then
        echo "  ✓ セグメンテーション成功"
        
        # STEP4: EAT/PAT抽出
        echo "  STEP4: EAT/PAT抽出中..."
        EAT_PAT_OUTPUT="$CASE_OUTPUT_DIR/eat_pat"
        mkdir -p "$EAT_PAT_OUTPUT"
        
        # Docker経由でEAT/PAT抽出
        docker run --rm --gpus all \
            -v "$HOME:/home/soya" \
            wasserth/totalsegmentator:2.10.0 \
            python3 -c "
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from pathlib import Path

# ファイルパス
ct_path = '$CT_FILE'
myocardium_path = '$SEGMENTATION_OUTPUT/heart_myocardium.nii.gz'
output_dir = '$EAT_PAT_OUTPUT'

# データ読み込み
ct_nii = nib.load(ct_path)
ct_data = ct_nii.get_fdata()
myocardium_nii = nib.load(myocardium_path)
myocardium_mask = myocardium_nii.get_fdata() > 0

# ボクセルサイズ
spacing = ct_nii.header.get_zooms()[:3]
voxel_volume_ml = np.prod(spacing) / 1000.0

# 脂肪マスク（HU値: -190 ~ -30）
fat_mask = (ct_data >= -190) & (ct_data <= -30)

# 膨張処理
struct_elem = generate_binary_structure(3, 2)
min_spacing = min(spacing)
dilation_5mm = int(np.ceil(5.0 / min_spacing))
dilation_10mm = int(np.ceil(10.0 / min_spacing))

# EAT（5mm膨張）
eat_roi = myocardium_mask.copy()
for _ in range(dilation_5mm):
    eat_roi = binary_dilation(eat_roi, struct_elem)

# PAT（10mm膨張）
pat_roi = myocardium_mask.copy()
for _ in range(dilation_10mm):
    pat_roi = binary_dilation(pat_roi, struct_elem)

# 脂肪抽出
eat_mask = fat_mask & eat_roi & ~myocardium_mask
pat_mask = fat_mask & pat_roi & ~myocardium_mask

# 体積計算
eat_volume_ml = np.sum(eat_mask) * voxel_volume_ml
pat_volume_ml = np.sum(pat_mask) * voxel_volume_ml

# 保存
Path(output_dir).mkdir(parents=True, exist_ok=True)
nib.save(nib.Nifti1Image(eat_mask.astype(np.uint8), ct_nii.affine, ct_nii.header), 
         f'{output_dir}/eat_5mm.nii.gz')
nib.save(nib.Nifti1Image(pat_mask.astype(np.uint8), ct_nii.affine, ct_nii.header), 
         f'{output_dir}/pat_10mm.nii.gz')

# 結果出力
print(f'EAT:{eat_volume_ml:.2f}ml,PAT:{pat_volume_ml:.2f}ml')
" 2>/dev/null | tail -1 > "$EAT_PAT_OUTPUT/volumes.txt"
        
        # 結果確認
        if [ -f "$EAT_PAT_OUTPUT/eat_5mm.nii.gz" ] && [ -f "$EAT_PAT_OUTPUT/pat_10mm.nii.gz" ]; then
            VOLUMES=$(cat "$EAT_PAT_OUTPUT/volumes.txt" 2>/dev/null)
            echo "  ✓ EAT/PAT抽出成功: $VOLUMES"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            
            # サマリーに記録
            echo "$CASE_ID: SUCCESS - $VOLUMES" >> "$SUMMARY_FILE"
        else
            echo "  ✗ EAT/PAT抽出失敗"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo "$CASE_ID: FAILED - EAT/PAT extraction error" >> "$SUMMARY_FILE"
        fi
    else
        echo "  ✗ セグメンテーション失敗"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "$CASE_ID: FAILED - Segmentation error" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

# 最終サマリー
echo "========================================="
echo "処理完了"
echo "========================================="
echo "総処理数: $TOTAL_FILES"
echo "成功: $SUCCESS_COUNT"
echo "失敗: $FAIL_COUNT"
echo ""
echo "結果保存先: $OUTPUT_BASE"
echo "サマリーファイル: $SUMMARY_FILE"

# サマリーファイルに最終結果追記
echo "" >> "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Completed at: $(date)" >> "$SUMMARY_FILE"
echo "Total: $TOTAL_FILES, Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT" >> "$SUMMARY_FILE"

# 結果一覧表示
if [ -f "$SUMMARY_FILE" ]; then
    echo ""
    echo "=== 処理結果一覧 ==="
    grep -E "valid_.*: " "$SUMMARY_FILE"
fi```

---

## 主要なアルゴリズム

### EAT計算式
```
EAT = HARTA_fat ∧ HU(-190,-30) ∧ Shell(0-6mm) ∧ ¬Myocardium
```

### シェル生成（修正版）
```python
# 重要: 心臓全体の外側のみを計算領域にする
outside = ~heart  # heart = myocardium + all chambers
dist_mm = distance_transform_edt(outside, sampling=spacing)
shell_eat = (dist_mm > 0) & (dist_mm <= shell_thickness_mm)
```

### 主な修正履歴
1. **PNG読み込み修正**: アルファチャンネル → RGBチャンネル
2. **シェル生成修正**: `~myo` → `~heart`（内側シェル除去）
3. **座標系保持**: NIfTI ↔ DICOM変換でaffine行列保持

### 現在の結果
- EAT体積: 約1.93ml（期待値50-200mlより低い）
- シェルボクセル数: 108,931（修正後）
- 心臓ボクセル数: 445,484

