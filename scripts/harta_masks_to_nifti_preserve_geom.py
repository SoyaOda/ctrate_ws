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
    main()