#!/usr/bin/env python3
"""
改善版：TotalSegmentatorとEDTを使用したEAT抽出
文献に基づく改善：
- 大血管を含む心臓集合
- 穴埋めと2Dクロージング
- 適応型シェル厚
- HARTAの条件付き使用
- Z範囲の自動クリップ
"""
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
import argparse
from scipy.ndimage import (
    distance_transform_edt, binary_dilation, generate_binary_structure,
    binary_fill_holes, binary_closing, binary_erosion, label
)
import subprocess
import sys

def _mm_to_px(mm, spacing):
    """mmをピクセル数に変換"""
    return tuple(int(np.ceil(mm / s)) for s in spacing)

def load_mask_from_png(png_path, rgb_threshold=127):
    """
    HARTA PNGからバイナリマスクを正しく抽出
    RGB値を使用（アルファチャンネルは常に255なので使わない）
    """
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im)
    rgb = arr[..., :3].astype(np.int16)
    mask = (rgb > rgb_threshold).any(axis=-1).astype(np.uint8)
    return mask

def reconstruct_3d_from_harta(harta_png_dir, slice_map_json):
    """HARTA PNGから3Dボリュームを再構成"""
    print(f"[INFO] Reconstructing 3D from HARTA PNGs...")
    
    with open(slice_map_json, 'r') as f:
        smap = json.load(f)
    
    geometry = smap["geometry"]
    size = tuple(geometry["size"])
    meta = smap["meta"]
    
    hdir = Path(harta_png_dir) / "fat"
    
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
    
    stacks = []
    for z in range(size[2]):
        if z in masks_dict:
            stacks.append(masks_dict[z])
        else:
            stacks.append(np.zeros((size[1], size[0]), dtype=np.uint8))
    
    volume = np.stack(stacks, axis=2).astype(np.uint8)
    print(f"[INFO] Reconstructed volume shape: {volume.shape}")
    return volume, geometry

def create_heart_roi_with_shell_improved(totalseg_dir, ct_nifti_path,
                                        base_shell_max_mm=15.0,
                                        seed_shell_mm=6.0):
    """
    改善版：心臓ROIとシェル作成
    1) 心筋+4腔+大血管（aorta, pulmonary_artery）を合算
    2) 3D穴埋め → 2Dクロージングで小裂けを塞ぐ
    3) EDTは心臓全体の外側のみで計算
    4) 適応型シェル厚：0-20mmバンドで脂肪距離の95パーセンタイル
    """
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    print("[INFO] Loading heart components including vessels...")
    
    # 心筋 + 4腔 + 大血管
    # totalタスクの場合は"heart.nii.gz"として全体が出力される
    comp_names = [
        "heart",  # totalタスクの場合
        "heart_myocardium",
        "heart_ventricle_left", "heart_ventricle_right",
        "heart_atrium_left", "heart_atrium_right",
        "aorta", "pulmonary_artery",  # 大血管
        "pulmonary_vein"  # 肺静脈も含める
    ]
    
    parts = []
    loaded_comps = []
    for name in comp_names:
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if p.exists():
            parts.append(nib.load(str(p)).get_fdata() > 0)
            loaded_comps.append(name)
            print(f"  Loaded: {name}")
        else:
            if name not in ["heart", "pulmonary_artery", "pulmonary_vein"]:
                print(f"  Not found: {name}")
    
    if not parts:
        print("[ERROR] No heart components found!")
        return None, None, None, None, None, None
    
    print(f"[INFO] Loaded {len(loaded_comps)} components")
    heart_all = np.logical_or.reduce(parts)
    
    # 心筋単独（除外用）
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
        print("[INFO] Using heart_myocardium for exclusion")
    else:
        # totalタスクでは心臓全体しか無いので、それを心筋として扱う
        # （保守的アプローチ：心臓全体を除外）
        myo = heart_all  # フォールバック
        print("[INFO] Using full heart mask as myocardium (fallback)")
    
    print("[INFO] Applying 3D hole filling...")
    heart_all = binary_fill_holes(heart_all)
    
    # 2Dクロージング（面内の小裂け対策）
    print("[INFO] Applying 2D closing per slice...")
    rad_mm = 1.5  # 半径1.5mm推奨
    rx, ry, _ = _mm_to_px(rad_mm, spacing)
    rx = max(1, rx)
    ry = max(1, ry)
    se2d = np.ones((2*ry+1, 2*rx+1), dtype=bool)
    
    for z in range(heart_all.shape[2]):
        if z % 50 == 0:
            print(f"  Processing slice {z}/{heart_all.shape[2]}...")
        heart_all[:, :, z] = binary_closing(heart_all[:, :, z], structure=se2d)
    
    # EDT（外側のみ）
    print("[INFO] Computing EDT for outer shell...")
    outside = ~heart_all
    dist_mm = distance_transform_edt(outside, sampling=spacing)
    
    # 0-20mmバンド（適応厚推定用）
    band20 = (dist_mm > 0) & (dist_mm <= 20.0)
    
    # 心臓集合の連結成分確認
    labeled, num_components = label(heart_all)
    print(f"[INFO] Heart components after processing: {num_components}")
    
    return dist_mm, band20, myo, heart_all, spacing, ct_data

def build_adaptive_shell_and_clip(dist_mm, band20, heart_all, ct_data, spacing,
                                 hu_min=-190, hu_max=-30,
                                 adaptive_cap_mm=15.0, seed_shell_mm=6.0):
    """
    適応型シェル厚とZ範囲クリップ
    - 0-20mmバンド内の脂肪距離分布の95パーセンタイルをシェル厚に
    - 心臓が存在するスライスのみに限定
    """
    print(f"[INFO] Computing adaptive shell thickness...")
    print(f"  HU range for fat: {hu_min} to {hu_max}")
    
    # 脂肪のHU判定
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    
    # 適応厚の推定
    cand = np.where(band20 & hu_fat, dist_mm, np.nan)
    dist_vals = cand[~np.isnan(cand)]
    
    if dist_vals.size > 500:
        t = float(np.percentile(dist_vals, 95.0))
        t = min(max(t, seed_shell_mm), adaptive_cap_mm)
        print(f"  Adaptive thickness: {t:.2f} mm (95th percentile)")
    else:
        t = seed_shell_mm
        print(f"  Using fallback thickness: {t:.2f} mm (insufficient fat samples)")
    
    # シェル作成
    shell = (dist_mm > 0) & (dist_mm <= t)
    
    # Z範囲クリップ：心臓があるスライスのみ
    z_any = np.any(heart_all, axis=(0, 1))
    z_idx = np.where(z_any)[0]
    
    if z_idx.size > 0:
        z0, z1 = int(z_idx[0]), int(z_idx[-1])
        print(f"[INFO] Z-range clipping: slice {z0} to {z1}")
        shell[:, :, :z0] = False
        shell[:, :, z1+1:] = False
    
    # シェルの連結成分確認
    shell_labeled, shell_components = label(shell)
    print(f"[INFO] Shell components: {shell_components}")
    
    return shell, t

def extract_eat_with_fallback(ct_data, harta_volume, shell, myo,
                             hu_min=-190, hu_max=-30,
                             use_harta_if_helpful=True,
                             min_harta_overlap_ratio=0.15):
    """
    改善版EAT抽出：HARTAの条件付き使用
    基本：HU ∧ Shell ∧ ¬Myocardium
    HARTAは有用な場合のみAND
    """
    print("[INFO] Extracting EAT with conditional HARTA...")
    
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    base_eat = hu_fat & shell & (~myo)
    
    base_voxels = np.sum(base_eat)
    print(f"  Base EAT candidates: {base_voxels:,} voxels")
    
    if use_harta_if_helpful and harta_volume is not None:
        harta = (harta_volume > 0)
        harta_voxels = np.sum(harta)
        overlap = np.sum(base_eat & harta)
        ratio = overlap / (base_voxels + 1e-6)
        
        print(f"  HARTA voxels: {harta_voxels:,}")
        print(f"  Overlap ratio: {ratio:.3f}")
        
        if ratio >= min_harta_overlap_ratio:
            eat = base_eat & harta
            print(f"  Using HARTA (helpful, ratio >= {min_harta_overlap_ratio})")
        else:
            eat = base_eat
            print(f"  Skipping HARTA (not helpful, ratio < {min_harta_overlap_ratio})")
    else:
        eat = base_eat
        print("  HARTA not used (disabled or not available)")
    
    final_voxels = np.sum(eat)
    print(f"  Final EAT: {final_voxels:,} voxels")
    
    return eat

def calculate_volume_ml(mask, spacing):
    """マスクの体積をml単位で計算"""
    voxel_vol_mm3 = np.prod(spacing)
    voxel_count = np.sum(mask)
    volume_ml = voxel_count * voxel_vol_mm3 / 1000.0
    return volume_ml, int(voxel_count)

def save_results(eat_mask, shell, ct_img, output_dir, shell_thickness_mm,
                hu_min, hu_max, harta_voxels=None):
    """結果の保存と統計出力"""
    os.makedirs(output_dir, exist_ok=True)
    
    spacing = ct_img.header.get_zooms()[:3]
    
    # EATマスク保存
    eat_img = nib.Nifti1Image(eat_mask.astype(np.uint8), ct_img.affine)
    eat_path = Path(output_dir) / "eat_mask.nii.gz"
    nib.save(eat_img, eat_path)
    
    # シェルマスク保存
    shell_img = nib.Nifti1Image(shell.astype(np.uint8), ct_img.affine)
    shell_path = Path(output_dir) / "shell_mask.nii.gz"
    nib.save(shell_img, shell_path)
    
    # 体積計算
    eat_vol_ml, eat_voxels = calculate_volume_ml(eat_mask, spacing)
    shell_vol_ml, shell_voxels = calculate_volume_ml(shell, spacing)
    
    # 統計保存
    stats = {
        "eat_volume_ml": float(eat_vol_ml),
        "eat_voxels": int(eat_voxels),
        "shell_volume_ml": float(shell_vol_ml),
        "shell_voxels": int(shell_voxels),
        "shell_thickness_mm": float(shell_thickness_mm),
        "hu_range": [float(hu_min), float(hu_max)],
    }
    
    if harta_voxels is not None:
        stats["harta_voxels"] = int(harta_voxels)
    
    stats_path = Path(output_dir) / "eat_summary.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("[RESULTS]")
    print(f"  EAT Volume: {eat_vol_ml:.2f} ml")
    print(f"  EAT Voxels: {eat_voxels:,}")
    print(f"  Shell Thickness: {shell_thickness_mm:.2f} mm")
    print(f"  Shell Voxels: {shell_voxels:,}")
    print(f"  HU Range: {hu_min} to {hu_max}")
    print(f"  Output: {output_dir}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="改善版EAT抽出")
    parser.add_argument("ct_nifti", help="入力CT NIfTI")
    parser.add_argument("totalseg_dir", help="TotalSegmentator出力ディレクトリ")
    parser.add_argument("output_dir", help="出力ディレクトリ")
    parser.add_argument("--harta-png-dir", help="HARTA PNG出力ディレクトリ")
    parser.add_argument("--slice-map", help="スライスマップJSON")
    parser.add_argument("--hu-min", type=float, default=-190,
                      help="脂肪HU下限（デフォルト: -190）")
    parser.add_argument("--hu-max", type=float, default=-30,
                      help="脂肪HU上限（デフォルト: -30、単純CT）")
    parser.add_argument("--shell-max", type=float, default=15.0,
                      help="適応シェル厚上限mm（デフォルト: 15）")
    parser.add_argument("--shell-seed", type=float, default=6.0,
                      help="フォールバックシェル厚mm（デフォルト: 6）")
    parser.add_argument("--harta-min-overlap", type=float, default=0.15,
                      help="HARTA使用の最小オーバーラップ率（デフォルト: 0.15）")
    parser.add_argument("--no-harta", action="store_true",
                      help="HARTAを使用しない")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("IMPROVED EAT EXTRACTION WITH TOTALSEGMENTATOR")
    print("="*60)
    
    # 1. 心臓ROI作成（改善版）
    result = create_heart_roi_with_shell_improved(
        args.totalseg_dir,
        args.ct_nifti,
        base_shell_max_mm=args.shell_max,
        seed_shell_mm=args.shell_seed
    )
    
    if result[0] is None:
        print("[ERROR] Failed to create heart ROI")
        sys.exit(1)
    
    dist_mm, band20, myo, heart_all, spacing, ct_data = result
    
    # 2. 適応型シェルとZクリップ
    shell, shell_thickness = build_adaptive_shell_and_clip(
        dist_mm, band20, heart_all, ct_data, spacing,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        adaptive_cap_mm=args.shell_max,
        seed_shell_mm=args.shell_seed
    )
    
    # 3. HARTA読み込み（オプション）
    harta_volume = None
    harta_voxels = None
    if args.harta_png_dir and args.slice_map and not args.no_harta:
        try:
            harta_volume, _ = reconstruct_3d_from_harta(
                args.harta_png_dir, args.slice_map
            )
            harta_voxels = np.sum(harta_volume > 0)
        except Exception as e:
            print(f"[WARNING] Failed to load HARTA: {e}")
    
    # 4. EAT抽出（フォールバック付き）
    eat = extract_eat_with_fallback(
        ct_data, harta_volume, shell, myo,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        use_harta_if_helpful=(not args.no_harta),
        min_harta_overlap_ratio=args.harta_min_overlap
    )
    
    # 5. 結果保存
    ct_img = nib.load(args.ct_nifti)
    save_results(
        eat, shell, ct_img, args.output_dir,
        shell_thickness, args.hu_min, args.hu_max,
        harta_voxels
    )

if __name__ == "__main__":
    main()