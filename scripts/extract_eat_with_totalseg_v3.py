#!/usr/bin/env python3
"""
改善版v3：TotalSegmentatorとEDTを使用したEAT抽出
過拡張対策版：
- EDTドメイン制限（肺・皮下・骨を除外）
- Z範囲を心筋+4腔ベースに
- スライス別適応型シェル厚
- 心臓表面からの連結制約
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
    binary_fill_holes, binary_closing, binary_erosion, label,
    binary_propagation
)
import sys

def _mm_to_px(mm, spacing):
    """mmをピクセル数に変換"""
    return int(np.ceil(mm / min(spacing)))

def load_mask_from_png(png_path, rgb_threshold=127):
    """HARTA PNGからバイナリマスク抽出"""
    im = Image.open(png_path).convert("RGBA")
    arr = np.array(im)
    rgb = arr[..., :3].astype(np.int16)
    mask = (rgb > rgb_threshold).any(axis=-1).astype(np.uint8)
    return mask

def reconstruct_3d_from_harta(harta_png_dir, slice_map_json):
    """HARTA PNGから3Dボリューム再構成"""
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

def build_domain_mask(totalseg_dir, ct_shape, spacing):
    """
    EDTドメインマスク作成（胸腔内のみ）
    肺・皮下・骨近傍を除外
    """
    def _load(name):
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if p.exists():
            return nib.load(str(p)).get_fdata() > 0
        return np.zeros(ct_shape, dtype=bool)
    
    print("[INFO] Building domain mask (thoracic cavity only)...")
    
    # 肺の除外
    lung_components = [
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
    ]
    lung = np.zeros(ct_shape, dtype=bool)
    for comp in lung_components:
        if Path(totalseg_dir, f"{comp}.nii.gz").exists():
            lung |= _load(comp)
            print(f"  Excluding: {comp}")
    
    # 体表面/皮下の除外（bodyがあれば体表から25mmバンド）
    body = _load("body") | _load("body_trunc")
    if body.any():
        r = _mm_to_px(25.0, spacing)
        subcut = body & ~binary_erosion(body, iterations=r)
        print(f"  Excluding subcutaneous ring (25mm from surface)")
    else:
        subcut = np.zeros(ct_shape, dtype=bool)
    
    # 骨近傍の除外（肋骨・胸骨・椎体の3mm近傍）
    ribs = np.zeros(ct_shape, dtype=bool)
    for side in ["left", "right"]:
        for k in range(1, 13):
            rib_name = f"rib_{side}_{k}"
            if Path(totalseg_dir, f"{rib_name}.nii.gz").exists():
                ribs |= _load(rib_name)
    
    sternum = _load("sternum")
    
    vertebrae = np.zeros(ct_shape, dtype=bool)
    for v in ["C1","C2","C3","C4","C5","C6","C7",
              "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
              "L1","L2","L3","L4","L5"]:
        v_name = f"vertebrae_{v}"
        if Path(totalseg_dir, f"{v_name}.nii.gz").exists():
            vertebrae |= _load(v_name)
    
    # 骨の3mm近傍を除外
    if (ribs | sternum | vertebrae).any():
        dil = _mm_to_px(3.0, spacing)
        se3d = generate_binary_structure(3, 2)
        bone_nbhd = binary_dilation(
            ribs | sternum | vertebrae,
            structure=se3d,
            iterations=dil
        )
        print(f"  Excluding bone neighborhood (3mm dilation)")
    else:
        bone_nbhd = np.zeros(ct_shape, dtype=bool)
    
    # 胸腔内ドメイン = 非肺 ∧ 非皮下 ∧ 非骨近傍
    domain = (~lung) & (~subcut) & (~bone_nbhd)
    
    # ドメインの統計
    domain_voxels = np.sum(domain)
    total_voxels = np.prod(ct_shape)
    print(f"[INFO] Domain mask created: {domain_voxels:,}/{total_voxels:,} voxels "
          f"({100*domain_voxels/total_voxels:.1f}%)")
    
    return domain

def keep_proximal_vessels(vessel_mask, heart_core, spacing, keep_mm=8.0, z_margin=2):
    """
    血管マスクを心臓近位部のみに限定
    心膜は大血管の根部のみを覆うため
    """
    if not vessel_mask.any():
        return vessel_mask
    
    # 心筋+4腔からの距離
    dist_to_heart = distance_transform_edt(~heart_core, sampling=spacing)
    
    # 近位部のみ保持（心臓から8mm以内）
    proximal = vessel_mask & (dist_to_heart <= keep_mm)
    
    # Z範囲も心臓コアの範囲±2スライスに限定
    z_any = np.any(heart_core, axis=(0, 1))
    if np.any(z_any):
        z_indices = np.where(z_any)[0]
        z0 = max(0, z_indices[0] - z_margin)
        z1 = min(vessel_mask.shape[2] - 1, z_indices[-1] + z_margin)
        proximal[:, :, :z0] = False
        proximal[:, :, z1+1:] = False
    
    return proximal

def create_heart_roi_with_domain_limited_edt(totalseg_dir, ct_nifti_path):
    """
    改善版v3：心臓ROI作成とドメイン限定EDT
    """
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    print("[INFO] Loading heart components...")
    
    # 心臓コンポーネント（totalタスクとheartchambersタスクの両対応）
    comp_names = [
        "heart",  # totalタスク
        "heart_myocardium",
        "heart_ventricle_left", "heart_ventricle_right",
        "heart_atrium_left", "heart_atrium_right"
    ]
    
    heart_parts = []
    loaded_comps = []
    for name in comp_names:
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if p.exists():
            heart_parts.append(nib.load(str(p)).get_fdata() > 0)
            loaded_comps.append(name)
            print(f"  Loaded: {name}")
    
    if not heart_parts:
        print("[ERROR] No heart components found!")
        return None, None, None, None, None, None, None
    
    # 心臓全体（心筋+4腔）
    heart_core = np.logical_or.reduce(heart_parts) if heart_parts else np.zeros_like(ct_data, dtype=bool)
    
    # 大血管（形状補助用、近位部のみ使用）
    vessel_names = ["aorta", "pulmonary_artery", "pulmonary_vein"]
    vessel_parts = []
    for name in vessel_names:
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if p.exists():
            vessel = nib.load(str(p)).get_fdata() > 0
            # 血管を心臓近位部のみに限定
            vessel_proximal = keep_proximal_vessels(vessel, heart_core, spacing, keep_mm=8.0, z_margin=2)
            vessel_parts.append(vessel_proximal)
            print(f"  Loaded vessel: {name} (proximal only)")
    
    # EDT計算用の心臓集合（近位血管のみを含む）
    if vessel_parts:
        heart_all = heart_core | np.logical_or.reduce(vessel_parts)
    else:
        heart_all = heart_core
    
    # 心筋（除外用）
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
        print("[INFO] Using heart_myocardium for exclusion")
    else:
        myo = heart_core  # フォールバック
        print("[INFO] Using full heart as myocardium (fallback)")
    
    # 3D穴埋めと2D/3Dクロージング
    print("[INFO] Applying 3D hole filling...")
    heart_all = binary_fill_holes(heart_all)
    
    # 軽い3Dクロージング（微小な穴を塞ぐ）
    print("[INFO] Applying 3D closing...")
    se3d_small = generate_binary_structure(3, 1)  # 6連結
    iterations_3d = max(1, int(np.ceil(1.0 / min(spacing))))
    heart_all = binary_closing(heart_all, structure=se3d_small, iterations=iterations_3d)
    
    print("[INFO] Applying 2D closing per slice...")
    rad_mm = 1.5
    rx = _mm_to_px(rad_mm, spacing[:2])
    ry = rx
    se2d = np.ones((2*ry+1, 2*rx+1), dtype=bool)
    
    for z in range(heart_all.shape[2]):
        if z % 50 == 0:
            print(f"  Processing slice {z}/{heart_all.shape[2]}...")
        heart_all[:, :, z] = binary_closing(heart_all[:, :, z], structure=se2d)
    
    # ドメインマスク作成
    domain = build_domain_mask(totalseg_dir, ct_data.shape, spacing)
    
    # 心臓表面からの距離を計算し、後でドメインを適用
    print("[INFO] Computing EDT from heart surface (domain applied afterward)...")
    dist_mm = distance_transform_edt(~heart_all, sampling=spacing)  # 背景=heart_allのみ
    dist_mm[~domain] = np.inf  # 胸腔外は無効化
    
    # 適応厚推定用バンド（自動的にドメイン内限定）
    band20 = (dist_mm > 0) & (dist_mm <= 20.0)
    
    return dist_mm, band20, myo, heart_core, heart_all, spacing, ct_data, domain

def build_adaptive_shell_with_connectivity(dist_mm, band20, heart_core, heart_all, 
                                          ct_data, spacing, domain,
                                          hu_min=-190, hu_max=-30,
                                          adaptive_cap_mm=10.0, seed_shell_mm=6.0):
    """
    スライス別適応型シェル厚と心臓表面連結制約
    """
    print("[INFO] Computing adaptive shell thickness per slice...")
    print(f"  HU range for fat: {hu_min} to {hu_max}")
    print(f"  Thickness cap: {adaptive_cap_mm} mm")
    
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    shell = np.zeros_like(dist_mm, dtype=bool)
    
    # Z範囲：心筋+4腔（heart_core）のみで決定
    z_any = np.any(heart_core, axis=(0, 1))
    z_idx = np.where(z_any)[0]
    
    if z_idx.size == 0:
        print("[WARNING] No heart_core slices found")
        return shell, seed_shell_mm
    
    z0, z1 = int(z_idx[0]), int(z_idx[-1])
    print(f"[INFO] Z-range clipping to heart_core: slice {z0} to {z1}")
    
    # スライスごとの適応厚
    thickness_per_slice = []
    for z in range(z0, z1 + 1):
        # このスライスのバンド内脂肪距離
        slice_band = band20[:, :, z]
        slice_fat = hu_fat[:, :, z]
        dv = dist_mm[:, :, z][slice_band & slice_fat]
        
        if dv.size > 100:  # 十分なサンプル
            t = float(np.percentile(dv, 85.0))  # 85パーセンタイル
            t = min(max(t, seed_shell_mm), adaptive_cap_mm)
        else:
            t = seed_shell_mm
        
        thickness_per_slice.append(t)
        
        # このスライスのシェル
        shell[:, :, z] = (dist_mm[:, :, z] > 0) & (dist_mm[:, :, z] <= t) & domain[:, :, z]
    
    avg_thickness = np.mean(thickness_per_slice)
    print(f"[INFO] Average shell thickness: {avg_thickness:.2f} mm")
    print(f"  Min: {min(thickness_per_slice):.2f} mm")
    print(f"  Max: {max(thickness_per_slice):.2f} mm")
    
    # 心臓表面からの連結制約
    print("[INFO] Applying connectivity constraint from heart surface...")
    se3d = generate_binary_structure(3, 2)
    
    # 心臓コア表面（heart_coreの1ボクセル外側）を種に（血管からの伝播を防ぐ）
    border = binary_dilation(heart_core, structure=se3d) & (~heart_core) & domain
    
    # 心臓表面から到達可能な部分のみ残す
    shell_connected = binary_propagation(input=border, mask=shell, structure=se3d)
    
    # propagationがうまく機能しない場合は元のシェルとのANDを取る
    shell_connected = shell_connected & shell
    
    # 連結前後の比較
    before_voxels = np.sum(shell)
    after_voxels = np.sum(shell_connected)
    print(f"[INFO] Connectivity constraint: {before_voxels:,} → {after_voxels:,} voxels")
    print(f"  Removed: {before_voxels-after_voxels:,} voxels ({100*(before_voxels-after_voxels)/max(before_voxels,1):.1f}%)")
    
    # 連結成分の確認
    labeled, num_components = label(shell_connected)
    print(f"[INFO] Shell components after connectivity: {num_components}")
    
    return shell_connected, avg_thickness

def extract_eat_with_fallback(ct_data, harta_volume, shell, myo,
                             hu_min=-190, hu_max=-30,
                             use_harta_if_helpful=True,
                             min_harta_overlap_ratio=0.15):
    """EAT抽出（HARTAの条件付き使用）"""
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
        print("  HARTA not used")
    
    final_voxels = np.sum(eat)
    print(f"  Final EAT: {final_voxels:,} voxels")
    
    # EAT密度（平均HU）の計算
    eat_hu_values = ct_data[eat]
    if eat_hu_values.size > 0:
        eat_mean_hu = float(np.mean(eat_hu_values))
        eat_std_hu = float(np.std(eat_hu_values))
        print(f"  EAT density: {eat_mean_hu:.1f} ± {eat_std_hu:.1f} HU")
    else:
        eat_mean_hu = np.nan
        eat_std_hu = np.nan
    
    return eat, eat_mean_hu, eat_std_hu

def calculate_volume_ml(mask, spacing):
    """体積計算"""
    voxel_vol_mm3 = np.prod(spacing)
    voxel_count = np.sum(mask)
    volume_ml = voxel_count * voxel_vol_mm3 / 1000.0
    return volume_ml, int(voxel_count)

def save_results(eat_mask, shell, ct_img, output_dir, shell_thickness_mm,
                hu_min, hu_max, eat_mean_hu, eat_std_hu, harta_voxels=None):
    """結果保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    spacing = ct_img.header.get_zooms()[:3]
    
    # マスク保存
    eat_img = nib.Nifti1Image(eat_mask.astype(np.uint8), ct_img.affine)
    eat_path = Path(output_dir) / "eat_mask.nii.gz"
    nib.save(eat_img, eat_path)
    
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
        "eat_mean_hu": float(eat_mean_hu) if not np.isnan(eat_mean_hu) else None,
        "eat_std_hu": float(eat_std_hu) if not np.isnan(eat_std_hu) else None,
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
    if not np.isnan(eat_mean_hu):
        print(f"  EAT Density: {eat_mean_hu:.1f} ± {eat_std_hu:.1f} HU")
    print(f"  Shell Thickness: {shell_thickness_mm:.2f} mm (avg)")
    print(f"  Shell Voxels: {shell_voxels:,}")
    print(f"  HU Range: {hu_min} to {hu_max}")
    print(f"  Output: {output_dir}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="改善版v3 EAT抽出（過拡張対策）")
    parser.add_argument("ct_nifti", help="入力CT NIfTI")
    parser.add_argument("totalseg_dir", help="TotalSegmentator出力ディレクトリ")
    parser.add_argument("output_dir", help="出力ディレクトリ")
    parser.add_argument("--harta-png-dir", help="HARTA PNG出力ディレクトリ")
    parser.add_argument("--slice-map", help="スライスマップJSON")
    parser.add_argument("--hu-min", type=float, default=-190,
                      help="脂肪HU下限（デフォルト: -190）")
    parser.add_argument("--hu-max", type=float, default=-30,
                      help="脂肪HU上限（デフォルト: -30、単純CT）")
    parser.add_argument("--shell-max", type=float, default=10.0,
                      help="適応シェル厚上限mm（デフォルト: 10）")
    parser.add_argument("--shell-seed", type=float, default=6.0,
                      help="フォールバックシェル厚mm（デフォルト: 6）")
    parser.add_argument("--harta-min-overlap", type=float, default=0.15,
                      help="HARTA使用の最小オーバーラップ率（デフォルト: 0.15）")
    parser.add_argument("--no-harta", action="store_true",
                      help="HARTAを使用しない")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("IMPROVED EAT EXTRACTION v3 (Domain-Limited)")
    print("="*60)
    
    # 1. 心臓ROIとドメイン限定EDT
    result = create_heart_roi_with_domain_limited_edt(
        args.totalseg_dir,
        args.ct_nifti
    )
    
    if result[0] is None:
        print("[ERROR] Failed to create heart ROI")
        sys.exit(1)
    
    dist_mm, band20, myo, heart_core, heart_all, spacing, ct_data, domain = result
    
    # 2. スライス別適応シェルと連結制約
    shell, shell_thickness = build_adaptive_shell_with_connectivity(
        dist_mm, band20, heart_core, heart_all, ct_data, spacing, domain,
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
    
    # 4. EAT抽出
    eat, eat_mean_hu, eat_std_hu = extract_eat_with_fallback(
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
        eat_mean_hu, eat_std_hu, harta_voxels
    )

if __name__ == "__main__":
    main()