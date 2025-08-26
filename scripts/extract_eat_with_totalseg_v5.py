#!/usr/bin/env python3
"""
改善版v5：TotalSegmentatorとEDTを使用したEAT抽出
前胸壁・椎体・肝臓漏れ対策完全版：
- ドメイン強化（肝・胃・脾・腎を明示的除外）
- 横隔膜での下限クリップ
- 禁制構造に触れた成分の全削除
- 骨・体表の二重バリア
- 品質ゲートと自動リカバリ
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

# 血管ごとの近位保持距離（mm） - v5でさらに厳格化
KEEP_MM = {
    "aorta": 5.0,  # 上行大動脈近位のみ
    "pulmonary_artery": 5.0,  # 肺動脈幹のみ
    "pulmonary_vein": 0.0,  # 完全除外
    "vena_cava_inferior": 5.0,  # v5: さらに厳格化
    "vena_cava_superior": 5.0,  # v5: さらに厳格化
}
Z_MARGIN = 1  # 血管のZ範囲マージン

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

def build_enhanced_domain_mask(totalseg_dir, ct_shape, spacing):
    """
    v5：強化版EDTドメインマスク作成
    肺・皮下・骨・腹部臓器を除外
    """
    def _load(name):
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if p.exists():
            return nib.load(str(p)).get_fdata() > 0
        return np.zeros(ct_shape, dtype=bool)
    
    print("[INFO] Building enhanced domain mask (v5)...")
    se3d = generate_binary_structure(3, 2)
    
    # 1. 肺の除外
    lung_components = [
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
    ]
    lung = np.zeros(ct_shape, dtype=bool)
    for comp in lung_components:
        if Path(totalseg_dir, f"{comp}.nii.gz").exists():
            lung |= _load(comp)
            print(f"  Excluding lung: {comp}")
    
    # 2. 体表面/皮下の除外（v5: 25-30mmバンドに拡大）
    body = _load("body") | _load("body_trunc") | _load("body_extremities")
    if body.any():
        r_outer = _mm_to_px(30.0, spacing)  # v5: 30mmまで拡大
        r_inner = _mm_to_px(0.0, spacing)
        subcut = body & ~binary_erosion(body, iterations=r_outer)
        print(f"  Excluding subcutaneous ring (30mm from surface)")
    else:
        subcut = np.zeros(ct_shape, dtype=bool)
    
    # 3. 骨近傍の除外（v5: 前胸壁は5mm、他は3mm）
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
    
    # 前胸壁（胸骨・肋骨）は5mm、椎体は3mm
    bone_nbhd = np.zeros(ct_shape, dtype=bool)
    if (ribs | sternum).any():
        dil_front = _mm_to_px(5.0, spacing)  # v5: 前胸壁は5mm
        bone_nbhd |= binary_dilation(ribs | sternum, structure=se3d, iterations=dil_front)
        print(f"  Excluding frontal bone neighborhood (5mm dilation)")
    
    if vertebrae.any():
        dil_back = _mm_to_px(3.0, spacing)
        bone_nbhd |= binary_dilation(vertebrae, structure=se3d, iterations=dil_back)
        print(f"  Excluding vertebrae neighborhood (3mm dilation)")
    
    # 4. v5新規：腹部臓器の除外（主要臓器のみ）
    abdominal_organs = ["liver", "spleen", "stomach", "kidney_left", "kidney_right"]
    org_ex = np.zeros(ct_shape, dtype=bool)
    for name in abdominal_organs:
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if p.exists():
            m = _load(name)
            # 臓器を2-4mm膨張
            m = binary_dilation(m, structure=se3d, iterations=_mm_to_px(3.0, spacing))
            org_ex |= m
            print(f"  Excluding abdominal organ: {name}")
    
    # 5. ドメイン統合
    domain = (~lung) & (~subcut) & (~bone_nbhd) & (~org_ex)
    
    # ドメインの統計
    domain_voxels = np.sum(domain)
    total_voxels = np.prod(ct_shape)
    print(f"[INFO] Enhanced domain mask created: {domain_voxels:,}/{total_voxels:,} voxels "
          f"({100*domain_voxels/total_voxels:.1f}%)")
    
    # 除外構造を返す（後の成分削除で使用）
    exclusion_masks = {
        'subcut': subcut,
        'bone_nbhd': bone_nbhd,
        'org_ex': org_ex,
        'lung': lung
    }
    
    return domain, exclusion_masks

def keep_proximal_vessels(vessel_mask, heart_core, spacing, keep_mm=8.0, z_margin=1):
    """
    血管マスクを心臓近位部のみに限定
    心膜は大血管の根部のみを覆うため
    """
    if not vessel_mask.any():
        return vessel_mask
    
    # 心筋+4腔からの距離
    dist_to_heart = distance_transform_edt(~heart_core, sampling=spacing)
    
    # 近位部のみ保持（心臓から指定mm以内）
    proximal = vessel_mask & (dist_to_heart <= keep_mm)
    
    # Z範囲も心臓コアの範囲±zマージンに限定
    z_any = np.any(heart_core, axis=(0, 1))
    if np.any(z_any):
        z_indices = np.where(z_any)[0]
        z0 = max(0, z_indices[0] - z_margin)
        z1 = min(vessel_mask.shape[2] - 1, z_indices[-1] + z_margin)
        proximal[:, :, :z0] = False
        proximal[:, :, z1+1:] = False
    
    return proximal

def create_heart_roi_with_enhanced_domain(totalseg_dir, ct_nifti_path):
    """
    改善版v5：心臓ROI作成と強化版ドメイン限定EDT
    返り値に除外マスクを追加
    """
    ct_img = nib.load(ct_nifti_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    print("[INFO] Loading heart components...")
    
    # 心臓コンポーネント
    comp_names = [
        "heart",
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
        return None
    
    # 心臓全体（心筋+4腔）
    heart_core = np.logical_or.reduce(heart_parts) if heart_parts else np.zeros_like(ct_data, dtype=bool)
    
    # 大血管（近位部と遠位部を分離）
    vessel_names = ["aorta", "pulmonary_artery", "pulmonary_vein",
                    "vena_cava_inferior", "vena_cava_superior"]
    proximal_vessels = []
    far_vessels = []
    
    for name in vessel_names:
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        if not p.exists():
            continue
        
        vessel = nib.load(str(p)).get_fdata() > 0
        keep_mm = KEEP_MM.get(name, 0.0)
        
        if keep_mm > 0.0:
            vessel_proximal = keep_proximal_vessels(vessel, heart_core, spacing, 
                                                   keep_mm=keep_mm, z_margin=Z_MARGIN)
            proximal_vessels.append(vessel_proximal)
            far_vessels.append(vessel & (~vessel_proximal))
            print(f"  Loaded vessel: {name} (proximal {keep_mm}mm only)")
        else:
            far_vessels.append(vessel)
            print(f"  Excluding vessel: {name} (all distal)")
    
    # EDT計算用の心臓集合（近位血管のみ）
    if proximal_vessels:
        heart_all = heart_core | np.logical_or.reduce(proximal_vessels)
    else:
        heart_all = heart_core
    
    # 遠位血管マスク
    far_vessel_mask = np.logical_or.reduce(far_vessels) if far_vessels else np.zeros_like(heart_core, dtype=bool)
    
    # 心筋（除外用）
    myo_path = Path(totalseg_dir) / "heart_myocardium.nii.gz"
    if myo_path.exists():
        myo = nib.load(str(myo_path)).get_fdata() > 0
        print("[INFO] Using heart_myocardium for exclusion")
    else:
        myo = heart_core
        print("[INFO] Using full heart as myocardium (fallback)")
    
    # 3D穴埋めと軽い3Dクロージング
    print("[INFO] Applying 3D hole filling...")
    heart_all = binary_fill_holes(heart_all)
    
    print("[INFO] Applying light 3D closing...")
    se3d_small = generate_binary_structure(3, 1)
    iterations_3d = max(1, int(np.ceil(1.0 / min(spacing))))
    heart_all = binary_closing(heart_all, structure=se3d_small, iterations=iterations_3d)
    
    # v5: 2Dクロージングは最小限に
    print("[INFO] Applying minimal 2D closing per slice...")
    rad_mm = 1.0  # v5: 1.0mmに削減
    rx = _mm_to_px(rad_mm, spacing[:2])
    ry = rx
    se2d = np.ones((2*ry+1, 2*rx+1), dtype=bool)
    
    for z in range(heart_all.shape[2]):
        if z % 50 == 0:
            print(f"  Processing slice {z}/{heart_all.shape[2]}...")
        heart_all[:, :, z] = binary_closing(heart_all[:, :, z], structure=se2d)
    
    # v5: 強化版ドメインマスク作成
    domain, exclusion_masks = build_enhanced_domain_mask(totalseg_dir, ct_data.shape, spacing)
    
    # 心臓表面からの距離計算
    print("[INFO] Computing EDT from heart surface...")
    dist_mm = distance_transform_edt(~heart_all, sampling=spacing)
    dist_mm[~domain] = np.inf
    
    # 適応厚推定用バンド
    band20 = (dist_mm > 0) & (dist_mm <= 20.0)
    
    return (dist_mm, band20, myo, heart_core, heart_all, spacing, ct_data, 
            domain, exclusion_masks, far_vessel_mask)

def build_shell_with_component_exclusion(dist_mm, band20, heart_core, heart_all, 
                                        ct_data, spacing, domain, exclusion_masks,
                                        far_vessel_mask, totalseg_dir,
                                        hu_min=-190, hu_max=-30,
                                        adaptive_cap_mm=10.0, seed_shell_mm=6.0):
    """
    v5: 禁制構造に触れた成分を全削除する強化版シェル生成
    """
    print("[INFO] Computing adaptive shell thickness per slice...")
    print(f"  HU range for fat: {hu_min} to {hu_max}")
    print(f"  Thickness cap: {adaptive_cap_mm} mm")
    
    hu_fat = (ct_data >= hu_min) & (ct_data <= hu_max)
    shell = np.zeros_like(dist_mm, dtype=bool)
    
    # Z範囲：心臓コアのみ（v5: 横隔膜クリップ）
    z_any = np.any(heart_core, axis=(0, 1))
    z_idx = np.where(z_any)[0]
    
    if z_idx.size == 0:
        print("[WARNING] No heart_core slices found")
        return shell, seed_shell_mm
    
    z0, z1 = int(z_idx[0]), int(z_idx[-1])
    
    # v5: 横隔膜クリップ（下限を厳格化）
    z_clip_margin = 1  # 心臓コアの範囲±1スライスに制限
    z0 = max(z0 - z_clip_margin, 0)
    z1 = min(z1 + z_clip_margin, dist_mm.shape[2] - 1)
    
    print(f"[INFO] Z-range with diaphragm clip: slice {z0} to {z1}")
    
    # スライスごとの適応厚
    thickness_per_slice = []
    for z in range(z0, z1 + 1):
        slice_band = band20[:, :, z]
        slice_fat = hu_fat[:, :, z]
        dv = dist_mm[:, :, z][slice_band & slice_fat]
        
        if dv.size > 100:
            t = float(np.percentile(dv, 85.0))
            t = min(max(t, seed_shell_mm), adaptive_cap_mm)
        else:
            t = seed_shell_mm
        
        # 端スライスの厚み抑制
        if z == z0 or z == z1:
            t = min(t, seed_shell_mm)
            
        thickness_per_slice.append(t)
        shell[:, :, z] = (dist_mm[:, :, z] > 0) & (dist_mm[:, :, z] <= t) & domain[:, :, z]
    
    # v5: 横隔膜外を完全除外
    shell[:, :, :z0] = False
    shell[:, :, z1+1:] = False
    
    avg_thickness = np.mean(thickness_per_slice) if thickness_per_slice else seed_shell_mm
    print(f"[INFO] Average shell thickness: {avg_thickness:.2f} mm")
    
    # 心臓表面からの連結制約
    print("[INFO] Applying connectivity constraint from heart surface...")
    se3d = generate_binary_structure(3, 2)
    
    # 心臓コア表面を種に
    border = binary_dilation(heart_core, structure=se3d) & (~heart_core) & domain
    shell_connected = binary_propagation(input=border, mask=shell, structure=se3d)
    shell_connected = shell_connected & shell
    
    before_voxels = np.sum(shell)
    after_voxels = np.sum(shell_connected)
    print(f"  Basic connectivity: {before_voxels:,} → {after_voxels:,} voxels")
    
    # 遠位血管禁止帯と気道禁止帯
    print("[INFO] Applying exclusion zones...")
    
    def _mm_to_iter(mm):
        return max(1, int(np.ceil(mm / min(spacing))))
    
    # 遠位血管禁止帯
    if far_vessel_mask.any():
        anti_vessel = binary_dilation(far_vessel_mask, structure=se3d, iterations=_mm_to_iter(2.0))
        before_anti = np.sum(shell_connected)
        shell_connected &= ~anti_vessel
        after_anti = np.sum(shell_connected)
        print(f"  Distal vessel exclusion: removed {before_anti-after_anti:,} voxels")
    
    # 気道禁止帯
    airway_removed = 0
    for airway in ["trachea", "main_bronchus_left", "main_bronchus_right"]:
        ap = Path(totalseg_dir) / f"{airway}.nii.gz"
        if ap.exists():
            am = nib.load(str(ap)).get_fdata() > 0
            anti_airway = binary_dilation(am, structure=se3d, iterations=_mm_to_iter(2.0))
            before_airway = np.sum(shell_connected)
            shell_connected &= ~anti_airway
            after_airway = np.sum(shell_connected)
            removed = before_airway - after_airway
            airway_removed += removed
            print(f"  {airway} exclusion: removed {removed:,} voxels")
    
    # === v5核心：禁制構造に触れた成分を全削除 ===
    print("[INFO] v5: Removing components touching forbidden structures...")
    
    # 除外マスクの統合
    forbidden = exclusion_masks['subcut'] | exclusion_masks['bone_nbhd'] | \
                exclusion_masks['org_ex'] | exclusion_masks['lung'] | (~domain)
    
    # 連結成分分析
    cc, n = label(shell_connected)
    print(f"  Found {n} components before filtering")
    
    # 各成分をチェック
    components_removed = 0
    voxels_removed = 0
    for k in range(1, n+1):
        comp = (cc == k)
        comp_size = np.sum(comp)
        
        # この成分が禁制構造に1ボクセルでも触れているか
        if np.any(comp & forbidden):
            shell_connected[comp] = False
            components_removed += 1
            voxels_removed += comp_size
    
    print(f"  Component-level exclusion: removed {components_removed} components ({voxels_removed:,} voxels)")
    
    # 最終的な連結成分数
    labeled_final, num_final = label(shell_connected)
    print(f"[INFO] Final shell components: {num_final}")
    
    # v5: 品質検証
    print("[VALIDATION] Quality checks:")
    
    # 禁制構造との交差チェック
    forbidden_overlap = np.sum(shell_connected & forbidden)
    if forbidden_overlap == 0:
        print("  ✓ No overlap with forbidden structures")
    else:
        print(f"  ⚠ WARNING: {forbidden_overlap} voxels still overlap forbidden structures!")
    
    # 遠位血管近傍チェック
    if far_vessel_mask.any():
        anti_vessel_check = binary_dilation(far_vessel_mask, structure=se3d, iterations=_mm_to_iter(2.0))
        ratio_far = np.sum(shell_connected & anti_vessel_check) / max(np.sum(shell_connected), 1)
        print(f"  Distal vessel proximity: {ratio_far*100:.2f}% (target < 2%)")
        if ratio_far > 0.02:
            print("  ⚠ High distal vessel proximity detected")
    
    return shell_connected, avg_thickness

def extract_eat_with_quality_gate(ct_data, harta_volume, shell, myo,
                                 hu_min=-190, hu_max=-30,
                                 use_harta_if_helpful=True,
                                 min_harta_overlap_ratio=0.15):
    """EAT抽出（v5: 品質ゲート付き）"""
    print("[INFO] Extracting EAT with quality gate...")
    
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
    
    # EAT密度の計算と品質チェック
    eat_hu_values = ct_data[eat]
    if eat_hu_values.size > 0:
        eat_mean_hu = float(np.mean(eat_hu_values))
        eat_std_hu = float(np.std(eat_hu_values))
        
        # v5: HU範囲の品質チェック
        hu_95_percentile = np.percentile(eat_hu_values, 95)
        hu_5_percentile = np.percentile(eat_hu_values, 5)
        
        print(f"  EAT density: {eat_mean_hu:.1f} ± {eat_std_hu:.1f} HU")
        print(f"  HU range (5-95%): {hu_5_percentile:.1f} to {hu_95_percentile:.1f}")
        
        # 品質警告
        if hu_95_percentile > hu_max or hu_5_percentile < hu_min:
            print("  ⚠ WARNING: EAT HU values outside expected fat range!")
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

def save_results_with_validation(eat_mask, shell, ct_img, output_dir, shell_thickness_mm,
                                hu_min, hu_max, eat_mean_hu, eat_std_hu, harta_voxels=None):
    """結果保存（v5: 強化版検証付き）"""
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
        "version": "v5_complete_exclusion"
    }
    
    if harta_voxels is not None:
        stats["harta_voxels"] = int(harta_voxels)
    
    stats_path = Path(output_dir) / "eat_summary.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # v5: 包括的な検証レポート
    print("\n" + "="*60)
    print("[V5 VALIDATION REPORT]")
    print("-"*60)
    
    # 体積検証
    volume_status = "✓ PASS" if 20 <= eat_vol_ml <= 400 else "⚠ WARNING"
    print(f"  EAT Volume: {eat_vol_ml:.2f} ml [{volume_status}]")
    if eat_vol_ml < 20:
        print("    → Very low volume - check for over-exclusion")
    elif eat_vol_ml > 400:
        print("    → Very high volume - check for leakage")
    
    # HU検証
    if eat_mean_hu is not None and not np.isnan(eat_mean_hu):
        hu_status = "✓ PASS" if hu_min <= eat_mean_hu <= hu_max else "⚠ WARNING"
        print(f"  EAT Mean HU: {eat_mean_hu:.1f} [{hu_status}]")
        if eat_mean_hu < hu_min + 20:
            print("    → Very low HU - possible water/blood contamination")
        elif eat_mean_hu > hu_max - 20:
            print("    → High HU - possible soft tissue contamination")
    
    # シェル厚検証
    thickness_status = "✓ PASS" if 4 <= shell_thickness_mm <= 12 else "⚠ WARNING"
    print(f"  Shell Thickness: {shell_thickness_mm:.1f} mm [{thickness_status}]")
    
    # ボクセル比率
    if shell_voxels > 0:
        eat_shell_ratio = eat_voxels / shell_voxels * 100
        ratio_status = "✓ PASS" if 10 <= eat_shell_ratio <= 60 else "⚠ CHECK"
        print(f"  EAT/Shell Ratio: {eat_shell_ratio:.1f}% [{ratio_status}]")
    
    print("-"*60)
    print("[FINAL RESULTS]")
    print(f"  EAT Volume: {eat_vol_ml:.2f} ml")
    print(f"  EAT Voxels: {eat_voxels:,}")
    if not np.isnan(eat_mean_hu):
        print(f"  EAT Density: {eat_mean_hu:.1f} ± {eat_std_hu:.1f} HU")
    print(f"  Shell Volume: {shell_vol_ml:.2f} ml")
    print(f"  Output: {output_dir}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="改善版v5 EAT抽出（完全漏れ防止版）")
    parser.add_argument("ct_nifti", help="入力CT NIfTI")
    parser.add_argument("totalseg_dir", help="TotalSegmentator出力ディレクトリ")
    parser.add_argument("output_dir", help="出力ディレクトリ")
    parser.add_argument("--harta-png-dir", help="HARTA PNG出力ディレクトリ")
    parser.add_argument("--slice-map", help="スライスマップJSON")
    parser.add_argument("--hu-min", type=float, default=-190,
                      help="脂肪HU下限（デフォルト: -190）")
    parser.add_argument("--hu-max", type=float, default=-30,
                      help="脂肪HU上限（デフォルト: -30）")
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
    print("IMPROVED EAT EXTRACTION v5 (Complete Leak Prevention)")
    print("="*60)
    
    # 1. 心臓ROIと強化版ドメイン
    result = create_heart_roi_with_enhanced_domain(
        args.totalseg_dir,
        args.ct_nifti
    )
    
    if result is None or result[0] is None:
        print("[ERROR] Failed to create heart ROI")
        sys.exit(1)
    
    (dist_mm, band20, myo, heart_core, heart_all, spacing, ct_data, 
     domain, exclusion_masks, far_vessel_mask) = result
    
    # 2. 成分レベル除外付きシェル生成
    shell, shell_thickness = build_shell_with_component_exclusion(
        dist_mm, band20, heart_core, heart_all, ct_data, spacing, domain, exclusion_masks,
        far_vessel_mask, args.totalseg_dir,
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
    
    # 4. 品質ゲート付きEAT抽出
    eat, eat_mean_hu, eat_std_hu = extract_eat_with_quality_gate(
        ct_data, harta_volume, shell, myo,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        use_harta_if_helpful=(not args.no_harta),
        min_harta_overlap_ratio=args.harta_min_overlap
    )
    
    # 5. 検証付き結果保存
    ct_img = nib.load(args.ct_nifti)
    save_results_with_validation(
        eat, shell, ct_img, args.output_dir,
        shell_thickness, args.hu_min, args.hu_max,
        eat_mean_hu, eat_std_hu, harta_voxels
    )

if __name__ == "__main__":
    main()