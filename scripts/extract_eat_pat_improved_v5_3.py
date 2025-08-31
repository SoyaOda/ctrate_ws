#!/usr/bin/env python3
"""
改善版v5.3 EAT+PAT抽出：修正版優勢接触ILAM実装
Spec5.3.mdのフィードバックに基づく改善：
- 接触判定と禁止帯の順序を入れ替え（最重要）
- シードだけZ範囲で制限し、伝播は3D全域を許可
- 心膜近傍との連結性を採用条件に追加
- 横隔膜の小孔をもう少し確実に塞ぐ
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
from scipy.ndimage import (
    binary_dilation, generate_binary_structure,
    distance_transform_edt, label, binary_propagation, binary_closing
)

# 定数
DILATION_RADIUS_MM = 15.0  # 心臓マスク膨張半径 [mm] (EAT+PATを含む範囲)

# ILAM (Inferior Lung-Adjacency Mode) デフォルトパラメータ
DEFAULT_INFERIOR_BAND_MM = 50.0  # 下部帯域の高さ [mm]
DEFAULT_LUNG_CONTACT_MM = 0.5    # 肺接触判定の許容距離 [mm] (δ)
DEFAULT_ABDOMEN_GAP_MM = 0.0     # 腹部臓器除外の膨張距離 [mm] (γ)
DEFAULT_MAX_HEART_DIST_MM = 35.0 # 心臓からの最大距離 [mm] (ρ)

# v5.3 新パラメータ
DEFAULT_CONTACT_EPS_MM = 3.0      # 接触判定の等方膨張 [mm]
DEFAULT_LUNG_DOM_RATIO = 1.0      # 肺接触が腹部臓器より何倍優勢か
DEFAULT_MIN_LUNG_TOUCH_VOX = 100  # 肺接触の最小ボクセル数
DEFAULT_POST_TRIM_MM = 1.5        # 最終整形用の安全帯 [mm]
DEFAULT_SEED_Z_MARGIN_MM = 10.0   # シード生成のZ範囲マージン [mm]
DEFAULT_SEAL_DIAPHRAGM_MM = 2.0   # 横隔膜穴塞ぎ [mm]
DEFAULT_REQUIRE_SHELL_TOUCH = True # Shell接触を必須とするか

def check_required_masks(totalseg_dir):
    """
    必要なマスクファイルの存在を確認
    
    Returns:
        missing_masks: 不足しているマスクのリスト
        mask_status: 各マスクの存在状況
    """
    totalseg_dir = Path(totalseg_dir)
    
    # 必須マスク
    required_masks = {
        'heart': ['heart_myocardium.nii.gz', 'myocardium.nii.gz', 'heart.nii.gz'],
        'lung_left': ['lung_left.nii.gz'],
        'lung_right': ['lung_right.nii.gz'],
        'torso_fat': ['torso_fat.nii.gz'],
        'liver': ['liver.nii.gz'],
        'stomach': ['stomach.nii.gz'],
        'spleen': ['spleen.nii.gz']
    }
    
    # 推奨マスク（オプション）
    optional_masks = {
        'aorta': ['aorta.nii.gz'],
        'pulmonary_artery': ['pulmonary_artery.nii.gz'],
        'inferior_vena_cava': ['inferior_vena_cava.nii.gz', 'vena_cava_inferior.nii.gz'],
        'pulmonary_vein': ['pulmonary_vein.nii.gz']
    }
    
    missing_required = []
    missing_optional = []
    mask_status = {}
    
    # 必須マスクのチェック
    for mask_type, filenames in required_masks.items():
        found = False
        for filename in filenames:
            if (totalseg_dir / filename).exists():
                mask_status[mask_type] = filename
                found = True
                break
        if not found:
            missing_required.append(mask_type)
            mask_status[mask_type] = None
    
    # オプショナルマスクのチェック
    for mask_type, filenames in optional_masks.items():
        found = False
        for filename in filenames:
            if (totalseg_dir / filename).exists():
                mask_status[mask_type] = filename
                found = True
                break
        if not found:
            missing_optional.append(mask_type)
            mask_status[mask_type] = None
    
    return missing_required, missing_optional, mask_status

def load_heart_components(totalseg_dir, verbose=False):
    """
    心臓コンポーネントを読み込んで統合
    フォールバックなし - 見つからない場合はエラー
    """
    totalseg_dir = Path(totalseg_dir)
    heart_mask = None
    affine = None
    
    # v2/v1 highres（先頭にheart_が付かない）
    names_v2 = ["myocardium", "atrium_left", "atrium_right", "ventricle_left", "ventricle_right"]
    # 旧来のプレフィックスあり表記
    names_v1like = [f"heart_{n}" for n in names_v2]
    
    components_found = []
    
    def _try_add(name):
        nonlocal heart_mask, affine
        p = totalseg_dir / f"{name}.nii.gz"
        if p.exists():
            img = nib.load(str(p))
            data = img.get_fdata() > 0
            if heart_mask is None:
                heart_mask = data
                affine = img.affine
            else:
                heart_mask |= data
            components_found.append(name)
            if verbose: 
                print(f"  Loaded: {name}")
            return True
        return False
    
    # highres（v2名→v1名の順で探す）
    for nm in names_v2 + names_v1like:
        _try_add(nm)
    
    # totalタスクのheart
    if heart_mask is None:
        p = totalseg_dir / "heart.nii.gz"
        if p.exists():
            img = nib.load(str(p))
            heart_mask = img.get_fdata() > 0
            affine = img.affine
            components_found.append("heart")
            if verbose: 
                print("  Using total task 'heart'")
    
    if heart_mask is None:
        raise FileNotFoundError(
            "\n[ERROR] Heart mask not found!\n"
            "Required files (at least one):\n"
            f"  - {totalseg_dir}/heart_myocardium.nii.gz (+ other chambers)\n"
            f"  - {totalseg_dir}/myocardium.nii.gz (+ other chambers)\n"
            f"  - {totalseg_dir}/heart.nii.gz\n"
            "\nRun TotalSegmentator:\n"
            "  docker run --rm -v /path:/workspace wasserth/totalsegmentator:2.10.0 \\\n"
            "    TotalSegmentator -i /workspace/input.nii.gz -o /workspace/output --task total"
        )
    
    if verbose:
        print(f"  Components used: {components_found}")
        print(f"  Heart voxels: {int(np.sum(heart_mask)):,}")
        
    return heart_mask, affine

def load_vessel_masks(totalseg_dir, verbose=False):
    """
    大血管マスクを読み込み（オプション）
    """
    totalseg_dir = Path(totalseg_dir)
    vessels = {}
    
    vessel_mapping = {
        "aorta": ["aorta.nii.gz"],
        "pulmonary_artery": ["pulmonary_artery.nii.gz"],
        "inferior_vena_cava": ["inferior_vena_cava.nii.gz", "vena_cava_inferior.nii.gz"],
        "pulmonary_vein": ["pulmonary_vein.nii.gz"]
    }
    
    for vessel_name, filenames in vessel_mapping.items():
        for filename in filenames:
            vessel_path = totalseg_dir / filename
            if vessel_path.exists():
                if verbose:
                    print(f"  Loading {vessel_name} from {filename}...")
                img = nib.load(str(vessel_path))
                vessels[vessel_name] = img.get_fdata() > 0
                break
    
    return vessels

def compute_intrapericardial_zmax(totalseg_dir, heart_mask, spacing,
                                  prox_mm=5.0, z_margin_slices=1, verbose=False):
    """
    心膜内領域の上端Z座標を計算（近位大血管を考慮）
    """
    # EDTによる心臓からの距離計算
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    prox_band = dist_to_heart <= prox_mm

    def _load(name):
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        return (nib.load(str(p)).get_fdata() > 0) if p.exists() else None

    aorta = _load("aorta")
    pa = _load("pulmonary_artery")

    intrapericardial = heart_mask.copy()
    for v in [aorta, pa]:
        if v is not None:
            intrapericardial |= (v & prox_band)

    z_any = np.any(intrapericardial, axis=(0,1))
    z_idx = np.where(z_any)[0]
    if z_idx.size == 0:
        # 心臓だけで決定
        z_idx = np.where(np.any(heart_mask, axis=(0,1)))[0]
        if z_idx.size == 0:
            raise ValueError("Heart mask is empty!")
        zmax = z_idx[-1]
    else:
        zmax = min(z_idx[-1] + z_margin_slices, heart_mask.shape[2]-1)
    
    if verbose:
        print(f"  Intrapericardial zmax: {zmax}")
        if aorta is None and pa is None:
            print("    WARNING: No great vessels found, using heart only")
    
    return zmax

def compute_intrapericardial_zmin(totalseg_dir, heart_mask, spacing,
                                  prox_mm=5.0, z_margin_slices=1, verbose=False):
    """
    IVC、肺静脈、portal veinを使った心膜内最尾側Z座標の計算
    """
    # 心臓からの距離画像（EDT）を計算
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    prox_band = dist_to_heart <= prox_mm  # 5mm以内の近接帯
    
    def _load(name):
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        return (nib.load(str(p)).get_fdata() > 0) if p.exists() else None
    
    # IVC、肺静脈、portal veinを読み込み
    ivc = _load("inferior_vena_cava")
    if ivc is None:
        ivc = _load("vena_cava_inferior")
    
    pv = _load("pulmonary_vein")
    
    # Portal veinも追加（肝臓付近の下部指標として）
    portal = _load("portal_vein_and_splenic_vein")
    
    # 心臓近接の血管部分を取得
    prox_vessels = np.zeros_like(heart_mask, dtype=bool)
    if ivc is not None:
        prox_vessels |= (ivc & prox_band)
    if pv is not None:
        prox_vessels |= (pv & prox_band)
    if portal is not None:
        # Portal veinは心臓から少し遠いので10mm以内でチェック
        portal_prox_band = dist_to_heart <= 10.0
        prox_vessels |= (portal & portal_prox_band)
    
    # 血管が存在する最初のZ座標を探す（最も尾側）
    if np.any(prox_vessels):
        z_any = np.any(prox_vessels, axis=(0,1))
        z_idx = np.where(z_any)[0]
        if z_idx.size > 0:
            zmin = max(0, z_idx[0] - z_margin_slices)  # 最も尾側（小さいZ）
            if verbose:
                print(f"  Intrapericardial zmin (IVC/PV/Portal based): {zmin}")
                print(f"    IVC found: {ivc is not None}")
                print(f"    PV found: {pv is not None}")
                print(f"    Portal vein found: {portal is not None}")
            return zmin
    
    # IVC/PV/Portalがない場合：心臓の最下端を使用
    z_heart = np.where(np.any(heart_mask, axis=(0,1)))[0]
    if z_heart.size > 0:
        zmin = max(0, z_heart[0] - 3)  # 少しマージンを追加
        if verbose:
            print(f"  WARNING: IVC/PV/Portal not found, using heart bottom: {zmin}")
    else:
        zmin = 0
    
    return zmin

def load_abdominal_organs(totalseg_dir, verbose=False):
    """
    腹部臓器マスクを読み込み（肝・胃・脾）
    必須 - 見つからない場合はエラー
    """
    totalseg_dir = Path(totalseg_dir)
    abdominal_mask = None
    
    organ_names = ["liver", "stomach", "spleen"]
    organs_found = []
    organs_missing = []
    
    for organ in organ_names:
        organ_path = totalseg_dir / f"{organ}.nii.gz"
        if organ_path.exists():
            img = nib.load(str(organ_path))
            data = img.get_fdata() > 0
            if abdominal_mask is None:
                abdominal_mask = data
            else:
                abdominal_mask |= data
            organs_found.append(organ)
            if verbose:
                print(f"  Loaded {organ}")
        else:
            organs_missing.append(organ)
    
    if organs_missing:
        raise FileNotFoundError(
            f"\n[ERROR] Abdominal organ masks not found: {organs_missing}\n"
            f"Required files:\n"
            f"  - {totalseg_dir}/liver.nii.gz\n"
            f"  - {totalseg_dir}/stomach.nii.gz\n"
            f"  - {totalseg_dir}/spleen.nii.gz\n"
            "\nRun TotalSegmentator with total task:\n"
            "  docker run --rm -v /path:/workspace wasserth/totalsegmentator:2.10.0 \\\n"
            "    TotalSegmentator -i /workspace/input.nii.gz -o /workspace/output --task total"
        )
    
    if verbose and organs_found:
        print(f"  Abdominal organs found: {organs_found}")
    
    return abdominal_mask

def load_lung_masks(totalseg_dir, verbose=False):
    """
    肺マスクを読み込み（縦隔限定用）
    必須 - 見つからない場合はエラー
    """
    totalseg_dir = Path(totalseg_dir)
    lungs = None
    
    # 左右の肺を探す
    lung_left_path = totalseg_dir / "lung_left.nii.gz"
    lung_right_path = totalseg_dir / "lung_right.nii.gz"
    lung_combined_path = totalseg_dir / "lung.nii.gz"
    
    if lung_left_path.exists() and lung_right_path.exists():
        # 左右の肺がある場合
        img_left = nib.load(str(lung_left_path))
        img_right = nib.load(str(lung_right_path))
        lungs = (img_left.get_fdata() > 0) | (img_right.get_fdata() > 0)
        if verbose:
            print(f"  Loaded lung_left and lung_right")
    elif lung_combined_path.exists():
        # 統合肺マスクがある場合
        img = nib.load(str(lung_combined_path))
        lungs = img.get_fdata() > 0
        if verbose:
            print(f"  Loaded lung (combined)")
    else:
        raise FileNotFoundError(
            f"\n[ERROR] Lung masks not found!\n"
            f"Required files (at least one set):\n"
            f"  - {totalseg_dir}/lung_left.nii.gz AND lung_right.nii.gz\n"
            f"  - {totalseg_dir}/lung.nii.gz\n"
            "\nRun TotalSegmentator with total task:\n"
            "  docker run --rm -v /path:/workspace wasserth/totalsegmentator:2.10.0 \\\n"
            "    TotalSegmentator -i /workspace/input.nii.gz -o /workspace/output --task total"
        )
    
    return lungs

def incorporate_proximal_vessels(heart_mask, vessels, proximity_mm=5.0, spacing=(1.0, 1.0, 1.0), verbose=False):
    """
    心臓に近接する大血管の根元部分を心臓マスクに追加
    """
    if not vessels:
        return heart_mask
    
    # EDTによる心臓からの距離計算
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    heart_proximity = dist_to_heart <= proximity_mm
    
    # 各血管の近接部分を追加
    for vessel_name, vessel_mask in vessels.items():
        proximal_part = vessel_mask & heart_proximity
        if np.any(proximal_part):
            heart_mask |= proximal_part
            if verbose:
                print(f"  Added proximal {vessel_name}: {np.sum(proximal_part):,} voxels")
    
    return heart_mask

def load_visceral_fat_mask(totalseg_dir, verbose=False):
    """
    tissue_typesの出力から内臓脂肪(torso_fat)を取得
    必須 - 見つからない場合はエラー
    """
    totalseg_dir = Path(totalseg_dir)
    
    # torso_fat.nii.gz を探す
    torso_fat_path = totalseg_dir / "torso_fat.nii.gz"
    
    if torso_fat_path.exists():
        if verbose: 
            print(f"  Found torso_fat: {torso_fat_path}")
        img = nib.load(str(torso_fat_path))
        return (img.get_fdata() > 0), img.affine
    
    # サブフォルダも探す
    for p in totalseg_dir.rglob("torso_fat.nii.gz"):
        if verbose: 
            print(f"  Found torso_fat: {p}")
        img = nib.load(str(p))
        return (img.get_fdata() > 0), img.affine
    
    raise FileNotFoundError(
        "\n[ERROR] Visceral fat (torso_fat) not found!\n"
        f"Required file:\n"
        f"  - {totalseg_dir}/torso_fat.nii.gz\n"
        "\nRun TotalSegmentator with tissue_types task:\n"
        "  docker run --rm -v /path:/workspace wasserth/totalsegmentator:2.10.0 \\\n"
        "    TotalSegmentator -i /workspace/input.nii.gz -o /workspace/output \\\n"
        "    --task tissue_types -ta <LICENSE_KEY>\n"
        "\nNote: tissue_types task requires a license key (free for research)"
    )

def iso_dilate(mask, radius_mm, spacing):
    """
    等方距離膨張：maskからradius_mm以内をTrueにする
    EDT (Euclidean Distance Transform)を使用して物理的距離を正確に計算
    
    Args:
        mask: 入力マスク (bool array)
        radius_mm: 膨張半径 [mm]
        spacing: ボクセルスペーシング (x, y, z) [mm]
    
    Returns:
        膨張後のマスク
    """
    if radius_mm <= 0:
        return mask.copy()
    dist = distance_transform_edt(~mask, sampling=spacing)
    return dist <= radius_mm

def load_abdominal_organs_dict(totalseg_dir, ct_shape, verbose=False):
    """
    腹部臓器を個別に読み込み（優勢接触判定用）
    
    Args:
        totalseg_dir: TotalSegmentatorの出力ディレクトリ
        ct_shape: CTデータの形状
        verbose: 詳細出力
    
    Returns:
        dict: 臓器名をキーとするマスク辞書
    """
    totalseg_dir = Path(totalseg_dir)
    masks = {}
    
    for organ in ["liver", "stomach", "spleen"]:
        organ_path = totalseg_dir / f"{organ}.nii.gz"
        if organ_path.exists():
            img = nib.load(str(organ_path))
            masks[organ] = img.get_fdata() > 0
            if verbose:
                print(f"    Loaded {organ}: {np.sum(masks[organ]):,} voxels")
        else:
            masks[organ] = np.zeros(ct_shape, dtype=bool)
            if verbose:
                print(f"    Warning: {organ} not found, using empty mask")
    
    return masks

def extract_eat_pat_improved_v5_3(ct_path, totalseg_dir, output_dir, 
                                     dilation_radius_mm=DILATION_RADIUS_MM,
                                     use_proximal_vessels=True,
                                     # ILAM parameters
                                     inferior_band_mm=DEFAULT_INFERIOR_BAND_MM,
                                     lung_contact_mm=DEFAULT_LUNG_CONTACT_MM,
                                     abdomen_gap_mm=DEFAULT_ABDOMEN_GAP_MM,
                                     max_heart_dist_mm=DEFAULT_MAX_HEART_DIST_MM,
                                     replace_by_contact=False,
                                     # v5.3 new parameters
                                     no_z_limit=False,
                                     contact_eps_mm=DEFAULT_CONTACT_EPS_MM,
                                     lung_dom_ratio=DEFAULT_LUNG_DOM_RATIO,
                                     min_lung_touch_voxels=DEFAULT_MIN_LUNG_TOUCH_VOX,
                                     seal_diaphragm_holes_mm=DEFAULT_SEAL_DIAPHRAGM_MM,
                                     post_trim_mm=DEFAULT_POST_TRIM_MM,
                                     seed_z_margin_mm=DEFAULT_SEED_Z_MARGIN_MM,
                                     require_shell_touch=DEFAULT_REQUIRE_SHELL_TOUCH,
                                     verbose=False):
    """
    改善版v5.3 EAT+PAT（心膜周囲脂肪）抽出 with 修正版優勢接触ILAM
    接触判定と禁止帯の順序を入れ替え、シードZ制限、Shell接触要件を追加
    """
    print("\n[Improved EAT+PAT Extraction v5.3 with Fixed Dominant Contact ILAM]")
    print("="*60)
    
    # 1. 必要なマスクの確認
    print("1. Checking required masks...")
    missing_required, missing_optional, mask_status = check_required_masks(totalseg_dir)
    
    if missing_required:
        error_msg = (
            f"\n[ERROR] Required masks missing: {missing_required}\n"
            "\nYou need to run the following commands:\n"
            "\n1. For organ segmentation (heart, lungs, liver, stomach, spleen):\n"
            "   docker run --rm -v /home/soya/ctrate_ws:/workspace \\\n"
            "     wasserth/totalsegmentator:2.10.0 \\\n"
            "     TotalSegmentator -i /workspace/<ct_file>.nii.gz \\\n"
            "     -o /workspace/<output_dir> --task total\n"
            "\n2. For tissue types (torso_fat):\n"
            "   docker run --rm -v /home/soya/ctrate_ws:/workspace \\\n"
            "     wasserth/totalsegmentator:2.10.0 \\\n"
            "     TotalSegmentator -i /workspace/<ct_file>.nii.gz \\\n"
            "     -o /workspace/<output_dir> --task tissue_types -ta <LICENSE_KEY>\n"
            "\nNote: tissue_types requires a license key (free for research)"
        )
        raise FileNotFoundError(error_msg)
    
    if missing_optional and verbose:
        print(f"  Optional masks missing (will continue): {missing_optional}")
    
    print("  ✓ All required masks found")
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    masks_dir = output_dir / "masks"
    stats_dir = output_dir / "statistics"
    masks_dir.mkdir(exist_ok=True)
    stats_dir.mkdir(exist_ok=True)
    
    # 2. CT画像読み込み
    print("\n2. Loading CT data...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    spacing = tuple(float(s) for s in spacing)
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # 3. 心臓マスクの構築
    print("\n3. Building heart mask...")
    heart_mask, affine = load_heart_components(totalseg_dir, verbose)
    
    # オプション: 大血管近位部の追加
    if use_proximal_vessels:
        print("\n4. Incorporating proximal vessels...")
        vessels = load_vessel_masks(totalseg_dir, verbose)
        if vessels:
            heart_mask = incorporate_proximal_vessels(
                heart_mask, vessels, proximity_mm=5.0, spacing=spacing, verbose=verbose
            )
        else:
            print("  No vessel masks found (optional)")
    
    # 心臓マスクを保存
    heart_path = masks_dir / "heart.nii.gz"
    heart_img = nib.Nifti1Image(heart_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(heart_img, str(heart_path))
    print(f"  Heart mask saved: {heart_path}")
    
    # 4. 内臓脂肪マスクの読み込み（必須）
    print("\n5. Loading visceral fat mask (torso_fat)...")
    fat_mask, _ = load_visceral_fat_mask(totalseg_dir, verbose=verbose)
    print(f"  Visceral fat voxels: {int(np.sum(fat_mask)):,}")
    
    # 内臓脂肪マスクも保存
    visceral_fat_path = masks_dir / "visceral_fat.nii.gz"
    visceral_fat_img = nib.Nifti1Image(fat_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(visceral_fat_img, str(visceral_fat_path))
    print(f"  Visceral fat mask saved: {visceral_fat_path}")
    
    # 5. 肺マスクの読み込み（必須）
    print("\n6. Loading lung masks...")
    lungs = load_lung_masks(totalseg_dir, verbose)
    print(f"  Lung voxels: {int(np.sum(lungs)):,}")
    
    # 6. 腹部臓器マスクの読み込み（必須）
    print("\n7. Loading abdominal organ masks...")
    abdominal_organs = load_abdominal_organs(totalseg_dir, verbose)
    print(f"  Abdominal organ voxels: {int(np.sum(abdominal_organs)):,}")
    
    # 7. 近位大血管ベースの下端・上端Z決定
    print("\n8. Computing anatomical boundaries...")
    
    # 下端：IVC/肺静脈ベースの判定
    z_min = compute_intrapericardial_zmin(
        totalseg_dir, heart_mask, spacing, 
        prox_mm=5.0, z_margin_slices=1, verbose=verbose
    )
    
    # 上端：近位大血管を考慮した解剖学的上限
    z_max = compute_intrapericardial_zmax(
        totalseg_dir, heart_mask, spacing, 
        prox_mm=5.0, z_margin_slices=1, verbose=verbose
    )
    
    print(f"  Anatomical Z-range: {z_min} to {z_max} (total: {z_max - z_min + 1} slices)")
    
    # 8. Shell（心臓周囲領域）の計算（EDTベースの等方的膨張）
    print(f"\n9. Computing Shell (EDT-based dilation: {dilation_radius_mm} mm)...")
    
    # iso_dilateヘルパー関数を使用した等方的膨張
    heart_dilated = iso_dilate(heart_mask, dilation_radius_mm, spacing)
    
    print(f"  Heart dilated voxels: {np.sum(heart_dilated):,}")
    
    # Shell（心臓周囲の殻状領域）を計算
    shell_mask = heart_dilated & ~heart_mask
    
    # 9. Zクリップを適用
    print("\n10. Applying Z-clipping...")
    shell_mask[:, :, :z_min] = False
    shell_mask[:, :, z_max+1:] = False
    
    fat_clipped = fat_mask.copy()
    fat_clipped[:, :, :z_min] = False
    fat_clipped[:, :, z_max+1:] = False
    
    print(f"  Shell and fat clipped to Z-range [{z_min}:{z_max+1}]")
    
    # 10. 縦隔クランプ（胸腔内限定）
    print("\n11. Applying mediastinal restriction...")
    mediastinum = ~lungs
    
    # Shellを縦隔に限定（特に下部で重要）
    shell_in_mediastinum = shell_mask & mediastinum
    
    # 脂肪候補：Shell内かつ縦隔内のtorso_fat
    fat_candidate = shell_in_mediastinum & fat_clipped
    
    if verbose:
        original_shell_voxels = np.sum(shell_mask)
        mediastinal_shell_voxels = np.sum(shell_in_mediastinum)
        print(f"  Shell restricted to mediastinum: {original_shell_voxels:,} → {mediastinal_shell_voxels:,} voxels")
        print(f"  Reduction: {original_shell_voxels - mediastinal_shell_voxels:,} voxels outside mediastinum")
    
    # Shellを縦隔限定版に更新
    shell_mask = shell_in_mediastinum
    
    # 11. 腹部臓器クランプ（偽陽性除去）
    print("\n12. Applying abdominal organ clamping...")
    
    # 腹部臓器を3mm膨張して禁止帯を作成
    struct = generate_binary_structure(3, 1)
    min_spacing = min(spacing)
    iterations = int(np.ceil(3.0 / min_spacing))  # 3mm膨張
    
    abdomen_forbidden = binary_dilation(abdominal_organs, structure=struct, iterations=iterations)
    
    # 腹部臓器接触部分を除去
    initial_fat_voxels = np.sum(fat_candidate)
    fat_candidate = fat_candidate & ~abdomen_forbidden
    removed_voxels = initial_fat_voxels - np.sum(fat_candidate)
    
    if verbose:
        print(f"  Removed {removed_voxels:,} voxels in contact with abdominal organs")
    
    # 12. 心膜ブリッジによる連結性制約（偽陰性救済）
    print("\n13. Applying pericardial bridge for connectivity...")
    
    # 心膜ブリッジ（心臓を2mm等方膨張）
    heart_dilated_small = iso_dilate(heart_mask, 2.0, spacing)
    bridge = heart_dilated_small & ~heart_mask
    
    # Binary propagationで心膜直外のPATを救済
    # 初期シード：Shell内の脂肪候補＋ブリッジ内の脂肪
    seed = fat_candidate | (bridge & fat_clipped & mediastinum)
    propagation_mask = (shell_mask | bridge) & fat_clipped & mediastinum
    
    # 連結成分の伝播
    reached = binary_propagation(seed, mask=propagation_mask, structure=struct)
    
    # Shellベースの EAT+PAT（ブリッジ領域を除外、Shell内のみ）
    eat_pat_shell = reached & shell_mask
    
    # 13.5. 個別臓器マスクの読み込み（v5.3: 優勢接触判定用）
    print("\n13.5. Loading individual abdominal organs for dominant contact rule...")
    abd_organs_dict = load_abdominal_organs_dict(totalseg_dir, ct_data.shape, verbose=verbose)
    
    # 14. ILAM (Inferior Lung-Adjacency Mode) v5.3 - 修正版優勢接触ルール
    print("\n14. Applying ILAM v5.3 (Fixed Dominant Contact Rule)...")
    print(f"  Parameters:")
    print(f"    - Inferior band: {inferior_band_mm} mm")
    print(f"    - Contact epsilon: {contact_eps_mm} mm")
    print(f"    - Abdomen gap: {abdomen_gap_mm} mm (γ)")
    print(f"    - Post-trim: {post_trim_mm} mm")
    print(f"    - Lung dominance ratio: {lung_dom_ratio}")
    print(f"    - Min lung touch voxels: {min_lung_touch_voxels}")
    print(f"    - No Z limit: {no_z_limit}")
    print(f"    - Seed Z margin: {seed_z_margin_mm} mm")
    print(f"    - Seal diaphragm holes: {seal_diaphragm_holes_mm} mm")
    print(f"    - Require shell touch: {require_shell_touch}")
    if max_heart_dist_mm is not None:
        print(f"    - Max heart distance: {max_heart_dist_mm} mm (ρ)")
    else:
        print(f"    - Max heart distance: None (no restriction)")
    print(f"    - Mode: {'replace' if replace_by_contact else 'add'}")
    
    # 設定の矛盾チェック（v5.3の重要な警告）
    if abdomen_gap_mm >= contact_eps_mm:
        print(f"\n  ⚠ WARNING: abdomen_gap_mm ({abdomen_gap_mm}) >= contact_eps_mm ({contact_eps_mm})")
        print(f"     This will cause abdominal organ contact counts to be 0!")
        print(f"     Recommended: abdomen_gap_mm < contact_eps_mm")
    
    # 下部帯域の計算（報告用、no_z_limitの場合は全Z）
    z_spacing = float(spacing[2])
    band_slices = int(np.ceil(inferior_band_mm / z_spacing))
    z_switch = min(z_min + band_slices, z_max) if not no_z_limit else z_max
    
    if no_z_limit:
        print(f"  ILAM Z-range: Full volume (no Z restriction)")
    else:
        print(f"  ILAM Z-range: {z_min} to {z_switch} ({z_switch - z_min + 1} slices)")
    print(f"  Z-spacing: {z_spacing:.2f} mm/slice")
    
    # 肺マスクの穴埋め（横隔膜の微小欠損対策）
    lungs_for_touch = lungs.copy()
    if seal_diaphragm_holes_mm > 0:
        print(f"  Sealing diaphragm holes with {seal_diaphragm_holes_mm} mm closing...")
        iterations = max(1, int(np.ceil(seal_diaphragm_holes_mm / min(spacing))))
        struct_closing = generate_binary_structure(3, 1)  # 6-connectivity for closing
        lungs_for_touch = binary_closing(lungs_for_touch, structure=struct_closing, iterations=iterations)
    
    # 等方膨張で接触帯を作成（mm単位）
    lungs_touch = iso_dilate(lungs_for_touch, contact_eps_mm, spacing)
    stomach_touch = iso_dilate(abd_organs_dict["stomach"], contact_eps_mm, spacing) if np.any(abd_organs_dict["stomach"]) else np.zeros_like(lungs_touch)
    liver_touch = iso_dilate(abd_organs_dict["liver"], contact_eps_mm, spacing) if np.any(abd_organs_dict["liver"]) else np.zeros_like(lungs_touch)
    spleen_touch = iso_dilate(abd_organs_dict["spleen"], contact_eps_mm, spacing) if np.any(abd_organs_dict["spleen"]) else np.zeros_like(lungs_touch)
    
    # v5.3の重要変更: 腹部禁止帯は接触判定後に適用
    # ILAMのドメイン：腹部禁止帯なしで連結成分解析（接触判定用）
    fat_for_ilam = fat_mask.copy() if no_z_limit else fat_clipped
    domain_cc = (fat_for_ilam & (~lungs))  # 禁止帯なしのドメイン
    
    # 心臓からの距離制約（オプションだが推奨）
    if max_heart_dist_mm is not None:
        dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
        domain_cc &= (dist_to_heart <= max_heart_dist_mm)
        print(f"  Applied heart distance constraint: {max_heart_dist_mm} mm")
    else:
        dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    
    # v5.3: シードだけZ範囲で制限（ドメインは全Z）
    # シード用のZ制限マスク
    seed_z_margin_slices = int(np.ceil(seed_z_margin_mm / spacing[2]))
    seed_z_mask = np.zeros_like(domain_cc, dtype=bool)
    seed_z_min = max(0, z_min - seed_z_margin_slices)
    seed_z_max = min(domain_cc.shape[2] - 1, z_max + seed_z_margin_slices)
    seed_z_mask[:, :, seed_z_min:seed_z_max+1] = True
    print(f"  Seed Z range: {seed_z_min} to {seed_z_max} (margin: ±{seed_z_margin_mm} mm)")
    
    # オプション: 遠位大動脈の除外（下行大動脈のPVATを除外）
    aorta_path = Path(totalseg_dir) / "aorta.nii.gz"
    if aorta_path.exists():
        aorta_img = nib.load(str(aorta_path))
        aorta_mask = aorta_img.get_fdata() > 0
        
        # 心臓から5mm以内を近位部とする
        proximal_aorta = aorta_mask & (dist_to_heart <= 5.0)
        distal_aorta = aorta_mask & ~proximal_aorta
        
        # 遠位大動脈を2.5mm等方膨張して禁止帯を作成
        distal_aorta_forbidden = iso_dilate(distal_aorta, 2.5, spacing)
        
        # domain_ccから遠位大動脈周囲を除外
        domain_cc &= ~distal_aorta_forbidden
        print(f"  Excluded distal aorta PVAT (2.5mm forbidden zone)")
    
    # v5.3: シード：肺接触している torso_fat（domain_cc内 かつ seed_z_mask内）
    seed = domain_cc & lungs_touch & seed_z_mask
    seed_voxels = np.sum(seed)
    print(f"  Seed voxels (lung-touching fat in Z range): {seed_voxels:,}")
    
    # 26-連結で連結成分ラベリング（禁止帯なしのdomain_cc）
    print("  Performing connected component labeling (without forbidden zones)...")
    conn = generate_binary_structure(3, 2)  # 26-connectivity
    lbl, n_components = label(domain_cc, structure=conn)
    print(f"  Found {n_components} connected components in domain")
    
    # 肺に触れている連結成分ID候補
    touch_ids = np.unique(lbl[seed])
    touch_ids = touch_ids[touch_ids > 0]  # 背景を除外
    print(f"  Components touching lungs (candidates): {len(touch_ids)}")
    
    # --- v5.3: 修正版優勢接触ルール + Shell接触要件 ---
    def count_contact(comp_id, touch_mask):
        """指定成分と臓器接触マスクの重なりボクセル数を数える"""
        return int(np.sum((lbl == comp_id) & touch_mask))
    
    def touches_shell(comp_id, shell_near_mask):
        """指定成分がShell近傍に接しているか"""
        return bool(np.any((lbl == comp_id) & shell_near_mask))
    
    # Shell近傍マスク（2mm膨張）
    shell_near = iso_dilate(shell_mask, 2.0, spacing) if require_shell_touch else None
    
    keep_ids = []
    rejected_stomach = 0
    rejected_liver = 0
    rejected_spleen = 0
    rejected_small = 0
    rejected_no_shell = 0
    
    # 接触による分類用マスク（可視化用）
    lung_contact_components = np.zeros_like(domain_cc, dtype=bool)
    stomach_contact_components = np.zeros_like(domain_cc, dtype=bool)
    
    print("\n  Evaluating dominant contact for each component...")
    for i, cid in enumerate(touch_ids):
        n_lung = count_contact(cid, lungs_touch)
        n_stomach = count_contact(cid, stomach_touch)
        n_liver = count_contact(cid, liver_touch)
        n_spleen = count_contact(cid, spleen_touch)
        n_abdmax = max(n_stomach, n_liver, n_spleen)
        
        # 1) 肺接触の絶対量が少ないものは捨てる
        if n_lung < min_lung_touch_voxels:
            rejected_small += 1
            if verbose and i < 5:  # 最初の5個だけ詳細表示
                print(f"    Component {cid}: Rejected (lung touch {n_lung} < {min_lung_touch_voxels})")
            continue
            
        # 2) 肺接触が優勢でなければ捨てる
        if n_lung < lung_dom_ratio * n_abdmax:
            if n_stomach == n_abdmax:
                rejected_stomach += 1
            elif n_liver == n_abdmax:
                rejected_liver += 1
            elif n_spleen == n_abdmax:
                rejected_spleen += 1
                
            if verbose and i < 5:
                print(f"    Component {cid}: Rejected (lung {n_lung} < {lung_dom_ratio} * abd_max {n_abdmax})")
            continue
        
        # 3) v5.3: Shell接触要件（オプション）
        if require_shell_touch and shell_near is not None:
            if not touches_shell(cid, shell_near):
                rejected_no_shell += 1
                if verbose and i < 5:
                    print(f"    Component {cid}: Rejected (no shell contact)")
                continue
            
        keep_ids.append(cid)
        
        # 採用された成分を接触優勢により分類（可視化用）
        comp_mask = (lbl == cid)
        if n_stomach > 0 and n_stomach == n_abdmax:
            # 胃接触が最大（ただし肺優勢条件は満たす）
            stomach_contact_components |= comp_mask
        else:
            # 肺接触優勢または腹部臓器接触なし
            lung_contact_components |= comp_mask
        
        if verbose and i < 5:
            print(f"    Component {cid}: Accepted (lung {n_lung}, stomach {n_stomach}, liver {n_liver}, spleen {n_spleen})")
    
    print(f"\n  Dominant contact filtering results:")
    print(f"    Accepted: {len(keep_ids)} components")
    print(f"    Rejected: {len(touch_ids) - len(keep_ids)} components")
    if rejected_small > 0:
        print(f"      - Small lung contact: {rejected_small}")
    if rejected_stomach > 0:
        print(f"      - Stomach dominant: {rejected_stomach}")
    if rejected_liver > 0:
        print(f"      - Liver dominant: {rejected_liver}")
    if rejected_spleen > 0:
        print(f"      - Spleen dominant: {rejected_spleen}")
    if rejected_no_shell > 0:
        print(f"      - No shell contact: {rejected_no_shell}")
    
    # 優勢接触を満たす成分だけを採用
    ilam_components = np.isin(lbl, np.array(keep_ids, dtype=lbl.dtype)) if keep_ids else np.zeros_like(domain_cc)
    
    # v5.3: 最終整形として薄い安全帯を差し引く（post-trim）
    print(f"\n  Applying post-trim safety margin ({post_trim_mm} mm)...")
    actual_post_trim = min(post_trim_mm, contact_eps_mm)  # 推奨: contact_eps_mmより小さく
    abdominal_trim = iso_dilate(abdominal_organs, actual_post_trim, spacing)
    ilam_components = ilam_components & ~abdominal_trim
    
    ilam_voxels = np.sum(ilam_components)
    print(f"  ILAM component voxels (after post-trim): {ilam_voxels:,}")
    
    # 診断情報：追加された部分
    ilam_addition = ilam_components & ~eat_pat_shell
    ilam_added_voxels = np.sum(ilam_addition)
    print(f"  ILAM voxels beyond shell: {ilam_added_voxels:,}")
    
    # 連結成分の統計
    if len(keep_ids) > 0:
        component_sizes = []
        for comp_id in keep_ids:
            comp_size = np.sum(lbl == comp_id)
            component_sizes.append(comp_size)
        
        max_comp_size = max(component_sizes)
        mean_comp_size = np.mean(component_sizes)
        print(f"  Largest component: {max_comp_size:,} voxels")
        print(f"  Mean component size: {mean_comp_size:.0f} voxels")
    
    # 最終的なEAT+PAT: Shellベース ∪ ILAM連結成分
    if replace_by_contact:
        # 下部帯域では肺接触ベースに置換（実験用）
        z_mask = np.zeros_like(eat_pat_shell, dtype=bool)
        z_mask[:, :, z_switch+1:] = True
        pericardial_fat_mask = (eat_pat_shell & z_mask) | ilam_components
    else:
        # 和集合モード（推奨）
        pericardial_fat_mask = eat_pat_shell | ilam_components
    
    # 体積比較
    shell_only_voxels = np.sum(eat_pat_shell)
    combined_voxels = np.sum(pericardial_fat_mask)
    added_by_ilam = combined_voxels - shell_only_voxels
    
    voxel_volume_ml = np.prod(spacing) / 1000.0
    ilam_added_volume_ml = ilam_added_voxels * voxel_volume_ml
    
    print(f"\n  Volume analysis:")
    print(f"    Shell-based EAT+PAT: {shell_only_voxels:,} voxels")
    print(f"    Added by ILAM: {added_by_ilam:,} voxels ({ilam_added_volume_ml:.2f} ml)")
    print(f"    Final EAT+PAT: {combined_voxels:,} voxels")
    
    # ILAMの影響を警告
    if added_by_ilam > shell_only_voxels * 0.5:
        print(f"  ⚠ ILAM added >50% volume - consider adjusting parameters")
        print(f"     Suggestions:")
        print(f"     - Reduce inferior_band_mm from {inferior_band_mm} to 20 mm")
        print(f"     - Reduce lung_contact_mm from {lung_contact_mm} to 1.0 mm")
        if max_heart_dist_mm is not None and max_heart_dist_mm > 25:
            print(f"     - Reduce max_heart_dist_mm from {max_heart_dist_mm} to 20-25 mm")
    
    # ILAM診断マスクを保存
    ilam_addition_path = masks_dir / "ilam_addition.nii.gz"
    ilam_addition_img = nib.Nifti1Image(ilam_addition.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(ilam_addition_img, str(ilam_addition_path))
    print(f"\n  ILAM addition mask saved: {ilam_addition_path}")
    
    # 接触分類マスクを保存（可視化用）
    lung_contact_path = masks_dir / "lung_contact_components.nii.gz"
    lung_contact_img = nib.Nifti1Image(lung_contact_components.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(lung_contact_img, str(lung_contact_path))
    print(f"  Lung contact components saved: {lung_contact_path}")
    
    stomach_contact_path = masks_dir / "stomach_contact_components.nii.gz"
    stomach_contact_img = nib.Nifti1Image(stomach_contact_components.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(stomach_contact_img, str(stomach_contact_path))
    print(f"  Stomach contact components saved: {stomach_contact_path}")
    
    # Shellマスクを保存
    shell_path = masks_dir / "shell.nii.gz"
    shell_img = nib.Nifti1Image(shell_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(shell_img, str(shell_path))
    print(f"  Shell mask saved: {shell_path}")
    
    # EAT+PATマスクを保存
    eat_pat_path = masks_dir / "eat_pat.nii.gz"
    eat_pat_img = nib.Nifti1Image(pericardial_fat_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(eat_pat_img, str(eat_pat_path))
    print(f"  EAT+PAT mask saved: {eat_pat_path}")
    
    # 15. 統計量の計算
    print("\n16. Calculating statistics...")
    
    # ボクセル体積 [mm³]
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # 各コンポーネントの統計
    heart_voxels = int(np.sum(heart_mask))
    heart_volume_ml = voxel_vol_mm3 * heart_voxels / 1000.0
    
    shell_voxels = int(np.sum(shell_mask))
    shell_volume_ml = voxel_vol_mm3 * shell_voxels / 1000.0
    
    eat_pat_voxels = int(np.sum(pericardial_fat_mask))
    eat_pat_volume_ml = voxel_vol_mm3 * eat_pat_voxels / 1000.0
    
    visceral_fat_voxels = int(np.sum(fat_mask))
    visceral_fat_volume_ml = voxel_vol_mm3 * visceral_fat_voxels / 1000.0
    
    # Shell内の脂肪割合
    fat_fraction_in_shell = (eat_pat_voxels / shell_voxels * 100) if shell_voxels > 0 else 0
    
    print(f"  Heart volume: {heart_volume_ml:.2f} ml")
    print(f"  Shell volume: {shell_volume_ml:.2f} ml")
    print(f"  EAT+PAT volume: {eat_pat_volume_ml:.2f} ml")
    print(f"  Total visceral fat volume: {visceral_fat_volume_ml:.2f} ml")
    print(f"  Fat fraction in shell: {fat_fraction_in_shell:.1f}%")
    print(f"  EAT+PAT / Visceral fat ratio: {(eat_pat_volume_ml/visceral_fat_volume_ml*100):.1f}%")
    
    # 16. 統計をJSON形式で保存
    stats = {
        "volumes_ml": {
            "heart": round(heart_volume_ml, 2),
            "shell": round(shell_volume_ml, 2),
            "eat_pat": round(eat_pat_volume_ml, 2),
            "visceral_fat": round(visceral_fat_volume_ml, 2)
        },
        "voxel_counts": {
            "heart": heart_voxels,
            "shell": shell_voxels,
            "eat_pat": eat_pat_voxels,
            "visceral_fat": visceral_fat_voxels
        },
        "ratios_percent": {
            "fat_fraction_in_shell": round(fat_fraction_in_shell, 2),
            "eat_pat_to_visceral": round(eat_pat_volume_ml/visceral_fat_volume_ml*100, 2) if visceral_fat_volume_ml > 0 else 0,
            "eat_pat_to_heart": round(eat_pat_volume_ml/heart_volume_ml*100, 2) if heart_volume_ml > 0 else 0
        },
        "parameters": {
            "dilation_radius_mm": float(dilation_radius_mm),
            "dilation_method": "EDT (Euclidean Distance Transform)",
            "processing_version": "v5.3 with Fixed Dominant Contact ILAM",
            "ilam_settings": {
                "inferior_band_mm": float(inferior_band_mm),
                "lung_contact_mm": float(lung_contact_mm),
                "contact_eps_mm": float(contact_eps_mm),
                "abdomen_gap_mm": float(abdomen_gap_mm),
                "max_heart_dist_mm": float(max_heart_dist_mm) if max_heart_dist_mm is not None else None,
                "no_z_limit": no_z_limit,
                "lung_dom_ratio": float(lung_dom_ratio),
                "min_lung_touch_voxels": int(min_lung_touch_voxels),
                "seal_diaphragm_holes_mm": float(seal_diaphragm_holes_mm),
                "post_trim_mm": float(actual_post_trim if 'actual_post_trim' in locals() else post_trim_mm),
                "seed_z_margin_mm": float(seed_z_margin_mm),
                "require_shell_touch": require_shell_touch,
                "mode": "replace" if replace_by_contact else "add",
                "z_range": [int(z_min), int(z_switch)] if not no_z_limit else "full",
                "z_spacing_mm": float(z_spacing),
                "added_voxels": int(added_by_ilam),
                "added_volume_ml": round(ilam_added_volume_ml, 2),
                "ilam_voxels": int(ilam_voxels),
                "components_accepted": len(keep_ids) if 'keep_ids' in locals() else 0,
                "components_rejected": len(touch_ids) - len(keep_ids) if 'touch_ids' in locals() and 'keep_ids' in locals() else 0
            },
            "improvements": [
                "IVC/PV-based inferior boundary",
                "Z-clipping before mediastinal restriction",
                "Shell restricted to mediastinum",
                "Abdominal organ clamping (3mm)",
                "Pericardial bridge connectivity",
                "ILAM v5.3: Fixed order of contact judgment and forbidden zones",
                "Seed-only Z restriction with full 3D propagation",
                "Shell contact requirement for component acceptance",
                "Improved diaphragm hole sealing",
                "Post-trim safety margin after contact judgment"
            ],
            "ct_shape": list(ct_data.shape),
            "ct_spacing_mm": list(spacing),
            "heart_z_range": [int(z_min), int(z_max)],
            "masks_used": mask_status
        }
    }
    
    stats_path = stats_dir / "eat_pat_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n17. Statistics saved: {stats_path}")
    
    # 検証：正常範囲チェック
    print("\n[Validation]")
    if 50 <= eat_pat_volume_ml <= 200:
        print(f"  ✓ EAT+PAT volume is within normal range (50-200 ml)")
    elif eat_pat_volume_ml < 50:
        print(f"  ⚠ EAT+PAT volume is below normal range (< 50 ml)")
    else:
        print(f"  ⚠ EAT+PAT volume is above normal range (> 200 ml)")
    
    if fat_fraction_in_shell < 30:
        print(f"  ✓ Fat fraction in shell is normal (< 30%)")
    else:
        print(f"  ⚠ Fat fraction in shell is high (≥ 30%)")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Improved EAT+PAT extraction v5.3 with Fixed Dominant Contact ILAM",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "ct_path",
        help="Path to CT NIfTI file"
    )
    
    parser.add_argument(
        "totalseg_dir",
        help="Path to TotalSegmentator output directory"
    )
    
    parser.add_argument(
        "output_dir",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--dilation-radius",
        type=float,
        default=15.0,
        help="Heart mask dilation radius in mm (default: 15.0)"
    )
    
    parser.add_argument(
        "--no-vessels",
        action="store_true",
        help="Do not include proximal vessels"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # ILAM parameters
    parser.add_argument(
        "--inferior-band-mm",
        type=float,
        default=DEFAULT_INFERIOR_BAND_MM,
        help=f"ILAM inferior band height in mm (default: {DEFAULT_INFERIOR_BAND_MM})"
    )
    
    parser.add_argument(
        "--lung-contact-mm",
        type=float,
        default=DEFAULT_LUNG_CONTACT_MM,
        help=f"ILAM lung contact tolerance in mm (δ) (default: {DEFAULT_LUNG_CONTACT_MM})"
    )
    
    parser.add_argument(
        "--abdomen-gap-mm",
        type=float,
        default=DEFAULT_ABDOMEN_GAP_MM,
        help=f"ILAM abdominal organ exclusion gap in mm (γ) (default: {DEFAULT_ABDOMEN_GAP_MM})"
    )
    
    parser.add_argument(
        "--max-heart-dist-mm",
        type=float,
        default=DEFAULT_MAX_HEART_DIST_MM,
        help=f"ILAM maximum distance from heart in mm (ρ). Use 0 or negative value to disable (default: {DEFAULT_MAX_HEART_DIST_MM})"
    )
    
    parser.add_argument(
        "--replace-by-contact",
        action="store_true",
        help="Replace shell-based EAT+PAT with lung contact in inferior band (experimental)"
    )
    
    # v5.2 new parameters
    parser.add_argument(
        "--no-z-limit",
        action="store_true",
        help="ILAM を全 Z で実行（inferior band によるZ制限を無効化）"
    )
    
    parser.add_argument(
        "--contact-eps-mm",
        type=float,
        default=DEFAULT_CONTACT_EPS_MM,
        help=f"肺/腹部臓器との接触判定に使う等方膨張距離 [mm] (default: {DEFAULT_CONTACT_EPS_MM})"
    )
    
    parser.add_argument(
        "--lung-dom-ratio",
        type=float,
        default=DEFAULT_LUNG_DOM_RATIO,
        help=f"採用条件: lung_contact >= ratio * max(abdominal organ contact) (default: {DEFAULT_LUNG_DOM_RATIO})"
    )
    
    parser.add_argument(
        "--min-lung-touch-voxels",
        type=int,
        default=DEFAULT_MIN_LUNG_TOUCH_VOX,
        help=f"採用するために必要な肺接触ボクセルの最小数 (default: {DEFAULT_MIN_LUNG_TOUCH_VOX})"
    )
    
    parser.add_argument(
        "--seal-diaphragm-holes-mm",
        type=float,
        default=DEFAULT_SEAL_DIAPHRAGM_MM,
        help=f"肺マスクの微小欠損を閉じる半径 [mm] (0で無効, default: {DEFAULT_SEAL_DIAPHRAGM_MM})"
    )
    
    # v5.3 新パラメータ
    parser.add_argument(
        "--post-trim-mm",
        type=float,
        default=DEFAULT_POST_TRIM_MM,
        help=f"最終整形用の安全帯 [mm] (default: {DEFAULT_POST_TRIM_MM})"
    )
    
    parser.add_argument(
        "--seed-z-margin-mm",
        type=float,
        default=DEFAULT_SEED_Z_MARGIN_MM,
        help=f"シード生成のZ範囲マージン [mm] (default: {DEFAULT_SEED_Z_MARGIN_MM})"
    )
    
    parser.add_argument(
        "--no-shell-touch",
        action="store_true",
        help="Shell接触要件を無効化 (default: required)"
    )
    
    args = parser.parse_args()
    
    # 実行
    try:
        # max_heart_dist_mmが0以下の場合はNoneに変換
        max_heart_dist = args.max_heart_dist_mm if args.max_heart_dist_mm > 0 else None
        
        stats = extract_eat_pat_improved_v5_3(
            ct_path=args.ct_path,
            totalseg_dir=args.totalseg_dir,
            output_dir=args.output_dir,
            dilation_radius_mm=args.dilation_radius,
            use_proximal_vessels=not args.no_vessels,
            # ILAM parameters
            inferior_band_mm=args.inferior_band_mm,
            lung_contact_mm=args.lung_contact_mm,
            abdomen_gap_mm=args.abdomen_gap_mm,
            max_heart_dist_mm=max_heart_dist,
            replace_by_contact=args.replace_by_contact,
            # v5.2 parameters
            no_z_limit=args.no_z_limit,
            contact_eps_mm=args.contact_eps_mm,
            lung_dom_ratio=args.lung_dom_ratio,
            min_lung_touch_voxels=args.min_lung_touch_voxels,
            seal_diaphragm_holes_mm=args.seal_diaphragm_holes_mm,
            post_trim_mm=args.post_trim_mm,
            seed_z_margin_mm=args.seed_z_margin_mm,
            require_shell_touch=not args.no_shell_touch,
            verbose=args.verbose
        )
        
        print("\n" + "="*60)
        print("Improved EAT+PAT extraction v5.3 (Fixed Dominant Contact Rule) completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print(f"Total EAT+PAT volume: {stats['volumes_ml']['eat_pat']:.2f} ml")
        print(f"Shell volume: {stats['volumes_ml']['shell']:.2f} ml")
        print(f"Fat fraction in shell: {stats['ratios_percent']['fat_fraction_in_shell']:.1f}%")
        
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()