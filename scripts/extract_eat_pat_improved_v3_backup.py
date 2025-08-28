#!/usr/bin/env python3
"""
改善版v3 EAT+PAT抽出：必要なマスクがない場合はエラーで停止
フォールバックを削除し、明確なエラーメッセージを提供
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
    distance_transform_edt, label, binary_propagation
)

# 定数
DILATION_RADIUS_MM = 15.0  # 心臓マスク膨張半径 [mm] (EAT+PATを含む範囲)

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
        'lungs': ['lung_left.nii.gz', 'lung_right.nii.gz', 'lung.nii.gz'],
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
    IVCと肺静脈を使った心膜内最尾側Z座標の計算
    """
    # 心臓からの距離画像（EDT）を計算
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    prox_band = dist_to_heart <= prox_mm  # 5mm以内の近接帯
    
    def _load(name):
        p = Path(totalseg_dir) / f"{name}.nii.gz"
        return (nib.load(str(p)).get_fdata() > 0) if p.exists() else None
    
    # IVCと肺静脈を読み込み
    ivc = _load("inferior_vena_cava")
    if ivc is None:
        ivc = _load("vena_cava_inferior")
    
    pv = _load("pulmonary_vein")
    
    # 心臓近接の血管部分を取得
    prox_vessels = np.zeros_like(heart_mask, dtype=bool)
    if ivc is not None:
        prox_vessels |= (ivc & prox_band)
    if pv is not None:
        prox_vessels |= (pv & prox_band)
    
    # 血管が存在する最初のZ座標を探す（最も尾側）
    if np.any(prox_vessels):
        z_any = np.any(prox_vessels, axis=(0,1))
        z_idx = np.where(z_any)[0]
        if z_idx.size > 0:
            zmin = max(0, z_idx[0] - z_margin_slices)  # 最も尾側（小さいZ）
            if verbose:
                print(f"  Intrapericardial zmin (IVC/PV based): {zmin}")
                print(f"    IVC found: {ivc is not None}")
                print(f"    PV found: {pv is not None}")
            return zmin
    
    # IVC/PVがない場合：心臓の最下端を使用
    z_heart = np.where(np.any(heart_mask, axis=(0,1)))[0]
    if z_heart.size > 0:
        zmin = max(0, z_heart[0] - 3)  # 少しマージンを追加
        if verbose:
            print(f"  WARNING: IVC/PV not found, using heart bottom: {zmin}")
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

def extract_eat_pat_improved_v3(ct_path, totalseg_dir, output_dir, 
                                dilation_radius_mm=DILATION_RADIUS_MM,
                                use_proximal_vessels=True,
                                verbose=False):
    """
    改善版v3 EAT+PAT（心膜周囲脂肪）抽出
    必要なマスクがない場合はエラーで停止（フォールバックなし）
    """
    print("\n[Improved EAT+PAT Extraction v3 - No Fallbacks]")
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
    
    # ユークリッド距離変換による等方的膨張
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    heart_dilated = dist_to_heart <= dilation_radius_mm
    
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
    
    # 心膜ブリッジ（心臓を2mm膨張）
    iterations = max(1, int(np.ceil(2.0 / min_spacing)))  # 2mm膨張
    heart_dilated_small = binary_dilation(heart_mask, structure=struct, iterations=iterations)
    bridge = heart_dilated_small & ~heart_mask
    
    # Binary propagationで心膜直外のPATを救済
    # 初期シード：Shell内の脂肪候補＋ブリッジ内の脂肪
    seed = fat_candidate | (bridge & fat_clipped & mediastinum)
    propagation_mask = (shell_mask | bridge) & fat_clipped & mediastinum
    
    # 連結成分の伝播
    reached = binary_propagation(seed, mask=propagation_mask, structure=struct)
    
    # 最終的なEAT+PATマスク（ブリッジ領域を除外、Shell内のみ）
    pericardial_fat_mask = reached & shell_mask
    
    print(f"  Final EAT+PAT voxels: {np.sum(pericardial_fat_mask):,}")
    
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
    
    # 13. 統計量の計算
    print("\n14. Calculating statistics...")
    
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
    
    # 14. 統計をJSON形式で保存
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
            "processing_version": "v3 (no fallbacks)",
            "improvements": [
                "IVC/PV-based inferior boundary",
                "Z-clipping before mediastinal restriction",
                "Shell restricted to mediastinum",
                "Abdominal organ clamping (3mm)",
                "Pericardial bridge connectivity",
                "No HU fallback (torso_fat only)"
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
    print(f"\n15. Statistics saved: {stats_path}")
    
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
        description="Improved EAT+PAT extraction v3 (no fallbacks)",
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
    
    args = parser.parse_args()
    
    # 実行
    try:
        stats = extract_eat_pat_improved_v3(
            ct_path=args.ct_path,
            totalseg_dir=args.totalseg_dir,
            output_dir=args.output_dir,
            dilation_radius_mm=args.dilation_radius,
            use_proximal_vessels=not args.no_vessels,
            verbose=args.verbose
        )
        
        print("\n" + "="*60)
        print("Improved EAT+PAT extraction v3 completed successfully!")
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