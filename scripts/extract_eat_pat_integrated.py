#!/usr/bin/env python3
"""
TotalSegmentatorを使用したEAT+PAT抽出（統合版）
Shell（心臓周囲領域）を含む全コンポーネントを出力
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
    distance_transform_edt, label
)

# 定数
DILATION_RADIUS_MM = 15.0  # 心臓マスク膨張半径 [mm] (EAT+PATを含む範囲)

def load_heart_components(totalseg_dir, verbose=False):
    """
    心臓コンポーネントを読み込んで統合（v1/v2名の両対応）
    
    Returns:
        heart_mask: 統合された心臓マスク
        affine: アフィン変換行列
    """
    totalseg_dir = Path(totalseg_dir)
    heart_mask = None
    affine = None
    
    # v2/v1 highres（先頭にheart_が付かない）
    names_v2 = ["myocardium", "atrium_left", "atrium_right", "ventricle_left", "ventricle_right"]
    # 旧来のプレフィックスあり表記に暫定対応
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
    
    # 見つからなければtotalタスクのheartを使用
    if heart_mask is None:
        p = totalseg_dir / "heart.nii.gz"
        if p.exists():
            img = nib.load(str(p))
            heart_mask = img.get_fdata() > 0
            affine = img.affine
            components_found.append("heart")
            if verbose: 
                print("  Fallback: total task 'heart'")
    
    if heart_mask is None:
        raise FileNotFoundError("No heart(-chambers) mask found in TotalSegmentator output")
    
    if verbose:
        print(f"  Components used: {components_found}")
        print(f"  Heart voxels: {int(np.sum(heart_mask)):,}")
        
    return heart_mask, affine

def load_vessel_masks(totalseg_dir, verbose=False):
    """
    大血管マスクを読み込み（オプション）
    
    Returns:
        vessels: 血管マスクの辞書
    """
    totalseg_dir = Path(totalseg_dir)
    vessels = {}
    
    vessel_names = [
        "aorta",                 # 大動脈
        "pulmonary_artery",      # 肺動脈
        "vena_cava_superior",    # 上大静脈
        "vena_cava_inferior"     # 下大静脈
    ]
    
    for vessel in vessel_names:
        vessel_path = totalseg_dir / f"{vessel}.nii.gz"
        if vessel_path.exists():
            if verbose:
                print(f"  Loading {vessel}...")
            img = nib.load(str(vessel_path))
            vessels[vessel] = img.get_fdata() > 0
    
    return vessels

def compute_intrapericardial_zmax(totalseg_dir, heart_mask, spacing,
                                  prox_mm=5.0, z_margin_slices=1, verbose=False):
    """
    心膜内領域の上端Z座標を計算（近位大血管を考慮）
    
    Args:
        totalseg_dir: TotalSegmentator出力ディレクトリ
        heart_mask: 心臓マスク
        spacing: ボクセルスペーシング
        prox_mm: 心臓からの近接距離 [mm]
        z_margin_slices: 安全マージン [スライス数]
        verbose: 詳細出力
    
    Returns:
        zmax: 上限Z座標
    """
    se3d = generate_binary_structure(3, 2)
    it = max(1, int(np.ceil(prox_mm / min(spacing))))
    prox_band = binary_dilation(heart_mask, structure=se3d, iterations=it)

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
        return np.where(np.any(heart_mask, axis=(0,1)))[0][-1]  # 心臓だけで決定
    
    zmax = min(z_idx[-1] + z_margin_slices, heart_mask.shape[2]-1)
    if verbose:
        print(f"  Intrapericardial zmax (with proximal great vessels): {zmax}")
    return zmax

def incorporate_proximal_vessels(heart_mask, vessels, proximity_mm=5.0, spacing=(1.0, 1.0, 1.0), verbose=False):
    """
    心臓に近接する大血管の根元部分を心臓マスクに追加
    
    Args:
        heart_mask: 心臓マスク
        vessels: 血管マスクの辞書
        proximity_mm: 心臓からの最大距離 [mm]
        spacing: ボクセルスペーシング [mm]
    
    Returns:
        拡張された心臓マスク
    """
    if not vessels:
        return heart_mask
    
    # 心臓を少し膨張させて近接領域を作成
    struct = generate_binary_structure(3, 1)  # 6近傍
    min_spacing = min(spacing)
    iterations = int(np.ceil(proximity_mm / min_spacing))
    
    heart_proximity = binary_dilation(heart_mask, structure=struct, iterations=iterations)
    
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
    tissue_types / tissue_4_types の出力から内臓脂肪(torso_fat)を取得。
    - 個別NIfTI (torso_fat.nii.gz) があればそれを使う
    - なければ --ml 出力（単一マルチラベルNIfTI）を探索して抽出
    
    Returns:
        fat_mask: 内臓脂肪マスク
        affine: アフィン変換行列
    """
    totalseg_dir = Path(totalseg_dir)
    
    # 1) 個別ファイル（標準）
    #   tissue_types/tissue_4_types では 'torso_fat.nii.gz' が出力される
    candidates = [
        totalseg_dir / "torso_fat.nii.gz",
    ]
    
    # サブフォルダに置かれる環境に備えて再帰探索
    if not any(p.exists() for p in candidates):
        for p in totalseg_dir.rglob("torso_fat.nii*"):
            candidates.append(p)
    
    for p in candidates:
        if p.exists():
            if verbose: 
                print(f"  Found torso_fat: {p}")
            img = nib.load(str(p))
            return (img.get_fdata() > 0), img.affine
    
    # 2) --ml 出力へのフォールバック
    #    1本のNIfTIに全クラスが入っているケース
    try:
        from totalsegmentator.nifti_ext_header import load_multilabel_nifti
    except Exception as e:
        if verbose: 
            print(f"  --ml fallback unavailable (install xmltodict): {e}")
        load_multilabel_nifti = None
    
    if load_multilabel_nifti is not None:
        # 出力ディレクトリ内の NIfTI を総当たりで試す
        for ml in list(totalseg_dir.glob("*.nii")) + list(totalseg_dir.glob("*.nii.gz")):
            try:
                ml_img, label_map = load_multilabel_nifti(str(ml))
            except Exception:
                continue
            
            # label_map の向きは環境で異なる可能性があるので両対応
            label_id = None
            if isinstance(label_map, dict):
                # 例1: {"0": "background", "1": "spleen", ..., "XYZ": "torso_fat"}
                if any(k.isdigit() for k in label_map.keys()):
                    for k, v in label_map.items():
                        if v == "torso_fat":
                            label_id = int(k)
                            break
                # 例2: {"background": 0, "spleen": 1, ..., "torso_fat": XYZ}
                else:
                    if "torso_fat" in label_map:
                        try:
                            label_id = int(label_map["torso_fat"])
                        except Exception:
                            pass
            
            if label_id is not None:
                if verbose: 
                    print(f"  Extract torso_fat from multilabel: {ml} (id={label_id})")
                data = ml_img.get_fdata()
                return (data == label_id), ml_img.affine
    
    raise FileNotFoundError(
        "\n[ERROR] Visceral fat (torso_fat) not found!\n"
        "Please run TotalSegmentator with tissue_types task:\n"
        "  TotalSegmentator -i <input.nii.gz> -o <output_dir> --task tissue_types\n"
        "\nExpected file:\n"
        f"  - {totalseg_dir}/torso_fat.nii.gz\n"
        "\nNote: Do NOT use --ml option for individual file output."
    )

def extract_eat_pat(ct_path, totalseg_dir, output_dir, 
                    dilation_radius_mm=DILATION_RADIUS_MM,
                    use_proximal_vessels=True,
                    verbose=False):
    """
    EAT+PAT（心膜周囲脂肪）を抽出（統合版）
    
    Args:
        ct_path: CT画像のパス
        totalseg_dir: TotalSegmentator出力ディレクトリ
        output_dir: 出力ディレクトリ
        dilation_radius_mm: 心臓マスク膨張半径 [mm]
        use_proximal_vessels: 大血管近位部を含めるか
        verbose: 詳細出力
    
    Returns:
        stats: 統計情報の辞書
    """
    print("\n[EAT+PAT Extraction with TotalSegmentator - Integrated]")
    print("="*60)
    
    # 出力ディレクトリ作成（統合ディレクトリ構造）
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サブディレクトリ作成
    masks_dir = output_dir / "masks"
    stats_dir = output_dir / "statistics"
    masks_dir.mkdir(exist_ok=True)
    stats_dir.mkdir(exist_ok=True)
    
    # 1. CT画像読み込み
    print("1. Loading CT data...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()[:3]
    
    # float32をfloatに変換
    spacing = tuple(float(s) for s in spacing)
    
    print(f"  CT shape: {ct_data.shape}")
    print(f"  Spacing: {spacing} mm")
    
    # 2. 心臓マスクの構築
    print("\n2. Building heart mask...")
    heart_mask, affine = load_heart_components(totalseg_dir, verbose)
    
    # オプション: 大血管近位部の追加
    if use_proximal_vessels:
        print("\n3. Incorporating proximal vessels...")
        vessels = load_vessel_masks(totalseg_dir, verbose)
        if vessels:
            heart_mask = incorporate_proximal_vessels(
                heart_mask, vessels, proximity_mm=5.0, spacing=spacing, verbose=verbose
            )
        else:
            print("  No vessel masks found")
    
    # 心臓マスクを保存
    heart_path = masks_dir / "heart.nii.gz"
    heart_img = nib.Nifti1Image(heart_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(heart_img, str(heart_path))
    print(f"  Heart mask saved: {heart_path}")
    
    # 3. 内臓脂肪マスクの読み込み（TotalSegmentator tissue_typesタスク - 必須）
    print("\n4. Loading visceral fat mask (torso_fat) ...")
    fat_mask, _ = load_visceral_fat_mask(totalseg_dir, verbose=verbose)
    print(f"  Visceral fat voxels: {int(np.sum(fat_mask)):,}")
    
    # 内臓脂肪マスクも保存
    visceral_fat_path = masks_dir / "visceral_fat.nii.gz"
    visceral_fat_img = nib.Nifti1Image(fat_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(visceral_fat_img, str(visceral_fat_path))
    print(f"  Visceral fat mask saved: {visceral_fat_path}")
    
    # 4. 心臓高さ近辺の脂肪に限定（改善版：解剖学的境界を使用）
    print("\n5. Restricting fat to anatomical boundaries...")
    
    # 下端：心臓マスクが存在する最も尾側のZ
    z_indices = np.where(np.any(heart_mask, axis=(0, 1)))[0]
    if len(z_indices) > 0:
        z_bottom_heart = z_indices[0]  # 最も尾側（小さいZ）
        z_basic_top = z_indices[-1]    # 基本的な上端（大きいZ）
    else:
        z_bottom_heart = 0
        z_basic_top = heart_mask.shape[2] - 1
    
    # 上端：近位大血管を考慮した解剖学的上限
    z_top = compute_intrapericardial_zmax(
        totalseg_dir, heart_mask, spacing, 
        prox_mm=5.0, z_margin_slices=1, verbose=verbose
    )
    
    # 下端には少しマージンを追加（横隔膜側への余裕）
    margin_bottom = 3
    z_min = max(0, z_bottom_heart - margin_bottom)
    z_max = z_top
    
    print(f"  Anatomical Z-range: {z_min} to {z_max} (total: {z_max - z_min + 1} slices)")
    print(f"    Bottom (diaphragm): {z_min} (heart bottom: {z_bottom_heart})")
    print(f"    Top (great vessels): {z_max}")
    
    # Z範囲外をマスク
    fat_near_heart = np.zeros_like(fat_mask, dtype=bool)
    fat_near_heart[:, :, z_min:z_max+1] = fat_mask[:, :, z_min:z_max+1]
    
    # 5. 心膜周囲脂肪の抽出（EDTベースの等方的膨張）
    print(f"\n6. Extracting pericardial fat (EDT-based dilation: {dilation_radius_mm} mm)...")
    
    # ユークリッド距離変換による等方的膨張
    print(f"  Computing Euclidean Distance Transform...")
    dist_to_heart = distance_transform_edt(~heart_mask, sampling=spacing)
    heart_dilated = dist_to_heart <= dilation_radius_mm
    
    print(f"  Heart dilated voxels: {np.sum(heart_dilated):,}")
    
    # Shell（心臓周囲の殻状領域）を計算
    shell_mask = heart_dilated & ~heart_mask
    
    # Shell領域も解剖学的境界でクリップ（上端・下端の制限を適用）
    shell_mask[:, :, :z_min] = False
    shell_mask[:, :, z_max+1:] = False
    fat_near_heart[:, :, :z_min] = False  # 念のため脂肪も再度クリップ
    fat_near_heart[:, :, z_max+1:] = False
    
    print(f"  Shell clipped to anatomical boundaries")
    
    # Shellマスクを保存
    shell_path = masks_dir / "shell.nii.gz"
    shell_img = nib.Nifti1Image(shell_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(shell_img, str(shell_path))
    print(f"  Shell mask saved: {shell_path}")
    
    # 心臓周囲の脂肪を抽出（Shell内の脂肪）
    pericardial_fat_mask = shell_mask & fat_near_heart
    
    # EAT+PATマスクを保存
    eat_pat_path = masks_dir / "eat_pat.nii.gz"
    eat_pat_img = nib.Nifti1Image(pericardial_fat_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(eat_pat_img, str(eat_pat_path))
    print(f"  EAT+PAT mask saved: {eat_pat_path}")
    
    # 6. 統計量の計算
    print("\n7. Calculating statistics...")
    
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
    
    # 7. 統計をJSON形式で保存
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
            "ct_shape": list(ct_data.shape),
            "ct_spacing_mm": list(spacing),
            "heart_z_range": [int(z_min), int(z_max)]
        }
    }
    
    stats_path = stats_dir / "eat_pat_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n8. Statistics saved: {stats_path}")
    
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
        description="Extract EAT+PAT using TotalSegmentator outputs (Integrated version)",
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
        stats = extract_eat_pat(
            ct_path=args.ct_path,
            totalseg_dir=args.totalseg_dir,
            output_dir=args.output_dir,
            dilation_radius_mm=args.dilation_radius,
            use_proximal_vessels=not args.no_vessels,
            verbose=args.verbose
        )
        
        print("\n" + "="*60)
        print("EAT+PAT extraction completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print(f"Total EAT+PAT volume: {stats['volumes_ml']['eat_pat']:.2f} ml")
        print(f"Shell volume: {stats['volumes_ml']['shell']:.2f} ml")
        print(f"Fat fraction in shell: {stats['ratios_percent']['fat_fraction_in_shell']:.1f}%")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()