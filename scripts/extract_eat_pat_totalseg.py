#!/usr/bin/env python3
"""
TotalSegmentatorを使用したEAT+PAT抽出
md_files/EAT_PAT_visceral_fat_totalsegmentator.mdに基づく実装

EAT (Epicardial Adipose Tissue): 心外膜脂肪（心膜内）
PAT (Paracardial Adipose Tissue): 心膜外脂肪
EAT+PAT: 心膜周囲脂肪の総量として測定
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
    EAT+PAT（心膜周囲脂肪）を抽出
    
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
    print("\n[EAT+PAT Extraction]")
    print("="*60)
    
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 3. 内臓脂肪マスクの読み込み（TotalSegmentator tissue_typesタスク - 必須）
    print("\n4. Loading visceral fat mask (torso_fat) ...")
    fat_mask, _ = load_visceral_fat_mask(totalseg_dir, verbose=verbose)
    print(f"  Visceral fat voxels: {int(np.sum(fat_mask)):,}")
    
    # 内臓脂肪マスクも保存（可視化用）
    visceral_fat_path = output_dir / "visceral_fat_mask.nii.gz"
    visceral_fat_img = nib.Nifti1Image(fat_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(visceral_fat_img, str(visceral_fat_path))
    print(f"  Visceral fat mask saved: {visceral_fat_path}")
    
    # 4. 心臓高さ近辺の脂肪に限定
    print("\n5. Restricting fat to heart level...")
    z_indices = np.where(np.any(heart_mask, axis=(0, 1)))[0]
    if len(z_indices) > 0:
        z_min, z_max = z_indices[0], z_indices[-1]
    else:
        z_min, z_max = 0, heart_mask.shape[2] - 1
    
    # 上下に少しマージンを追加
    margin_slices = 5
    z_min = max(0, z_min - margin_slices)
    z_max = min(heart_mask.shape[2] - 1, z_max + margin_slices)
    
    print(f"  Heart Z-range: {z_min} to {z_max} (total: {z_max - z_min + 1} slices)")
    
    # Z範囲外をマスク
    fat_near_heart = np.zeros_like(fat_mask, dtype=bool)
    fat_near_heart[:, :, z_min:z_max+1] = fat_mask[:, :, z_min:z_max+1]
    
    # 5. 心膜周囲脂肪の抽出（心臓マスクを膨張）
    print(f"\n6. Extracting pericardial fat (dilation: {dilation_radius_mm} mm)...")
    
    # 膨張回数の計算
    min_spacing = min(spacing)
    iterations = int(np.ceil(dilation_radius_mm / min_spacing))
    print(f"  Dilation iterations: {iterations} (min spacing: {min_spacing} mm)")
    
    # 3D構造要素（26近傍）
    struct = generate_binary_structure(3, 2)
    
    # 心臓マスクを膨張
    heart_dilated = binary_dilation(heart_mask, structure=struct, iterations=iterations)
    
    # 心臓周囲の脂肪を抽出
    pericardial_fat_mask = heart_dilated & fat_near_heart
    
    # 心臓内部の脂肪は除外（心筋内の脂肪浸潤は除く）
    pericardial_fat_mask &= ~heart_mask
    
    # 6. 統計量の計算
    print("\n7. Calculating statistics...")
    
    # ボクセル体積 [mm³]
    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
    
    # EAT+PATの統計
    eat_pat_voxels = int(np.sum(pericardial_fat_mask))
    eat_pat_volume_ml = voxel_vol_mm3 * eat_pat_voxels / 1000.0
    
    # 内臓脂肪全体の統計
    visceral_fat_voxels = int(np.sum(fat_mask))
    visceral_fat_volume_ml = voxel_vol_mm3 * visceral_fat_voxels / 1000.0
    
    print(f"  EAT+PAT volume: {eat_pat_volume_ml:.2f} ml")
    print(f"  EAT+PAT voxels: {eat_pat_voxels:,}")
    print(f"  Total visceral fat volume: {visceral_fat_volume_ml:.2f} ml")
    print(f"  EAT+PAT / Visceral fat ratio: {(eat_pat_volume_ml/visceral_fat_volume_ml*100):.1f}%")
    
    # 7. 結果の保存
    print("\n8. Saving results...")
    
    # マスクをNIfTI形式で保存
    mask_path = output_dir / "eat_pat_mask.nii.gz"
    mask_img = nib.Nifti1Image(pericardial_fat_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(mask_img, str(mask_path))
    print(f"  Mask saved: {mask_path}")
    
    # 心臓膨張マスクも保存（デバッグ用）
    heart_dilated_path = output_dir / "heart_dilated.nii.gz"
    heart_dilated_img = nib.Nifti1Image(heart_dilated.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(heart_dilated_img, str(heart_dilated_path))
    
    # 統計をJSON形式で保存（float32をfloatに変換）
    stats = {
        "eat_pat_volume_ml": round(eat_pat_volume_ml, 2),
        "eat_pat_voxels": eat_pat_voxels,
        "visceral_fat_volume_ml": round(visceral_fat_volume_ml, 2),
        "visceral_fat_voxels": visceral_fat_voxels,
        "eat_pat_to_visceral_ratio": round(eat_pat_volume_ml/visceral_fat_volume_ml*100, 2) if visceral_fat_volume_ml > 0 else 0,
        "dilation_radius_mm": float(dilation_radius_mm),
        "ct_shape": list(ct_data.shape),
        "ct_spacing": list(spacing),
        "heart_voxels": int(np.sum(heart_mask)),
        "z_range": [int(z_min), int(z_max)]
    }
    
    stats_path = output_dir / "eat_pat_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved: {stats_path}")
    
    # 検証：正常範囲チェック
    print("\n[Validation]")
    if 50 <= eat_pat_volume_ml <= 200:
        print(f"  ✓ EAT+PAT volume is within normal range (50-200 ml)")
    elif eat_pat_volume_ml < 50:
        print(f"  ⚠ EAT+PAT volume is below normal range (< 50 ml)")
    else:
        print(f"  ⚠ EAT+PAT volume is above normal range (> 200 ml)")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Extract EAT+PAT using TotalSegmentator outputs",
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
        print(f"Total EAT+PAT volume: {stats['eat_pat_volume_ml']:.2f} ml")
        print(f"Total visceral fat volume: {stats['visceral_fat_volume_ml']:.2f} ml")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()