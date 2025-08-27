#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EAT segmentation with DeepHeartSeg pericardial segmentation (wrapper).
- Runs DeepHeartSeg (optional) and reads its outputs.
- Reconstructs a watertight 'inside-pericardium' volume.
- Computes EAT = (inside-pericardium) ∧ (fat HU) ∧ (not myocardium),
  with connectivity to myocardium (optional) and strong exclusion gates.
"""

import os, sys, re, json, subprocess
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_dilation, binary_erosion, binary_fill_holes, label,
    distance_transform_edt, generate_binary_structure
)

def load_nii(path):
    img = nib.load(str(path))
    return img, img.get_fdata(), img.header.get_zooms()[:3]

def save_mask_like(ref_img, mask, out_path):
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine), str(out_path))

def find_first_matching(base_dir: Path, patterns):
    for pat in patterns:
        for p in sorted(base_dir.glob(pat)):
            return p
    return None

def mm_to_iters(mm, spacing):
    return max(1, int(np.ceil(mm / min(spacing))))

def run_deepheartseg(ct_path: Path, out_dir: Path, repo_root: Path):
    script = repo_root / "run_deepheartseg.sh"
    if not script.exists():
        raise FileNotFoundError(f"DeepHeartSeg runner not found: {script}")
    cmd = [str(script), str(ct_path), str(out_dir)]
    subprocess.check_call(cmd)

def reconstruct_inside_from_pericardium(peri_mask, heart_seed, spacing):
    """
    peri_mask: 心膜（境界 or 薄い帯）or 心膜内（filled）のどちらでも OK
    heart_seed: 既知の"内側"シード（心筋＋心腔など）
    """
    peri = peri_mask.astype(bool)
    # "線状/薄膜"かどうか推定：体積比と厚みでざっくり判定
    peri_ratio = peri.mean()
    is_thin = peri_ratio < 0.02  # かなり薄いなら境界の可能性が高い

    if is_thin:
        # 薄い境界→ 少し太らせて閉曲面化
        it = mm_to_iters(1.0, spacing)
        peri_thick = binary_dilation(peri, iterations=it)
        # Heart seed から域内を決定：境界外には漏れないように
        inside = binary_fill_holes(peri_thick)  # 万一の穴塞ぎ
        # "心筋＋心腔"シードが必ず inside に入ることを保証
        inside = inside | heart_seed
        # 外の連結成分は除去
        # （境界が完全でない場合の保険：心筋シードと同一 CC のみ残す）
        cc, n = label(inside)
        if n > 1:
            seed_label = np.argmax(np.bincount(cc[heart_seed].ravel().astype(int))[1:]) + 1
            inside = (cc == seed_label)
    else:
        # 既に filled とみなす
        inside = binary_fill_holes(peri)

    return inside

def main():
    import argparse
    ap = argparse.ArgumentParser(description="EAT via DeepHeartSeg pericardial segmentation")
    ap.add_argument("ct_nifti")
    ap.add_argument("totalseg_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--deepheartseg-root", default=str(Path("tools/DeepHeartSeg").absolute()))
    ap.add_argument("--deepheartseg-out", default=None, help="where DeepHeartSeg drops its NIfTI masks")
    ap.add_argument("--run-deepheartseg", action="store_true", help="actually run DeepHeartSeg")
    ap.add_argument("--hu-min", type=float, default=-190.0)
    ap.add_argument("--hu-max", type=float, default=-30.0)
    ap.add_argument("--contact-mm", type=float, default=3.0, help="EAT must contact myocardium within this distance")
    ap.add_argument("--strict-exclusion", action="store_true", help="remove any component touching forbidden structures")
    args = ap.parse_args()

    ct_path = Path(args.ct_nifti).absolute()
    out_dir = Path(args.output_dir).absolute(); out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.deepheartseg_root).absolute()
    dh_out = Path(args.deepheartseg_out).absolute() if args.deepheartseg_out else (out_dir / "deepheartseg")

    if args.run_deepheartseg:
        dh_out.mkdir(parents=True, exist_ok=True)
        run_deepheartseg(ct_path, dh_out, repo_root)

    # --- load CT ---
    ct_img, ct, spacing = load_nii(ct_path)

    # --- load myocardium (TotalSegmentator) ---
    ts = Path(args.totalseg_dir)
    myo_path = ts / "heart_myocardium.nii.gz"
    if not myo_path.exists():
        # フォールバック
        myo_path = ts / "heart.nii.gz"
    if not myo_path.exists():
        raise FileNotFoundError("TotalSegmentator myocardium/heart mask not found")

    _, myo, _ = load_nii(myo_path)
    myo = myo > 0

    # --- DeepHeartSeg outputs: pericardium or heart/EAT ---
    # 優先：pericardium* / *pericardial* / *heart_sac*
    if not dh_out.exists():
        raise FileNotFoundError(f"[ERROR] DeepHeartSeg output directory does not exist: {dh_out}")
    
    cand_peri = find_first_matching(dh_out, ["*pericardium*.nii*", "*pericardial*.nii*", "*heart_sac*.nii*"])
    cand_eat  = find_first_matching(dh_out, ["*EAT*.nii*","*epicardial*fat*.nii*"])
    cand_heart= find_first_matching(dh_out, ["heart.nii*", "*heart*mask*.nii*"])

    if cand_peri is None and cand_heart is None and cand_eat is None:
        raise FileNotFoundError(f"[ERROR] No DeepHeartSeg outputs found in: {dh_out}\n"
                               "Please run DeepHeartSeg first or check the output directory.")

    if cand_peri is not None:
        _, peri_mask, _ = load_nii(cand_peri); peri_mask = peri_mask > 0
        # "心臓コア"シード：心筋＋（あれば）心腔
        heart_seed = myo.copy()
        # 心膜内再構成
        inside_peri = reconstruct_inside_from_pericardium(peri_mask, heart_seed, spacing)
    elif cand_heart is not None:
        # 心臓マスクがある場合
        _, heart_core, _ = load_nii(cand_heart)
        heart_core = heart_core > 0
        band = binary_dilation(heart_core, iterations=mm_to_iters(2.0, spacing))
        inside_peri = binary_fill_holes(band)
    elif cand_eat is not None:
        # EATが直接ある場合
        _, eat_dh, _ = load_nii(cand_eat)
        # EAT が直接ある場合は、内包領域として "心筋 + EAT + 心腔（仮に myo 膨張）"を使う
        heart_core = binary_dilation(myo, iterations=mm_to_iters(2.0, spacing))
        band = binary_dilation(heart_core, iterations=mm_to_iters(2.0, spacing))
        inside_peri = binary_fill_holes(band)

    # --- fat HU ---
    hu_fat = (ct >= args.hu_min) & (ct <= args.hu_max)

    # --- EAT base ---
    eat_base = inside_peri & hu_fat & (~myo)

    # --- 接触制約：心筋から contact-mm 以内
    if args.contact_mm > 0:
        contact = binary_dilation(myo, iterations=mm_to_iters(args.contact_mm, spacing))
        eat_base &= contact

    # --- 禁制構造の除外（漏れ対策） ---
    # 利用可能であれば TotalSegmentator のマスクで除外（肺・皮下・骨・肝など）
    forbid = np.zeros_like(eat_base, dtype=bool)
    def add_mask(name, dil_mm=0.0):
        p = ts / f"{name}.nii.gz"
        if p.exists():
            _, m, _ = load_nii(p)
            m = m > 0
            if dil_mm > 0:
                m = binary_dilation(m, iterations=mm_to_iters(dil_mm, spacing))
            return m
        return np.zeros_like(eat_base, dtype=bool)

    # 胸腔外・肺・気道・骨近傍・肝・脾・胃・腎を広めに除外
    forbid |= add_mask("lung_upper_lobe_left")
    forbid |= add_mask("lung_lower_lobe_left")
    forbid |= add_mask("lung_upper_lobe_right")
    forbid |= add_mask("lung_middle_lobe_right")
    forbid |= add_mask("lung_lower_lobe_right")
    forbid |= add_mask("trachea", 2.0) | add_mask("main_bronchus_left", 2.0) | add_mask("main_bronchus_right", 2.0)
    # 骨近傍
    forbid |= add_mask("sternum", 3.0)
    for v in ["T%d"%i for i in range(1,13)] + ["L%d"%i for i in range(1,6)]:
        forbid |= add_mask(f"vertebrae_{v}", 2.0)
    # 腹部臓器
    for org in ["liver","spleen","stomach","kidney_left","kidney_right","gallbladder","pancreas"]:
        forbid |= add_mask(org, 2.0)

    eat = eat_base & (~forbid)

    # 触れた成分を丸ごと落とす（厳格モード）
    if args.strict_exclusion:
        cc, n = label(eat)
        if n > 0:
            kept = np.zeros_like(eat, bool)
            for k in range(1, n+1):
                comp = (cc == k)
                if not np.any(comp & forbid):
                    kept |= comp
            eat = kept

    # --- 保存とサマリ ---
    out_mask = out_dir / "eat_mask.nii.gz"
    out_inside = out_dir / "pericardial_inside.nii.gz"
    save_mask_like(ct_img, eat, out_mask)
    save_mask_like(ct_img, inside_peri, out_inside)

    vox_vol = np.prod(spacing)
    eat_vox = int(eat.sum()); eat_ml = eat_vox * vox_vol / 1000.0
    vals = ct[eat]; mu = float(np.mean(vals)) if vals.size else None; sd = float(np.std(vals)) if vals.size else None

    with open(out_dir / "eat_summary.json", "w") as f:
        json.dump({
            "eat_volume_ml": round(eat_ml, 2),
            "eat_voxels": eat_vox,
            "mean_hu": mu,
            "std_hu": sd,
            "hu_range": [args.hu_min, args.hu_max],
            "contact_mm": args.contact_mm,
            "strict_exclusion": bool(args.strict_exclusion)
        }, f, indent=2)

    print(f"[DONE] EAT {eat_ml:.2f} ml  (voxels={eat_vox})  meanHU={mu}±{sd}")

if __name__ == "__main__":
    main()