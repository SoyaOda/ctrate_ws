#!/usr/bin/env python3
"""
LPS正規化機能
CT と Mask を厳密 LPS に正規化し、配列とaffineの同期変換を行う
"""

import numpy as np
import nibabel as nib
from nibabel import orientations as nio
from nibabel.affines import voxel_sizes
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_lps_array_and_spacing(img):
    """
    nibabel 画像を LPS へ正規化し、(data, spacing, aff_lps, to_lps) を返す
    
    Args:
        img: nibabel.Nifti1Image
        
    Returns:
        tuple: (data_lps, spacing, aff_lps, to_lps)
    """
    logger.info(f"Input image shape: {img.shape}")
    logger.info(f"Input image affine:\n{img.affine}")
    
    # 現在の orientation を取得
    in_ornt = nio.io_orientation(img.affine)
    logger.info(f"Input orientation: {in_ornt}")
    
    # LPS orientation を定義
    out_ornt = nio.axcodes2ornt(('L','P','S'))
    logger.info(f"Target orientation (LPS): {out_ornt}")
    
    # 変換行列を計算
    to_lps = nio.ornt_transform(in_ornt, out_ornt)
    logger.info(f"Transformation matrix: {to_lps}")
    
    # データをLPSに変換
    data_lps = nio.apply_orientation(img.get_fdata().astype(np.float32), to_lps)
    logger.info(f"LPS data shape: {data_lps.shape}")
    
    # affineをLPSに変換
    aff_lps = img.affine @ nio.inv_ornt_aff(to_lps, img.shape)
    logger.info(f"LPS affine:\n{aff_lps}")
    
    # spacing を計算
    spacing = voxel_sizes(aff_lps)[:3]  # (sx, sy, sz) [mm]
    logger.info(f"Voxel spacing [mm]: {spacing}")
    
    return data_lps, spacing, aff_lps, to_lps

def test_lps_normalization(ct_path, mask_path):
    """
    LPS正規化のテスト関数
    
    Args:
        ct_path: CTファイルのパス
        mask_path: マスクファイルのパス
    """
    logger.info("=== LPS正規化テスト開始 ===")
    
    # CTファイルを読み込み
    logger.info(f"Loading CT from: {ct_path}")
    ct_img = nib.load(str(ct_path))
    
    # マスクファイルを読み込み
    logger.info(f"Loading mask from: {mask_path}")
    mask_img = nib.load(str(mask_path))
    
    # CTをLPS正規化
    logger.info("\n--- CT LPS正規化 ---")
    ct_lps, ct_spacing, ct_aff_lps, ct_to_lps = to_lps_array_and_spacing(ct_img)
    
    # マスクを同じ変換でLPS正規化
    logger.info("\n--- マスク LPS正規化 ---")
    logger.info("CTと同じ変換行列を使用してマスクを変換")
    mask_lps = nio.apply_orientation(mask_img.get_fdata().astype(np.float32), ct_to_lps)
    
    # 結果の確認
    logger.info("\n=== 変換結果 ===")
    logger.info(f"CT原画像形状: {ct_img.shape}")
    logger.info(f"CT LPS形状: {ct_lps.shape}")
    logger.info(f"マスク原画像形状: {mask_img.shape}")
    logger.info(f"マスク LPS形状: {mask_lps.shape}")
    
    # データ値域の確認
    logger.info(f"\nCT値域: {ct_lps.min():.2f} ~ {ct_lps.max():.2f}")
    logger.info(f"マスク値域: {mask_lps.min():.2f} ~ {mask_lps.max():.2f}")
    logger.info(f"マスク内のユニークな値: {np.unique(mask_lps)}")
    
    return ct_lps, mask_lps, ct_spacing, ct_aff_lps, ct_to_lps

if __name__ == "__main__":
    # テスト実行
    ct_path = Path("/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    mask_path = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    
    if ct_path.exists() and mask_path.exists():
        test_lps_normalization(ct_path, mask_path)
    else:
        logger.error("必要なファイルが見つかりません")
        if not ct_path.exists():
            logger.error(f"CT file not found: {ct_path}")
        if not mask_path.exists():
            logger.error(f"Mask file not found: {mask_path}")