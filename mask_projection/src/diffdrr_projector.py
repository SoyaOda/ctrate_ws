#!/usr/bin/env python3
"""
DiffDRR基本マスク投影機能
マスクのSiddon線積分投影により厚みマップを生成
"""

import numpy as np
import torch
import torchio as tio
import nibabel as nib
from pathlib import Path
import imageio.v2 as imageio
import logging
from diffdrr.drr import DRR
from diffdrr.data import read

# LPS正規化機能をインポート
from lps_normalizer import to_lps_array_and_spacing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_diffdrr_subject_from_file(mask_path):
    """
    diffdrr.data.readを使ってSubject構築（gpt5pro2.mdの最小コード準拠）
    
    Args:
        mask_path: マスクファイルのパス
        
    Returns:
        Subject: DiffDRR用のSubjectオブジェクト
    """
    logger.info(f"マスクファイルから読み込み: {mask_path}")
    
    # diffdrr.data.readを使用
    subject = read(
        volume=str(mask_path),    # 0/1 の三次元マスク（1=目的部位）
        orientation="PA",         # radiographic view 規約を使う
        center_volume=True        # ワールド原点に移す（既定）
    )
    
    logger.info(f"Subject loaded: {type(subject)}")
    logger.info(f"Volume shape: {subject.volume.shape}")
    logger.info(f"Volume spacing: {subject.volume.spacing}")
    
    return subject

def save_temp_mask(mask_lps, aff_lps, temp_path):
    """
    一時的なマスクファイルを保存
    
    Args:
        mask_lps: LPS正規化されたマスクデータ
        aff_lps: LPS affine行列  
        temp_path: 保存先パス
    """
    # nibabelでNifTI形式で保存
    mask_img = nib.Nifti1Image(mask_lps, aff_lps)
    nib.save(mask_img, str(temp_path))
    logger.info(f"一時マスクファイル保存: {temp_path}")
    logger.info(f"保存データ形状: {mask_lps.shape}")
    logger.info(f"保存データ値域: {mask_lps.min():.3f} ~ {mask_lps.max():.3f}")
    logger.info(f"保存データユニーク値: {np.unique(mask_lps)}")
    
    return temp_path

def project_mask_basic(subj, sdd=1800.0, height=512, width=512, pixel_size=0.7):
    """
    基本的なマスク投影（Siddon線積分）
    
    Args:
        subj: TorchIO Subject
        sdd: Source-to-Detector Distance [mm]
        height: 検出器高さ [pixels]
        width: 検出器幅 [pixels] 
        pixel_size: ピクセルサイズ [mm/pixel]
        
    Returns:
        np.ndarray: 投影された厚みマップ
    """
    logger.info(f"投影パラメータ:")
    logger.info(f"  SDD: {sdd} mm")
    logger.info(f"  検出器サイズ: {width} x {height} pixels")
    logger.info(f"  ピクセルサイズ: {pixel_size} mm/pixel")
    
    # DiffDRRモジュール（Siddon：線積分）
    drr = DRR(
        subject=subj,
        sdd=sdd,
        height=height,
        width=width,
        delx=pixel_size,
        dely=pixel_size,
        reverse_x_axis=True,  # radiographic view に準拠（心陰影は右）
        renderer='siddon',    # 厚み＝線積分
        reducefn='sum'        # 線積分の総和
    )
    
    logger.info("DRRモジュール初期化完了")
    
    # center_volume=Trueで既に中心化されているため、原点中心を使用
    logger.info("center_volume=Trueで既に中心化済み、原点中心で投影")
    
    # 基本的な正面投影（PA）
    # 角度：yaw=0, pitch=0, roll=0（正面）
    pose_angles = torch.tensor([[0., 0., 0.]], dtype=torch.float32)
    translation = torch.tensor([[0., 0., 0.]], dtype=torch.float32)  # 原点中心
    
    logger.info("マスク投影実行中...")
    
    try:
        with torch.no_grad():
            # マスク体積の線積分投影（直接厚み取得）
            thickness_proj = drr.forward(
                pose_angles, translation,
                parameterization="euler_angles",
                convention="ZXY"
            )
        
        # 厚み [mm] を取得
        thickness_mm = thickness_proj[0, 0].cpu().numpy()
        
        logger.info(f"投影結果形状: {thickness_mm.shape}")
        logger.info(f"厚み値域: {thickness_mm.min():.3f} ~ {thickness_mm.max():.3f} mm")
        
        return thickness_mm
        
    except Exception as e:
        logger.error(f"投影エラー: {e}")
        raise

def test_basic_mask_projection():
    """
    基本的なマスク投影のテスト
    """
    logger.info("=== DiffDRR基本マスク投影テスト開始 ===")
    
    # ファイルパス
    ct_path = Path("/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz")
    mask_path = Path("/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz")
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not ct_path.exists() or not mask_path.exists():
        logger.error("必要なファイルが見つかりません")
        return
    
    try:
        # ファイル読み込み
        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))
        
        # LPS正規化
        ct_lps, ct_spacing, ct_aff_lps, ct_to_lps = to_lps_array_and_spacing(ct_img)
        mask_lps = mask_img.get_fdata().astype(np.float32)
        
        # 一時マスクファイルを保存
        temp_mask_path = output_dir / "temp_mask_lps.nii.gz"
        save_temp_mask(mask_lps, ct_aff_lps, temp_mask_path)
        
        # DiffDRR Subject 構築（ファイルから読み込み）
        subj = build_diffdrr_subject_from_file(temp_mask_path)
        
        # 基本投影（小さなサイズでテスト）
        thickness_map = project_mask_basic(
            subj,
            sdd=1800.0,
            height=256,  # テスト用に小さなサイズ
            width=256,
            pixel_size=1.4  # テスト用に大きなピクセルサイズ
        )
        
        # 結果を16bit PNGで保存
        thickness_normalized = np.clip(thickness_map / max(thickness_map.max(), 1e-6), 0, 1)
        thickness_u16 = (thickness_normalized * 65535).astype(np.uint16)
        
        output_file = output_dir / "basic_mask_projection_test.png"
        imageio.imwrite(output_file, thickness_u16)
        
        # 一時ファイルをクリーンアップ
        if temp_mask_path.exists():
            temp_mask_path.unlink()
            logger.info("一時マスクファイルを削除")
        
        logger.info(f"✅ 基本投影テスト完了")
        logger.info(f"出力ファイル: {output_file}")
        logger.info(f"最大厚み: {thickness_map.max():.3f} mm")
        
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        return False

if __name__ == "__main__":
    test_basic_mask_projection()