#!/usr/bin/env python3
"""
最小限テスト: 非常に低解像度でDiffDRR動作確認
実用解像度での問題を切り分けるため、まず動作する最小構成を確認
"""

import torch
from diffdrr.data import read
from diffdrr.drr import DRR
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 最小設定 ============
MASK_NII = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"

# 非常に低解像度でテスト
SID = 1800.0                
PIX = 2.0                   # 大きなピクセルサイズ
W, H = 128, 128             # 超低解像度

alpha_deg, beta_deg, gamma_deg = 0.0, 0.0, 90.0

def main():
    """最小限テスト"""
    logger.info("=== 最小限DiffDRRテスト開始 ===")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    logger.info(f"🚀 デバイス: {device}, 解像度: {W}x{H}")
    
    # メモリクリア
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # マスク読み込み
    logger.info("📁 マスク読み込み中...")
    subject = read(
        volume=MASK_NII,
        orientation="PA",
        center_volume=False
    )
    logger.info(f"✅ マスク: {subject['volume'].shape}")
    
    # DRRセットアップ（最小構成）
    logger.info("⚙️  最小DRRセットアップ中...")
    try:
        drr = DRR(
            subject,
            sdd=SID,
            height=H,
            width=W,
            delx=PIX, 
            dely=PIX,
            renderer="siddon",              
            reducefn="sum",
        ).to(device=device, dtype=dtype)
        
        logger.info(f"✅ DRR OK: {W}x{H}")
        
    except Exception as e:
        logger.error(f"❌ DRRセットアップ失敗: {e}")
        return
    
    # 角度設定
    alpha = np.deg2rad(alpha_deg)
    beta  = np.deg2rad(90.0 - beta_deg) 
    gamma = np.deg2rad(-gamma_deg)
    
    angles = torch.tensor([[alpha, beta, gamma]], device=device, dtype=dtype)
    translations = torch.zeros((1, 3), device=device, dtype=dtype)
    
    logger.info(f"📐 角度: DiffDRR({np.degrees(alpha):.1f}°,{np.degrees(beta):.1f}°,{np.degrees(gamma):.1f}°)")
    
    # 投影実行
    logger.info("🎯 最小投影実行中...")
    try:
        with torch.inference_mode():
            img = drr(angles, translations, parameterization="euler_angles", convention="YZX")
            img = img.squeeze().to(torch.float32)
        
        logger.info(f"✅ 投影完了: {img.shape}, 範囲[{img.min():.3f}, {img.max():.3f}]")
        
        # 結果保存
        output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        x = img.cpu().numpy()
        
        # 正規化保存
        if x.max() > 0:
            x_norm = x / x.max()
            u16 = (x_norm * 65535).astype(np.uint16)
            
            test_png = output_dir / f"minimal_test_{W}x{H}.png"
            imageio.imwrite(test_png, u16)
            logger.info(f"💾 テスト画像保存: {test_png}")
            
            # 統計情報
            nonzero = x[x > 0]
            if len(nonzero) > 0:
                logger.info(f"📊 統計: 非ゼロ={len(nonzero)}, max={x.max():.3f}, mean={nonzero.mean():.3f}")
            else:
                logger.warning("⚠️  全ピクセルがゼロです")
        else:
            logger.warning("⚠️  投影結果が全てゼロです")
        
        logger.info("🎉 最小限テスト成功!")
        
    except Exception as e:
        logger.error(f"❌ 投影失敗: {e}")
        return

if __name__ == "__main__":
    main()