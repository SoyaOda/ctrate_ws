#!/usr/bin/env python3
"""
クイックテスト: 修正された設定で低解像度テスト
center_volume=True + by=-SOD が正しく機能するかを確認
"""

import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import logging

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """クイックテスト"""
    logger.info("=== クイックテスト: 修正設定確認 ===")
    
    # 低解像度設定
    SID = 1800.0
    SOD = 1720.0  # SID - OID
    PX = 2.0      # 大きなピクセル
    W, H = 256, 256  # 低解像度
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    logger.info(f"📐 テスト設定: {W}×{H}, SOD={SOD}, PX={PX}")
    
    # マスク読み込み（修正版）
    subject = read(
        volume="/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz",
        orientation="PA",
        center_volume=True,  # 修正
    )
    
    # DRR（パッチなし）
    drr = DRR(
        subject,
        sdd=SID,
        height=H, width=W,
        delx=PX, dely=PX,
        renderer="siddon",
        reducefn="sum",
        reverse_x_axis=True,
    ).to(device=device, dtype=dtype)
    
    # 角度・平行移動（修正版）
    rot = torch.tensor([[0.0, np.deg2rad(90.0), np.deg2rad(-90.0)]], device=device, dtype=dtype)
    xyz = torch.tensor([[0.0, -SOD, 0.0]], device=device, dtype=dtype)  # 修正
    
    logger.info(f"📍 by=-SOD: {-SOD}")
    
    pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
    
    # 投影
    logger.info("🎯 投影実行中...")
    with torch.inference_mode():
        thickness = drr(pose).squeeze().detach().cpu().numpy()
    
    # 結果確認
    max_val = thickness.max()
    nonzero = np.count_nonzero(thickness)
    
    logger.info(f"📊 結果: 範囲[{thickness.min():.3f}, {max_val:.3f}]")
    logger.info(f"   非ゼロ: {nonzero}/{thickness.size} ({100*nonzero/thickness.size:.1f}%)")
    
    if max_val > 0:
        logger.info("🎉 SUCCESS: 非ゼロ投影成功!")
        mean_nonzero = thickness[thickness > 0].mean()
        logger.info(f"   平均厚み: {mean_nonzero:.3f}mm")
        
        # 保存
        output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        thickness_norm = thickness / max_val
        u16 = (thickness_norm * 65535).astype(np.uint16)
        
        quick_png = output_dir / f"quick_test_{W}x{H}_success.png"
        imageio.imwrite(quick_png, u16)
        logger.info(f"💾 成功画像保存: {quick_png}")
        
    else:
        logger.error("❌ FAILED: まだ全てゼロ")
        
    logger.info("✅ クイックテスト完了")

if __name__ == "__main__":
    main()