#!/usr/bin/env python3
"""
Query回答に基づく高速・正確なマスク厚みDRR実装
- center_volume=False で原点ズレ解決
- GPU + Siddon で高速処理  
- 公式角度変換とPA水平反転で完全位置一致
- Beer–Lambert無しの純粋線積分（厚みmm）
"""

import torch
from diffdrr.data import read
from diffdrr.drr import DRR
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 設定 ============
MASK_NII = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
DEEPDRR_PA_PNG = "/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png"

# DeepDRRと同一幾何設定
SID = 1800.0                # Source-to-Detector distance [mm]
PIX = 0.194                 # pixel size [mm] (既存設定に合わせる)
W, H = 2500, 3070           # detector width/height [px]

# DeepDRRのPA最終姿勢: α=0, β=0, γ=90deg
alpha_deg, beta_deg, gamma_deg = 0.0, 0.0, 90.0

# radiographic view合わせ
PA_HORIZONTAL_FLIP = True   # 既存PA画像に合わせる

def main():
    """メイン処理"""
    logger.info("=== 高速・正確マスク厚みDRR開始 ===")
    
    # ============ デバイス設定 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.error("❌ CUDA利用不可。GPU必須です。")
        raise RuntimeError("GPU (CUDA) required for practical performance")
    
    dtype = torch.float16  # GPU半精度で省メモリ高速化
    logger.info(f"🚀 デバイス: {device}, 精度: {dtype}")
    
    # ============ 1) マスクをVolumeとして読む ============
    logger.info(f"📁 マスク読み込み: {MASK_NII}")
    
    # 🔑 KEY: center_volume=False で原点ズレ防止
    subject = read(
        volume=MASK_NII,
        orientation="PA",         # PA表記
        center_volume=False       # ★原点ズレ防止
    )
    logger.info(f"✅ マスク読み込み完了: 形状 {subject['volume'].shape}")
    
    # ============ 2) DRRモジュール（GPU + Siddon） ============
    logger.info("⚙️  DRRセットアップ中...")
    
    try:
        drr = DRR(
            subject,
            sdd=SID,
            height=H,
            width=W,
            delx=PIX, 
            dely=PIX,
            renderer="siddon",              # 正確な線積分
            reducefn="sum",                 # Beer–Lambert無し
            reverse_x_axis=True,            # radiographic view
        ).to(device=device, dtype=dtype)
        
        # torch.compile は互換性問題のため無効化
        logger.info("⚠️  torch.compile無効化（互換性問題回避）")
        
        logger.info(f"✅ DRRセットアップ完了: {W}x{H}, GPU={device.type}")
        
    except Exception as e:
        logger.error(f"❌ DRRセットアップ失敗: {e}")
        raise
    
    # ============ 3) 角度変換（公式準拠） ============
    logger.info("🔄 DeepDRR→DiffDRR角度変換")
    
    # 公式: α_diff = α, β_diff = 90°-β, γ_diff = -γ
    alpha = np.deg2rad(alpha_deg)           # 0°
    beta  = np.deg2rad(90.0 - beta_deg)     # 90° - 0° = 90°
    gamma = np.deg2rad(-gamma_deg)          # -90°
    
    angles = torch.tensor([[alpha, beta, gamma]], device=device, dtype=dtype)
    translations = torch.zeros((1, 3), device=device, dtype=dtype)  # 原点ズレなし
    
    logger.info(f"📐 角度変換: DeepDRR({alpha_deg}°,{beta_deg}°,{gamma_deg}°) → DiffDRR({np.degrees(alpha):.1f}°,{np.degrees(beta):.1f}°,{np.degrees(gamma):.1f}°)")
    
    # ============ 4) 投影実行 ============
    logger.info("🎯 マスク厚み投影実行中...")
    
    torch.set_float32_matmul_precision("high")
    
    try:
        with torch.inference_mode():
            img = drr(angles, translations, parameterization="euler_angles", convention="YZX")
            img = img.squeeze().to(torch.float32)  # (H,W)
        
        logger.info(f"✅ 投影完了: 形状{img.shape}, 範囲[{img.min():.3f}, {img.max():.3f}]")
        
    except Exception as e:
        logger.error(f"❌ 投影実行失敗: {e}")
        raise
    
    # ============ 5) PA水平反転（既存DeepDRRに合わせる） ============
    if PA_HORIZONTAL_FLIP:
        img = torch.flip(img, dims=[1])
        logger.info("🔄 PA水平反転適用")
    
    # ============ 6) 保存とオーバーレイ ============
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # マスク厚み画像保存
    x = img.cpu().numpy()
    
    # 正規化して16bit保存
    x_norm = np.clip(x / (x.max() + 1e-6), 0, 1)
    u16 = (x_norm * 65535).astype(np.uint16)
    
    thickness_png = output_dir / "mask_thickness_drr.png"
    imageio.imwrite(thickness_png, u16)
    logger.info(f"💾 マスク厚み画像保存: {thickness_png}")
    
    # ============ 7) DeepDRR画像読み込みとオーバーレイ ============
    if Path(DEEPDRR_PA_PNG).exists():
        logger.info("📖 既存DeepDRR画像読み込み")
        deepdrr_img = imageio.imread(DEEPDRR_PA_PNG)
        
        if deepdrr_img.ndim == 3:
            deepdrr_img = deepdrr_img[:,:,0]  # グレースケール化
        
        logger.info(f"✅ DeepDRR画像: 形状{deepdrr_img.shape}, 型{deepdrr_img.dtype}")
        
        # オーバーレイ可視化作成
        create_overlay_visualization(deepdrr_img, x, output_dir)
        
    else:
        logger.warning(f"⚠️  DeepDRR画像が見つかりません: {DEEPDRR_PA_PNG}")
    
    logger.info("🎉 高速・正確マスク厚みDRR完了!")

def create_overlay_visualization(deepdrr_img, mask_thickness, output_dir):
    """オーバーレイ可視化作成"""
    logger.info("🎨 オーバーレイ可視化作成中...")
    
    # 形状確認
    if deepdrr_img.shape != mask_thickness.shape:
        logger.warning(f"⚠️  形状不一致: DeepDRR{deepdrr_img.shape} vs マスク{mask_thickness.shape}")
        # 必要に応じてリサイズやクロップ処理を追加
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # DeepDRR画像
    axes[0].imshow(deepdrr_img, cmap='gray')
    axes[0].set_title('DeepDRR PA (既存)')
    axes[0].axis('off')
    
    # マスク厚み
    im1 = axes[1].imshow(mask_thickness, cmap='hot')
    axes[1].set_title('マスク厚み [mm]')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # オーバーレイ
    # DeepDRRを背景、マスクを半透明カラーで重ね合わせ
    deepdrr_norm = deepdrr_img / deepdrr_img.max()
    mask_norm = mask_thickness / (mask_thickness.max() + 1e-6)
    
    axes[2].imshow(deepdrr_norm, cmap='gray', alpha=0.7)
    axes[2].imshow(mask_norm, cmap='hot', alpha=0.5)
    axes[2].set_title('オーバーレイ (DeepDRR + マスク厚み)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    overlay_png = output_dir / "deepdrr_mask_overlay.png"
    plt.savefig(overlay_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"🎨 オーバーレイ保存: {overlay_png}")

if __name__ == "__main__":
    main()