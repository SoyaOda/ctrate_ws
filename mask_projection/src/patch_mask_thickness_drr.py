#!/usr/bin/env python3
"""
Patch分割によるマスク厚みDRR実装
CUDAメモリ不足を回避するため、適切なpatch_sizeで分割処理
"""

import torch
from diffdrr.data import read
from diffdrr.drr import DRR
import numpy as np
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 設定 ============
MASK_NII = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
DEEPDRR_PA_PNG = "/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png"

# DeepDRRと同一幾何設定
SID = 1800.0                # Source-to-Detector distance [mm]
PIX = 0.194                 # pixel size [mm]
W, H = 2500, 3070           # detector width/height [px]

# パッチサイズの候補（両辺を割り切る必要）
# 2500の約数: 1, 2, 4, 5, 10, 20, 25, 50, 100, 125, 250, 500, 625, 1250, 2500
# 3070の約数: 1, 2, 5, 10, 307, 614, 1535, 3070
PATCH_SIZE_CANDIDATES = [10, 5, 2, 1]  # 両辺を割り切る数値

# DeepDRRのPA最終姿勢
alpha_deg, beta_deg, gamma_deg = 0.0, 0.0, 90.0

# radiographic view合わせ
PA_HORIZONTAL_FLIP = True

def find_optimal_patch_size(width, height, candidates):
    """適切なパッチサイズを見つける"""
    for patch_size in candidates:
        if width % patch_size == 0 and height % patch_size == 0:
            logger.info(f"📐 パッチサイズ決定: {patch_size} (分割数: {width//patch_size}×{height//patch_size})")
            return patch_size
    
    logger.error("❌ 適切なパッチサイズが見つかりません")
    return None

def main():
    """メイン処理"""
    logger.info("=== Patch分割マスク厚みDRR開始 ===")
    
    # ============ デバイス設定 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.error("❌ CUDA利用不可。GPU必須です。")
        raise RuntimeError("GPU (CUDA) required for practical performance")
    
    # メモリ使用量を抑えるため float32 を使用
    dtype = torch.float32
    logger.info(f"🚀 デバイス: {device}, 精度: {dtype}")
    
    # メモリクリア
    torch.cuda.empty_cache()
    
    # ============ パッチサイズ決定 ============
    patch_size = find_optimal_patch_size(W, H, PATCH_SIZE_CANDIDATES)
    if patch_size is None:
        raise ValueError("No valid patch size found")
    
    # ============ 1) マスクをVolumeとして読む ============
    logger.info(f"📁 マスク読み込み: {MASK_NII}")
    
    subject = read(
        volume=MASK_NII,
        orientation="PA",
        center_volume=False       # 原点ズレ防止
    )
    logger.info(f"✅ マスク読み込み完了: 形状 {subject['volume'].shape}")
    
    # ============ 2) DRRモジュール（GPU + Siddon + Patch） ============
    logger.info("⚙️  DRRセットアップ中...")
    
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
            reverse_x_axis=True,            
            patch_size=patch_size,          # ★メモリ節約のパッチ分割
        ).to(device=device, dtype=dtype)
        
        logger.info(f"✅ DRRセットアップ完了: {W}x{H}, パッチ{patch_size}, GPU={device.type}")
        
    except Exception as e:
        logger.error(f"❌ DRRセットアップ失敗: {e}")
        raise
    
    # ============ 3) 角度変換（公式準拠） ============
    logger.info("🔄 DeepDRR→DiffDRR角度変換")
    
    # 公式: α_diff = α, β_diff = 90°-β, γ_diff = -γ
    alpha = np.deg2rad(alpha_deg)           # 0°
    beta  = np.deg2rad(90.0 - beta_deg)     # 90°
    gamma = np.deg2rad(-gamma_deg)          # -90°
    
    angles = torch.tensor([[alpha, beta, gamma]], device=device, dtype=dtype)
    translations = torch.zeros((1, 3), device=device, dtype=dtype)
    
    logger.info(f"📐 角度変換: DeepDRR({alpha_deg}°,{beta_deg}°,{gamma_deg}°) → DiffDRR({np.degrees(alpha):.1f}°,{np.degrees(beta):.1f}°,{np.degrees(gamma):.1f}°)")
    
    # ============ 4) 投影実行（メモリ効率重視） ============
    logger.info("🎯 Patchマスク厚み投影実行中...")
    
    try:
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=False):  # AMP無効でメモリ使用量安定化
                img = drr(angles, translations, parameterization="euler_angles", convention="YZX")
                img = img.squeeze().to(torch.float32)
        
        logger.info(f"✅ 投影完了: 形状{img.shape}, 範囲[{img.min():.3f}, {img.max():.3f}]")
        
    except torch.OutOfMemoryError as e:
        logger.error(f"❌ CUDAメモリ不足: {e}")
        logger.error("💡 より小さなpatch_sizeを試すか、解像度を下げてください")
        raise
    except Exception as e:
        logger.error(f"❌ 投影実行失敗: {e}")
        raise
    
    # メモリクリア
    torch.cuda.empty_cache()
    
    # ============ 5) PA水平反転 ============
    if PA_HORIZONTAL_FLIP:
        img = torch.flip(img, dims=[1])
        logger.info("🔄 PA水平反転適用")
    
    # ============ 6) 保存とオーバーレイ ============
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # マスク厚み画像保存
    x = img.cpu().numpy()
    
    # 正規化して16bit保存
    max_val = x.max()
    logger.info(f"📊 マスク厚み統計: max={max_val:.3f}mm, mean={x.mean():.3f}mm")
    
    x_norm = np.clip(x / (max_val + 1e-6), 0, 1)
    u16 = (x_norm * 65535).astype(np.uint16)
    
    thickness_png = output_dir / "patch_mask_thickness_drr.png"
    imageio.imwrite(thickness_png, u16)
    logger.info(f"💾 マスク厚み画像保存: {thickness_png}")
    
    # 生データも保存
    thickness_raw = output_dir / "patch_mask_thickness_raw.npy"
    np.save(thickness_raw, x)
    logger.info(f"💾 生データ保存: {thickness_raw}")
    
    # ============ 7) DeepDRR画像読み込みとオーバーレイ ============
    if Path(DEEPDRR_PA_PNG).exists():
        logger.info("📖 既存DeepDRR画像読み込み")
        deepdrr_img = imageio.imread(DEEPDRR_PA_PNG)
        
        if deepdrr_img.ndim == 3:
            deepdrr_img = deepdrr_img[:,:,0]  # グレースケール化
        
        logger.info(f"✅ DeepDRR画像: 形状{deepdrr_img.shape}, 型{deepdrr_img.dtype}")
        
        # オーバーレイ可視化作成
        create_overlay_visualization(deepdrr_img, x, output_dir, patch_size)
        
    else:
        logger.warning(f"⚠️  DeepDRR画像が見つかりません: {DEEPDRR_PA_PNG}")
    
    logger.info("🎉 Patch分割マスク厚みDRR完了!")
    logger.info(f"📈 最大厚み: {max_val:.2f}mm")

def create_overlay_visualization(deepdrr_img, mask_thickness, output_dir, patch_size):
    """オーバーレイ可視化作成"""
    logger.info("🎨 オーバーレイ可視化作成中...")
    
    # 形状確認
    if deepdrr_img.shape != mask_thickness.shape:
        logger.warning(f"⚠️  形状不一致: DeepDRR{deepdrr_img.shape} vs マスク{mask_thickness.shape}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'DeepDRR vs DiffDRRマスク厚み投影 (patch_size={patch_size})', fontsize=14)
    
    # DeepDRR画像
    axes[0,0].imshow(deepdrr_img, cmap='gray')
    axes[0,0].set_title('DeepDRR PA (既存)')
    axes[0,0].axis('off')
    
    # マスク厚み
    im1 = axes[0,1].imshow(mask_thickness, cmap='hot')
    axes[0,1].set_title('マスク厚み [mm]')
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], shrink=0.8)
    
    # オーバーレイ
    deepdrr_norm = deepdrr_img / (deepdrr_img.max() + 1e-6)
    mask_norm = mask_thickness / (mask_thickness.max() + 1e-6)
    
    axes[1,0].imshow(deepdrr_norm, cmap='gray', alpha=0.7)
    axes[1,0].imshow(mask_norm, cmap='hot', alpha=0.5)
    axes[1,0].set_title('オーバーレイ (DeepDRR + マスク厚み)')
    axes[1,0].axis('off')
    
    # 厚みヒストグラム
    mask_flat = mask_thickness.flatten()
    mask_nonzero = mask_flat[mask_flat > 0]
    
    axes[1,1].hist(mask_nonzero, bins=50, alpha=0.7, color='red')
    axes[1,1].set_xlabel('厚み [mm]')
    axes[1,1].set_ylabel('ピクセル数')
    axes[1,1].set_title(f'マスク厚み分布 (非ゼロ値のみ)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    overlay_png = output_dir / f"patch_deepdrr_mask_overlay_{patch_size}.png"
    plt.savefig(overlay_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"🎨 オーバーレイ保存: {overlay_png}")

if __name__ == "__main__":
    main()