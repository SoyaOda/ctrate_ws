#!/usr/bin/env python3
"""
修正版マスク厚みDRR実装
根本原因解決: center_volume=True + by=-SOD でカメラ視錐台とボリュームの交差確保
DeepDRR test_deepdrr_final_pa_fixed.py の幾何学設定に完全準拠
"""

import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# DiffDRR
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """修正版メイン処理"""
    logger.info("=== 修正版マスク厚みDRR開始 ===")
    logger.info("🔧 根本原因修正: center_volume=True + by=-SOD")
    
    # ============ DeepDRRと完全同一の幾何学設定 ============
    # test_deepdrr_final_pa_fixed.py から抽出
    SID = 1800.0    # source-to-detector distance
    OID = 80.0      # object-to-detector distance  
    SOD = SID - OID # source-to-object distance = 1720.0
    PX = 0.14       # pixel size (mm)
    
    # 検出器サイズ (plate_mm = 350.0 x 430.0)
    W = int(round(350.0 / PX)) & ~1  # 2500
    H = int(round(430.0 / PX)) & ~1  # 3070
    
    # DeepDRR角度設定
    ALPHA, BETA, GAMMA_ROLL = 0.0, 0.0, 90.0  # PA設定
    
    logger.info(f"📐 DeepDRR準拠幾何学:")
    logger.info(f"  SID={SID}mm, SOD={SOD}mm, OID={OID}mm")
    logger.info(f"  検出器: {W}×{H}, ピクセル: {PX}mm") 
    logger.info(f"  PA角度: α={ALPHA}°, β={BETA}°, γ={GAMMA_ROLL}°")
    
    # ============ デバイス設定 ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.error("❌ CUDA必須です")
        raise RuntimeError("GPU required")
    
    dtype = torch.float32
    logger.info(f"🚀 デバイス: {device}, 精度: {dtype}")
    
    # メモリクリア
    torch.cuda.empty_cache()
    
    # ============ 1) マスク読み込み（重要修正！） ============
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    logger.info(f"📁 マスク読み込み: {mask_path}")
    
    # 🔑 KEY FIX: center_volume=True で体積中心を原点へ
    subject = read(
        volume=mask_path,
        orientation="PA",
        center_volume=True,     # ★修正: 体積中心を原点に移動
    )
    logger.info(f"✅ マスク読み込み完了: 形状 {subject['volume'].shape}")
    logger.info("🔧 center_volume=True → 体積中心を原点に配置")
    
    # ============ 2) DRR設定（patch分割でメモリ節約） ============
    logger.info("⚙️  DRRセットアップ中...")
    
    # 適切なパッチサイズ決定
    patch_candidates = [64, 50, 25, 20, 10, 5]
    patch_size = None
    for p in patch_candidates:
        if W % p == 0 and H % p == 0:
            patch_size = p
            break
    
    if patch_size is None:
        patch_size = 5  # 最小値
    
    logger.info(f"📦 パッチサイズ: {patch_size} (分割: {W//patch_size}×{H//patch_size})")
    
    try:
        drr = DRR(
            subject,
            sdd=SID,
            height=H,
            width=W,
            delx=PX, 
            dely=PX,
            renderer="siddon",                   # 正確な線積分
            reducefn="sum",                      # Beer–Lambert無し
            reverse_x_axis=True,                 # radiographic view
            patch_size=patch_size,               # メモリ節約
        ).to(device=device, dtype=dtype)
        
        logger.info(f"✅ DRRセットアップ完了: {W}×{H}")
        
    except Exception as e:
        logger.error(f"❌ DRRセットアップ失敗: {e}")
        raise
    
    # ============ 3) 角度・平行移動設定（重要修正！） ============
    logger.info("🔄 DeepDRR→DiffDRR変換（修正版）")
    
    # DeepDRR→DiffDRR角度変換（公式準拠）
    # DeepDRR: (α,β,γ) → DiffDRR: (α, 90°-β, -γ) with YZX
    alpha_diff = np.deg2rad(ALPHA)                   # 0°
    beta_diff = np.deg2rad(90.0 - BETA)             # 90°
    gamma_diff = np.deg2rad(-GAMMA_ROLL)            # -90°
    
    rot = torch.tensor([[alpha_diff, beta_diff, gamma_diff]], 
                       device=device, dtype=dtype)
    
    # 🔑 KEY FIX: PA では by = -SOD （視錐台がボリュームを横切る）
    xyz = torch.tensor([[0.0, -SOD, 0.0]], device=device, dtype=dtype)
    
    logger.info(f"📐 角度変換: DeepDRR({ALPHA}°,{BETA}°,{GAMMA_ROLL}°) → DiffDRR({np.degrees(alpha_diff):.1f}°,{np.degrees(beta_diff):.1f}°,{np.degrees(gamma_diff):.1f}°)")
    logger.info(f"📍 平行移動: (0.0, {-SOD:.1f}, 0.0) - PA用SOD補正")
    
    # pose変換
    pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
    
    # ============ 4) 投影実行 ============
    logger.info("🎯 修正版マスク厚み投影実行中...")
    
    try:
        with torch.inference_mode():
            thickness = drr(pose).squeeze().detach().cpu().numpy()  # (H, W)
        
        logger.info(f"✅ 投影完了: 形状{thickness.shape}")
        
        # 統計情報
        nonzero_count = np.count_nonzero(thickness)
        max_thickness = thickness.max()
        mean_thickness = thickness[thickness > 0].mean() if nonzero_count > 0 else 0.0
        
        logger.info(f"📊 厚み統計:")
        logger.info(f"  範囲: {thickness.min():.3f} - {max_thickness:.3f} mm")
        logger.info(f"  非ゼロピクセル: {nonzero_count:,}/{thickness.size:,} ({100*nonzero_count/thickness.size:.2f}%)")
        logger.info(f"  平均厚み: {mean_thickness:.3f} mm (非ゼロのみ)")
        
        if max_thickness == 0.0:
            logger.error("❌ まだ全てゼロです。さらなるデバッグが必要")
            return
        else:
            logger.info("🎉 非ゼロ投影成功！")
        
    except Exception as e:
        logger.error(f"❌ 投影失敗: {e}")
        raise
    
    # メモリクリア
    torch.cuda.empty_cache()
    
    # ============ 5) 結果保存 ============
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 正規化して保存
    if max_thickness > 0:
        thickness_norm = thickness / max_thickness
        u16 = (thickness_norm * 65535).astype(np.uint16)
        
        fixed_png = output_dir / "fixed_mask_thickness_drr.png"
        imageio.imwrite(fixed_png, u16)
        logger.info(f"💾 修正版マスク厚み保存: {fixed_png}")
        
        # 生データも保存
        thickness_raw = output_dir / "fixed_mask_thickness_raw.npy"
        np.save(thickness_raw, thickness)
        logger.info(f"💾 生データ保存: {thickness_raw}")
    
    # ============ 6) DeepDRR画像との比較 ============
    deepdrr_png = Path("/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png")
    
    if deepdrr_png.exists() and max_thickness > 0:
        logger.info("📖 DeepDRR画像読み込み中...")
        
        deepdrr_img = imageio.imread(deepdrr_png)
        if deepdrr_img.ndim == 3:
            deepdrr_img = deepdrr_img[:,:,0]
        
        logger.info(f"✅ DeepDRR画像: {deepdrr_img.shape}, {deepdrr_img.dtype}")
        
        # 形状確認
        if deepdrr_img.shape == thickness.shape:
            logger.info("✅ 形状一致 - オーバーレイ可視化作成")
            create_comparison_visualization(deepdrr_img, thickness, output_dir, max_thickness)
        else:
            logger.warning(f"⚠️  形状不一致: DeepDRR{deepdrr_img.shape} vs マスク{thickness.shape}")
    
    logger.info(f"🎉 修正版マスク厚みDRR完了! 最大厚み: {max_thickness:.2f}mm")

def create_comparison_visualization(deepdrr_img, thickness, output_dir, max_thickness):
    """比較可視化作成"""
    logger.info("🎨 比較可視化作成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('修正版: DeepDRR vs DiffDRRマスク厚み投影比較', fontsize=14)
    
    # DeepDRR画像
    axes[0,0].imshow(deepdrr_img, cmap='gray')
    axes[0,0].set_title('DeepDRR PA (Original)', fontsize=12)
    axes[0,0].axis('off')
    
    # マスク厚み
    im1 = axes[0,1].imshow(thickness, cmap='hot', vmin=0, vmax=max_thickness)
    axes[0,1].set_title(f'Mask Thickness [mm] (max: {max_thickness:.1f})', fontsize=12)
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], shrink=0.8)
    
    # オーバーレイ
    deepdrr_norm = deepdrr_img / (deepdrr_img.max() + 1e-6)
    thickness_norm = thickness / (max_thickness + 1e-6)
    
    axes[1,0].imshow(deepdrr_norm, cmap='gray', alpha=0.7)
    axes[1,0].imshow(thickness_norm, cmap='hot', alpha=0.5)
    axes[1,0].set_title('Overlay (DeepDRR + Mask)', fontsize=12)
    axes[1,0].axis('off')
    
    # 厚みヒストグラム
    thickness_nonzero = thickness[thickness > 0]
    if len(thickness_nonzero) > 0:
        axes[1,1].hist(thickness_nonzero, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1,1].set_xlabel('Thickness [mm]')
        axes[1,1].set_ylabel('Pixel Count')
        axes[1,1].set_title(f'Thickness Distribution ({len(thickness_nonzero):,} pixels)')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_png = output_dir / "fixed_deepdrr_mask_comparison.png"
    plt.savefig(comparison_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"🎨 比較可視化保存: {comparison_png}")

if __name__ == "__main__":
    main()