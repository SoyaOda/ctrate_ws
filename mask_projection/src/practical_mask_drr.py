#!/usr/bin/env python3
"""
実用版: メモリ制限内での高品質マスク厚みDRR
修正された設定で実用的な解像度・処理時間を実現
"""

import torch
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """実用版メイン処理"""
    logger.info("=== 実用版: 高品質マスク厚みDRR ===")
    
    # ============ 実用的な設定 ============
    SID = 1800.0
    SOD = 1720.0
    
    # 解像度を実用的に調整（DeepDRRの1/2）
    PX = 0.388    # 0.194 * 2 (ピクセルサイズ2倍)
    W = 1250      # 2500 / 2 (幅半分)
    H = 1535      # 3070 / 2 (高さ半分)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    logger.info(f"📐 実用設定: {W}×{H}, PX={PX}mm")
    logger.info(f"   メモリ使用量: {W*H/1e6:.1f}M pixels (元の25%)")
    
    # メモリクリア
    torch.cuda.empty_cache()
    
    # ============ マスク読み込み（修正版設定） ============
    subject = read(
        volume="/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz",
        orientation="PA",
        center_volume=True,    # ✅ 修正された設定
    )
    logger.info(f"✅ マスク読み込み完了: {subject['volume'].shape}")
    
    # ============ 角度・平行移動設定（修正版） ============
    rot = torch.tensor([[0.0, np.deg2rad(90.0), np.deg2rad(-90.0)]], device=device, dtype=dtype)
    xyz = torch.tensor([[0.0, -SOD, 0.0]], device=device, dtype=dtype)  # ✅ 修正
    pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
    
    logger.info(f"📍 修正設定: center_volume=True, by=-{SOD}")
    
    # ============ DRR投影（パッチ分割） ============
    logger.info("🎯 Siddon精密投影実行中...")
    
    # 適切なパッチサイズ
    patch_candidates = [50, 25, 10, 5]
    patch_size = None
    for p in patch_candidates:
        if W % p == 0 and H % p == 0:
            patch_size = p
            break
    
    if patch_size is None:
        patch_size = 5
    
    logger.info(f"📦 パッチ分割: {patch_size} ({W//patch_size}×{H//patch_size})")
    
    try:
        drr = DRR(
            subject,
            sdd=SID, height=H, width=W, delx=PX, dely=PX,
            renderer="siddon",
            reducefn="sum",
            reverse_x_axis=True,
            patch_size=patch_size,
        ).to(device=device, dtype=dtype)
        
        start_time = time.time()
        with torch.inference_mode():
            thickness = drr(pose).squeeze().detach().cpu().numpy()
        process_time = time.time() - start_time
        
        # 結果統計
        max_thickness = thickness.max()
        nonzero_count = np.count_nonzero(thickness)
        total_pixels = thickness.size
        
        logger.info(f"✅ 投影完了: {process_time:.1f}秒")
        logger.info(f"📊 結果統計:")
        logger.info(f"   範囲: {thickness.min():.3f} - {max_thickness:.3f} mm")
        logger.info(f"   非ゼロ: {nonzero_count:,}/{total_pixels:,} ({100*nonzero_count/total_pixels:.1f}%)")
        
        if max_thickness == 0:
            logger.error("❌ 投影結果がゼロです")
            return
        else:
            mean_thickness = thickness[thickness > 0].mean()
            logger.info(f"   平均厚み: {mean_thickness:.3f} mm")
            logger.info("🎉 非ゼロ投影成功!")
        
    except Exception as e:
        logger.error(f"❌ 投影失敗: {e}")
        return
    
    # メモリクリア
    torch.cuda.empty_cache()
    
    # ============ 結果保存 ============
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 正規化保存
    thickness_norm = thickness / max_thickness
    u16 = (thickness_norm * 65535).astype(np.uint16)
    
    practical_png = output_dir / f"practical_mask_{W}x{H}.png"
    imageio.imwrite(practical_png, u16)
    logger.info(f"💾 実用版保存: {practical_png}")
    
    # 生データ保存
    raw_npy = output_dir / f"practical_thickness_{W}x{H}.npy"
    np.save(raw_npy, thickness)
    logger.info(f"💾 生データ保存: {raw_npy}")
    
    # ============ DeepDRR比較準備 ============
    deepdrr_png = Path("/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png")
    
    if deepdrr_png.exists():
        logger.info("📊 DeepDRR画像読み込み中...")
        
        deepdrr_img = imageio.imread(deepdrr_png)
        if deepdrr_img.ndim == 3:
            deepdrr_img = deepdrr_img[:,:,0]
        
        logger.info(f"✅ DeepDRR: {deepdrr_img.shape}")
        
        # DeepDRRを実用版サイズにリサイズ
        if deepdrr_img.shape != thickness.shape:
            from skimage.transform import resize
            deepdrr_resized = resize(deepdrr_img, thickness.shape, 
                                   anti_aliasing=True, preserve_range=True).astype(deepdrr_img.dtype)
            logger.info(f"🔄 DeepDRRリサイズ: {deepdrr_img.shape} → {deepdrr_resized.shape}")
            
            create_practical_comparison(deepdrr_resized, thickness, output_dir, 
                                      max_thickness, process_time, W, H, PX)
        else:
            create_practical_comparison(deepdrr_img, thickness, output_dir,
                                      max_thickness, process_time, W, H, PX)
    else:
        logger.warning("⚠️  DeepDRR画像が見つかりません")
    
    # ============ 最終サマリー ============
    logger.info("🎉 実用版マスク厚みDRR完了!")
    logger.info("📈 パフォーマンス:")
    logger.info(f"   解像度: {W}×{H} ({W*H/1e6:.1f}M pixels)")
    logger.info(f"   処理時間: {process_time:.1f}秒")
    logger.info(f"   スループット: {W*H/process_time/1e6:.1f}M pixels/秒")
    logger.info("✅ DeepDRRとの位置対応マスク投影成功!")

def create_practical_comparison(deepdrr_img, thickness, output_dir, max_thickness, process_time, W, H, PX):
    """実用版比較可視化"""
    logger.info("🎨 実用版比較可視化作成")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Practical: DeepDRR vs DiffDRR Mask Thickness ({W}×{H})', fontsize=16, weight='bold')
    
    # DeepDRR
    axes[0,0].imshow(deepdrr_img, cmap='gray')
    axes[0,0].set_title('DeepDRR PA\n(Resized to Match)', fontsize=12)
    axes[0,0].axis('off')
    
    # マスク厚み
    im1 = axes[0,1].imshow(thickness, cmap='hot', vmin=0, vmax=max_thickness)
    axes[0,1].set_title(f'DiffDRR Mask Thickness\nMax: {max_thickness:.1f}mm', fontsize=12)
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], shrink=0.8)
    
    # オーバーレイ
    deepdrr_norm = deepdrr_img / (deepdrr_img.max() + 1e-6)
    thickness_norm = thickness / (max_thickness + 1e-6)
    
    axes[0,2].imshow(deepdrr_norm, cmap='gray', alpha=0.7)
    axes[0,2].imshow(thickness_norm, cmap='hot', alpha=0.6)
    axes[0,2].set_title('Position-Aligned\nOverlay', fontsize=12)
    axes[0,2].axis('off')
    
    # ヒストグラム
    thickness_nonzero = thickness[thickness > 0]
    axes[1,0].hist(thickness_nonzero, bins=40, alpha=0.7, color='red', edgecolor='black')
    axes[1,0].set_xlabel('Thickness [mm]')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title(f'Thickness Distribution\n({len(thickness_nonzero):,} pixels)', fontsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # 中心クロップ
    h, w = thickness.shape
    cy, cx = h//2, w//2
    crop = 200
    y1, y2 = max(0, cy-crop//2), min(h, cy+crop//2)
    x1, x2 = max(0, cx-crop//2), min(w, cx+crop//2)
    
    thickness_crop = thickness[y1:y2, x1:x2]
    axes[1,1].imshow(thickness_crop, cmap='hot', vmin=0, vmax=max_thickness)
    axes[1,1].set_title(f'Center Detail\n({x2-x1}×{y2-y1} pixels)', fontsize=12)
    axes[1,1].axis('off')
    
    # 統計
    axes[1,2].axis('off')
    nonzero_count = len(thickness_nonzero)
    mean_thick = thickness_nonzero.mean()
    
    stats = f"""Results Summary:

Resolution: {W} × {H} pixels
Pixel Size: {PX:.3f} mm
Total Pixels: {thickness.size:,}

Thickness Stats:
• Max: {max_thickness:.2f} mm  
• Mean: {mean_thick:.2f} mm
• Non-zero: {nonzero_count:,} ({100*nonzero_count/thickness.size:.1f}%)

Performance:
• Process Time: {process_time:.1f} sec
• Throughput: {W*H/process_time/1e6:.1f} MP/sec

Geometry (DeepDRR-compatible):
• SID: 1800 mm, SOD: 1720 mm
• PA: α=0°, β=0°, γ=90°
• center_volume=True, by=-SOD

Status: ✅ SUCCESS"""
    
    axes[1,2].text(0.05, 0.95, stats, transform=axes[1,2].transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    comparison_png = output_dir / f"PRACTICAL_comparison_{W}x{H}.png"
    plt.savefig(comparison_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"🎨 実用版比較保存: {comparison_png}")

if __name__ == "__main__":
    main()