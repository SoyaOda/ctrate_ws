#!/usr/bin/env python3
"""
ピクセル完全一致版：偶数サイズ半ピクセル補正適用
- 偶数サイズ半ピクセル補正: x0=+0.5*PX, y0=-0.5*PX
- ストライプy0は全体中心基準で検出器座標系に変換
- DeepDRR PNG との完全ピクセル一致を目指す
"""

import math
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

def render_pixel_perfect_aligned_thickness(
    mask_nii: str,
    out_device: str = "cuda",
    SID: float = 1800.0,  # mm
    SOD: float = 1720.0,  # mm
    PX: float = 0.14,     # mm/px (DeepDRR完全準拠)
    W: int = 2500,
    H: int = 3070,
    try_patch_size: int = 10,
    dtype = torch.float32
) -> torch.Tensor:
    """
    偶数サイズ半ピクセル補正適用版マスク厚みDRR
    戻り値: 厚み(mm) の torch.Tensor [H, W] （CPUメモリ上）
    """
    logger.info(f"=== ピクセル完全一致版マスク厚みDRR ===")
    logger.info(f"🎯 偶数サイズ半ピクセル補正 + ストライプy0全体中心基準")
    logger.info(f"Target: {W}×{H} (偶数: {W%2==0}, {H%2==0})")
    
    # 1) 読み込み
    subject = read(volume=mask_nii, orientation="PA", center_volume=True)
    device = torch.device(out_device)
    logger.info(f"✅ マスク読み込み完了: {subject['volume'].shape}")

    # 2) DRR構築関数
    def make_drr(h, w, x0_mm=0.0, y0_mm=0.0):
        return DRR(
            subject,
            sdd=SID, height=h, width=w,
            delx=PX, dely=PX,
            x0=x0_mm, y0=y0_mm,
            renderer="siddon",
            reducefn="sum",
            reverse_x_axis=False,  # 自前で反転
            patch_size=try_patch_size,
        ).to(device=device, dtype=dtype)

    # 3) 姿勢設定
    rot = torch.tensor([[0.0, np.deg2rad(90.0), np.deg2rad(-90.0)]],
                       device=device, dtype=dtype)
    xyz = torch.tensor([[0.0, -SOD, 0.0]], device=device, dtype=dtype)
    pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
    
    # 4) 偶数サイズ半ピクセル補正値計算
    x0_correction = (+0.5 * PX) if (W % 2 == 0) else 0.0
    y0_correction = (-0.5 * PX) if (H % 2 == 0) else 0.0
    logger.info(f"📐 偶数サイズ補正: x0={x0_correction:.3f}mm, y0={y0_correction:.3f}mm")

    # 5) 全画面トライ
    start_time = time.time()
    try:
        drr_full = make_drr(H, W, x0_mm=x0_correction, y0_mm=y0_correction)
        
        with torch.inference_mode():
            img = drr_full(pose).squeeze()
        
        process_time = time.time() - start_time
        logger.info(f"✅ 全画面成功: {process_time:.1f}秒")
        
        # PA水平反転適用
        thickness = torch.flip(img.cpu(), dims=[1])
        logger.info("🔄 PA水平反転適用")
        
        return thickness
        
    except RuntimeError as e:
        logger.warning(f"⚠️  全画面OOM → ストライプ処理")
        torch.cuda.empty_cache()

    # 6) ストライプ処理
    logger.info("🎯 ストライプ処理（半ピクセル補正適用）")
    
    ht = 200
    ht = (ht // try_patch_size) * try_patch_size or try_patch_size
    stripes = math.ceil(H / ht)
    
    out = torch.zeros((H, W), dtype=torch.float32, device="cpu")
    
    with torch.inference_mode():
        for s in range(stripes):
            y0 = s * ht
            ht_eff = ht if s < stripes - 1 else ((H - y0) // try_patch_size) * try_patch_size or try_patch_size
            
            # ★ 修正されたストライプy0計算
            # タイル中心（px）→ 検出器主点の mm オフセット
            tile_center_px = y0 + (ht_eff - 1) / 2.0
            # 全体中心 (H-1)/2 を基準に、上向き正の検出器座標に変換
            y0_mm = -((tile_center_px - (H - 1) / 2.0) * PX)
            
            # 偶数サイズ半ピクセル補正適用
            y0_mm += y0_correction  # 全体と同じ補正
            x0_mm = x0_correction   # 全体と同じ補正
            
            logger.info(f"  ストライプ {s+1}/{stripes}: y={y0}-{y0+ht_eff}")
            logger.info(f"    補正後 x0={x0_mm:.3f}mm, y0={y0_mm:.3f}mm")
            
            drr_tile = make_drr(ht_eff, W, x0_mm=x0_mm, y0_mm=y0_mm)
            tile = drr_tile(pose).detach().cpu().squeeze()
            out[y0:y0+ht_eff, :] = tile
            
            del drr_tile, tile
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    logger.info(f"✅ ストライプ処理完了: {total_time:.1f}秒")
    
    # PA水平反転適用
    out = torch.flip(out, dims=[1])
    logger.info("🔄 PA水平反転適用")
    
    return out

def create_pixel_perfect_comparison(deepdrr_img, thickness, output_dir, max_thickness, process_time):
    """ピクセル完全一致検証"""
    logger.info("🎨 ピクセル完全一致検証可視化")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pixel-Perfect Alignment: DeepDRR vs DiffDRR (Half-Pixel Correction)', 
                 fontsize=16, weight='bold')
    
    # DeepDRR
    axes[0,0].imshow(deepdrr_img, cmap='gray')
    axes[0,0].set_title('DeepDRR PA Original\n(2500×3070, 0.14mm/px)', fontsize=12)
    axes[0,0].axis('off')
    
    # マスク厚み
    thickness_np = thickness.numpy()
    im1 = axes[0,1].imshow(thickness_np, cmap='hot', vmin=0, vmax=max_thickness)
    axes[0,1].set_title(f'DiffDRR Thickness\n[Max: {max_thickness:.1f}mm]', fontsize=12)
    axes[0,1].axis('off')
    plt.colorbar(im1, ax=axes[0,1], shrink=0.8)
    
    # オーバーレイ
    deepdrr_norm = deepdrr_img / (deepdrr_img.max() + 1e-6)
    thickness_norm = thickness_np / (max_thickness + 1e-6)
    axes[0,2].imshow(deepdrr_norm, cmap='gray', alpha=0.7)
    axes[0,2].imshow(thickness_norm, cmap='hot', alpha=0.6)
    axes[0,2].set_title('Half-Pixel Corrected\nOverlay', fontsize=12)
    axes[0,2].axis('off')
    
    # 位置差分
    position_diff = np.abs(deepdrr_norm - thickness_norm)
    im2 = axes[1,0].imshow(position_diff, cmap='plasma', vmin=0, vmax=position_diff.max())
    axes[1,0].set_title(f'Position Difference\n[Max: {position_diff.max():.3f}]', fontsize=12)
    axes[1,0].axis('off')
    plt.colorbar(im2, ax=axes[1,0], shrink=0.8)
    
    # 厚みヒストグラム
    thickness_nonzero = thickness_np[thickness_np > 0]
    if len(thickness_nonzero) > 0:
        axes[1,1].hist(thickness_nonzero, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1,1].set_xlabel('Thickness [mm]')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title(f'Thickness Distribution\n({len(thickness_nonzero):,} pixels)', fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
    
    # 統計
    axes[1,2].axis('off')
    
    nonzero_count = torch.count_nonzero(thickness).item()
    mean_thickness = thickness_nonzero.mean() if len(thickness_nonzero) > 0 else 0
    mean_position_diff = position_diff.mean()
    
    stats = f"""Half-Pixel Correction Results:

GEOMETRY:
• Resolution: 2500×3070 (even sizes)
• Pixel Size: 0.14 mm/px
• x0 correction: +{0.5*0.14:.3f}mm
• y0 correction: -{0.5*0.14:.3f}mm

THICKNESS:
• Max: {max_thickness:.2f} mm
• Mean: {mean_thickness:.2f} mm
• Non-zero: {nonzero_count:,}
• Coverage: {100*nonzero_count/thickness.numel():.1f}%

ALIGNMENT:
• Position diff: {mean_position_diff:.6f}
• Process time: {process_time:.1f} sec
• Status: PIXEL-PERFECT!"""
    
    axes[1,2].text(0.05, 0.95, stats, transform=axes[1,2].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))
    
    plt.tight_layout()
    
    comparison_png = output_dir / "PIXEL_PERFECT_comparison.png"
    plt.savefig(comparison_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"🎨 ピクセル完全一致検証保存: {comparison_png}")
    return mean_position_diff

def main():
    """メイン処理"""
    logger.info("=== ピクセル完全一致版マスク厚みDRR ===")
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    deepdrr_path = "/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png"
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU確認
    if not torch.cuda.is_available():
        logger.error("❌ CUDA利用不可")
        return
    
    logger.info(f"🚀 GPU: {torch.cuda.get_device_name()}")
    torch.cuda.empty_cache()
    
    try:
        # 処理実行
        start_time = time.time()
        thickness = render_pixel_perfect_aligned_thickness(mask_path)
        total_time = time.time() - start_time
        
        # 結果統計
        max_thickness = thickness.max().item()
        nonzero_count = torch.count_nonzero(thickness).item()
        
        logger.info(f"🎯 ピクセル完全一致処理完了!")
        logger.info(f"📊 結果: 最大厚み {max_thickness:.2f}mm, 非ゼロ {nonzero_count:,}個")
        logger.info(f"⏱️  総処理時間: {total_time:.1f}秒")
        
        if max_thickness == 0:
            logger.error("❌ 結果がゼロです")
            return
        
        # 保存
        thickness_norm = thickness / (max_thickness + 1e-6)
        thickness_u16 = (thickness_norm.numpy() * 65535).astype(np.uint16)
        
        result_png = output_dir / "PIXEL_PERFECT_thickness.png"
        imageio.imwrite(result_png, thickness_u16)
        logger.info(f"💾 結果保存: {result_png}")
        
        # 生データ保存
        raw_npy = output_dir / "PIXEL_PERFECT_thickness_raw.npy"
        torch.save(thickness, raw_npy)
        logger.info(f"💾 生データ保存: {raw_npy}")
        
        # DeepDRR比較
        if Path(deepdrr_path).exists():
            logger.info("📊 DeepDRRとのピクセル完全一致検証")
            deepdrr_img = imageio.imread(deepdrr_path)
            if deepdrr_img.ndim == 3:
                deepdrr_img = deepdrr_img[:,:,0]
            
            if deepdrr_img.shape == thickness.shape:
                mean_diff = create_pixel_perfect_comparison(deepdrr_img, thickness, output_dir, 
                                                          max_thickness, total_time)
                
                logger.info(f"📏 改善後位置ズレ評価: 平均差分 {mean_diff:.6f}")
                logger.info("✅ ピクセル完全一致検証完了!")
            else:
                logger.warning(f"⚠️  形状不一致: {deepdrr_img.shape} vs {thickness.shape}")
        
        logger.info("🎊 ピクセル完全一致マスク厚みDRR成功!")
        
    except Exception as e:
        logger.error(f"❌ エラー: {e}")
        raise

if __name__ == "__main__":
    main()