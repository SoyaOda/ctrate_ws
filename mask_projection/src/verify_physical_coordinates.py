#!/usr/bin/env python3
"""
マスクデータの物理座標詳細検証
Slicerで確認された275.2999mm→276.7999mmでの急激な途切れを数値的に確認
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_physical_coordinates_detailed(mask_path):
    """物理座標での詳細解析"""
    logger.info(f"🔍 マスク物理座標詳細解析開始: {mask_path}")
    
    # NIfTI読み込み
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    affine = mask_nii.affine
    
    logger.info(f"マスク形状: {mask_data.shape}")
    logger.info(f"非零ボクセル数: {np.count_nonzero(mask_data)}")
    
    # Affine行列の詳細
    logger.info(f"\nAffine行列:")
    logger.info(f"{affine}")
    
    # ボクセル→物理座標変換の確認
    voxel_size = mask_nii.header.get_zooms()[:3]
    logger.info(f"\nボクセルサイズ: {voxel_size} mm")
    
    # Z軸方向の物理座標を詳細に解析
    nz = mask_data.shape[2]
    
    # 全Zスライスの物理座標を計算
    z_physical_coords = []
    z_densities = []
    
    for z_idx in range(nz):
        # ボクセル座標 (0, 0, z_idx) → 物理座標
        voxel_coord = np.array([0, 0, z_idx, 1])
        physical_coord = affine @ voxel_coord
        z_physical = physical_coord[2]
        z_physical_coords.append(z_physical)
        
        # このスライスでの密度
        slice_density = np.sum(mask_data[:, :, z_idx])
        z_densities.append(slice_density)
    
    z_physical_coords = np.array(z_physical_coords)
    z_densities = np.array(z_densities)
    
    logger.info(f"\nZ軸物理座標範囲: {z_physical_coords.min():.3f} - {z_physical_coords.max():.3f} mm")
    logger.info(f"Z軸物理座標ステップ: {voxel_size[2]:.3f} mm")
    
    # 非零スライスのみ抽出
    nonzero_indices = np.where(z_densities > 0)[0]
    if len(nonzero_indices) == 0:
        logger.error("❌ 非零スライスが見つかりません")
        return None
    
    nonzero_z_physical = z_physical_coords[nonzero_indices]
    nonzero_z_densities = z_densities[nonzero_indices]
    
    logger.info(f"\n非零スライス数: {len(nonzero_indices)}")
    logger.info(f"非零スライス物理範囲: {nonzero_z_physical.min():.3f} - {nonzero_z_physical.max():.3f} mm")
    
    # Slicerで確認された275.2999mm→276.7999mmの範囲を詳細解析
    target_start = 275.2999
    target_end = 276.7999
    
    logger.info(f"\n🎯 Slicer確認範囲解析: {target_start}mm - {target_end}mm")
    
    # この範囲のスライスを特定
    in_range_mask = (nonzero_z_physical >= target_start) & (nonzero_z_physical <= target_end)
    in_range_indices = nonzero_indices[in_range_mask]
    in_range_z_physical = nonzero_z_physical[in_range_mask]
    in_range_densities = nonzero_z_densities[in_range_mask]
    
    if len(in_range_indices) > 0:
        logger.info(f"範囲内スライス数: {len(in_range_indices)}")
        logger.info(f"範囲内物理座標:")
        for i, (z_idx, z_phys, density) in enumerate(zip(in_range_indices, in_range_z_physical, in_range_densities)):
            logger.info(f"  スライス{z_idx}: {z_phys:.4f}mm, 密度: {density:.1f}")
    else:
        logger.warning("⚠️ 指定範囲にスライスが見つかりません")
    
    # 範囲前後での急激な変化を検出
    before_range = nonzero_z_physical < target_start
    after_range = nonzero_z_physical > target_end
    
    if np.any(before_range) and np.any(after_range):
        before_avg = nonzero_z_densities[before_range].mean()
        after_avg = nonzero_z_densities[after_range].mean()
        
        logger.info(f"\n急激変化解析:")
        logger.info(f"範囲前平均密度: {before_avg:.1f}")
        logger.info(f"範囲後平均密度: {after_avg:.1f}")
        logger.info(f"変化率: {(after_avg/before_avg - 1)*100:.1f}%")
        
        if after_avg < before_avg * 0.1:  # 90%以上の減少
            logger.warning("⚠️ 急激な途切れを検出! (90%以上の減少)")
            sharp_cutoff_detected = True
        else:
            logger.info("✅ 急激な途切れは検出されませんでした")
            sharp_cutoff_detected = False
    else:
        logger.warning("⚠️ 範囲前後の比較ができません")
        sharp_cutoff_detected = False
    
    # 最も密度の高い連続領域と途切れの検出
    logger.info(f"\n📊 密度分布詳細解析:")
    
    # 密度の連続性解析
    density_threshold = nonzero_z_densities.max() * 0.1  # 最大の10%以下で途切れと判定
    high_density_mask = nonzero_z_densities > density_threshold
    
    if np.any(high_density_mask):
        high_density_physical = nonzero_z_physical[high_density_mask]
        high_density_range = (high_density_physical.min(), high_density_physical.max())
        
        logger.info(f"高密度領域(>{density_threshold:.1f}): {high_density_range[0]:.3f} - {high_density_range[1]:.3f} mm")
        
        # 高密度領域の終端をチェック
        if high_density_range[1] <= target_end:
            logger.warning("⚠️ 高密度領域がSlicer確認範囲内で終了")
            logger.warning("   → 急激な途切れの可能性が高い")
    
    return {
        'z_physical_coords': z_physical_coords,
        'z_densities': z_densities,
        'nonzero_indices': nonzero_indices,
        'nonzero_z_physical': nonzero_z_physical,
        'nonzero_z_densities': nonzero_z_densities,
        'target_range': (target_start, target_end),
        'in_range_indices': in_range_indices,
        'in_range_z_physical': in_range_z_physical,
        'in_range_densities': in_range_densities,
        'sharp_cutoff_detected': sharp_cutoff_detected,
        'voxel_size': voxel_size,
        'affine': affine
    }

def detailed_slice_by_slice_analysis(analysis_result):
    """スライス単位での詳細解析"""
    logger.info("\n🔬 スライス単位詳細解析")
    
    nonzero_indices = analysis_result['nonzero_indices']
    nonzero_z_physical = analysis_result['nonzero_z_physical']
    nonzero_z_densities = analysis_result['nonzero_z_densities']
    target_start, target_end = analysis_result['target_range']
    
    # 最後の10スライスを詳細に見る
    last_10_indices = nonzero_indices[-10:]
    last_10_physical = nonzero_z_physical[-10:]
    last_10_densities = nonzero_z_densities[-10:]
    
    logger.info("最後の10スライス詳細:")
    for i, (z_idx, z_phys, density) in enumerate(zip(last_10_indices, last_10_physical, last_10_densities)):
        marker = "🎯" if target_start <= z_phys <= target_end else "  "
        logger.info(f"{marker} スライス{z_idx:3d}: {z_phys:8.4f}mm, 密度: {density:8.1f}")
    
    # 前スライスとの密度変化率
    logger.info("\nスライス間密度変化率:")
    for i in range(1, len(last_10_densities)):
        prev_density = last_10_densities[i-1]
        curr_density = last_10_densities[i]
        change_rate = (curr_density / prev_density - 1) * 100 if prev_density > 0 else 0
        
        z_phys = last_10_physical[i]
        marker = "⚠️" if abs(change_rate) > 50 else "  "  # 50%以上の変化で警告
        
        logger.info(f"{marker} {last_10_physical[i-1]:8.4f}mm → {z_phys:8.4f}mm: {change_rate:+6.1f}%")

def create_physical_coordinate_report(analysis_result, output_dir):
    """物理座標解析レポート作成"""
    logger.info("📋 物理座標解析レポート作成")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('マスク物理座標詳細解析: Slicer確認範囲検証', fontsize=16, weight='bold')
    
    nonzero_z_physical = analysis_result['nonzero_z_physical']
    nonzero_z_densities = analysis_result['nonzero_z_densities']
    target_start, target_end = analysis_result['target_range']
    
    # 物理座標 vs 密度（全体）
    axes[0,0].plot(nonzero_z_physical, nonzero_z_densities, 'b-', linewidth=2, marker='o', markersize=3)
    axes[0,0].axvspan(target_start, target_end, alpha=0.3, color='red', label=f'Slicer確認範囲\n{target_start}-{target_end}mm')
    axes[0,0].set_title('物理座標 vs 密度分布（全体）')
    axes[0,0].set_xlabel('Z座標 (mm)')
    axes[0,0].set_ylabel('ボクセル密度')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 物理座標 vs 密度（拡大：末端付近）
    # 最後の20%の範囲を拡大表示
    tail_20pct = int(len(nonzero_z_physical) * 0.8)
    tail_z_physical = nonzero_z_physical[tail_20pct:]
    tail_densities = nonzero_z_densities[tail_20pct:]
    
    axes[0,1].plot(tail_z_physical, tail_densities, 'g-', linewidth=3, marker='o', markersize=4)
    axes[0,1].axvspan(target_start, target_end, alpha=0.4, color='red', label='Slicer確認範囲')
    axes[0,1].set_title('物理座標 vs 密度（末端拡大表示）')
    axes[0,1].set_xlabel('Z座標 (mm)')
    axes[0,1].set_ylabel('ボクセル密度')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # スライス間変化率
    if len(nonzero_z_densities) > 1:
        change_rates = []
        change_z_coords = []
        
        for i in range(1, len(nonzero_z_densities)):
            if nonzero_z_densities[i-1] > 0:
                rate = (nonzero_z_densities[i] / nonzero_z_densities[i-1] - 1) * 100
                change_rates.append(rate)
                change_z_coords.append(nonzero_z_physical[i])
        
        change_rates = np.array(change_rates)
        change_z_coords = np.array(change_z_coords)
        
        axes[1,0].plot(change_z_coords, change_rates, 'r-', linewidth=2, marker='s', markersize=3)
        axes[1,0].axvspan(target_start, target_end, alpha=0.3, color='red')
        axes[1,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1,0].axhline(-50, color='orange', linestyle='--', alpha=0.7, label='50%減少ライン')
        axes[1,0].axhline(-90, color='red', linestyle='--', alpha=0.7, label='90%減少ライン')
        axes[1,0].set_title('スライス間密度変化率')
        axes[1,0].set_xlabel('Z座標 (mm)')
        axes[1,0].set_ylabel('変化率 (%)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
    
    # 解析結果サマリー
    axes[1,1].axis('off')
    
    sharp_cutoff = analysis_result['sharp_cutoff_detected']
    in_range_count = len(analysis_result['in_range_indices'])
    
    summary_text = f"""物理座標解析結果:

マスク範囲:
  Z座標: {nonzero_z_physical.min():.3f} - {nonzero_z_physical.max():.3f} mm
  スライス数: {len(nonzero_z_physical)}

Slicer確認範囲解析:
  範囲: {target_start} - {target_end} mm
  範囲内スライス数: {in_range_count}

急激途切れ検出:
  {'Yes (確認済み)' if sharp_cutoff else 'No (未検出)'}

最終スライス:
  物理座標: {nonzero_z_physical[-1]:.4f} mm
  密度: {nonzero_z_densities[-1]:.1f}

判定:
  {'⚠️ Slicerの観察と一致' if sharp_cutoff else '❌ Slicerの観察と不一致'}
  {'→ 2D投射で線状境界になるはず' if sharp_cutoff else '→ 解析手法の見直しが必要'}"""
    
    axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.4", 
                           facecolor="lightgreen" if sharp_cutoff else "lightcoral", 
                           alpha=0.9))
    
    plt.tight_layout()
    
    report_path = output_dir / "PHYSICAL_COORDINATE_ANALYSIS.png"
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📋 物理座標解析レポート保存: {report_path}")

def main():
    """メイン処理"""
    logger.info("🔍 マスク物理座標詳細検証開始")
    logger.info("=== Slicer確認：275.2999mm→276.7999mmでの急激途切れ検証 ===")
    
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # マスクパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(mask_path).exists():
        logger.error(f"❌ マスクパスが見つかりません: {mask_path}")
        return
    
    try:
        # 物理座標詳細解析
        analysis_result = analyze_physical_coordinates_detailed(mask_path)
        
        if analysis_result is None:
            logger.error("❌ 解析失敗")
            return
        
        # スライス単位詳細解析
        detailed_slice_by_slice_analysis(analysis_result)
        
        # レポート作成
        create_physical_coordinate_report(analysis_result, output_dir)
        
        # 最終判定
        logger.info("\n" + "="*60)
        logger.info("🎯 物理座標解析最終判定")
        logger.info("="*60)
        
        sharp_cutoff = analysis_result['sharp_cutoff_detected']
        
        if sharp_cutoff:
            logger.info("✅ Slicerでの観察と一致: 急激な途切れを確認")
            logger.info("   → 2D投射で線状境界として現れるはず")
            logger.info("   → 前回の2D解析結果との不整合を要調査")
        else:
            logger.warning("⚠️ Slicerでの観察と不一致: 急激な途切れが数値的に確認できない")
            logger.warning("   → 解析手法またはデータ読み込みを見直し")
        
        logger.info("🎊 物理座標詳細検証完了!")
        
    except Exception as e:
        logger.error(f"❌ 検証エラー: {e}")
        raise

if __name__ == "__main__":
    main()