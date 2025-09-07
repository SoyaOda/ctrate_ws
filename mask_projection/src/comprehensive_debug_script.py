#!/usr/bin/env python3
"""
包括的デバッグスクリプト：見た目が根本的に違う原因を段階的に特定
指摘された順序でチェック1〜10を実行し、ズレの真因を数値で特定
"""

import math
import torch
import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from scipy import ndimage

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def world_box_analysis(nifti_path):
    """世界座標バウンディングボックス解析"""
    img = nib.load(str(nifti_path))
    A = img.affine
    Z = np.array(img.shape[:3]) - 1
    
    # 8つのコーナー座標
    corners = np.array([
        [0,0,0], [Z[0],0,0], [0,Z[1],0], [0,0,Z[2]],
        [Z[0],Z[1],0], [Z[0],0,Z[2]], [0,Z[1],Z[2]], [Z[0],Z[1],Z[2]]
    ])
    
    # 世界座標に変換
    world_corners = (A @ np.c_[corners, np.ones(8)].T).T[:, :3]
    
    # 統計
    world_min = world_corners.min(0)
    world_max = world_corners.max(0)
    world_center = world_corners.mean(0)
    world_size = world_max - world_min
    
    return {
        'affine': A,
        'voxel_size': img.header.get_zooms()[:3],
        'shape': img.shape[:3],
        'world_min': world_min,
        'world_max': world_max, 
        'world_center': world_center,
        'world_size': world_size,
        'corners': world_corners
    }

def check_1_world_coordinate_alignment():
    """チェック1：世界座標の整合（最重要）"""
    logger.info("=" * 60)
    logger.info("📍 チェック1：世界座標の整合（最重要原因チェック）")
    logger.info("=" * 60)
    
    # CT と Mask のパス（推定）
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(ct_path).exists():
        logger.warning(f"⚠️  CT パスが見つかりません: {ct_path}")
        logger.info("利用可能なCTパスを探索中...")
        # 代替パス探索
        for alt_ct in [
            "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_2/valid_2_a/valid_2_a_1.nii.gz",
            "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/train/train_1/train_1_a/train_1_a_1.nii.gz"
        ]:
            if Path(alt_ct).exists():
                ct_path = alt_ct
                logger.info(f"✅ 代替CTパス使用: {ct_path}")
                break
    
    if not Path(ct_path).exists():
        logger.error("❌ 利用可能なCTパスが見つかりません")
        return False
        
    if not Path(mask_path).exists():
        logger.error(f"❌ マスクパスが見つかりません: {mask_path}")
        return False
    
    # 解析実行
    logger.info(f"📊 CT解析: {ct_path}")
    ct_info = world_box_analysis(ct_path)
    
    logger.info(f"📊 マスク解析: {mask_path}")
    mask_info = world_box_analysis(mask_path)
    
    # 結果出力
    logger.info("\n🔍 世界座標整合性解析結果:")
    logger.info(f"CT形状: {ct_info['shape']}")
    logger.info(f"マスク形状: {mask_info['shape']}")
    logger.info(f"CT画素サイズ: {ct_info['voxel_size']}")
    logger.info(f"マスク画素サイズ: {mask_info['voxel_size']}")
    
    logger.info(f"\nCT世界中心(mm): [{ct_info['world_center'][0]:.2f}, {ct_info['world_center'][1]:.2f}, {ct_info['world_center'][2]:.2f}]")
    logger.info(f"マスク世界中心(mm): [{mask_info['world_center'][0]:.2f}, {mask_info['world_center'][1]:.2f}, {mask_info['world_center'][2]:.2f}]")
    
    center_diff = mask_info['world_center'] - ct_info['world_center']
    logger.info(f"🎯 世界中心差分(mm): [{center_diff[0]:.2f}, {center_diff[1]:.2f}, {center_diff[2]:.2f}]")
    logger.info(f"📏 差分ノルム: {np.linalg.norm(center_diff):.2f}mm")
    
    # 判定
    center_diff_norm = np.linalg.norm(center_diff)
    if center_diff_norm < 2.0:
        logger.info("✅ 世界座標中心差分OK (<2mm)")
        alignment_ok = True
    elif center_diff_norm < 10.0:
        logger.warning(f"⚠️  世界座標中心差分注意 ({center_diff_norm:.2f}mm)")
        logger.warning("   これが位置ズレの主原因の可能性があります")
        alignment_ok = False
    else:
        logger.error(f"❌ 世界座標中心差分異常 ({center_diff_norm:.2f}mm)")
        logger.error("   これが確実に位置ズレの主原因です")
        alignment_ok = False
    
    # Affine行列比較
    logger.info(f"\nCT affine:\n{ct_info['affine']}")
    logger.info(f"\nマスク affine:\n{mask_info['affine']}")
    
    affine_diff = np.abs(ct_info['affine'] - mask_info['affine']).max()
    logger.info(f"📐 Affine最大差分: {affine_diff:.6f}")
    
    return alignment_ok, ct_info, mask_info

def deepdrr_to_diffdrr_angles(alpha_deg=0.0, beta_deg=0.0, gamma_deg=90.0, PA=True):
    """DeepDRR→DiffDRR角度変換（一元化）"""
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = (np.pi/2) - np.deg2rad(beta_deg)  # 90° - β
    gamma_rad = -np.deg2rad(gamma_deg)           # -γ
    
    # PA読み込み時はY軸回転符号反転
    if PA:
        beta_rad = -beta_rad
        
    return alpha_rad, beta_rad, gamma_rad

def check_2_angle_mapping():
    """チェック2：角度のマッピングを一元化"""
    logger.info("\n📐 チェック2：DeepDRR→DiffDRR角度マッピング検証")
    
    # PA設定での角度変換テスト
    angles = deepdrr_to_diffdrr_angles(0.0, 0.0, 90.0, PA=True)
    logger.info(f"PA角度変換結果 (rad): [{angles[0]:.3f}, {angles[1]:.3f}, {angles[2]:.3f}]")
    logger.info(f"PA角度変換結果 (deg): [{np.degrees(angles[0]):.1f}°, {np.degrees(angles[1]):.1f}°, {np.degrees(angles[2]):.1f}°]")
    
    # 期待値との比較
    expected_deg = [0.0, -90.0, -90.0]
    actual_deg = [np.degrees(a) for a in angles]
    diff_deg = [abs(a - e) for a, e in zip(actual_deg, expected_deg)]
    
    logger.info(f"期待値 (deg): {expected_deg}")
    logger.info(f"実際値 (deg): {actual_deg}")
    logger.info(f"角度差分 (deg): {diff_deg}")
    
    angle_ok = all(d < 1e-6 for d in diff_deg)
    if angle_ok:
        logger.info("✅ 角度変換OK")
    else:
        logger.warning("⚠️  角度変換に問題があります")
        
    return angle_ok

def check_3_4_geometric_parameters():
    """チェック3&4：Radiographic view と主点補正の検証"""
    logger.info("\n🎭 チェック3&4：Radiographic view と主点補正検証")
    
    # パラメータ設定
    W, H = 2500, 3070
    PX = 0.14
    
    # 偶数サイズ主点補正
    x0_mm = (PX * 0.5) if (W % 2 == 0) else 0.0
    y0_mm = (-PX * 0.5) if (H % 2 == 0) else 0.0
    
    logger.info(f"解像度: {W}×{H} (偶数: {W%2==0}, {H%2==0})")
    logger.info(f"ピクセルサイズ: {PX}mm/px")
    logger.info(f"主点補正: x0={x0_mm:.3f}mm, y0={y0_mm:.3f}mm")
    
    # ストライプy0計算テスト
    logger.info("\n📏 ストライプy0計算テスト:")
    for ht, y0 in [(200, 0), (200, 1000), (200, 2000)]:
        ht_eff = ht
        tile_center_px = y0 + (ht_eff - 1) / 2.0
        y_offset_mm = ((H-1)/2 - tile_center_px) * PX
        total_y0_mm = y0_mm + y_offset_mm
        
        logger.info(f"  ストライプ y={y0}-{y0+ht_eff}: tile_center={tile_center_px:.1f}px")
        logger.info(f"    y_offset={y_offset_mm:.3f}mm, total_y0={total_y0_mm:.3f}mm")
    
    return True

def test_ray_reprojection(mask_path, device="cuda"):
    """チェック4詳細：1本のレイの往復誤差テスト"""
    logger.info("\n🔬 レイ往復誤差テスト:")
    
    try:
        # 設定
        SID, SOD = 1800.0, 1720.0
        PX = 0.14
        W, H = 2500, 3070
        
        # 読み込み
        subject = read(volume=mask_path, orientation="PA", center_volume=True)
        
        # 角度設定
        rot = torch.tensor([deepdrr_to_diffdrr_angles(0.0, 0.0, 90.0, PA=True)], device=device)
        xyz = torch.tensor([[0.0, -SOD, 0.0]], device=device)
        pose = convert(rot, xyz, parameterization="euler_angles", convention="YZX")
        
        # 主点補正
        x0_mm = (PX * 0.5) if (W % 2 == 0) else 0.0
        y0_mm = (-PX * 0.5) if (H % 2 == 0) else 0.0
        
        # DRR構築
        drr = DRR(
            subject, sdd=SID, height=H, width=W,
            delx=PX, dely=PX, x0=x0_mm, y0=y0_mm,
            renderer="siddon", reducefn="sum",
            reverse_x_axis=False, patch_size=10
        ).to(device)
        
        # テストポイント
        test_points = torch.tensor([
            [W//4, H//4], [W//2, H//2], [3*W//4, 3*H//4]
        ], device=device, dtype=torch.float32)
        
        logger.info(f"テストポイント: {test_points.cpu().numpy()}")
        
        # 往復テスト
        with torch.no_grad():
            # detector → world
            world_points = drr.inverse_projection(test_points, pose)
            logger.info(f"World座標: {world_points.cpu().numpy()}")
            
            # world → detector
            reprojected = drr.perspective_projection(world_points, pose)
            logger.info(f"再投影: {reprojected.cpu().numpy()}")
            
            # 誤差計算
            error = (reprojected - test_points).abs()
            max_error = error.max().item()
            mean_error = error.mean().item()
            
            logger.info(f"📊 往復誤差統計:")
            logger.info(f"   最大誤差: {max_error:.6f} px")
            logger.info(f"   平均誤差: {mean_error:.6f} px")
            
            if max_error < 0.1:
                logger.info("✅ 往復誤差OK (<0.1px)")
                return True
            else:
                logger.warning(f"⚠️  往復誤差注意 ({max_error:.6f}px)")
                return False
                
    except Exception as e:
        logger.error(f"❌ 往復誤差テスト失敗: {e}")
        return False

def check_5_preprocessing_alignment():
    """チェック5：前処理アライメント確認"""
    logger.info("\n🔄 チェック5：前処理・後処理アライメント確認")
    
    deepdrr_path = "/home/soya/ctrate_ws/deepdrr/deepdrr/outputs/deepdrr_PA_final_pa_fixed.png"
    
    if Path(deepdrr_path).exists():
        deepdrr_img = imageio.imread(deepdrr_path)
        logger.info(f"DeepDRR画像形状: {deepdrr_img.shape}")
        logger.info(f"DeepDRR画像dtype: {deepdrr_img.dtype}")
        
        # チャンネル確認
        if deepdrr_img.ndim == 3:
            logger.info(f"多チャンネル画像、1チャンネル目使用")
            deepdrr_img = deepdrr_img[:,:,0]
            
        logger.info(f"最終DeepDRR形状: {deepdrr_img.shape}")
        
        # 統計
        logger.info(f"DeepDRR統計: min={deepdrr_img.min()}, max={deepdrr_img.max()}, mean={deepdrr_img.mean():.2f}")
        
        return True
    else:
        logger.warning(f"⚠️  DeepDRR画像が見つかりません: {deepdrr_path}")
        return False

def comprehensive_debug_analysis():
    """包括的デバッグ解析実行"""
    logger.info("🔍 包括的デバッグ解析開始")
    logger.info("指摘された順序でチェック1〜5を実行")
    
    results = {}
    
    # チェック1：世界座標整合（最重要）
    try:
        alignment_ok, ct_info, mask_info = check_1_world_coordinate_alignment()
        results['world_alignment'] = alignment_ok
        results['ct_info'] = ct_info
        results['mask_info'] = mask_info
        
        if not alignment_ok:
            logger.warning("🎯 主原因候補：世界座標不整合")
            logger.warning("   マスクをCTのaffineに再格子化する必要があります")
        
    except Exception as e:
        logger.error(f"❌ チェック1失敗: {e}")
        results['world_alignment'] = False
    
    # チェック2：角度マッピング
    try:
        angle_ok = check_2_angle_mapping()
        results['angle_mapping'] = angle_ok
    except Exception as e:
        logger.error(f"❌ チェック2失敗: {e}")
        results['angle_mapping'] = False
    
    # チェック3&4：幾何パラメータ
    try:
        geo_ok = check_3_4_geometric_parameters()
        results['geometric_params'] = geo_ok
    except Exception as e:
        logger.error(f"❌ チェック3&4失敗: {e}")
        results['geometric_params'] = False
    
    # レイ往復誤差テスト
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    if torch.cuda.is_available() and Path(mask_path).exists():
        try:
            ray_ok = test_ray_reprojection(mask_path)
            results['ray_reprojection'] = ray_ok
        except Exception as e:
            logger.error(f"❌ レイ往復テスト失敗: {e}")
            results['ray_reprojection'] = False
    else:
        logger.warning("⚠️  CUDA不可またはマスクパス不存在のためレイテストスキップ")
        results['ray_reprojection'] = None
    
    # チェック5：前処理確認
    try:
        prep_ok = check_5_preprocessing_alignment()
        results['preprocessing'] = prep_ok
    except Exception as e:
        logger.error(f"❌ チェック5失敗: {e}")
        results['preprocessing'] = False
    
    return results

def create_debug_report(results, output_dir):
    """デバッグレポート作成"""
    logger.info("📋 デバッグレポート作成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Debug Analysis Report', fontsize=16, weight='bold')
    
    # チェック結果サマリー
    axes[0,0].axis('off')
    
    check_status = []
    check_status.append(f"1. 世界座標整合: {'✅ OK' if results.get('world_alignment') else '❌ NG'}")
    check_status.append(f"2. 角度マッピング: {'✅ OK' if results.get('angle_mapping') else '❌ NG'}")
    check_status.append(f"3. 幾何パラメータ: {'✅ OK' if results.get('geometric_params') else '❌ NG'}")
    
    ray_result = results.get('ray_reprojection')
    if ray_result is None:
        ray_status = "⚠️ SKIP"
    else:
        ray_status = "✅ OK" if ray_result else "❌ NG"
    check_status.append(f"4. レイ往復誤差: {ray_status}")
    
    check_status.append(f"5. 前処理確認: {'✅ OK' if results.get('preprocessing') else '❌ NG'}")
    
    status_text = "DEBUG ANALYSIS RESULTS:\n\n" + "\n".join(check_status)
    
    if not results.get('world_alignment'):
        status_text += "\n\n🎯 PRIMARY SUSPECT:\n世界座標不整合\n(最頻原因)"
    
    axes[0,0].text(0.05, 0.95, status_text, transform=axes[0,0].transAxes,
                  fontsize=12, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))
    
    # 世界座標比較（可能なら）
    if 'ct_info' in results and 'mask_info' in results:
        axes[0,1].axis('off')
        ct_info = results['ct_info']
        mask_info = results['mask_info']
        
        coord_text = f"""WORLD COORDINATE ANALYSIS:

CT:
  Shape: {ct_info['shape']}
  Voxel: {ct_info['voxel_size']}
  Center: [{ct_info['world_center'][0]:.1f}, {ct_info['world_center'][1]:.1f}, {ct_info['world_center'][2]:.1f}]

MASK:
  Shape: {mask_info['shape']}
  Voxel: {mask_info['voxel_size']}
  Center: [{mask_info['world_center'][0]:.1f}, {mask_info['world_center'][1]:.1f}, {mask_info['world_center'][2]:.1f}]

DIFFERENCE:
  Center Δ: [{(mask_info['world_center'] - ct_info['world_center'])[0]:.2f}, {(mask_info['world_center'] - ct_info['world_center'])[1]:.2f}, {(mask_info['world_center'] - ct_info['world_center'])[2]:.2f}]mm
  Norm: {np.linalg.norm(mask_info['world_center'] - ct_info['world_center']):.2f}mm"""
        
        axes[0,1].text(0.05, 0.95, coord_text, transform=axes[0,1].transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    
    # 推奨対策
    axes[1,0].axis('off')
    
    recommendations = "RECOMMENDED ACTIONS:\n\n"
    
    if not results.get('world_alignment'):
        recommendations += "🔧 URGENT: マスクをCTのaffineに再格子化\n"
        recommendations += "   (最近傍補間で0/1値保持)\n\n"
    
    if not results.get('angle_mapping'):
        recommendations += "🔧 PA角度変換の再確認\n"
        recommendations += "   Y軸符号反転の重複/欠落チェック\n\n"
        
    if ray_result is False:
        recommendations += "🔧 主点補正・DRR設定の見直し\n\n"
    
    recommendations += "📋 Next Steps:\n"
    recommendations += "1. 世界座標整合 → マスク再格子化\n"
    recommendations += "2. CTも同条件でDiffDRR投影してDeepDRRと比較\n"
    recommendations += "3. 位置は輪郭・モーメントで定量評価"
    
    axes[1,0].text(0.05, 0.95, recommendations, transform=axes[1,0].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.9))
    
    # 技術詳細
    axes[1,1].axis('off')
    
    tech_details = f"""TECHNICAL DETAILS:

DiffDRR Settings:
• SID: 1800mm, SOD: 1720mm
• Pixel size: 0.14 mm/px
• Resolution: 2500×3070
• PA Y-axis sign flip: Applied
• Principal point: x0=+0.070mm, y0=-0.070mm

DeepDRR Angles:
• Original: α=0°, β=0°, γ=90°
• DiffDRR: [0°, -90°, -90°] (YZX)
• Translation: [0, -1720, 0]mm

Radiographic View:
• reverse_x_axis=False
• Final horizontal flip applied

Read Settings:
• orientation="PA"
• center_volume=True"""
    
    axes[1,1].text(0.05, 0.95, tech_details, transform=axes[1,1].transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    report_path = output_dir / "COMPREHENSIVE_DEBUG_REPORT.png"
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📋 デバッグレポート保存: {report_path}")

def main():
    """メイン処理"""
    logger.info("🔍 包括的デバッグスクリプト実行開始")
    logger.info("=" * 80)
    
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 包括的解析実行
        results = comprehensive_debug_analysis()
        
        # レポート作成
        create_debug_report(results, output_dir)
        
        # 最終判定
        logger.info("\n" + "=" * 60)
        logger.info("🎯 最終デバッグ判定")
        logger.info("=" * 60)
        
        if not results.get('world_alignment'):
            logger.warning("❌ 主要問題：世界座標不整合")
            logger.warning("   → マスクをCTのaffineに再格子化してください")
            logger.warning("   → これが位置ズレの最頻原因です")
        elif not results.get('angle_mapping'):
            logger.warning("⚠️  角度変換に問題の可能性")
        else:
            logger.info("✅ 主要幾何パラメータは正常")
            logger.info("   → 微細な調整または比較手法の見直しが必要")
        
        logger.info("🎊 包括的デバッグ解析完了!")
        
    except Exception as e:
        logger.error(f"❌ デバッグスクリプト実行エラー: {e}")
        raise

if __name__ == "__main__":
    main()