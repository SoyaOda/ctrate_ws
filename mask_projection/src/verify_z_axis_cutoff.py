#!/usr/bin/env python3
"""
Z軸方向の途切れ検証：3D構造予測と2D投射結果の比較
頭方向での線状表現が正しく再現されているかを確認
"""

import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_z_cutoff_discrepancy(mask_path, projection_2d_path):
    """Z軸途切れの3D vs 2D比較解析"""
    logger.info("🔍 Z軸途切れの3D vs 2D比較解析開始")
    
    # 3Dマスクデータ読み込み
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    
    # 2D投射画像読み込み
    if Path(projection_2d_path).exists():
        proj_2d = imageio.imread(projection_2d_path)
        if proj_2d.ndim == 3:
            proj_2d = proj_2d[:,:,0]
    else:
        logger.error(f"❌ 2D投射画像が見つかりません: {projection_2d_path}")
        return None
    
    logger.info(f"3Dマスク形状: {mask_data.shape}")
    logger.info(f"2D投射形状: {proj_2d.shape}")
    
    # Z軸プロファイル詳細解析
    z_profile = np.sum(mask_data, axis=(0, 1))
    z_nonzero = np.nonzero(z_profile)[0]
    
    if len(z_nonzero) == 0:
        logger.error("❌ マスクデータに非零ボクセルが見つかりません")
        return None
        
    z_start, z_end = z_nonzero[0], z_nonzero[-1]
    
    # 頭方向での急激な変化を詳細に分析
    z_density = z_profile[z_nonzero]
    
    # 上位10%, 5%, 1%での密度変化
    len_z = len(z_density)
    top_10pct_idx = int(len_z * 0.9)
    top_5pct_idx = int(len_z * 0.95)
    top_1pct_idx = int(len_z * 0.99)
    
    body_avg = z_density[:top_10pct_idx].mean() if top_10pct_idx > 0 else z_density.mean()
    top_10pct_avg = z_density[top_10pct_idx:].mean()
    top_5pct_avg = z_density[top_5pct_idx:].mean()
    top_1pct_avg = z_density[top_1pct_idx:].mean()
    
    logger.info(f"📊 Z軸密度分布詳細解析:")
    logger.info(f"  全体範囲: Z={z_start}-{z_end} ({len_z} スライス)")
    logger.info(f"  体部(90%まで)平均: {body_avg:.1f}")
    logger.info(f"  上位10%平均: {top_10pct_avg:.1f} (比率: {top_10pct_avg/body_avg:.3f})")
    logger.info(f"  上位5%平均: {top_5pct_avg:.1f} (比率: {top_5pct_avg/body_avg:.3f})")
    logger.info(f"  上位1%平均: {top_1pct_avg:.1f} (比率: {top_1pct_avg/body_avg:.3f})")
    
    # 急激な途切れの判定（より厳格）
    sharp_cutoff = False
    if top_1pct_avg < body_avg * 0.1:  # 1%が10%以下に落ちる
        sharp_cutoff = True
        logger.warning("⚠️ 検出: 頭方向で急激な途切れ (上位1%が10%以下)")
    elif top_5pct_avg < body_avg * 0.3:  # 5%が30%以下に落ちる
        sharp_cutoff = True
        logger.warning("⚠️ 検出: 頭方向で明確な途切れ (上位5%が30%以下)")
    elif top_10pct_avg < body_avg * 0.5:  # 10%が50%以下に落ちる
        logger.info("📝 検出: 頭方向で緩やかな途切れ (上位10%が50%以下)")
    else:
        logger.info("✅ 検出: 比較的均一な分布")
    
    # 2D投射での境界の鋭さを解析
    logger.info("\n🔍 2D投射境界解析:")
    
    # 2D画像の非零領域
    proj_nonzero = proj_2d > (proj_2d.max() * 0.05)  # 5%閾値
    
    # 上下境界の検出
    y_profile = np.sum(proj_nonzero, axis=1)
    y_nonzero = np.nonzero(y_profile)[0]
    
    if len(y_nonzero) > 0:
        y_top = y_nonzero[0]  # 上端
        y_bottom = y_nonzero[-1]  # 下端
        
        logger.info(f"2D投射Y範囲: {y_top}-{y_bottom} (高さ: {y_bottom-y_top+1}px)")
        
        # 上端近傍での境界の急激さを評価
        # 上端から10%の範囲での変化を確認
        top_10pct_range = int((y_bottom - y_top) * 0.1)
        if top_10pct_range > 5:
            top_region_profile = y_profile[y_top:y_top+top_10pct_range]
            if len(top_region_profile) > 0:
                top_gradient = np.gradient(top_region_profile)
                max_gradient = np.max(np.abs(top_gradient))
                
                logger.info(f"上端10%領域の最大勾配: {max_gradient:.2f}")
                
                if max_gradient > 100:  # 急激な変化
                    logger.info("✅ 2D投射で線状境界を確認")
                    line_boundary_detected = True
                else:
                    logger.warning("⚠️ 2D投射で緩やかな境界（線状でない）")
                    line_boundary_detected = False
            else:
                line_boundary_detected = False
        else:
            logger.warning("⚠️ 2D投射領域が小さすぎて境界解析不可")
            line_boundary_detected = False
    else:
        logger.error("❌ 2D投射で非零領域が見つかりません")
        line_boundary_detected = False
    
    # 矛盾の検出
    logger.info("\n🎯 3D vs 2D整合性判定:")
    
    if sharp_cutoff and line_boundary_detected:
        logger.info("✅ 整合性OK: 3D急激途切れ → 2D線状境界")
        consistency = "consistent"
    elif sharp_cutoff and not line_boundary_detected:
        logger.error("❌ 不整合検出: 3D急激途切れ → 2D緩やかな境界")
        logger.error("   → 投射処理または幾何設定に問題あり")
        consistency = "inconsistent_3d_sharp_2d_smooth"
    elif not sharp_cutoff and line_boundary_detected:
        logger.warning("⚠️ 予想外: 3D緩やか → 2D線状境界")
        logger.warning("   → 投射過程で境界が強調された可能性")
        consistency = "unexpected_2d_enhancement"
    else:
        logger.info("✅ 整合性OK: 3D緩やか → 2D緩やか")
        consistency = "consistent"
    
    return {
        'z_range': (z_start, z_end),
        'z_length': len_z,
        'body_avg': body_avg,
        'top_10pct_avg': top_10pct_avg,
        'top_5pct_avg': top_5pct_avg,
        'top_1pct_avg': top_1pct_avg,
        'sharp_cutoff': sharp_cutoff,
        'line_boundary_detected': line_boundary_detected,
        'consistency': consistency,
        'y_range': (y_top, y_bottom) if len(y_nonzero) > 0 else None,
        'z_profile': z_profile,
        'proj_2d': proj_2d
    }

def create_cutoff_verification_report(analysis_result, output_dir):
    """途切れ検証レポート作成"""
    logger.info("📋 Z軸途切れ検証レポート作成")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Z軸途切れ検証: 3D構造 vs 2D投射整合性', fontsize=16, weight='bold')
    
    z_profile = analysis_result['z_profile']
    proj_2d = analysis_result['proj_2d']
    z_start, z_end = analysis_result['z_range']
    
    # Z軸プロファイル（詳細）
    z_nonzero = np.nonzero(z_profile)[0]
    z_density = z_profile[z_nonzero]
    
    axes[0,0].plot(z_nonzero, z_density, 'b-', linewidth=2)
    axes[0,0].axvline(z_start, color='green', linestyle='--', alpha=0.7, label=f'開始: {z_start}')
    axes[0,0].axvline(z_end, color='red', linestyle='--', alpha=0.7, label=f'終了: {z_end}')
    
    # 境界領域の強調
    len_z = len(z_density)
    top_10pct_idx = int(len_z * 0.9)
    if top_10pct_idx < len_z:
        axes[0,0].axvspan(z_nonzero[top_10pct_idx], z_end, alpha=0.3, color='red', 
                          label='上位10%領域')
    
    axes[0,0].set_title('3D: Z軸密度プロファイル\n頭方向途切れ詳細')
    axes[0,0].set_xlabel('Z インデックス')
    axes[0,0].set_ylabel('ボクセル密度')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2D投射画像
    axes[0,1].imshow(proj_2d, cmap='hot', origin='upper')
    axes[0,1].set_title('2D投射結果\n境界の鋭さ確認')
    axes[0,1].axis('off')
    
    # Y方向プロファイル
    y_profile = np.sum(proj_2d > (proj_2d.max() * 0.05), axis=1)
    axes[0,2].plot(y_profile, np.arange(len(y_profile)), 'g-', linewidth=2)
    axes[0,2].invert_yaxis()
    axes[0,2].set_title('2D: Y方向プロファイル\n境界の急峻さ')
    axes[0,2].set_xlabel('幅 (px)')
    axes[0,2].set_ylabel('Y インデックス')
    axes[0,2].grid(True, alpha=0.3)
    
    # 数値解析結果
    axes[1,0].axis('off')
    
    body_avg = analysis_result['body_avg']
    top_10pct = analysis_result['top_10pct_avg']
    top_5pct = analysis_result['top_5pct_avg']
    top_1pct = analysis_result['top_1pct_avg']
    
    numerical_text = f"""3D密度解析結果:

体部平均密度: {body_avg:.1f}
上位10%平均: {top_10pct:.1f} ({top_10pct/body_avg:.3f}倍)
上位5%平均:  {top_5pct:.1f} ({top_5pct/body_avg:.3f}倍)  
上位1%平均:  {top_1pct:.1f} ({top_1pct/body_avg:.3f}倍)

急激途切れ: {'Yes' if analysis_result['sharp_cutoff'] else 'No'}
線状境界(2D): {'検出' if analysis_result['line_boundary_detected'] else '未検出'}

範囲: Z={z_start}-{z_end} ({analysis_result['z_length']} slices)"""
    
    axes[1,0].text(0.05, 0.95, numerical_text, transform=axes[1,0].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.9))
    
    # 整合性判定
    axes[1,1].axis('off')
    
    consistency = analysis_result['consistency']
    if consistency == "consistent":
        status_color = "lightgreen"
        status_icon = "✅"
        status_text = "整合性: 良好"
    elif consistency == "inconsistent_3d_sharp_2d_smooth":
        status_color = "lightcoral" 
        status_icon = "❌"
        status_text = "整合性: 不良"
    else:
        status_color = "lightyellow"
        status_icon = "⚠️"
        status_text = "整合性: 要注意"
    
    consistency_text = f"""{status_icon} {status_text}

3D構造予測:
{'急激な途切れ' if analysis_result['sharp_cutoff'] else '緩やかな変化'}

2D投射結果:  
{'線状境界検出' if analysis_result['line_boundary_detected'] else '緩やかな境界'}

判定:
{consistency}

推奨アクション:
{'幾何設定・投射処理の見直し' if consistency.startswith('inconsistent') else '現在の実装OK'}"""
    
    axes[1,1].text(0.05, 0.95, consistency_text, transform=axes[1,1].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.9))
    
    # 結論
    axes[1,2].axis('off')
    
    if consistency == "inconsistent_3d_sharp_2d_smooth":
        conclusion_text = f"""🔍 問題検出!

マスクデータは頭方向で
急激に途切れているが、
2D投射では線状にならず
緩やかな境界となっている。

考えられる原因:
• 投射幾何の設定ミス
• レンダリングの補間効果
• 座標系の不整合
• 解像度の問題

対策:
• 幾何パラメータ再確認
• 最近傍補間の使用
• 高解像度投射テスト"""
    else:
        conclusion_text = f"""✅ 検証完了

3D構造と2D投射結果の
整合性が確認されました。

Z軸方向の途切れは
適切に再現されています。

現在の実装は
正しく動作しています。"""
    
    axes[1,2].text(0.05, 0.95, conclusion_text, transform=axes[1,2].transAxes,
                  fontsize=9, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor="lightcoral" if consistency == "inconsistent_3d_sharp_2d_smooth" else "lightgreen", 
                           alpha=0.9))
    
    plt.tight_layout()
    
    report_path = output_dir / "Z_AXIS_CUTOFF_VERIFICATION.png"
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📋 Z軸途切れ検証レポート保存: {report_path}")

def main():
    """メイン処理"""
    logger.info("🔍 Z軸途切れ検証開始")
    logger.info("=== 3D構造予測 vs 2D投射結果の整合性確認 ===")
    
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    projection_path = "/home/soya/ctrate_ws/mask_projection/outputs/ALIGNED_2500x3070_thickness.png"
    
    if not Path(mask_path).exists():
        logger.error(f"❌ マスクパスが見つかりません: {mask_path}")
        return
        
    if not Path(projection_path).exists():
        logger.error(f"❌ 2D投射画像が見つかりません: {projection_path}")
        return
    
    try:
        # 比較解析実行
        analysis_result = analyze_z_cutoff_discrepancy(mask_path, projection_path)
        
        if analysis_result is None:
            logger.error("❌ 解析失敗")
            return
        
        # レポート作成
        create_cutoff_verification_report(analysis_result, output_dir)
        
        # 最終判定
        logger.info("\n" + "="*60)
        logger.info("🎯 Z軸途切れ検証結果")
        logger.info("="*60)
        
        consistency = analysis_result['consistency']
        
        if consistency == "consistent":
            logger.info("✅ 検証完了: 3D構造と2D投射の整合性確認")
        elif consistency == "inconsistent_3d_sharp_2d_smooth":
            logger.error("❌ 問題検出: 3D急激途切れが2Dで緩やかに")
            logger.error("   → 投射処理の見直しが必要")
        else:
            logger.warning(f"⚠️  要注意: {consistency}")
        
        logger.info("🎊 Z軸途切れ検証完了!")
        
    except Exception as e:
        logger.error(f"❌ 検証エラー: {e}")
        raise

if __name__ == "__main__":
    main()