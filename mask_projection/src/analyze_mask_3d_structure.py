#!/usr/bin/env python3
"""
マスクデータの3D構造解析
Z軸方向の途切れを確認し、2D投射での線状表現の予測
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_mask_3d_structure(mask_path):
    """マスク3D構造の詳細解析"""
    logger.info(f"📊 マスク3D構造解析開始: {mask_path}")
    
    # NIfTI読み込み
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    
    logger.info(f"マスク形状: {mask_data.shape}")
    logger.info(f"マスク値域: {mask_data.min()} - {mask_data.max()}")
    logger.info(f"非零ボクセル数: {np.count_nonzero(mask_data)}")
    
    # 軸ごとの分布解析
    z_profile = np.sum(mask_data, axis=(0, 1))  # Z方向プロファイル
    y_profile = np.sum(mask_data, axis=(0, 2))  # Y方向プロファイル  
    x_profile = np.sum(mask_data, axis=(1, 2))  # X方向プロファイル
    
    logger.info(f"Z軸非零スライス数: {np.count_nonzero(z_profile)}")
    logger.info(f"Y軸非零スライス数: {np.count_nonzero(y_profile)}")
    logger.info(f"X軸非零スライス数: {np.count_nonzero(x_profile)}")
    
    # Z軸範囲の詳細解析
    z_nonzero_indices = np.nonzero(z_profile)[0]
    if len(z_nonzero_indices) > 0:
        z_start, z_end = z_nonzero_indices[0], z_nonzero_indices[-1]
        z_range = z_end - z_start + 1
        
        logger.info(f"🔍 Z軸範囲詳細:")
        logger.info(f"  開始スライス: {z_start} ({z_start/mask_data.shape[2]*100:.1f}%)")
        logger.info(f"  終了スライス: {z_end} ({z_end/mask_data.shape[2]*100:.1f}%)")
        logger.info(f"  範囲: {z_range} スライス ({z_range/mask_data.shape[2]*100:.1f}%)")
        
        # 頭方向(高Z値)での急激な途切れをチェック
        z_density = z_profile[z_nonzero_indices]
        
        # 最後の10%での変化率
        tail_10pct = int(len(z_density) * 0.1)
        if tail_10pct > 0:
            tail_start_val = z_density[-tail_10pct:].mean()
            tail_end_val = z_density[-5:].mean() if len(z_density) >= 5 else z_density[-1]
            
            logger.info(f"📈 頭方向(高Z)での変化:")
            logger.info(f"  最後10%平均密度: {tail_start_val:.1f}")
            logger.info(f"  最後5スライス平均密度: {tail_end_val:.1f}")
            logger.info(f"  変化率: {(tail_end_val/tail_start_val - 1)*100:.1f}%")
            
            # 急激な途切れの判定
            if tail_end_val < tail_start_val * 0.5:
                logger.warning("⚠️ 頭方向で急激な途切れを検出!")
                logger.warning("  → 2D投射で線状になる可能性が高い")
            else:
                logger.info("✅ 頭方向で緩やかな変化")
    
    return {
        'shape': mask_data.shape,
        'data': mask_data,
        'z_profile': z_profile,
        'y_profile': y_profile,
        'x_profile': x_profile,
        'z_range': (z_start, z_end) if len(z_nonzero_indices) > 0 else (0, 0)
    }

def create_3d_analysis_visualization(mask_info, output_dir):
    """3D構造解析の可視化"""
    logger.info("📊 3D構造解析結果の可視化作成")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('マスク3D構造解析: Z軸途切れの確認', fontsize=16, weight='bold')
    
    mask_data = mask_info['data']
    z_profile = mask_info['z_profile']
    y_profile = mask_info['y_profile'] 
    x_profile = mask_info['x_profile']
    z_start, z_end = mask_info['z_range']
    
    # 軸方向プロファイル
    axes[0,0].plot(z_profile, 'b-', linewidth=2)
    axes[0,0].axvline(z_start, color='green', linestyle='--', label=f'開始: {z_start}')
    axes[0,0].axvline(z_end, color='red', linestyle='--', label=f'終了: {z_end}')
    axes[0,0].set_title('Z軸方向プロファイル\n(PA投射方向)')
    axes[0,0].set_xlabel('Z インデックス')
    axes[0,0].set_ylabel('ボクセル密度')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    axes[0,1].plot(y_profile, 'g-', linewidth=2)
    axes[0,1].set_title('Y軸方向プロファイル\n(上下方向)')
    axes[0,1].set_xlabel('Y インデックス')
    axes[0,1].set_ylabel('ボクセル密度')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].plot(x_profile, 'r-', linewidth=2)
    axes[0,2].set_title('X軸方向プロファイル\n(左右方向)')
    axes[0,2].set_xlabel('X インデックス')
    axes[0,2].set_ylabel('ボクセル密度')
    axes[0,2].grid(True, alpha=0.3)
    
    # サンプルスライス表示
    # 中央スライス
    mid_z = (z_start + z_end) // 2
    axes[1,0].imshow(mask_data[:, :, mid_z], cmap='hot', origin='lower')
    axes[1,0].set_title(f'中央Zスライス (z={mid_z})')
    axes[1,0].axis('off')
    
    # 開始付近スライス
    start_z = min(z_start + 5, z_end)
    axes[1,1].imshow(mask_data[:, :, start_z], cmap='hot', origin='lower')
    axes[1,1].set_title(f'開始付近Zスライス (z={start_z})')
    axes[1,1].axis('off')
    
    # 終了付近スライス (頭方向)
    end_z = max(z_end - 5, z_start)
    axes[1,2].imshow(mask_data[:, :, end_z], cmap='hot', origin='lower')
    axes[1,2].set_title(f'終了付近Zスライス (z={end_z})\n頭方向途切れ確認')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    viz_path = output_dir / "MASK_3D_STRUCTURE_ANALYSIS.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 3D構造解析図保存: {viz_path}")

def predict_2d_projection_appearance(mask_info):
    """2D投射での見た目を予測"""
    logger.info("🔮 2D投射での見た目予測")
    
    z_profile = mask_info['z_profile']
    z_start, z_end = mask_info['z_range']
    
    # Z軸範囲の密度分布解析
    z_nonzero_indices = np.nonzero(z_profile)[0]
    z_density = z_profile[z_nonzero_indices]
    
    # 頭方向での急激な変化をチェック
    if len(z_density) >= 10:
        tail_10pct = int(len(z_density) * 0.1)
        head_region = z_density[-tail_10pct:]
        body_region = z_density[:-tail_10pct]
        
        head_avg = head_region.mean()
        body_avg = body_region.mean()
        
        logger.info(f"📊 投射予測解析:")
        logger.info(f"  体部領域平均密度: {body_avg:.1f}")
        logger.info(f"  頭部領域平均密度: {head_avg:.1f}")
        logger.info(f"  密度比: {head_avg/body_avg:.3f}")
        
        if head_avg < body_avg * 0.3:
            logger.warning("⚠️ 予測: 頭方向で線状または薄い境界として表現される可能性が高い")
            logger.warning("  → 2D投射で頭方向境界が明瞭な線として現れるはず")
            return "linear_boundary"
        elif head_avg < body_avg * 0.7:
            logger.info("📝 予測: 頭方向で薄くなるがある程度の厚みを持つ")
            return "gradual_thinning"
        else:
            logger.info("✅ 予測: 比較的均一な密度分布")
            return "uniform_density"
    
    return "unknown"

def main():
    """メイン処理"""
    logger.info("🔍 マスクデータ3D構造解析開始")
    logger.info("=== Z軸方向の途切れと2D投射予測 ===")
    
    output_dir = Path("/home/soya/ctrate_ws/mask_projection/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # マスクパス
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    
    if not Path(mask_path).exists():
        logger.error(f"❌ マスクパスが見つかりません: {mask_path}")
        return
    
    try:
        # 3D構造解析
        mask_info = analyze_mask_3d_structure(mask_path)
        
        # 可視化作成
        create_3d_analysis_visualization(mask_info, output_dir)
        
        # 2D投射予測
        projection_type = predict_2d_projection_appearance(mask_info)
        
        # 最終まとめ
        logger.info("\n" + "="*60)
        logger.info("🎯 マスク3D構造解析結果まとめ")
        logger.info("="*60)
        
        z_start, z_end = mask_info['z_range']
        z_total = mask_info['shape'][2]
        
        logger.info(f"📊 マスク範囲: Z={z_start}-{z_end} ({z_end-z_start+1}/{z_total} スライス)")
        logger.info(f"📈 2D投射予測タイプ: {projection_type}")
        
        if projection_type == "linear_boundary":
            logger.info("✅ 確認: 頭方向で線状境界として現れることが予測される")
            logger.info("   → もし2D投射で線状になっていない場合、幾何的整合性に問題あり")
        
        logger.info("🎊 マスク3D構造解析完了!")
        
    except Exception as e:
        logger.error(f"❌ 解析エラー: {e}")
        raise

if __name__ == "__main__":
    main()