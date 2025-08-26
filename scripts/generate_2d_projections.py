#!/usr/bin/env python3
"""
STEP6: 2D投影（DRR/脂肪マップ）生成スクリプト

3D CTデータと脂肪マスクから2D投影画像を生成します。
- DRR (Digitally Reconstructed Radiograph): CTデータの投影
- Fat Map: 脂肪マスクの投影
"""

import os
import sys
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def load_nifti(filepath):
    """NIfTIファイルを読み込み"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")
    
    img = nib.load(filepath)
    data = img.get_fdata()
    return img, data


def generate_drr(ct_data, projection_axis='AP', window_center=40, window_width=400):
    """
    CTデータからDRR（デジタル再構成X線画像）を生成
    
    Args:
        ct_data: 3D CTデータ (HU値)
        projection_axis: 投影方向 ('AP', 'PA', 'LAT', 'RLAT')
        window_center: ウィンドウ中心
        window_width: ウィンドウ幅
    
    Returns:
        2D DRR画像
    """
    print(f"DRR生成中 (軸: {projection_axis})")
    
    # HUからリニア減衰係数への変換（簡易版）
    # 水を基準とした相対的な減衰係数
    mu_water = 0.2  # 任意単位
    linear_atten = ct_data * (mu_water / 1000.0) + mu_water
    linear_atten = np.maximum(linear_atten, 0)  # 負の値を0にクリップ
    
    # 投影方向に応じた積分
    if projection_axis in ['AP', 'PA']:
        # Anterior-Posterior / Posterior-Anterior (Y軸方向)
        projection = np.sum(linear_atten, axis=1)
        if projection_axis == 'PA':
            projection = np.flip(projection, axis=0)
    elif projection_axis == 'LAT':
        # Lateral (X軸方向、左から右)
        projection = np.sum(linear_atten, axis=0)
    elif projection_axis == 'RLAT':
        # Right Lateral (X軸方向、右から左)
        projection = np.sum(linear_atten, axis=0)
        projection = np.flip(projection, axis=1)
    else:
        raise ValueError(f"不明な投影軸: {projection_axis}")
    
    # 対数変換（X線の減衰をシミュレート）
    # I = I0 * exp(-∫μ dx) → log(I0/I) = ∫μ dx
    projection = np.log(np.maximum(projection, 1e-6))
    
    # ウィンドウレベル調整
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    projection = np.clip(projection, min_val, max_val)
    
    # 0-255にスケーリング
    projection = (projection - min_val) / (max_val - min_val) * 255
    projection = projection.astype(np.uint8)
    
    return projection


def generate_fat_projection(fat_mask, projection_axis='AP', smoothing=True):
    """
    脂肪マスクから2D投影マップを生成
    
    Args:
        fat_mask: 3D脂肪マスク (バイナリ)
        projection_axis: 投影方向 ('AP', 'PA', 'LAT', 'RLAT')
        smoothing: ガウシアンフィルタでスムージングするか
    
    Returns:
        2D脂肪投影マップ
    """
    print(f"脂肪マップ生成中 (軸: {projection_axis})")
    
    # バイナリマスクを確実にfloat型に変換
    fat_mask = fat_mask.astype(np.float32)
    
    # 投影方向に応じた積分
    if projection_axis in ['AP', 'PA']:
        # Y軸方向の和
        projection = np.sum(fat_mask, axis=1)
        if projection_axis == 'PA':
            projection = np.flip(projection, axis=0)
    elif projection_axis == 'LAT':
        # X軸方向の和（左から右）
        projection = np.sum(fat_mask, axis=0)
    elif projection_axis == 'RLAT':
        # X軸方向の和（右から左）
        projection = np.sum(fat_mask, axis=0)
        projection = np.flip(projection, axis=1)
    else:
        raise ValueError(f"不明な投影軸: {projection_axis}")
    
    # スムージング（オプション）
    if smoothing and projection.max() > 0:
        projection = gaussian_filter(projection, sigma=1.0)
    
    # 正規化して0-255にスケーリング
    if projection.max() > 0:
        projection = projection / projection.max() * 255
    
    projection = projection.astype(np.uint8)
    
    return projection


def create_overlay(drr, fat_map, alpha=0.3):
    """
    DRRと脂肪マップをオーバーレイ
    
    Args:
        drr: DRR画像
        fat_map: 脂肪マップ
        alpha: 脂肪マップの透明度
    
    Returns:
        オーバーレイ画像（RGB）
    """
    # DRRをRGBに変換（グレースケール）
    drr_rgb = np.stack([drr, drr, drr], axis=-1)
    
    # 脂肪マップをカラーマップで着色（赤系）
    fat_colored = np.zeros_like(drr_rgb)
    fat_colored[:, :, 0] = fat_map  # 赤チャンネル
    fat_colored[:, :, 1] = fat_map * 0.3  # 緑チャンネル（少し）
    
    # アルファブレンディング
    overlay = (1 - alpha) * drr_rgb + alpha * fat_colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


def save_projection_images(drr, fat_map, overlay, output_dir, case_name, axis):
    """
    投影画像を保存
    
    Args:
        drr: DRR画像
        fat_map: 脂肪マップ
        overlay: オーバーレイ画像
        output_dir: 出力ディレクトリ
        case_name: 症例名
        axis: 投影軸名
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイルパス
    drr_path = os.path.join(output_dir, f"{case_name}_drr_{axis}.png")
    fat_path = os.path.join(output_dir, f"{case_name}_fat_{axis}.png")
    overlay_path = os.path.join(output_dir, f"{case_name}_overlay_{axis}.png")
    
    # 画像保存
    plt.imsave(drr_path, drr, cmap='gray')
    plt.imsave(fat_path, fat_map, cmap='hot')
    plt.imsave(overlay_path, overlay)
    
    print(f"保存: {drr_path}")
    print(f"保存: {fat_path}")
    print(f"保存: {overlay_path}")
    
    return drr_path, fat_path, overlay_path


def create_visualization(drr, fat_map, overlay, output_path, title="2D Projections"):
    """
    3つの画像を並べた可視化図を作成
    
    Args:
        drr: DRR画像
        fat_map: 脂肪マップ
        overlay: オーバーレイ画像
        output_path: 出力パス
        title: 図のタイトル
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # DRR
    axes[0].imshow(drr, cmap='gray')
    axes[0].set_title('DRR')
    axes[0].axis('off')
    
    # 脂肪マップ
    im = axes[1].imshow(fat_map, cmap='hot')
    axes[1].set_title('Fat Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # オーバーレイ
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可視化保存: {output_path}")


def process_single_case(ct_path, fat_mask_path, output_dir, 
                       projection_axis='AP', create_viz=True):
    """
    単一症例の2D投影を生成
    
    Args:
        ct_path: CT画像のパス
        fat_mask_path: 脂肪マスクのパス（EATまたはPAT）
        output_dir: 出力ディレクトリ
        projection_axis: 投影方向
        create_viz: 可視化図を作成するか
    
    Returns:
        (DRRパス, 脂肪マップパス, オーバーレイパス)
    """
    print(f"\n処理中:")
    print(f"  CT: {ct_path}")
    print(f"  脂肪マスク: {fat_mask_path}")
    
    # データ読み込み
    ct_img, ct_data = load_nifti(ct_path)
    fat_img, fat_data = load_nifti(fat_mask_path)
    
    # データ情報
    print(f"\nデータ情報:")
    print(f"  CT形状: {ct_data.shape}")
    print(f"  HU範囲: [{ct_data.min():.0f}, {ct_data.max():.0f}]")
    print(f"  脂肪ボクセル数: {(fat_data > 0).sum()}")
    
    # DRR生成
    drr = generate_drr(ct_data, projection_axis)
    
    # 脂肪マップ生成
    fat_map = generate_fat_projection(fat_data > 0, projection_axis)
    
    # オーバーレイ作成
    overlay = create_overlay(drr, fat_map, alpha=0.4)
    
    # ケース名取得
    case_name = Path(ct_path).stem.replace('.nii', '')
    fat_type = 'eat' if 'eat' in fat_mask_path.lower() else 'pat'
    
    # 画像保存
    drr_path, fat_path, overlay_path = save_projection_images(
        drr, fat_map, overlay, 
        output_dir, 
        f"{case_name}_{fat_type}",
        projection_axis.lower()
    )
    
    # 可視化図作成
    if create_viz:
        viz_path = os.path.join(
            output_dir, 
            f"{case_name}_{fat_type}_{projection_axis.lower()}_viz.png"
        )
        create_visualization(
            drr, fat_map, overlay, viz_path,
            f"{case_name} - {fat_type.upper()} - {projection_axis}"
        )
    
    print(f"\n投影生成完了!")
    
    return drr_path, fat_path, overlay_path


def main():
    parser = argparse.ArgumentParser(
        description='2D投影（DRR/脂肪マップ）生成'
    )
    parser.add_argument('ct_path', help='CT画像ファイル (NIfTI)')
    parser.add_argument('fat_mask_path', help='脂肪マスクファイル (NIfTI, EATまたはPAT)')
    parser.add_argument('-o', '--output', default='./projection_output',
                       help='出力ディレクトリ (デフォルト: ./projection_output)')
    parser.add_argument('-a', '--axis', default='AP',
                       choices=['AP', 'PA', 'LAT', 'RLAT'],
                       help='投影軸 (デフォルト: AP)')
    parser.add_argument('--all-axes', action='store_true',
                       help='全ての投影軸で生成')
    parser.add_argument('--no-viz', action='store_true',
                       help='可視化図を作成しない')
    parser.add_argument('--window-center', type=float, default=40,
                       help='DRRウィンドウ中心 (デフォルト: 40)')
    parser.add_argument('--window-width', type=float, default=400,
                       help='DRRウィンドウ幅 (デフォルト: 400)')
    
    args = parser.parse_args()
    
    try:
        if args.all_axes:
            # 全投影軸で生成
            axes = ['AP', 'PA', 'LAT', 'RLAT']
            print(f"全投影軸で生成: {axes}")
            
            for axis in axes:
                print(f"\n{'='*50}")
                print(f"投影軸: {axis}")
                print(f"{'='*50}")
                
                axis_output_dir = os.path.join(args.output, axis.lower())
                process_single_case(
                    args.ct_path,
                    args.fat_mask_path,
                    axis_output_dir,
                    axis,
                    not args.no_viz
                )
        else:
            # 単一投影軸で生成
            process_single_case(
                args.ct_path,
                args.fat_mask_path,
                args.output,
                args.axis,
                not args.no_viz
            )
        
        print(f"\n全処理完了!")
        print(f"出力ディレクトリ: {args.output}")
        
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()