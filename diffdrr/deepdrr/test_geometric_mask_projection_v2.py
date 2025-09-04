#!/usr/bin/env python3
"""
DiffDRRを使った幾何学的マスク投影のテストV2
Beer-Lambert則を回避し、純粋な線積分（厚み）を取得する実装
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import matplotlib.pyplot as plt
from pathlib import Path

def test_geometric_mask_projection_v2():
    """
    マスクだけを使って純粋な厚み投影を実行
    """
    
    # GPU設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("/home/soya/ctrate_ws/deepdrr/deepdrr/pure_geometric_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) マスクの3D体積を計算
    print("\n=== Step 1: 3D Volume Calculation ===")
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    
    voxel_dims = mask_nii.header.get_zooms()[:3]  # (dx, dy, dz) [mm]
    voxel_volume = np.prod(voxel_dims)  # mm³
    true_volume_3d = np.sum(mask_binary) * voxel_volume
    print(f"Voxel dimensions: {voxel_dims} mm")
    print(f"Voxel volume: {voxel_volume:.6f} mm³")
    print(f"True 3D volume: {true_volume_3d:.2f} mm³")
    print(f"Number of non-zero voxels: {np.sum(mask_binary):.0f}")
    
    # Z方向の厚み分布（真値）
    thickness_z_true = np.sum(mask_binary, axis=2) * voxel_dims[2]
    thickness_z_nonzero = thickness_z_true[thickness_z_true > 0]
    print(f"\nTrue Z-direction thickness:")
    print(f"  Range: {thickness_z_nonzero.min():.2f} - {thickness_z_nonzero.max():.2f} mm")
    print(f"  Mean: {thickness_z_nonzero.mean():.2f} mm")
    print(f"  Std: {thickness_z_nonzero.std():.2f} mm")
    print(f"  Unique values: {len(np.unique(thickness_z_nonzero))}")
    
    # 2) マスクをvolumeとして読み込む
    print("\n=== Step 2: Loading Mask with DiffDRR ===")
    subject = read(
        volume=mask_path,
        orientation="PA",    # PAビューの撮影向き
        center_volume=True,
    )
    print(f"Subject loaded successfully")
    print(f"Volume shape: {subject.volume.shape}")
    
    # 3) DRRをSiddonレンダラで初期化（reducefn='sum'で線積分）
    print("\n=== Step 3: Initializing DRR with Geometric Settings ===")
    drr = DRR(
        subject,
        sdd=1800.0,        # Source-to-detector distance
        height=512,
        width=512,
        delx=0.7,          # Pixel spacing in mm
        renderer="siddon",  # Siddonレンダラを使用
    ).to(device)
    
    # rendererのreducefnを確認・設定
    if hasattr(drr.renderer, 'reducefn'):
        print(f"Current reducefn: {drr.renderer.reducefn}")
        drr.renderer.reducefn = 'sum'  # 線積分に設定
        print(f"Set reducefn to: {drr.renderer.reducefn}")
    
    # modeを設定
    if hasattr(drr.renderer, 'mode'):
        print(f"Current mode: {drr.renderer.mode}")
        drr.renderer.mode = 'nearest'  # バイナリマスクの補間を防ぐ
        print(f"Set mode to: {drr.renderer.mode}")
    
    print("DRR initialized")
    
    # 4) ポーズの設定（正面PA view）
    print("\n=== Step 4: Rendering Thickness Map ===")
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # 通常のDRRレンダリング
    thickness_map = drr(rotations, translations, 
                       parameterization="euler_angles", 
                       convention="ZXY")
    thickness_np = thickness_map.squeeze().detach().cpu().numpy()
    
    print(f"Thickness map shape: {thickness_np.shape}")
    print(f"Thickness range: {thickness_np.min():.6f} - {thickness_np.max():.6f}")
    print(f"Non-zero pixels: {np.sum(thickness_np > 0)}")
    
    # 5) reducefn='sum'の効果を確認するための実験
    print("\n=== Step 5: Testing Different Reducefn Settings ===")
    
    # Mean版
    if hasattr(drr.renderer, 'reducefn'):
        drr.renderer.reducefn = 'mean'
        thickness_mean = drr(rotations, translations, 
                           parameterization="euler_angles", 
                           convention="ZXY")
        thickness_mean_np = thickness_mean.squeeze().detach().cpu().numpy()
        print(f"Mean mode - Range: {thickness_mean_np.min():.6f} - {thickness_mean_np.max():.6f}")
        
        # Sum版に戻す
        drr.renderer.reducefn = 'sum'
        thickness_sum = drr(rotations, translations, 
                          parameterization="euler_angles", 
                          convention="ZXY")
        thickness_sum_np = thickness_sum.squeeze().detach().cpu().numpy()
        print(f"Sum mode - Range: {thickness_sum_np.min():.6f} - {thickness_sum_np.max():.6f}")
        
        # 最終的に使用するもの
        thickness_np = thickness_sum_np
    
    # 6) 体積の計算
    print("\n=== Step 6: Volume Analysis ===")
    pixel_area = 0.7 * 0.7  # mm²
    
    # 生データの統計
    if np.any(thickness_np > 1e-6):
        # ゼロでない部分のみ
        thickness_nonzero = thickness_np[thickness_np > 1e-6]
        volume_2d_raw = np.sum(thickness_np) * pixel_area
        
        print(f"2D projection statistics:")
        print(f"  Non-zero pixels: {len(thickness_nonzero)}")
        print(f"  Thickness range: {thickness_nonzero.min():.6f} - {thickness_nonzero.max():.6f}")
        print(f"  Mean thickness: {thickness_nonzero.mean():.6f}")
        print(f"  Std thickness: {thickness_nonzero.std():.6f}")
        print(f"  2D Volume (raw): {volume_2d_raw:.2f} mm³")
        print(f"  Ratio (2D/3D): {volume_2d_raw / true_volume_3d:.6f}")
        
        # 単純なスケーリング係数
        if volume_2d_raw > 0:
            k = true_volume_3d / volume_2d_raw
            thickness_calibrated = thickness_np * k
            volume_2d_calibrated = np.sum(thickness_calibrated) * pixel_area
            
            print(f"\nCalibration:")
            print(f"  Factor k: {k:.6f}")
            print(f"  2D Volume (calibrated): {volume_2d_calibrated:.2f} mm³")
            print(f"  Error: {abs(volume_2d_calibrated - true_volume_3d) / true_volume_3d * 100:.4f}%")
            
            # キャリブレート後の厚み統計
            thick_cal_nonzero = thickness_calibrated[thickness_calibrated > 1e-6]
            print(f"\nCalibrated thickness statistics:")
            print(f"  Range: {thick_cal_nonzero.min():.3f} - {thick_cal_nonzero.max():.3f} mm")
            print(f"  Mean: {thick_cal_nonzero.mean():.3f} mm")
            print(f"  Std: {thick_cal_nonzero.std():.3f} mm")
        else:
            thickness_calibrated = thickness_np
            volume_2d_calibrated = 0
    else:
        print("WARNING: No thickness detected in projection!")
        thickness_calibrated = thickness_np
        volume_2d_raw = 0
        volume_2d_calibrated = 0
    
    # 7) 可視化
    print("\n=== Step 7: Visualization ===")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 3D厚み真値（Z方向投影）
    im0 = axes[0, 0].imshow(thickness_z_true.T, cmap='hot', origin='lower')
    axes[0, 0].set_title(f'True Z-thickness\nMax: {thickness_z_true.max():.1f} mm')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], label='mm')
    
    # 生の2D投影
    im1 = axes[0, 1].imshow(thickness_np, cmap='hot')
    axes[0, 1].set_title(f'Raw DRR Thickness\nSum: {volume_2d_raw:.0f} mm³')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Value')
    
    # キャリブレート後
    if volume_2d_calibrated > 0:
        im2 = axes[0, 2].imshow(thickness_calibrated, cmap='hot')
        axes[0, 2].set_title(f'Calibrated Thickness\nSum: {volume_2d_calibrated:.0f} mm³')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], label='mm')
    else:
        axes[0, 2].text(0.5, 0.5, 'No calibration\n(no thickness detected)', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].axis('off')
    
    # ヒストグラム（真値）
    if len(thickness_z_nonzero) > 0:
        axes[1, 0].hist(thickness_z_nonzero, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('True Z-thickness Distribution')
        axes[1, 0].set_xlabel('Thickness (mm)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
    
    # ヒストグラム（投影）
    if np.any(thickness_np > 1e-6):
        axes[1, 1].hist(thickness_np[thickness_np > 1e-6], bins=50, alpha=0.7, color='blue')
        axes[1, 1].set_title('DRR Thickness Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
    
    # 体積比較
    volumes = [true_volume_3d, volume_2d_raw, volume_2d_calibrated]
    labels = ['3D True', '2D Raw', '2D Calibrated']
    colors = ['green', 'blue', 'red']
    bars = axes[1, 2].bar(labels, volumes, color=colors, alpha=0.7)
    axes[1, 2].set_title('Volume Comparison')
    axes[1, 2].set_ylabel('Volume (mm³)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, vol in zip(bars, volumes):
        if vol > 0:
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{vol:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "pure_geometric_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    
    # NumPy配列として保存
    np.save(output_dir / "thickness_raw.npy", thickness_np)
    np.save(output_dir / "thickness_calibrated.npy", thickness_calibrated)
    np.save(output_dir / "thickness_z_true.npy", thickness_z_true)
    
    plt.show()
    
    return thickness_calibrated, true_volume_3d

if __name__ == "__main__":
    thickness_map, true_volume = test_geometric_mask_projection_v2()
    print("\n=== Test Complete ===")