#!/usr/bin/env python3
"""
DiffDRRを使った幾何学的マスク投影のテスト
Beer-Lambert則を回避し、純粋な線積分（厚み）を取得する実装
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
from diffdrr.pose import convert
import matplotlib.pyplot as plt
from pathlib import Path

def test_geometric_mask_projection():
    """
    Siddonレンダラとreducefn='sum'を使って、
    Beer-Lambert則を回避した純粋な厚み投影を実行
    """
    
    # GPU設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = Path("/home/soya/ctrate_ws/deepdrr/deepdrr/geometric_projection_output")
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
    
    # 2) DiffDRRのSubjectを作成（CTとマスクを一緒に読み込む）
    print("\n=== Step 2: Loading Data with DiffDRR ===")
    try:
        # CTとマスクを一緒に読み込む
        subject = read(
            volume=ct_path,
            labelmap=mask_path,  # マスクはlabelmapとして渡す
            orientation="PA",    # PAビューの撮影向き
            center_volume=True,
        )
        print("Subject loaded successfully")
        print(f"Volume shape: {subject.volume.shape}")
        print(f"Mask shape: {subject.mask.shape if hasattr(subject, 'mask') else 'N/A'}")
    except Exception as e:
        print(f"Error loading with labelmap: {e}")
        # 代替方法：マスクだけを読み込む
        subject = read(
            volume=mask_path,
            orientation="PA",
            center_volume=True,
        )
        print("Loaded mask as volume")
    
    # 3) DRRをSiddon・線積分（sum）で初期化
    print("\n=== Step 3: Initializing DRR with Siddon Renderer ===")
    drr = DRR(
        subject,
        sdd=1800.0,        # Source-to-detector distance
        height=512,
        width=512,
        delx=0.7,          # Pixel spacing in mm
        dely=0.7,
        renderer="siddon",  # Siddonレンダラを使用
        mode="nearest",     # バイナリマスクの補間を防ぐ
        reducefn="sum",     # 線積分（Beer-Lambertなし）
        filter_intersections_outside_volume=True,
    ).to(device)
    print("DRR initialized with Siddon renderer")
    
    # 4) ポーズの設定（PA view）
    print("\n=== Step 4: Setting Pose ===")
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device, dtype=torch.float32)
    
    # 5) 低レベルAPIを使用して厚みを直接取得
    print("\n=== Step 5: Rendering Thickness Map ===")
    try:
        # Detectorを使ってソースとターゲット位置を取得
        pose = convert(rotations, translations, parameterization="euler_angles", convention="ZXY")
        src, tgt = drr.detector(pose, calibration=None)
        
        # アフィン変換の逆変換
        src = drr.affine_inverse(src)
        tgt = drr.affine_inverse(tgt)
        
        # マスクデータを密度として使用
        if hasattr(subject, 'mask'):
            # labelmapとして読み込まれた場合
            mask_density = subject.mask.data.float().to(device)
        else:
            # volumeとして読み込まれた場合
            mask_density = subject.volume.data.float().to(device)
        
        # 厚み（線積分）を計算
        thickness = drr.renderer(mask_density, src, tgt)
        thickness_np = thickness.squeeze().detach().cpu().numpy()
        
        print(f"Thickness map shape: {thickness_np.shape}")
        print(f"Thickness range: {thickness_np.min():.3f} - {thickness_np.max():.3f} mm")
        print(f"Non-zero pixels: {np.sum(thickness_np > 0)}")
        
    except Exception as e:
        print(f"Error with low-level API: {e}")
        print("Trying alternative approach...")
        
        # 代替アプローチ：高レベルAPIを使用
        if hasattr(subject, 'mask'):
            # マスクを密度として設定
            subject.volume.data = subject.mask.data.float()
        
        thickness = drr(rotations, translations, 
                       parameterization="euler_angles", 
                       convention="ZXY")
        thickness_np = thickness.squeeze().detach().cpu().numpy()
        print(f"Alternative thickness map shape: {thickness_np.shape}")
    
    # 6) 体積の計算と比較
    print("\n=== Step 6: Volume Calculation ===")
    pixel_area = 0.7 * 0.7  # mm²
    volume_2d_raw = np.sum(thickness_np) * pixel_area
    
    print(f"2D Volume (raw): {volume_2d_raw:.2f} mm³")
    print(f"Ratio (2D/3D): {volume_2d_raw / true_volume_3d:.4f}")
    
    # 7) 単一スカラ係数による体積キャリブレーション
    print("\n=== Step 7: Volume Calibration ===")
    if volume_2d_raw > 0:
        k = true_volume_3d / volume_2d_raw
        thickness_calibrated = thickness_np * k
        volume_2d_calibrated = np.sum(thickness_calibrated) * pixel_area
        
        print(f"Calibration factor k: {k:.6f}")
        print(f"2D Volume (calibrated): {volume_2d_calibrated:.2f} mm³")
        print(f"Volume error: {abs(volume_2d_calibrated - true_volume_3d) / true_volume_3d * 100:.4f}%")
        
        # 厚み分布の統計
        print("\n=== Thickness Distribution ===")
        thick_nonzero = thickness_calibrated[thickness_calibrated > 0]
        if len(thick_nonzero) > 0:
            print(f"Calibrated thickness range: {thick_nonzero.min():.3f} - {thick_nonzero.max():.3f} mm")
            print(f"Mean thickness: {thick_nonzero.mean():.3f} mm")
            print(f"Std thickness: {thick_nonzero.std():.3f} mm")
            print(f"Unique thickness values: {len(np.unique(thick_nonzero))}")
    else:
        print("Warning: No thickness detected!")
        thickness_calibrated = thickness_np
    
    # 8) 結果の可視化
    print("\n=== Step 8: Visualization ===")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元の3Dマスク（中央スライス）
    mid_slice = mask_binary.shape[2] // 2
    axes[0, 0].imshow(mask_binary[:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title(f'3D Mask (slice {mid_slice})')
    axes[0, 0].axis('off')
    
    # 生の厚みマップ
    im1 = axes[0, 1].imshow(thickness_np, cmap='hot')
    axes[0, 1].set_title(f'Raw Thickness Map\nSum: {volume_2d_raw:.0f} mm³')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], label='Thickness (mm)')
    
    # キャリブレート後の厚みマップ
    im2 = axes[0, 2].imshow(thickness_calibrated, cmap='hot')
    axes[0, 2].set_title(f'Calibrated Thickness Map\nSum: {volume_2d_calibrated:.0f} mm³')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], label='Thickness (mm)')
    
    # 厚みのヒストグラム（生データ）
    if np.any(thickness_np > 0):
        axes[1, 0].hist(thickness_np[thickness_np > 0], bins=50, alpha=0.7, color='blue')
        axes[1, 0].set_title('Raw Thickness Distribution')
        axes[1, 0].set_xlabel('Thickness (mm)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 厚みのヒストグラム（キャリブレート後）
    if np.any(thickness_calibrated > 0):
        axes[1, 1].hist(thickness_calibrated[thickness_calibrated > 0], bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('Calibrated Thickness Distribution')
        axes[1, 1].set_xlabel('Thickness (mm)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 体積比較
    volumes = [true_volume_3d, volume_2d_raw, volume_2d_calibrated if volume_2d_raw > 0 else 0]
    labels = ['3D True', '2D Raw', '2D Calibrated']
    colors = ['green', 'blue', 'red']
    bars = axes[1, 2].bar(labels, volumes, color=colors, alpha=0.7)
    axes[1, 2].set_title('Volume Comparison')
    axes[1, 2].set_ylabel('Volume (mm³)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, vol in zip(bars, volumes):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{vol:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "geometric_thickness_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_path}")
    
    # NumPy配列として保存
    np.save(output_dir / "thickness_raw.npy", thickness_np)
    np.save(output_dir / "thickness_calibrated.npy", thickness_calibrated)
    
    plt.show()
    
    return thickness_calibrated, true_volume_3d

if __name__ == "__main__":
    thickness_map, true_volume = test_geometric_mask_projection()
    print("\n=== Test Complete ===")