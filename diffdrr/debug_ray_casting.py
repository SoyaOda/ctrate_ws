#!/usr/bin/env python3
"""
デバッグ用レイキャスティング実装
DiffDRRで投影できることを確認しながら、純粋な幾何学的投影を実装
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import os
import matplotlib.pyplot as plt
import tempfile


def debug_diffdrr_projection():
    """DiffDRRでマスクを投影して座標系を確認"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # カメラパラメータ
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    print("\n" + "="*60)
    print("1. Load and analyze mask")
    print("="*60)
    
    # マスクデータを読み込み
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    mask_binary = (mask_data > 0).astype(np.float32)
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    
    print(f"Mask shape: {mask_binary.shape}")
    print(f"Voxel spacing: {voxel_spacing} mm")
    print(f"Non-zero voxels: {np.sum(mask_binary)}")
    
    # 3D体積
    voxel_volume = np.prod(voxel_spacing)
    volume_3d = np.sum(mask_binary) * voxel_volume
    print(f"3D Volume: {volume_3d:.2f} mm³")
    
    # Z軸方向の厚み分布を確認
    z_thickness = np.sum(mask_binary, axis=(0, 1))  # 各Z スライスでのマスクピクセル数
    max_z_thickness = np.max(z_thickness)
    print(f"Max thickness in Z direction: {max_z_thickness * voxel_spacing[2]:.2f} mm")
    
    print("\n" + "="*60)
    print("2. DiffDRR projection (baseline)")
    print("="*60)
    
    # DiffDRR用に一時ファイルを作成
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        mask_nifti_binary = nib.Nifti1Image(mask_binary, mask_nifti.affine)
        nib.save(mask_nifti_binary, temp_file.name)
        temp_path = temp_file.name
    
    # DiffDRRで投影
    mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
    print(f"DiffDRR volume shape: {mask_subject.volume.shape}")
    print(f"DiffDRR spacing: {mask_subject.spacing}")
    
    drr = DRR(
        mask_subject,
        sdd=sdd,
        height=height,
        width=width,
        delx=delx,
        dely=dely,
    ).to(device)
    
    # 投影
    transmittance = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")
    transmittance_np = transmittance.squeeze().cpu().numpy()
    
    # 厚みに変換
    thickness_diffdrr = -np.log(np.maximum(transmittance_np, 1e-9))
    
    print(f"Transmittance range: [{transmittance_np.min():.6f}, {transmittance_np.max():.6f}]")
    print(f"Thickness range: [{thickness_diffdrr.min():.3f}, {thickness_diffdrr.max():.3f}]")
    print(f"Non-zero pixels: {np.sum(thickness_diffdrr > 0.1)}")
    
    # 投影された領域の位置を確認
    mask_projected = thickness_diffdrr > 0.1
    if np.any(mask_projected):
        rows, cols = np.where(mask_projected)
        print(f"Projected region: rows [{rows.min()}, {rows.max()}], cols [{cols.min()}, {cols.max()}]")
        center_row = (rows.min() + rows.max()) // 2
        center_col = (cols.min() + cols.max()) // 2
        print(f"Center of projected region: ({center_row}, {center_col})")
    
    print("\n" + "="*60)
    print("3. Custom ray casting (debugging)")
    print("="*60)
    
    # カスタムレイキャスティング（シンプル版）
    # まず単純にZ軸方向の投影を試す
    simple_projection = np.sum(mask_binary, axis=2) * voxel_spacing[2]  # Z方向の厚み
    
    # 画像サイズにリサイズ（簡易的に中央部分を切り出し）
    h_orig, w_orig = simple_projection.shape
    h_start = max(0, (h_orig - height) // 2)
    w_start = max(0, (w_orig - width) // 2)
    h_end = min(h_orig, h_start + height)
    w_end = min(w_orig, w_start + width)
    
    simple_projection_resized = np.zeros((height, width))
    h_target_start = max(0, (height - (h_end - h_start)) // 2)
    w_target_start = max(0, (width - (w_end - w_start)) // 2)
    h_target_end = h_target_start + (h_end - h_start)
    w_target_end = w_target_start + (w_end - w_start)
    
    simple_projection_resized[h_target_start:h_target_end, w_target_start:w_target_end] = \
        simple_projection[h_start:h_end, w_start:w_end]
    
    print(f"Simple projection range: [{simple_projection_resized.min():.3f}, {simple_projection_resized.max():.3f}] mm")
    print(f"Simple projection non-zero: {np.sum(simple_projection_resized > 0)}")
    
    # 体積を計算
    pixel_area = delx * dely
    volume_simple = np.sum(simple_projection_resized) * pixel_area
    print(f"Simple projection volume: {volume_simple:.2f} mm³")
    
    print("\n" + "="*60)
    print("4. Save visualizations")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # DiffDRR投影（生の透過率）
    im0 = axes[0, 0].imshow(transmittance_np, cmap='gray')
    axes[0, 0].set_title(f'DiffDRR Transmittance\n[{transmittance_np.min():.3f}, {transmittance_np.max():.3f}]')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # DiffDRR厚み
    im1 = axes[0, 1].imshow(thickness_diffdrr, cmap='hot')
    axes[0, 1].set_title(f'DiffDRR Thickness\n[{thickness_diffdrr.min():.1f}, {thickness_diffdrr.max():.1f}]')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # DiffDRRマスク
    axes[0, 2].imshow(thickness_diffdrr > 0.1, cmap='gray')
    axes[0, 2].set_title('DiffDRR Mask (>0.1)')
    
    # シンプル投影
    im3 = axes[1, 0].imshow(simple_projection_resized, cmap='hot')
    axes[1, 0].set_title(f'Simple Z-projection\n[{simple_projection_resized.min():.1f}, {simple_projection_resized.max():.1f}] mm')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # シンプル投影マスク
    axes[1, 1].imshow(simple_projection_resized > 0, cmap='gray')
    axes[1, 1].set_title('Simple projection mask')
    
    # 元のマスクの中間スライス
    mid_slice = mask_binary.shape[2] // 2
    axes[1, 2].imshow(mask_binary[:, :, mid_slice], cmap='gray')
    axes[1, 2].set_title(f'Original mask (slice {mid_slice})')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'debug_comparison.png'), dpi=150)
    print(f"Saved to {os.path.join(output_dir, 'debug_comparison.png')}")
    
    # 一時ファイルを削除
    os.unlink(temp_path)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Original 3D volume: {volume_3d:.2f} mm³")
    print(f"Simple projection volume: {volume_simple:.2f} mm³")
    print(f"DiffDRR successfully projects the mask")
    print(f"Need to implement proper ray-casting with DiffDRR's coordinate system")


if __name__ == "__main__":
    debug_diffdrr_projection()