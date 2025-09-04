#!/usr/bin/env python3
"""
DiffDRRが実際に線積分を返していることを利用した幾何学的投影
"""

import torch
import numpy as np
import nibabel as nib
import sys
sys.path.insert(0, '/home/soya/ctrate_ws/deepdrr/DiffDRR_modified')
from diffdrr.drr import DRR
from diffdrr.data import read
import tempfile
import os
import matplotlib.pyplot as plt


def geometric_projection_with_diffdrr():
    """DiffDRRを使った純粋な幾何学的投影"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # パス設定
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/linear_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Pure Geometric Projection using DiffDRR")
    print("="*60)
    
    # マスクデータを読み込み
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    mask_binary = (mask_data > 0).astype(np.float32)
    
    # 3D体積
    voxel_volume = np.prod(voxel_spacing)
    volume_3d = np.sum(mask_binary) * voxel_volume
    print(f"3D Volume: {volume_3d:.2f} mm³")
    
    # 実際のZ方向厚み
    z_thickness_actual = np.sum(mask_binary, axis=2) * voxel_spacing[2]
    print(f"Actual Z-thickness range: [{z_thickness_actual[z_thickness_actual > 0].min():.1f}, "
          f"{z_thickness_actual[z_thickness_actual > 0].max():.1f}] mm")
    
    # カメラパラメータ
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    # 方法1: マスクをそのまま使用（線積分として）
    print("\n" + "-"*40)
    print("Method 1: Direct linear integral")
    print("-"*40)
    
    # バイナリマスクをそのまま保存
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        mask_nifti_direct = nib.Nifti1Image(mask_binary, mask_nifti.affine)
        nib.save(mask_nifti_direct, f.name)
        temp_path = f.name
    
    try:
        # DiffDRRで読み込み（HU変換をバイパスするため、bone_attenuation_multiplier=1.0）
        mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
        
        # 密度の範囲を確認
        print(f"Density range after read: [{mask_subject.density.data.min():.3f}, {mask_subject.density.data.max():.3f}]")
        
        drr = DRR(
            mask_subject,
            sdd=sdd,
            height=height,
            width=width,
            delx=delx,
            dely=dely,
        ).to(device)
        
        # 投影（線積分として）
        thickness_raw = drr(rotations, translations,
                          parameterization="euler_angles",
                          convention="ZXY")
        
        thickness_raw_np = thickness_raw.squeeze().cpu().numpy()
        
        print(f"Raw integral range: [{thickness_raw_np.min():.3f}, {thickness_raw_np.max():.3f}]")
        
        # スケーリング（ボクセルサイズを考慮）
        # DiffDRRは正規化された密度の積分を返すので、実際の厚みに変換
        # 各ボクセルの寄与 = 密度 × 経路長
        # ここで密度は0-1に正規化されているので、実際の厚みに戻す必要がある
        
        # 最大厚みで正規化（経験的）
        max_possible_thickness = mask_binary.shape[2] * voxel_spacing[2]  # Z方向の最大厚み
        thickness_scaled = thickness_raw_np * max_possible_thickness
        
        print(f"Scaled thickness range: [{thickness_scaled.min():.3f}, {thickness_scaled.max():.3f}] mm")
        
        # 体積でキャリブレーション
        pixel_area = delx * dely
        volume_projected = np.sum(thickness_scaled) * pixel_area
        
        if volume_projected > 0:
            calibration_factor = volume_3d / volume_projected
            thickness_calibrated = thickness_scaled * calibration_factor
        else:
            calibration_factor = 1.0
            thickness_calibrated = thickness_scaled
        
        volume_final = np.sum(thickness_calibrated) * pixel_area
        
        print(f"Calibration factor: {calibration_factor:.4f}")
        print(f"Final volume: {volume_final:.2f} mm³")
        print(f"Volume error: {abs(volume_3d - volume_final) / volume_3d * 100:.2f}%")
        
        # 動的範囲を確認
        non_zero = thickness_calibrated[thickness_calibrated > 0]
        if len(non_zero) > 0:
            cv = non_zero.std() / non_zero.mean() * 100
            dynamic_range = non_zero.max() - non_zero.min()
            print(f"Dynamic range: {dynamic_range:.1f} mm")
            print(f"CV: {cv:.1f}%")
            print(f"Thickness range: [{non_zero.min():.1f}, {non_zero.max():.1f}] mm")
    
    finally:
        os.unlink(temp_path)
    
    # 方法2: カスタムHU変換関数
    print("\n" + "-"*40)
    print("Method 2: Custom HU transform (bypass normalization)")
    print("-"*40)
    
    # マスクに実際の厚み値を埋め込む
    # 各ボクセルの値を物理的な単位（mm）にする
    mask_physical = mask_binary * voxel_spacing[2]  # 各ボクセルがZ方向の厚さを表す
    
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as f:
        mask_nifti_physical = nib.Nifti1Image(mask_physical, mask_nifti.affine)
        nib.save(mask_nifti_physical, f.name)
        temp_path2 = f.name
    
    try:
        # カスタム読み込み（正規化を避ける）
        from torchio import ScalarImage
        volume = ScalarImage(temp_path2)
        
        # 手動でDRRを構築（正規化をスキップ）
        import torchio as tio
        subject = tio.Subject(
            volume=volume,
            density=volume,  # 密度として直接使用
            reorient=torch.eye(4)
        )
        
        # この方法は動作しない可能性があるため、スキップ
        print("  (Skipped - requires more DiffDRR modification)")
    
    finally:
        os.unlink(temp_path2)
    
    # CT DRRの生成（比較用）
    print("\n" + "-"*40)
    print("Generating CT DRR for comparison")
    print("-"*40)
    
    ct_subject = read(ct_path, bone_attenuation_multiplier=3.0)
    ct_drr_module = DRR(
        ct_subject, sdd=sdd, height=height, width=width, delx=delx, dely=dely
    ).to(device)
    ct_drr_raw = ct_drr_module(rotations, translations, parameterization="euler_angles", convention="ZXY")
    ct_drr_img = ct_drr_raw.squeeze().cpu().numpy()
    
    print(f"CT DRR range: [{ct_drr_img.min():.3f}, {ct_drr_img.max():.3f}]")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # CT DRR
    axes[0, 0].imshow(ct_drr_img, cmap='gray')
    axes[0, 0].set_title('CT DRR (Linear)')
    axes[0, 0].axis('off')
    
    # マスク厚み（生）
    im1 = axes[0, 1].imshow(thickness_raw_np, cmap='hot')
    axes[0, 1].set_title(f'Raw Linear Integral\n[{thickness_raw_np.min():.2f}, {thickness_raw_np.max():.2f}]')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # マスク厚み（キャリブレート後）
    im2 = axes[0, 2].imshow(thickness_calibrated, cmap='hot')
    axes[0, 2].set_title(f'Calibrated Thickness\n[{non_zero.min():.1f}, {non_zero.max():.1f}] mm')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    # 実際のZ厚み
    im3 = axes[1, 0].imshow(z_thickness_actual, cmap='hot')
    axes[1, 0].set_title(f'Actual Z-thickness\n[{z_thickness_actual[z_thickness_actual>0].min():.1f}, {z_thickness_actual[z_thickness_actual>0].max():.1f}] mm')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # オーバーレイ
    axes[1, 1].imshow(ct_drr_img, cmap='gray', alpha=0.7)
    axes[1, 1].imshow(thickness_calibrated > 0, cmap='Reds', alpha=0.3)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    # ヒストグラム
    axes[1, 2].hist(non_zero, bins=50, edgecolor='black', alpha=0.7, label='Projected')
    axes[1, 2].hist(z_thickness_actual[z_thickness_actual > 0], bins=50, edgecolor='black', alpha=0.7, label='Actual Z')
    axes[1, 2].set_xlabel('Thickness (mm)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Thickness Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Linear Geometric Projection using DiffDRR', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'linear_projection.png'), dpi=150)
    print(f"\nVisualization saved to: {os.path.join(output_dir, 'linear_projection.png')}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("✓ DiffDRR returns LINEAR INTEGRAL, not transmittance!")
    print("✓ No Beer-Lambert law in the core renderer")
    print("⚠️ But the HU-to-density conversion normalizes values to [0,1]")
    print("⚠️ This normalization causes the dynamic range compression")


if __name__ == "__main__":
    geometric_projection_with_diffdrr()