#!/usr/bin/env python3
"""
ハイブリッド投影アプローチ
DiffDRRの座標系を利用しつつ、Beer-Lambert則を回避する実装
"""

import torch
import numpy as np
import nibabel as nib
from diffdrr.drr import DRR
from diffdrr.data import read
import os
import matplotlib.pyplot as plt
import tempfile
from scipy import ndimage


def create_thickness_encoded_volume(mask_data, voxel_spacing):
    """
    マスクボリュームを厚みエンコードボリュームに変換
    各ボクセルに、そのレイが通過する総厚みの情報を埋め込む
    """
    print("Creating thickness-encoded volume...")
    
    # Z方向の累積厚み
    mask_binary = (mask_data > 0).astype(np.float32)
    
    # 各Z位置での厚み（そのZ位置より下のマスクボクセル数）
    thickness_volume = np.zeros_like(mask_binary)
    
    for z in range(mask_binary.shape[2]):
        # Z方向の厚みを計算（このスライスから最後までのマスクの合計）
        thickness_at_z = np.sum(mask_binary[:, :, z:], axis=2) * voxel_spacing[2]
        thickness_volume[:, :, z] = thickness_at_z
    
    return thickness_volume


def hybrid_geometric_projection(
    mask_path: str,
    ct_path: str,
    output_dir: str,
    device: torch.device
):
    """
    ハイブリッド方式による幾何学的投影
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # カメラパラメータ
    sdd = 1800.0
    height, width = 512, 512
    delx, dely = 0.7, 0.7
    
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, 850.0, 0.0]], device=device)
    
    print("\n" + "="*60)
    print("Hybrid Geometric Projection")
    print("="*60)
    
    # ========================================
    # 1. データ読み込み
    # ========================================
    print("\nStep 1: Load data")
    print("-"*40)
    
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    voxel_spacing = mask_nifti.header.get_zooms()[:3]
    
    mask_binary = (mask_data > 0).astype(np.float32)
    
    print(f"Mask shape: {mask_binary.shape}")
    print(f"Voxel spacing: {voxel_spacing} mm")
    
    # 3D体積
    voxel_volume = np.prod(voxel_spacing)
    volume_3d = np.sum(mask_binary) * voxel_volume
    print(f"3D Volume: {volume_3d:.2f} mm³")
    
    # ========================================
    # 2. 方法1: 線形スケーリング
    # ========================================
    print("\nStep 2: Method 1 - Linear scaling of DiffDRR output")
    print("-"*40)
    
    # DiffDRR用に一時ファイルを作成
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        # マスクを異なる値でスケール（1.0ではなく0.01など）
        # これによりBeer-Lambert則の影響を軽減
        mask_scaled = mask_binary * 0.01
        mask_nifti_scaled = nib.Nifti1Image(mask_scaled, mask_nifti.affine)
        nib.save(mask_nifti_scaled, temp_file.name)
        temp_path = temp_file.name
    
    # DiffDRRで投影
    mask_subject = read(temp_path, bone_attenuation_multiplier=1.0)
    
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
    
    # 線形変換で厚みを推定
    # 透過率 I = exp(-μ*t) の場合、小さいμでは I ≈ 1 - μ*t
    # したがって、t ≈ (1 - I) / μ
    mu = 0.01  # スケーリング係数
    thickness_linear = (1.0 - transmittance_np) / mu
    thickness_linear = np.maximum(thickness_linear, 0)
    
    print(f"Linear thickness range: [{thickness_linear.min():.3f}, {thickness_linear.max():.3f}] mm")
    
    # 体積を計算
    pixel_area = delx * dely
    volume_linear = np.sum(thickness_linear) * pixel_area
    print(f"Linear method volume: {volume_linear:.2f} mm³")
    
    # キャリブレーション
    if volume_linear > 0:
        calibration_factor = volume_3d / volume_linear
        thickness_calibrated = thickness_linear * calibration_factor
        volume_calibrated = np.sum(thickness_calibrated) * pixel_area
        print(f"Calibration factor: {calibration_factor:.4f}")
        print(f"Calibrated volume: {volume_calibrated:.2f} mm³")
    else:
        thickness_calibrated = thickness_linear
    
    # ========================================
    # 3. 方法2: 多層投影の統合
    # ========================================
    print("\nStep 3: Method 2 - Multi-layer projection")
    print("-"*40)
    
    # マスクを複数の薄いスライスに分割して投影
    num_layers = 5
    layer_thickness = mask_binary.shape[2] // num_layers
    
    layer_projections = []
    
    for i in range(num_layers):
        z_start = i * layer_thickness
        z_end = min((i + 1) * layer_thickness, mask_binary.shape[2])
        
        # レイヤーを作成
        layer_mask = np.zeros_like(mask_binary)
        layer_mask[:, :, z_start:z_end] = mask_binary[:, :, z_start:z_end]
        
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
            layer_scaled = layer_mask * 0.01
            layer_nifti = nib.Nifti1Image(layer_scaled, mask_nifti.affine)
            nib.save(layer_nifti, temp_file.name)
            layer_path = temp_file.name
        
        # DiffDRRで投影
        layer_subject = read(layer_path, bone_attenuation_multiplier=1.0)
        
        drr_layer = DRR(
            layer_subject,
            sdd=sdd,
            height=height,
            width=width,
            delx=delx,
            dely=dely,
        ).to(device)
        
        # 投影
        trans_layer = drr_layer(rotations, translations, parameterization="euler_angles", convention="ZXY")
        trans_layer_np = trans_layer.squeeze().cpu().numpy()
        
        # 厚みに変換
        thickness_layer = (1.0 - trans_layer_np) / 0.01
        layer_projections.append(thickness_layer)
        
        # 一時ファイル削除
        os.unlink(layer_path)
        
        print(f"  Layer {i+1}/{num_layers}: thickness range [{thickness_layer.min():.3f}, {thickness_layer.max():.3f}]")
    
    # レイヤーを合成
    thickness_multilayer = np.sum(layer_projections, axis=0)
    
    print(f"Multi-layer thickness range: [{thickness_multilayer.min():.3f}, {thickness_multilayer.max():.3f}] mm")
    
    # 体積を計算
    volume_multilayer = np.sum(thickness_multilayer) * pixel_area
    print(f"Multi-layer volume: {volume_multilayer:.2f} mm³")
    
    # ========================================
    # 4. CT DRRの生成（比較用）
    # ========================================
    print("\nStep 4: Generate CT DRR for comparison")
    print("-"*40)
    
    ct_subject = read(ct_path, bone_attenuation_multiplier=3.0)
    ct_drr_module = DRR(
        ct_subject, sdd=sdd, height=height, width=width, delx=delx, dely=dely
    ).to(device)
    ct_drr_raw = ct_drr_module(rotations, translations, parameterization="euler_angles", convention="ZXY")
    ct_drr_img = -torch.log(torch.clamp(ct_drr_raw, 1e-9)).squeeze().cpu().numpy()
    
    print(f"CT DRR range: [{ct_drr_img.min():.3f}, {ct_drr_img.max():.3f}]")
    
    # ========================================
    # 5. 結果の可視化
    # ========================================
    print("\nStep 5: Visualization")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # CT DRR
    axes[0, 0].imshow(ct_drr_img, cmap='gray')
    axes[0, 0].set_title('CT DRR')
    axes[0, 0].axis('off')
    
    # 線形法の厚みマップ
    im1 = axes[0, 1].imshow(thickness_calibrated, cmap='hot')
    axes[0, 1].set_title(f'Linear Method\nMax: {thickness_calibrated.max():.1f} mm')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # 線形法のオーバーレイ
    axes[0, 2].imshow(ct_drr_img, cmap='gray', alpha=0.7)
    axes[0, 2].imshow(thickness_calibrated > 0, cmap='Reds', alpha=0.3)
    axes[0, 2].set_title('Linear Method Overlay')
    axes[0, 2].axis('off')
    
    # 多層法の厚みマップ
    im3 = axes[1, 0].imshow(thickness_multilayer, cmap='hot')
    axes[1, 0].set_title(f'Multi-layer Method\nMax: {thickness_multilayer.max():.1f} mm')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # 多層法のオーバーレイ
    axes[1, 1].imshow(ct_drr_img, cmap='gray', alpha=0.7)
    axes[1, 1].imshow(thickness_multilayer > 0, cmap='Blues', alpha=0.3)
    axes[1, 1].set_title('Multi-layer Method Overlay')
    axes[1, 1].axis('off')
    
    # 比較
    diff = thickness_calibrated - thickness_multilayer
    im5 = axes[1, 2].imshow(diff, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1, 2].set_title('Difference (Linear - Multi-layer)')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_results.png'), dpi=150)
    print(f"Saved to {os.path.join(output_dir, 'hybrid_results.png')}")
    
    # データ保存
    np.save(os.path.join(output_dir, 'thickness_linear.npy'), thickness_calibrated)
    np.save(os.path.join(output_dir, 'thickness_multilayer.npy'), thickness_multilayer)
    np.save(os.path.join(output_dir, 'ct_drr.npy'), ct_drr_img)
    
    # 一時ファイル削除
    os.unlink(temp_path)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original 3D volume: {volume_3d:.2f} mm³")
    print(f"Linear method (calibrated): {volume_calibrated:.2f} mm³")
    print(f"Multi-layer method: {volume_multilayer:.2f} mm³")
    print(f"Results saved to: {output_dir}")
    
    return {
        'thickness_linear': thickness_calibrated,
        'thickness_multilayer': thickness_multilayer,
        'ct_drr': ct_drr_img,
        'volume_3d': volume_3d
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ct_path = "/home/soya/ctrate_ws/data/CT-RATE-v2/dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz"
    mask_path = "/home/soya/ctrate_ws/outputs/eat_pat_v5_4_valid_1_a_1/masks/eat_pat.nii.gz"
    output_dir = "/home/soya/ctrate_ws/deepdrr/hybrid_output"
    
    results = hybrid_geometric_projection(mask_path, ct_path, output_dir, device)


if __name__ == "__main__":
    main()